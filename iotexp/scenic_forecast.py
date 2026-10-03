"""Auditable NEW forecasting experiments for ScenicIoT-Rebuild.

This is an independently specified implementation, not a recovered E-STGNN.
All fitted transforms use training days; validation alone selects checkpoints,
ridge regularization, and the predictor frozen for the control experiments.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .telemetry import TrainingStats
from .util import write_json


DEFAULTS = {
    "models": ["persistence", "historical_average", "ridge", "lstm", "gru", "edge_stgru"],
    "seeds": [0, 1, 2], "hidden": 32, "epochs": 50, "patience": 8,
    "batch_size": 128, "learning_rate": 0.002, "weight_decay": 0.0001,
    "ridge_alphas": [0.1, 1.0, 10.0, 100.0], "selected_model": "validation_best", "selected_seed": None,
    "device": "cpu", "torch_threads": 1,
}


def forecast_config(cfg):
    result = dict(DEFAULTS)
    result.update(cfg.get("forecaster", {}))
    result["history"] = int(cfg.get("scenic", cfg).get("history", 12))
    result["horizons"] = list(cfg.get("scenic", cfg).get("horizons", [1, 2, 4, 6]))
    result["period"] = int(cfg.get("scenic", cfg).get("episode_steps", 72))
    if not result["models"] or any(s not in DEFAULTS["models"] for s in result["models"]):
        raise ValueError("Unknown or empty forecasting model list")
    if result["history"] < 1 or min(result["horizons"]) < 1:
        raise ValueError("Forecast history/horizons must be positive")
    if result["selected_model"] != "validation_best" and result["selected_model"] not in result["models"]:
        raise ValueError("selected_model must be trained or validation_best")
    return result


def split_day_indices(dataset):
    split = dataset["split_days"]
    if isinstance(split, dict):
        if all(np.isscalar(split[k]) for k in ("train", "validation", "test")):
            split = [split[k] for k in ("train", "validation", "test")]
        else:
            result = {k: np.asarray(split[k], dtype=int) for k in ("train", "validation", "test")}
            merged = np.concatenate(list(result.values()))
            if len(np.unique(merged)) != len(merged):
                raise ValueError("Forecast splits overlap")
            return result
    a, b, c = map(int, split)
    if min(a, b, c) < 1 or a + b + c != len(dataset["inflow"]):
        raise ValueError("Invalid day split")
    return {"train": np.arange(a), "validation": np.arange(a, a+b), "test": np.arange(a+b, a+b+c)}


def fit_training_stats(dataset, days, period):
    observations = dataset.get("observations")
    if observations is None:
        observations = np.asarray(dataset["inflow"])[..., None]
    observations = np.asarray(observations)
    samples = observations[days].reshape(-1, observations.shape[-2], observations.shape[-1])
    times = np.tile(np.asarray(dataset["times"], dtype=int), len(days))
    return TrainingStats.fit(samples, times, period)


def build_windows(dataset, days, cfg, stats):
    """One sample per day and eligible decision origin, without cross-day windows."""
    flows = np.asarray(dataset["inflow"], dtype=np.float32)
    times = np.asarray(dataset["times"], dtype=int)
    history, horizons = cfg["history"], np.asarray(cfg["horizons"], dtype=int)
    indices = [i for i, t in enumerate(times)
               if t >= 0 and i >= history-1 and i+max(horizons) < len(times)]
    if not indices or not np.array_equal(np.diff(times), np.ones(len(times)-1, dtype=int)):
        raise ValueError("Need a consecutive time grid with valid forecast targets")
    x, y, day_ids, origins = [], [], [], []
    for day in days:
        for i in indices:
            x.append(flows[day, i-history+1:i+1])
            y.append(flows[day, i+horizons].T)
            day_ids.append(day)
            origins.append(times[i])
    x, y = np.asarray(x), np.asarray(y)
    mean, scale = stats.mean[:, 0].astype(np.float32), stats.scale[:, 0].astype(np.float32)
    return {"x": ((x-mean)/scale).astype(np.float32), "y": y,
            "yn": ((y-mean[None, :, None])/scale[None, :, None]).astype(np.float32),
            "day": np.asarray(day_ids), "origin": np.asarray(origins),
            "history_times": np.asarray(origins)[:, None]-np.arange(history-1, -1, -1)[None]}


def edge_arrays(dataset):
    """Undirected physical edges plus self loops; lengths are normalized by median."""
    adjacency = np.asarray(dataset["adjacency"], dtype=float)
    n = len(adjacency)
    mask = (adjacency > 0) | (adjacency.T > 0) | np.eye(n, dtype=bool)
    distances = np.ones((n, n), dtype=np.float32)
    if "edge_index" in dataset and "edge_lengths_m" in dataset:
        edges = np.asarray(dataset["edge_index"], dtype=int)
        lengths = np.asarray(dataset["edge_lengths_m"], dtype=float)
        if edges.shape != (len(lengths), 2) or (lengths <= 0).any():
            raise ValueError("Expected [E,2] physical edges with positive lengths")
        median = np.median(lengths)
        for (i, j), length in zip(edges, lengths):
            distances[i, j] = distances[j, i] = length / median
    np.fill_diagonal(distances, 0)
    return mask, np.log1p(distances).astype(np.float32)


class TemporalForecaster(nn.Module):
    """Shared node GRU/LSTM, with an optional scalar dynamic edge-attention channel.

    a_ij(t) = softmax_j(LeakyReLU(q*z_i(t)+k*z_j(t)+w*log(1+d_ij)+b)).
    The normalized inflow and known sin/cos time are recurrent inputs. The graph
    version adds sum_j a_ij*z_j; no graph version keeps every other component.
    Outputs are four simultaneous residuals from the last observed inflow.
    """
    def __init__(self, name, hidden, horizons, edge_mask, edge_lengths):
        super().__init__()
        if name not in ("lstm", "gru", "edge_stgru"):
            raise ValueError(name)
        self.name, self.hidden, self.horizons = name, int(hidden), int(horizons)
        self.register_buffer("edge_mask", torch.as_tensor(edge_mask, dtype=torch.bool))
        self.register_buffer("edge_lengths", torch.as_tensor(edge_lengths, dtype=torch.float32))
        if name == "edge_stgru":
            self.edge_scores = nn.Parameter(torch.tensor([0.1, 0.1, -0.1, 0.0]))
        recurrent = nn.LSTM if name == "lstm" else nn.GRU
        self.recurrent = recurrent(4 if name == "edge_stgru" else 3, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizons)

    def forward(self, x, clock):
        b, length, n = x.shape
        clock = clock[:, :, None, :].expand(b, length, n, 2)
        channels = [x[..., None], clock]
        if self.name == "edge_stgru":
            q, k, w, offset = self.edge_scores
            logits = q*x[..., :, None]+k*x[..., None, :]+w*self.edge_lengths+offset
            logits = torch.nn.functional.leaky_relu(logits, negative_slope=0.2)
            weights = torch.softmax(logits.masked_fill(~self.edge_mask, -1e9), dim=-1)
            neighbor = (weights*x[..., None, :]).sum(-1)
            channels.append(neighbor[..., None])
        features = torch.cat(channels, dim=-1).permute(0, 2, 1, 3).reshape(b*n, length, -1)
        sequence, _ = self.recurrent(features)
        result = self.head(sequence[:, -1]).reshape(b, n, self.horizons)
        return result+x[:, -1, :, None]


def _clock(history_times, period):
    angle = 2*np.pi*np.asarray(history_times)/period
    return np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)


def _ridge_features(windows, period):
    angle = 2*np.pi*windows["origin"]/period
    return np.column_stack((windows["x"].reshape(len(angle), -1), np.sin(angle), np.cos(angle), np.ones(len(angle))))


def _unscale(values, stats):
    return np.maximum(0, np.asarray(values)*stats.scale[None, :, 0, None]+stats.mean[None, :, 0, None]).astype(np.float32)


def _predict_neural(model, windows, stats, cfg):
    device = next(model.parameters()).device
    prediction = []
    clock = _clock(windows["history_times"], cfg["period"])
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(windows["x"]), cfg["batch_size"]):
            x = torch.as_tensor(windows["x"][i:i+cfg["batch_size"]], device=device)
            c = torch.as_tensor(clock[i:i+cfg["batch_size"]], device=device)
            prediction.append(model(x, c).cpu().numpy())
    return _unscale(np.concatenate(prediction), stats)


def _save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scores(truth, prediction):
    error = prediction-truth
    denominator = float(np.abs(truth).sum())
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))),
            "wape_pct": float(100*np.abs(error).sum()/denominator) if denominator else None,
            "observations": int(truth.size)}


def _metrics(rows, daily, name, seed, split, windows, prediction, cfg):
    for j, horizon in enumerate(cfg["horizons"]):
        descriptor = {"model": name, "seed": seed, "split": split, "horizon_steps": horizon}
        rows.append({**descriptor, **_scores(windows["y"][:, :, j], prediction[:, :, j])})
        for day in np.unique(windows["day"]):
            mask = windows["day"] == day
            daily.append({**descriptor, "day": int(day),
                          **_scores(windows["y"][mask, :, j], prediction[mask, :, j])})


def _train_neural(name, seed, train, validation, stats, cfg, edge_mask, edge_lengths):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TemporalForecaster(name, cfg["hidden"], len(cfg["horizons"]), edge_mask, edge_lengths).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    x = torch.as_tensor(train["x"], device=cfg["device"])
    target = torch.as_tensor(train["yn"], device=cfg["device"])
    clock = torch.as_tensor(_clock(train["history_times"], cfg["period"]), device=cfg["device"])
    rng = np.random.default_rng(seed)
    best_score, best_epoch, best_state, logs = float("inf"), 0, None, []
    started = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"])+1):
        model.train()
        order, total_loss = rng.permutation(len(x)), 0.0
        for begin in range(0, len(x), cfg["batch_size"]):
            indices = torch.as_tensor(order[begin:begin+cfg["batch_size"]], device=cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            estimate = model(x[indices], clock[indices])
            loss = torch.nn.functional.mse_loss(estimate, target[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite forecast loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        prediction = _predict_neural(model, validation, stats, cfg)
        score = float(np.mean(np.abs(prediction-validation["y"])))
        logs.append({"model": name, "seed": seed, "epoch": epoch,
                     "train_standardized_mse": total_loss/len(x), "validation_mae": score,
                     "elapsed_seconds": time.perf_counter()-started})
        if score < best_score-1e-7:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch-best_epoch >= cfg["patience"]:
            break
    if best_state is None:
        raise ValueError("At least one forecasting epoch is required")
    model.load_state_dict(best_state)
    model.eval()
    return model, logs, {"best_epoch": best_epoch, "validation_mae": best_score,
                         "training_seconds": time.perf_counter()-started,
                         "parameters": sum(p.numel() for p in model.parameters())}


def prepare_forecaster(dataset, cfg, output_dir):
    """Train baselines, export predictions and diagnostics, return frozen winner.

    Output files live in output_dir/forecast/. ``training_stats.npz`` is also at
    output_dir for compatibility with the environment/telemetry interface.
    No simulation or forecasting result from the lost implementation is used.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    folder = output/"forecast"
    folder.mkdir(exist_ok=True)
    fc = forecast_config(cfg)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(int(fc["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    days = split_day_indices(dataset)
    stats = fit_training_stats(dataset, days["train"], fc["period"])
    stats.save(output/"training_stats.npz")
    windows = {split: build_windows(dataset, ids, fc, stats) for split, ids in days.items()}
    edge_mask, lengths = edge_arrays(dataset)
    records, daily, logs, models, selections = [], [], [], {}, []
    write_json(folder/"configuration.json", fc)
    write_json(folder/"provenance.json", {
        "implementation": "new independent ScenicIoT-Rebuild forecasting experiment; not recovered E-STGNN",
        "inflow_sha256": hashlib.sha256(np.asarray(dataset["inflow"]).tobytes()).hexdigest(),
        "fit_days": days["train"].tolist(), "validation_days": days["validation"].tolist(),
        "test_days": days["test"].tolist(), "history": fc["history"], "horizons": fc["horizons"],
        "samples_per_split": {s: len(w["x"]) for s, w in windows.items()},
        "normalization": "node/channel training-day mean and sample standard deviation",
        "target": "future node inflow, visits per 10-minute interval",
        "objective": "standardized MSE; checkpoint and final model chosen by validation raw MAE",
        "test_usage": "metrics only, no tuning or model selection",
        "neural_inputs": "12 inflow observations plus deterministic sin/cos clock; edge_stgru adds attention-weighted neighbors",
        "graph_free_comparator": "identical GRU hidden/head and clock; removes neighbor channel and four edge-score parameters",
        "negative_predictions": "clipped to zero before metrics and downstream use",
    })
    for split in ("validation", "test"):
        np.savez_compressed(folder/f"{split}_targets.npz", truth=windows[split]["y"],
                            day=windows[split]["day"], origin=windows[split]["origin"],
                            horizons=np.asarray(fc["horizons"]))
    for name in fc["models"]:
        seeds = fc["seeds"] if name in ("lstm", "gru", "edge_stgru") else [-1]
        for seed in seeds:
            key = f"{name}_seed{seed}"
            predictions = {}
            details = {"model": name, "seed": int(seed)}
            if name == "persistence":
                model = {"kind": name}
                for split in ("validation", "test"):
                    raw = _unscale(np.repeat(windows[split]["x"][:, -1, :, None], len(fc["horizons"]), axis=-1), stats)
                    predictions[split] = raw
                details.update(parameters=0, training_seconds=0)
                np.savez_compressed(folder/f"{key}.npz", kind=name)
            elif name == "historical_average":
                values = np.asarray(dataset["inflow"])[days["train"]].mean(axis=0)
                model = {"kind": name, "values": values, "first_time": int(dataset["times"][0])}
                for split in ("validation", "test"):
                    indices = windows[split]["origin"][:, None]+np.asarray(fc["horizons"])[None]-model["first_time"]
                    predictions[split] = values[indices].transpose(0, 2, 1).astype(np.float32)
                details.update(parameters=int(values.size), training_seconds=0)
                np.savez_compressed(folder/f"{key}.npz", kind=name, values=values, first_time=model["first_time"])
            elif name == "ridge":
                begin = time.perf_counter()
                x = _ridge_features(windows["train"], fc["period"])
                y = windows["train"]["yn"].reshape(len(x), -1)
                xv = _ridge_features(windows["validation"], fc["period"])
                gram, xy = x.T@x, x.T@y
                best = float("inf")
                for alpha in fc["ridge_alphas"]:
                    penalty = np.eye(x.shape[1])*alpha
                    penalty[-1, -1] = 0
                    coef = np.linalg.solve(gram+penalty, xy)
                    estimate = _unscale((xv@coef).reshape(windows["validation"]["y"].shape), stats)
                    score = float(np.mean(np.abs(estimate-windows["validation"]["y"])))
                    if score < best:
                        best, best_coef, best_alpha = score, coef, alpha
                model = {"kind": name, "coefficients": best_coef}
                for split in ("validation", "test"):
                    predictions[split] = _unscale((_ridge_features(windows[split], fc["period"])@best_coef).reshape(windows[split]["y"].shape), stats)
                details.update(parameters=int(best_coef.size), alpha=best_alpha, training_seconds=time.perf_counter()-begin)
                np.savez_compressed(folder/f"{key}.npz", kind=name, coefficients=best_coef)
            else:
                network, model_logs, info = _train_neural(name, int(seed), windows["train"], windows["validation"], stats, fc, edge_mask, lengths)
                model = {"kind": name, "network": network}
                logs.extend(model_logs)
                details.update(info)
                for split in ("validation", "test"):
                    predictions[split] = _predict_neural(network, windows[split], stats, fc)
                torch.save({"model_name": name, "state_dict": network.cpu().state_dict(), "hidden": fc["hidden"],
                            "horizons": len(fc["horizons"]), "edge_mask": torch.as_tensor(edge_mask),
                            "edge_lengths": torch.as_tensor(lengths), "seed": seed}, folder/f"{key}.pt")
            model["config"] = fc
            models[key] = model
            details["validation_mae"] = float(np.mean(np.abs(predictions["validation"]-windows["validation"]["y"])))
            details["checkpoint"] = f"{key}.pt" if name in ("lstm", "gru", "edge_stgru") else f"{key}.npz"
            selections.append(details)
            for split, prediction in predictions.items():
                _metrics(records, daily, name, int(seed), split, windows[split], prediction, fc)
                np.savez_compressed(folder/f"{key}_{split}_predictions.npz", prediction=prediction)
            # Incremental outputs survive an interrupted training run.
            write_json(folder/"model_details.json", selections)
            _save_csv(folder/"metrics.csv", records)
            _save_csv(folder/"daily_metrics.csv", daily)
            _save_csv(folder/"training_log.csv", logs)
            print(f"forecast {key}: validation MAE={details['validation_mae']:.4f}", flush=True)
    # Aggregate by family before choosing a family; select its seed on validation.
    candidates = [r for r in selections if fc["selected_model"] == "validation_best" or r["model"] == fc["selected_model"]]
    family_scores = {name: float(np.mean([r["validation_mae"] for r in candidates if r["model"] == name]))
                     for name in sorted(set(r["model"] for r in candidates))}
    family = min(family_scores, key=family_scores.get)
    finalists = [r for r in candidates if r["model"] == family and
                 (fc["selected_seed"] is None or r["seed"] == fc["selected_seed"])]
    if not finalists:
        raise ValueError("Requested frozen forecast seed was not trained")
    winner = min(finalists, key=lambda r: r["validation_mae"])
    selected = {**winner, "selection": fc["selected_model"], "family_validation_mae": family_scores,
                "frozen_for_control": True}
    write_json(folder/"selected.json", selected)
    frozen = models[f"{winner['model']}_seed{winner['seed']}"]
    # Single-window inference uses only observations, on CPU like PPO control.
    from types import SimpleNamespace
    raw = windows["test"]["x"][0]*stats.scale[:, 0]+stats.mean[:, 0]
    example = SimpleNamespace(values=raw[..., None], decision_time=int(windows["test"]["origin"][0]))
    for _ in range(10):
        forecast_window(frozen, example, stats)
    timings = []
    for _ in range(100):
        begin = time.perf_counter()
        forecast_window(frozen, example, stats)
        timings.append(1000*(time.perf_counter()-begin))
    write_json(folder/"inference_latency.json", {
        "model": winner["model"], "seed": winner["seed"], "device": "cpu",
        "torch_threads": torch.get_num_threads(), "python": platform.python_version(),
        "processor": platform.processor(), "torch": torch.__version__,
        "warmup_calls": 10, "measured_calls": 100, "batch_size": 1,
        "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
        "minimum_ms": float(np.min(timings)), "scope": "forecast_window only; excludes telemetry, communication and policy inference",
    })
    return stats, frozen


def load_forecaster(output_dir):
    output = Path(output_dir)
    folder = output/"forecast"
    stats = TrainingStats.load(output/"training_stats.npz")
    cfg = json.loads((folder/"configuration.json").read_text(encoding="utf-8"))
    selected = json.loads((folder/"selected.json").read_text(encoding="utf-8"))
    name = selected["model"]
    path = folder/selected["checkpoint"]
    if name in ("lstm", "gru", "edge_stgru"):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        network = TemporalForecaster(name, saved["hidden"], saved["horizons"], saved["edge_mask"], saved["edge_lengths"])
        network.load_state_dict(saved["state_dict"])
        network.eval()
        model = {"kind": name, "network": network}
    else:
        with np.load(path, allow_pickle=False) as z:
            model = {k: z[k].copy() for k in z.files if k != "kind"}
        model["kind"] = name
    model["config"] = cfg
    return stats, model


def forecast_window(model, window, stats, cfg=None):
    """Only a causal telemetry Window is accepted; no simulator state is read."""
    fc = model["config"]
    values = np.asarray(window.values)[-fc["history"]:, :, 0]
    if values.shape != (fc["history"], len(stats.mean)) or not np.isfinite(values).all():
        raise ValueError("Invalid forecast observation history")
    x = ((values-stats.mean[:, 0])/stats.scale[:, 0]).astype(np.float32)[None]
    times = np.arange(window.decision_time-fc["history"]+1, window.decision_time+1)[None]
    windows = {"x": x, "origin": np.asarray([window.decision_time]), "history_times": times}
    name = model["kind"]
    if name == "persistence":
        return np.repeat(values[-1, :, None], len(fc["horizons"]), axis=-1).astype(np.float32)
    if name == "historical_average":
        indices = window.decision_time+np.asarray(fc["horizons"])-int(model["first_time"])
        # Beyond closing, use the final operating-bin expectation, explicitly.
        return model["values"][np.clip(indices, 0, len(model["values"])-1)].T.astype(np.float32)
    if name == "ridge":
        normalized = (_ridge_features(windows, fc["period"])@model["coefficients"]).reshape(1, len(stats.mean), -1)
        return _unscale(normalized, stats)[0]
    return _predict_neural(model["network"], windows, stats, fc)[0]


def main():
    import argparse
    from .scenic import build_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    prepare_forecaster(build_dataset(config), config, args.output)


if __name__ == "__main__":
    main()
