"""V3 forecast benchmark: author-architecture adaptations and controlled ablations.

This module leaves the frozen v2 implementation/results untouched. Graph WaveNet,
STID and iTransformer are adaptations of pinned author sources recorded in the
run's source/source_manifest.json; see architecture_notes() for differences.
They are not claims to reproduce the authors' original benchmark numbers.
"""
from __future__ import annotations

import copy
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
from torch.nn import functional as F

from .scenic import build_dataset
from .scenic_forecast import (
    TemporalForecaster, _clock, _ridge_features, _scores, _unscale,
    build_windows, edge_arrays, fit_training_stats, split_day_indices,
)
from .telemetry import TrainingStats


NEURAL_MODELS = ["lstm", "gru", "edge_stgru", "graph_wavenet", "stid",
                 "itransformer", "uniform_neighbor", "no_edge_length"]
CANDIDATES = [
    {"id": "C0", "hidden": 32, "learning_rate": 0.002},
    {"id": "C1", "hidden": 64, "learning_rate": 0.001},
    {"id": "C2", "hidden": 32, "learning_rate": 0.0005},
]
FINAL_DATASET_SEEDS = [20261001, 20261002, 20261003]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, content, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def architecture_notes():
    return {
        "common_information": "12 historical inflow values at each of 24 nodes; historical operating-time sin/cos; known node identities and fixed physical graph. No day-of-week, weather, event indicators, hidden intensity, future labels or queue/resource variables are forecast inputs.",
        "not_capacity_matched": "The available observation tensor is common, but architectures use different subsets/connectivity and parameter counts. Nodewise LSTM/GRU and spatial ablations are mechanism controls, not equally connected global models.",
        "edge_stgru": "Unchanged v2 TemporalForecaster architecture; only the shared validation hyperparameter protocol changes. Four scalar attention scores, physical neighbors/self, edge-length log feature, one GRU, residual four-horizon head. Not the lost E-STGNN.",
        "uniform_neighbor": "Same four-channel GRU/head as EdgeSTGRU; replace learned attention with a uniform mean over the same physical neighbors and self. No attention score parameters.",
        "no_edge_length": "Same EdgeSTGRU with q,k,b attention scores, omitting w*log(1+distance); physical graph remains. This is an edge-feature ablation, not removal of graph information.",
        "graph_wavenet": "Pinned author architecture: 4 blocks x 2 gated temporal layers, dilation 1/2 reset per block (receptive field 13, left padding for 12 observations), residual/skip paths, BatchNorm, order-2 graph diffusion. One fixed row-normalized physical adjacency plus learned rank-10 softmax(ReLU(E1E2)) support. Input channels inflow/sin/cos; four direct target channels [1,2,4,6]. residual/dilation width=hidden, skip=2*hidden, end=4*hidden, dropout=.1. Replace legacy Conv1d receiving 4D tensors by dimensionally equivalent Conv2d and omit the final post-skip diffusion/BatchNorm branch, whose output is unused in the author forward function. Common unmasked standardized MSE replaces author's masked MAE that excludes valid zero labels.",
        "stid": "Pinned author architecture: shared 1x1 history embedding; learned node identity (8 dims), operating-time identity (72 slots, 8 dims), three residual MLP blocks with dropout=.15, 1x1 horizon head. History embedding receives inflow/sin/cos, explicitly adapting the author's dataset-dependent feature channels. Last historical origin selects time ID; day-of-week embedding disabled. Embed width=hidden; output four nonconsecutive horizons.",
        "itransformer": "Pinned author architecture: history-length linear inverted embedding (one token/node plus two historical clock tokens), two full variate-attention encoder layers, four heads, FFN=4*hidden with GELU, dropout=.1, post-attention and post-FFN LayerNorm plus final LayerNorm; direct four-horizon projection. Author per-window mean/variance normalization and output inverse retained inside train-only external scaling. Clock-token outputs discarded; no decoder/future input.",
        "training": "All neural families use identical 3 validation candidates, seed0 search, then the chosen candidate's seed0 plus fresh seeds1/2. AdamW, standardized unmasked MSE, weight decay1e-4, batch128, gradient norm1, maximum100 epochs, validation raw MAE early stop patience12. Common hyperparameter budget is not exhaustive or proof of optimal tuning.",
    }


class RecurrentAdapter(nn.Module):
    def __init__(self, name, hidden, horizons, mask, lengths):
        super().__init__()
        self.core = TemporalForecaster(name, hidden, horizons, mask, lengths)

    def forward(self, x, clock, origin):
        return self.core(x, clock)


class SpatialAblation(nn.Module):
    def __init__(self, name, hidden, horizons, mask):
        super().__init__()
        self.name, self.horizons = name, horizons
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.bool))
        if name == "no_edge_length":
            self.edge_scores = nn.Parameter(torch.tensor([0.1, 0.1, 0.0]))
        self.recurrent = nn.GRU(4, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizons)

    def forward(self, x, clock, origin):
        b, length, n = x.shape
        if self.name == "uniform_neighbor":
            weights = self.mask.to(x.dtype) / self.mask.sum(-1, keepdim=True)
            neighbor = torch.einsum("blj,ij->bli", x, weights)
        else:
            q, k, bias = self.edge_scores
            logits = F.leaky_relu(q*x[..., :, None]+k*x[..., None, :]+bias, .2)
            weights = torch.softmax(logits.masked_fill(~self.mask, -1e9), -1)
            neighbor = (weights*x[..., None, :]).sum(-1)
        time_features = clock[:, :, None, :].expand(b, length, n, 2)
        features = torch.cat([x[..., None], time_features, neighbor[..., None]], -1)
        features = features.permute(0, 2, 1, 3).reshape(b*n, length, 4)
        sequence, _ = self.recurrent(features)
        return self.head(sequence[:, -1]).reshape(b, n, self.horizons)+x[:, -1, :, None]


class DiffusionConv(nn.Module):
    def __init__(self, hidden, supports=2, order=2, dropout=.1):
        super().__init__()
        self.order, self.dropout = order, dropout
        self.projection = nn.Conv2d(hidden*(1+supports*order), hidden, (1, 1))

    def forward(self, x, supports):
        values = [x]
        for support in supports:
            current = x
            for _ in range(self.order):
                current = torch.einsum("bcvt,vw->bcwt", current, support).contiguous()
                values.append(current)
        return F.dropout(self.projection(torch.cat(values, 1)), self.dropout, self.training)


class GraphWaveNet(nn.Module):
    """Author-architecture adaptation; pinned source and license stored with run."""
    def __init__(self, hidden, horizons, adjacency):
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.register_buffer("physical", adjacency / adjacency.sum(1, keepdim=True).clamp_min(1))
        nodes = len(adjacency)
        self.nodevec1 = nn.Parameter(torch.randn(nodes, 10))
        self.nodevec2 = nn.Parameter(torch.randn(10, nodes))
        self.start = nn.Conv2d(3, hidden, (1, 1))
        self.filters, self.gates, self.skips = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.diffusions, self.norms = nn.ModuleList(), nn.ModuleList()
        self.receptive_field = 13
        for _ in range(4):
            for dilation in [1, 2]:
                self.filters.append(nn.Conv2d(hidden, hidden, (1, 2), dilation=(1, dilation)))
                self.gates.append(nn.Conv2d(hidden, hidden, (1, 2), dilation=(1, dilation)))
                self.skips.append(nn.Conv2d(hidden, 2*hidden, (1, 1)))
                # The author's last post-skip branch never reaches the output.
                # Omit only that dead branch so counted parameters all train.
                if len(self.filters) < 8:
                    self.diffusions.append(DiffusionConv(hidden))
                    self.norms.append(nn.BatchNorm2d(hidden))
        self.end1, self.end2 = nn.Conv2d(2*hidden, 4*hidden, 1), nn.Conv2d(4*hidden, horizons, 1)

    def forward(self, x, clock, origin):
        b, length, nodes = x.shape
        clock = clock[:, :, None, :].expand(b, length, nodes, 2)
        x = torch.cat([x[..., None], clock], -1).permute(0, 3, 2, 1)
        x = F.pad(x, (max(0, self.receptive_field-length), 0, 0, 0))
        x = self.start(x)
        adaptive = torch.softmax(F.relu(self.nodevec1@self.nodevec2), dim=1)
        supports, skip = [self.physical, adaptive], None
        for index, (filter_conv, gate_conv, skip_conv) in enumerate(zip(
                self.filters, self.gates, self.skips)):
            residual = x
            x = torch.tanh(filter_conv(x))*torch.sigmoid(gate_conv(x))
            part = skip_conv(x)
            skip = part if skip is None else skip[..., -part.shape[-1]:]+part
            if index < len(self.diffusions):
                x = self.diffusions[index](x, supports)
                x = self.norms[index](x+residual[..., -x.shape[-1]:])
        result = self.end2(F.relu(self.end1(F.relu(skip))))
        return result[..., -1].transpose(1, 2)


class ResidualMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.first, self.second = nn.Conv2d(width, width, 1), nn.Conv2d(width, width, 1)
        self.dropout = nn.Dropout(.15)

    def forward(self, x):
        return x+self.second(self.dropout(F.relu(self.first(x))))


class STID(nn.Module):
    def __init__(self, hidden, horizons, nodes, history=12, period=72):
        super().__init__()
        self.period = period
        self.node_embedding = nn.Parameter(torch.empty(nodes, 8))
        self.time_embedding = nn.Parameter(torch.empty(period, 8))
        nn.init.xavier_uniform_(self.node_embedding)
        nn.init.xavier_uniform_(self.time_embedding)
        self.history_embedding = nn.Conv2d(history*3, hidden, 1)
        self.encoder = nn.Sequential(*[ResidualMLP(hidden+16) for _ in range(3)])
        self.head = nn.Conv2d(hidden+16, horizons, 1)

    def forward(self, x, clock, origin):
        b, length, n = x.shape
        clock = clock[:, :, None, :].expand(b, length, n, 2)
        history = torch.cat([x[..., None], clock], -1).transpose(1, 2).contiguous()
        history = history.reshape(b, n, -1).transpose(1, 2).unsqueeze(-1)
        value = self.history_embedding(history)
        node = self.node_embedding.T[None, :, :, None].expand(b, -1, -1, 1)
        tod = self.time_embedding[origin.long().remainder(self.period)]
        tod = tod[:, :, None, None].expand(b, -1, n, 1)
        return self.head(self.encoder(torch.cat([value, node, tod], 1))).squeeze(-1).transpose(1, 2)


class VariateEncoderLayer(nn.Module):
    def __init__(self, hidden, heads=4):
        super().__init__()
        self.heads, self.head_dim = heads, hidden//heads
        self.query, self.key, self.value = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.first, self.second = nn.Linear(hidden, 4*hidden), nn.Linear(4*hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        b, tokens, hidden = x.shape
        project = lambda layer: layer(x).reshape(b, tokens, self.heads, self.head_dim)
        q, k, v = project(self.query), project(self.key), project(self.value)
        weights = torch.softmax(torch.einsum("blhe,bshe->bhls", q, k)/(self.head_dim**.5), -1)
        context = torch.einsum("bhls,bshd->blhd", self.dropout(weights), v).contiguous().reshape(b, tokens, hidden)
        x = self.norm1(x+self.dropout(self.output(context)))
        feedforward = self.dropout(F.gelu(self.first(x)))
        return self.norm2(x+self.dropout(self.second(feedforward)))


class ITransformer(nn.Module):
    def __init__(self, hidden, horizons, history=12):
        super().__init__()
        self.embedding = nn.Linear(history, hidden)
        self.dropout = nn.Dropout(.1)
        self.layers = nn.ModuleList([VariateEncoderLayer(hidden) for _ in range(2)])
        self.norm, self.head = nn.LayerNorm(hidden), nn.Linear(hidden, horizons)

    def forward(self, x, clock, origin):
        nodes = x.shape[-1]
        means = x.mean(1, keepdim=True).detach()
        centered = x-means
        stdev = torch.sqrt(centered.var(1, keepdim=True, unbiased=False)+1e-5)
        tokens = torch.cat([centered/stdev, clock], -1).transpose(1, 2)
        values = self.dropout(self.embedding(tokens))
        for layer in self.layers:
            values = layer(values)
        prediction = self.head(self.norm(values))[:, :nodes]
        return prediction*stdev[:, 0, :, None]+means[:, 0, :, None]


def make_model(name, candidate, dataset, configuration):
    mask, lengths = edge_arrays(dataset)
    hidden, horizons = candidate["hidden"], len(configuration["horizons"])
    if name in ["lstm", "gru", "edge_stgru"]:
        return RecurrentAdapter(name, hidden, horizons, mask, lengths)
    if name in ["uniform_neighbor", "no_edge_length"]:
        return SpatialAblation(name, hidden, horizons, mask)
    if name == "graph_wavenet":
        # Same FP32 operations; NHWC-compatible storage avoids a very slow
        # small-spatial-convolution kernel on the available laptop GPU.
        return GraphWaveNet(hidden, horizons, dataset["adjacency"]).to(memory_format=torch.channels_last)
    if name == "stid":
        return STID(hidden, horizons, len(mask), configuration["history"], configuration["period"])
    if name == "itransformer":
        return ITransformer(hidden, horizons, configuration["history"])
    raise ValueError(name)


def setup_runtime(configuration, seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(configuration["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_windows(windows, configuration):
    device = configuration["device"]
    return {
        "x": torch.as_tensor(windows["x"], device=device),
        "target": torch.as_tensor(windows["yn"], device=device),
        "clock": torch.as_tensor(_clock(windows["history_times"], configuration["period"]), device=device),
        "origin": torch.as_tensor(windows["origin"], device=device),
    }


def predict(model, windows, stats, configuration):
    tensors = tensor_windows(windows, configuration)
    model.eval()
    results = []
    with torch.inference_mode():
        for begin in range(0, len(windows["x"]), configuration["batch_size"]):
            batch = slice(begin, begin+configuration["batch_size"])
            results.append(model(tensors["x"][batch], tensors["clock"][batch], tensors["origin"][batch]).cpu().numpy())
    return _unscale(np.concatenate(results), stats)


def train_one(name, candidate, seed, dataset, train, validation, stats, configuration, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory/"complete.json"
    if complete.exists():
        result = json.loads(complete.read_text(encoding="utf8"))
        if result["checkpoint_sha256"] != sha256(directory/"checkpoint.pt"):
            raise ValueError("Checkpoint changed after completed training")
        return result
    setup_runtime(configuration, seed)
    model = make_model(name, candidate, dataset, configuration).to(configuration["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=configuration["weight_decay"])
    tensors = tensor_windows(train, configuration)
    rng = np.random.default_rng(seed)
    best_score, best_epoch, best_state, logs = float("inf"), 0, None, []
    started = time.perf_counter()
    for epoch in range(1, configuration["epochs"]+1):
        model.train()
        order, total_loss = rng.permutation(len(train["x"])), 0.
        for begin in range(0, len(order), configuration["batch_size"]):
            indices = torch.as_tensor(order[begin:begin+configuration["batch_size"]], device=configuration["device"])
            optimizer.zero_grad(set_to_none=True)
            estimate = model(tensors["x"][indices], tensors["clock"][indices], tensors["origin"][indices])
            loss = F.mse_loss(estimate, tensors["target"][indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite forecast loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        validation_prediction = predict(model, validation, stats, configuration)
        validation_mae = float(np.abs(validation_prediction-validation["y"]).mean())
        logs.append({"epoch": epoch, "train_standardized_mse": total_loss/len(order),
                     "validation_mae": validation_mae, "elapsed_seconds": time.perf_counter()-started})
        if validation_mae < best_score-1e-7:
            best_score, best_epoch = validation_mae, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0:
            write_csv(directory/"training_log.csv", logs)
            print(f"{name}/{candidate['id']}/seed{seed} epoch={epoch} val_mae={validation_mae:.5f} best={best_score:.5f}", flush=True)
        if epoch-best_epoch >= configuration["patience"]:
            break
    model.load_state_dict(best_state)
    validation_prediction = predict(model, validation, stats, configuration)
    np.savez_compressed(directory/"validation_predictions.npz", prediction=validation_prediction)
    torch.save({"name": name, "candidate": candidate, "seed": seed, "state_dict": best_state}, directory/"checkpoint.pt")
    write_csv(directory/"training_log.csv", logs)
    result = {"model": name, "seed": seed, "candidate": candidate, "epochs_run": epoch,
              "best_epoch": best_epoch, "validation_mae": best_score,
              "parameters": sum(value.numel() for value in model.parameters()),
              "seconds": time.perf_counter()-started,
              "checkpoint_sha256": sha256(directory/"checkpoint.pt"),
              "validation_predictions_sha256": sha256(directory/"validation_predictions.npz")}
    write_json(complete, result, exclusive=True)
    print(f"COMPLETE {name}/{candidate['id']}/seed{seed}: {best_score:.5f}; {result['seconds']:.1f}s", flush=True)
    return result


def load_v2_dataset(protocol):
    with np.load(protocol["source_dataset"], allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def protocol_for(root=Path.cwd(), output=None):
    root = Path(root).resolve()
    output = Path(output or root/"runs/supplement_v3/forecast").resolve()
    config_path = root/"runs/scenic_rebuild_v2/assets/simulator_config.json"
    dataset_path = root/"runs/scenic_rebuild_v2/assets/synthetic_dataset.npz"
    source_paths = [root/name for name in ["iotexp/scenic.py", "iotexp/scenic_forecast.py", "iotexp/telemetry.py",
                    "iotexp/forecast_supplement.py", "scripts/run_forecast_supplement.py"]]
    source_paths += sorted((output/"source").glob("*"))
    return {
        "version": "supplement_v3_forecast_1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root), "output": str(output), "source_dataset": str(dataset_path),
        "source_dataset_sha256": sha256(dataset_path), "simulator_config": json.loads(config_path.read_text(encoding="utf8")),
        "source_hashes": {str(p): sha256(p) for p in source_paths},
        "neural_models": NEURAL_MODELS, "candidates": CANDIDATES,
        "configuration": {"history": 12, "horizons": [1, 2, 4, 6], "period": 72, "device": "cuda",
                          "torch_threads": 2, "batch_size": 128, "epochs": 100, "patience": 12, "weight_decay": .0001},
        "ridge_alphas": [1., 100., 10000.], "training_seeds": [0, 1, 2],
        "final_dataset_seeds": FINAL_DATASET_SEEDS,
        "selection": "For each neural family train all three candidate configurations on training days 0-125 with seed0; choose lowest pooled raw validation MAE across days126-152, all24 nodes and all4 horizons (tie: candidate order). Reuse winner seed0 and train seeds1/2. Early-stop/checkpoint solely on validation MAE. Ridge fits training data for three alphas, selected by same validation criterion. Deterministic HA and persistence have no hyperparameters.",
        "test_rule": "Create selection_freeze.json with checkpoint/selection hashes before generating or loading any new final dataset. For each new seed generate the unchanged v2 simulator with only dataset_seed changed; evaluate frozen models on days153-179. No final data, test metrics or downstream rewards may tune any model. Original v2 test is secondary compatibility evidence only. New datasets are independent stochastic realizations of one generator/graph, not real-world or cross-topology validation.",
        "normalization": "Reuse fitting function on v2 training days only. Same scaler for validation, old-test and all new datasets. All observed zeros are valid and included in unmasked loss/metrics; inverse predictions clipped at zero.",
        "numerics": "FP32 tensors and no mixed-precision/autocast training; cuDNN TF32 convolutions permitted, matmul TF32 disabled (explicit common runtime settings). Deterministic PyTorch algorithms, CUDA, two CPU threads. Graph WaveNet convolution parameters use channels_last memory layout; this changes storage/kernel selection, not architecture or tensors' semantic axes. Unit-tested against contiguous-layout inference under identical weights.",
        "metrics": "MAE/RMSE/WAPE all4 horizons, pooled and perday, each selected training seed. Seed-family summaries average metrics, not predictions. Error bars are sample SD, no small-n confidence/superiority claim.",
        "architecture_notes": architecture_notes(),
    }


def freeze_protocol(output):
    output = Path(output).resolve()
    path = output/"protocol.json"
    if path.exists():
        return verify_protocol(output)
    protocol = protocol_for(output=output)
    if not (output/"source/source_manifest.json").exists():
        raise ValueError("Pinned author source provenance must exist before protocol freeze")
    write_json(path, protocol, exclusive=True)
    (output/"protocol.sha256").write_text(sha256(path)+"\n", encoding="ascii")
    return protocol


def verify_protocol(output):
    output = Path(output).resolve()
    path = output/"protocol.json"
    if sha256(path) != (output/"protocol.sha256").read_text().strip():
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(path.read_text(encoding="utf8"))
    if sha256(protocol["source_dataset"]) != protocol["source_dataset_sha256"]:
        raise ValueError("Frozen v2 data changed")
    for path, expected in protocol["source_hashes"].items():
        if sha256(path) != expected:
            raise ValueError("Frozen source changed: "+path)
    return protocol


def train_experiment(output):
    output = Path(output).resolve()
    protocol = verify_protocol(output)
    if (output/"selection_freeze.json").exists():
        print("All selections already frozen; no retraining", flush=True)
        return
    dataset = load_v2_dataset(protocol)
    configuration = protocol["configuration"]
    days = split_day_indices(dataset)
    stats = fit_training_stats(dataset, days["train"], configuration["period"])
    stats.save(output/"training_stats.npz")
    # No test windows or final-generator realizations are constructed in this stage.
    train = build_windows(dataset, days["train"], configuration, stats)
    validation = build_windows(dataset, days["validation"], configuration, stats)
    np.savez_compressed(output/"validation_targets.npz", truth=validation["y"], day=validation["day"], origin=validation["origin"])
    setup_runtime(configuration)
    write_json(output/"runtime.json", {"python": platform.python_version(), "numpy": np.__version__,
               "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
               "platform": platform.platform(), "threads": torch.get_num_threads(),
               "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
               "cudnn_version": torch.backends.cudnn.version(),
               "train_windows": len(train["x"]), "validation_windows": len(validation["x"])})
    selected, searches = [], []
    for name in protocol["neural_models"]:
        results = []
        for candidate in protocol["candidates"]:
            directory = output/"training"/name/candidate["id"]/"seed0"
            info = train_one(name, candidate, 0, dataset, train, validation, stats, configuration, directory)
            results.append(info)
            searches.append({"model": name, "candidate": candidate["id"], "hidden": candidate["hidden"],
                             "learning_rate": candidate["learning_rate"], "validation_mae": info["validation_mae"],
                             "best_epoch": info["best_epoch"], "parameters": info["parameters"]})
            write_csv(output/"hyperparameter_search.csv", searches)
        winner = min(results, key=lambda result: result["validation_mae"])
        for seed in protocol["training_seeds"]:
            directory = output/"training"/name/winner["candidate"]["id"]/f"seed{seed}"
            info = train_one(name, winner["candidate"], seed, dataset, train, validation, stats, configuration, directory)
            selected.append({**info, "checkpoint": str((directory/"checkpoint.pt").relative_to(output))})
        write_json(output/"selected_progress.json", selected)
    x, xv = _ridge_features(train, configuration["period"]), _ridge_features(validation, configuration["period"])
    y = train["yn"].reshape(len(x), -1)
    gram, xy, ridge_results = x.T@x, x.T@y, []
    for alpha in protocol["ridge_alphas"]:
        penalty = np.eye(x.shape[1])*alpha
        penalty[-1, -1] = 0
        coefficients = np.linalg.solve(gram+penalty, xy)
        prediction = _unscale((xv@coefficients).reshape(validation["y"].shape), stats)
        score = float(np.abs(prediction-validation["y"]).mean())
        ridge_results.append((score, alpha, coefficients))
        np.savez_compressed(output/f"ridge_alpha{alpha:g}_validation_predictions.npz", prediction=prediction)
    score, alpha, coefficients = min(ridge_results, key=lambda result: result[0])
    np.savez_compressed(output/"ridge.npz", coefficients=coefficients)
    historical = np.asarray(dataset["inflow"])[days["train"]].mean(0)
    np.savez_compressed(output/"historical_average.npz", values=historical, first_time=int(dataset["times"][0]))
    selected += [{"model": "ridge", "seed": -1, "alpha": alpha, "parameters": int(coefficients.size),
                  "validation_mae": score, "checkpoint": "ridge.npz", "checkpoint_sha256": sha256(output/"ridge.npz")},
                 {"model": "historical_average", "seed": -1, "parameters": int(historical.size),
                  "checkpoint": "historical_average.npz", "checkpoint_sha256": sha256(output/"historical_average.npz")},
                 {"model": "persistence", "seed": -1, "parameters": 0, "checkpoint": None, "checkpoint_sha256": None}]
    write_json(output/"ridge_search.json", [{"alpha": alpha, "validation_mae": score} for score, alpha, _ in ridge_results])
    write_json(output/"selected_models.json", selected)
    freeze = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "protocol_sha256": sha256(output/"protocol.json"), "selected_models_sha256": sha256(output/"selected_models.json"),
              "training_stats_sha256": sha256(output/"training_stats.npz"),
              "checkpoints": {entry["checkpoint"]: entry["checkpoint_sha256"] for entry in selected if entry["checkpoint"]},
              "independent_final_datasets_generated_or_inspected": False}
    write_json(output/"selection_freeze.json", freeze, exclusive=True)
    (output/"selection_freeze.sha256").write_text(sha256(output/"selection_freeze.json")+"\n", encoding="ascii")


def verify_selection(output):
    output = Path(output)
    path = output/"selection_freeze.json"
    if not path.exists():
        raise ValueError("Final evaluation is forbidden before selection is frozen")
    if sha256(path) != (output/"selection_freeze.sha256").read_text().strip():
        raise ValueError("Selection freeze changed")
    freeze = json.loads(path.read_text(encoding="utf8"))
    for name, expected in freeze["checkpoints"].items():
        if sha256(output/name) != expected:
            raise ValueError("Selected checkpoint changed")
    for name, key in [("selected_models.json", "selected_models_sha256"), ("training_stats.npz", "training_stats_sha256")]:
        if sha256(output/name) != freeze[key]:
            raise ValueError("Selection asset changed")
    return freeze


def evaluate_experiment(output):
    output = Path(output).resolve()
    protocol = verify_protocol(output)
    verify_selection(output)
    if (output/"evaluation_complete.json").exists():
        print("Final evaluation already complete; not rerunning", flush=True)
        return
    configuration = protocol["configuration"]
    setup_runtime(configuration)
    selected = json.loads((output/"selected_models.json").read_text(encoding="utf8"))
    stats = TrainingStats.load(output/"training_stats.npz")
    datasets = {"legacy_test": load_v2_dataset(protocol)}
    dataset_records = []
    data_directory = output/"final_datasets"
    data_directory.mkdir(exist_ok=True)
    for seed in protocol["final_dataset_seeds"]:
        path = data_directory/f"dataset_seed{seed}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                data = {key: saved[key] for key in saved.files}
        else:
            cfg = copy.deepcopy(protocol["simulator_config"])
            cfg["dataset_seed"] = seed
            data = build_dataset(cfg)
            np.savez_compressed(path, **data)
        datasets[f"independent_{seed}"] = data
        dataset_records.append({"dataset_seed": seed, "npz_sha256": sha256(path),
                                "inflow_sha256": hashlib.sha256(data["inflow"].tobytes()).hexdigest(),
                                "test_days": list(range(153, 180))})
    write_json(output/"final_dataset_manifest.json", dataset_records, exclusive=not (output/"final_dataset_manifest.json").exists())
    windows = {tag: build_windows(data, split_day_indices(data)["test"], configuration, stats) for tag, data in datasets.items()}
    target_directory, prediction_directory = output/"targets", output/"predictions"
    target_directory.mkdir(exist_ok=True)
    prediction_directory.mkdir(exist_ok=True)
    for tag, window in windows.items():
        np.savez_compressed(target_directory/f"{tag}.npz", truth=window["y"], day=window["day"], origin=window["origin"], horizons=configuration["horizons"])
    metrics, daily, manifest = [], [], []
    for entry in selected:
        name, seed = entry["model"], entry["seed"]
        model = None
        if name in NEURAL_MODELS:
            checkpoint = torch.load(output/entry["checkpoint"], map_location="cpu", weights_only=False)
            model = make_model(name, checkpoint["candidate"], datasets["legacy_test"], configuration).to(configuration["device"])
            model.load_state_dict(checkpoint["state_dict"])
        elif name == "ridge":
            coefficients = np.load(output/"ridge.npz")["coefficients"]
        elif name == "historical_average":
            saved = np.load(output/"historical_average.npz")
            values, first_time = saved["values"], int(saved["first_time"])
        independent_truth, independent_prediction = [], []
        for tag, window in windows.items():
            if model is not None:
                prediction = predict(model, window, stats, configuration)
            elif name == "ridge":
                prediction = _unscale((_ridge_features(window, configuration["period"])@coefficients).reshape(window["y"].shape), stats)
            elif name == "historical_average":
                indices = window["origin"][:, None]+np.asarray(configuration["horizons"])[None]-first_time
                prediction = values[indices].transpose(0, 2, 1).astype(np.float32)
            else:
                prediction = _unscale(np.repeat(window["x"][:, -1, :, None], 4, -1), stats)
            path = prediction_directory/f"{name}_seed{seed}_{tag}.npz"
            np.savez_compressed(path, prediction=prediction)
            manifest.append({"model": name, "training_seed": seed, "dataset": tag,
                             "file": str(path.relative_to(output)), "sha256": sha256(path)})
            for j, horizon in enumerate(configuration["horizons"]):
                descriptor = {"model": name, "training_seed": seed, "dataset": tag, "horizon_steps": horizon}
                metrics.append({**descriptor, **_scores(window["y"][:, :, j], prediction[:, :, j])})
                for day in np.unique(window["day"]):
                    mask = window["day"] == day
                    daily.append({**descriptor, "day": int(day), **_scores(window["y"][mask, :, j], prediction[mask, :, j])})
            if tag.startswith("independent_"):
                independent_truth.append(window["y"])
                independent_prediction.append(prediction)
        for j, horizon in enumerate(configuration["horizons"]):
            metrics.append({"model": name, "training_seed": seed, "dataset": "independent_pooled", "horizon_steps": horizon,
                            **_scores(np.concatenate(independent_truth)[:, :, j], np.concatenate(independent_prediction)[:, :, j])})
        print(f"EVALUATED {name} seed{seed}", flush=True)
    write_csv(output/"metrics.csv", metrics)
    write_csv(output/"daily_metrics.csv", daily)
    write_json(output/"prediction_manifest.json", manifest)
    write_json(output/"evaluation_complete.json", {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "selection_freeze_sha256": sha256(output/"selection_freeze.json"), "metrics_rows": len(metrics),
               "daily_metric_rows": len(daily), "prediction_arrays": len(manifest),
               "metrics_sha256": sha256(output/"metrics.csv"), "prediction_manifest_sha256": sha256(output/"prediction_manifest.json")}, exclusive=True)


def report_experiment(output):
    import pandas as pd
    output = Path(output).resolve()
    protocol = verify_protocol(output)
    verify_selection(output)
    metrics = pd.read_csv(output/"metrics.csv")
    pooled = metrics[metrics.dataset == "independent_pooled"]
    summary = pooled.groupby(["model", "horizon_steps"])[["mae", "rmse", "wape_pct"]].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().to_csv(output/"independent_summary.csv", index=False)
    selected = json.loads((output/"selected_models.json").read_text(encoding="utf8"))
    rows = []
    for name in NEURAL_MODELS:
        entries = [entry for entry in selected if entry["model"] == name]
        rows.append({"model": name, "candidate": entries[0]["candidate"]["id"], "hidden": entries[0]["candidate"]["hidden"],
                     "learning_rate": entries[0]["candidate"]["learning_rate"], "parameters": entries[0]["parameters"],
                     "training_seeds": len(entries), "validation_mae_mean": float(np.mean([entry["validation_mae"] for entry in entries])),
                     "best_epochs": "/".join(str(entry["best_epoch"]) for entry in entries),
                     "epochs_run": "/".join(str(entry["epochs_run"]) for entry in entries)})
    write_csv(output/"selected_configuration_summary.csv", rows)
    title = ["# Supplementary forecasting experiment", "", "Status: completed; no old experiment results were overwritten.", "",
             "Models and checkpoints were selected on the original training/validation days before generating the three independent final data realizations. Each final seed contributes 27 held-out days; all share the fixed synthetic graph/generator. This is not external field validation.", "",
             "Neural entries below are means over three separately trained seeds; dispersion is sample SD, not a confidence interval. Deterministic baseline SD is undefined, not zero evidence of uncertainty.", "",
             "| Model | 10-min MAE | 20-min MAE | 40-min MAE | 60-min MAE | 60-min RMSE | 60-min WAPE % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(pooled.model.unique()):
        per_model = summary.loc[name]
        values = [per_model.loc[h, "mae_mean"] for h in [1, 2, 4, 6]]
        values += [per_model.loc[6, "rmse_mean"], per_model.loc[6, "wape_pct_mean"]]
        title.append("| "+name+" | "+" | ".join(f"{value:.6f}" for value in values)+" |")
    title += ["", "Configuration and fairness boundaries:", ""]
    title += ["- "+key+": "+value for key, value in protocol["architecture_notes"].items()]
    title += ["", "Audit files: protocol.json/sha256, source/source_manifest.json and retained licenses, selection_freeze.json/sha256, all candidate epoch logs and validation predictions, selected checkpoints, per-dataset targets/predictions, metrics.csv and daily_metrics.csv. Legacy-test rows are compatibility diagnostics only. No superiority claim is automatic from the table.", ""]
    (output/"FORECAST_SUPPLEMENT_REPORT.md").write_text("\n".join(title), encoding="utf8")


def smoke_experiment(output):
    """Small synthetic training-only smoke; never constructs final-seed datasets."""
    output = Path(output)
    cfg = json.loads(Path("runs/scenic_rebuild_v2/assets/simulator_config.json").read_text(encoding="utf8"))
    cfg.update(days=8, split_days=[4, 2, 2], dataset_seed=20260000)
    dataset = build_dataset(cfg)
    configuration = {"history": 12, "horizons": [1, 2, 4, 6], "period": 72, "device": "cuda",
                     "torch_threads": 2, "batch_size": 128, "epochs": 2, "patience": 2, "weight_decay": .0001}
    stats = fit_training_stats(dataset, np.arange(4), 72)
    train = build_windows(dataset, np.arange(4), configuration, stats)
    validation = build_windows(dataset, np.arange(4, 6), configuration, stats)
    records = []
    for name in NEURAL_MODELS:
        records.append(train_one(name, CANDIDATES[0], 0, dataset, train, validation, stats, configuration, output/name))
    write_json(output/"smoke_summary.json", records)
    return records
