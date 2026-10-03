"""Prospectively appended, validation-config-matched spatial mechanism comparison.

Does not edit the primary frozen protocol/module/results. Freeze this appendix
before generating independent final datasets; train only after the primary
validation selections exist, and evaluate only after both selections are frozen.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from iotexp.forecast_supplement import (
    sha256, write_json, write_csv, verify_protocol, verify_selection,
    load_v2_dataset, build_windows, split_day_indices, TrainingStats,
    train_one, setup_runtime, make_model, predict, _scores,
)

MODELS = ["edge_stgru", "uniform_neighbor", "no_edge_length"]


def assert_final_not_generated(output):
    if list((Path(output)/"final_datasets").glob("*.npz")):
        raise ValueError("Matched-ablation training/protocol must precede new final data generation")


def freeze(output):
    output = Path(output).resolve()
    verify_protocol(output)
    folder = output/"matched_ablation"
    folder.mkdir(exist_ok=True)
    path = folder/"protocol.json"
    if path.exists():
        return verify_appendix(output)
    assert_final_not_generated(output)
    protocol = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_protocol_sha256": sha256(output/"protocol.json"),
        "script_sha256": sha256(__file__), "models": MODELS, "seeds": [0, 1, 2],
        "configuration_rule": "Use only the candidate chosen for edge_stgru from the primary three-candidate seed0 validation search. Apply that exact hidden width and learning rate, plus unchanged common trainer/100epoch/patience12, to EdgeSTGRU, uniform-neighbor and no-edge-length for seeds0,1,2. Each model's checkpoint epoch remains validation-only. No ablation-specific tuning in this matched table.",
        "reuse": "Reuse a completed primary checkpoint only when model, candidate and training seed match exactly; record its relative path and SHA256. Otherwise train the exact configuration in this appendix. Edge checkpoints always reuse the primary selected models.",
        "evaluation_rule": "Freeze all nine selected checkpoint identities before any independent final dataset is generated. Then evaluate on the same three new nominal generator seeds and days153-179 as the primary comparison. Keep this matched mechanism table separate from independently tuned family-winner comparisons.",
        "scope": "Same-width/recurrent-head mechanism ablation, with parameter differences limited to removed attention scalars/edge feature. Common optimization settings do not imply identical early-stopping epochs or exact equal parameter counts. No tests or final outcomes choose configuration.",
    }
    write_json(path, protocol, exclusive=True)
    (folder/"protocol.sha256").write_text(sha256(path)+"\n", encoding="ascii")
    return protocol


def verify_appendix(output):
    output = Path(output).resolve()
    folder = output/"matched_ablation"
    path = folder/"protocol.json"
    if sha256(path) != (folder/"protocol.sha256").read_text().strip():
        raise ValueError("Appended protocol changed")
    protocol = json.loads(path.read_text(encoding="utf8"))
    if protocol["parent_protocol_sha256"] != sha256(output/"protocol.json"):
        raise ValueError("Parent protocol mismatch")
    if protocol["script_sha256"] != sha256(__file__):
        raise ValueError("Frozen appendix script changed")
    return protocol


def train(output):
    output = Path(output).resolve()
    primary = verify_protocol(output)
    verify_selection(output)
    appendix = verify_appendix(output)
    folder = output/"matched_ablation"
    if (folder/"selection_freeze.json").exists():
        verify_matched_selection(output)
        return
    assert_final_not_generated(output)
    selected = json.loads((output/"selected_models.json").read_text(encoding="utf8"))
    candidate = next(entry["candidate"] for entry in selected if entry["model"] == "edge_stgru" and entry["seed"] == 0)
    dataset = load_v2_dataset(primary)
    configuration = primary["configuration"]
    stats = TrainingStats.load(output/"training_stats.npz")
    days = split_day_indices(dataset)
    train_windows = build_windows(dataset, days["train"], configuration, stats)
    validation = build_windows(dataset, days["validation"], configuration, stats)
    records = []
    for name in appendix["models"]:
        for seed in appendix["seeds"]:
            primary_directory = output/"training"/name/candidate["id"]/f"seed{seed}"
            reused = (primary_directory/"complete.json").exists()
            directory = primary_directory if reused else folder/"training"/name/f"seed{seed}"
            info = train_one(name, candidate, seed, dataset, train_windows, validation, stats, configuration, directory)
            records.append({**info, "checkpoint": str((directory/"checkpoint.pt").relative_to(output)),
                            "reused_primary_checkpoint": reused})
    assert_final_not_generated(output)
    write_json(folder/"selected_models.json", records, exclusive=True)
    freeze_record = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "candidate": candidate, "protocol_sha256": sha256(folder/"protocol.json"),
                     "selected_models_sha256": sha256(folder/"selected_models.json"),
                     "checkpoints": {entry["checkpoint"]: entry["checkpoint_sha256"] for entry in records},
                     "final_datasets_not_yet_generated": True}
    write_json(folder/"selection_freeze.json", freeze_record, exclusive=True)
    (folder/"selection_freeze.sha256").write_text(sha256(folder/"selection_freeze.json")+"\n", encoding="ascii")


def verify_matched_selection(output):
    output = Path(output).resolve()
    verify_appendix(output)
    folder = output/"matched_ablation"
    path = folder/"selection_freeze.json"
    if sha256(path) != (folder/"selection_freeze.sha256").read_text().strip():
        raise ValueError("Matched selection freeze changed")
    record = json.loads(path.read_text(encoding="utf8"))
    if sha256(folder/"selected_models.json") != record["selected_models_sha256"]:
        raise ValueError("Matched models changed")
    for checkpoint, digest in record["checkpoints"].items():
        if sha256(output/checkpoint) != digest:
            raise ValueError("Matched checkpoint changed")
    return record


def evaluate(output):
    output = Path(output).resolve()
    primary = verify_protocol(output)
    verify_selection(output)
    verify_matched_selection(output)
    folder = output/"matched_ablation"
    if (folder/"evaluation_complete.json").exists():
        return
    if not (output/"evaluation_complete.json").exists():
        raise ValueError("Run primary final evaluation only after matched selection freeze; then evaluate matched models")
    configuration = primary["configuration"]
    setup_runtime(configuration)
    selected = json.loads((folder/"selected_models.json").read_text(encoding="utf8"))
    primary_selected = json.loads((output/"selected_models.json").read_text(encoding="utf8"))
    stats = TrainingStats.load(output/"training_stats.npz")
    structure = load_v2_dataset(primary)
    windows = {}
    for dataset_seed in primary["final_dataset_seeds"]:
        with np.load(output/"final_datasets"/f"dataset_seed{dataset_seed}.npz", allow_pickle=False) as saved:
            dataset = {key: saved[key] for key in saved.files}
        windows[f"independent_{dataset_seed}"] = build_windows(dataset, split_day_indices(dataset)["test"], configuration, stats)
    (folder/"predictions").mkdir(exist_ok=True)
    metrics, daily, manifest = [], [], []
    for entry in selected:
        name, seed = entry["model"], entry["seed"]
        reference = next((other for other in primary_selected if other["model"] == name and other["seed"] == seed
                          and other["checkpoint_sha256"] == entry["checkpoint_sha256"]), None)
        model = None
        if reference is None:
            checkpoint = torch.load(output/entry["checkpoint"], map_location="cpu", weights_only=False)
            model = make_model(name, checkpoint["candidate"], structure, configuration).to(configuration["device"])
            model.load_state_dict(checkpoint["state_dict"])
        pooled_truth, pooled_prediction = [], []
        for tag, window in windows.items():
            if reference is not None:
                path = output/"predictions"/f"{name}_seed{seed}_{tag}.npz"
                prediction = np.load(path)["prediction"]
            else:
                prediction = predict(model, window, stats, configuration)
                path = folder/"predictions"/f"{name}_seed{seed}_{tag}.npz"
                np.savez_compressed(path, prediction=prediction)
            manifest.append({"model": name, "training_seed": seed, "dataset": tag,
                             "file": str(path.relative_to(output)), "sha256": sha256(path),
                             "reused_primary_prediction": reference is not None})
            for j, horizon in enumerate(configuration["horizons"]):
                identifier = {"model": name, "training_seed": seed, "dataset": tag, "horizon_steps": horizon}
                metrics.append({**identifier, **_scores(window["y"][:, :, j], prediction[:, :, j])})
                for day in np.unique(window["day"]):
                    mask = window["day"] == day
                    daily.append({**identifier, "day": int(day), **_scores(window["y"][mask, :, j], prediction[mask, :, j])})
            pooled_truth.append(window["y"])
            pooled_prediction.append(prediction)
        for j, horizon in enumerate(configuration["horizons"]):
            metrics.append({"model": name, "training_seed": seed, "dataset": "independent_pooled", "horizon_steps": horizon,
                            **_scores(np.concatenate(pooled_truth)[:, :, j], np.concatenate(pooled_prediction)[:, :, j])})
    write_csv(folder/"metrics.csv", metrics)
    write_csv(folder/"daily_metrics.csv", daily)
    write_json(folder/"prediction_manifest.json", manifest)
    import pandas as pd
    frame = pd.DataFrame(metrics)
    means = frame[frame.dataset == "independent_pooled"].groupby(["model", "horizon_steps"])[["mae", "rmse", "wape_pct"]].agg(["mean", "std"])
    means.columns = ["_".join(col) for col in means.columns]
    means.reset_index().to_csv(folder/"summary.csv", index=False)
    pivot = frame.pivot(index=["training_seed", "dataset", "horizon_steps"], columns="model", values="mae")
    pairs = []
    for (training_seed, dataset_tag, horizon), row in pivot.iterrows():
        for ablation in ["uniform_neighbor", "no_edge_length"]:
            pairs.append({"ablation": ablation, "training_seed": training_seed, "dataset": dataset_tag,
                          "horizon_steps": horizon, "ablation_minus_edge_mae": row[ablation]-row["edge_stgru"]})
    write_csv(folder/"paired_differences.csv", pairs)
    write_json(folder/"evaluation_complete.json", {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "selection_freeze_sha256": sha256(folder/"selection_freeze.json"),
               "metrics_sha256": sha256(folder/"metrics.csv"), "rows": len(metrics),
               "daily_rows": len(daily), "prediction_arrays_including_reuse": len(manifest)}, exclusive=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "train", "evaluate"])
    parser.add_argument("--output", default="runs/supplement_v3/forecast")
    arguments = parser.parse_args()
    {"freeze": freeze, "train": train, "evaluate": evaluate}[arguments.stage](arguments.output)
