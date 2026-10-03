"""Read-only numerical/protocol audit of completed V3 forecasting outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from iotexp.forecast_supplement import verify_protocol, verify_selection, NEURAL_MODELS
from run_forecast_matched_ablation import verify_matched_selection


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measures(truth, predicted):
    difference = predicted-truth
    return [float(np.abs(difference).mean()), float(np.sqrt((difference*difference).mean())),
            float(100*np.abs(difference).sum()/float(np.abs(truth).sum()))]


def check_summary(folder, filename):
    metrics = pd.read_csv(folder/"metrics.csv")
    pooled = metrics[metrics.dataset == "independent_pooled"]
    expected = pooled.groupby(["model", "horizon_steps"])[["mae", "rmse", "wape_pct"]].agg(["mean", "std"])
    expected.columns = ["_".join(item) for item in expected.columns]
    actual = pd.read_csv(folder/filename).set_index(["model", "horizon_steps"]).sort_index()
    assert actual.index.equals(expected.index)
    assert actual.columns.tolist() == expected.columns.tolist()
    assert np.allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-10, rtol=0, equal_nan=True)
    return len(actual)


def check_metrics(folder, output, target_map, matched=False):
    table = pd.read_csv(folder/"metrics.csv")
    daily = pd.read_csv(folder/"daily_metrics.csv")
    manifest = json.loads((folder/"prediction_manifest.json").read_text())
    predictions = {}
    for item in manifest:
        path = (output if matched else folder)/item["file"]
        assert digest(path) == item["sha256"], str(path)
        predicted = np.load(path)["prediction"]
        assert predicted.shape == (1809, 24, 4)
        assert np.isfinite(predicted).all() and (predicted >= 0).all()
        predictions[item["model"], item["training_seed"], item["dataset"]] = predicted
    assert len(predictions) == len(manifest)
    max_error, comparisons = 0., 0
    horizons = [1, 2, 4, 6]
    independent_tags = sorted(tag for tag in target_map if tag.startswith("independent_"))
    model_seeds = {(name, seed) for name, seed, tag in predictions}
    tags = independent_tags if matched else ["legacy_test"]+independent_tags
    assert set(predictions) == {(name, seed, tag) for name, seed in model_seeds for tag in tags}
    metric_keys = ["model", "training_seed", "dataset", "horizon_steps"]
    assert not table.duplicated(metric_keys).any()
    assert not daily.duplicated(metric_keys+["day"]).any()
    assert set(table[metric_keys].itertuples(index=False, name=None)) == {
        (name, seed, tag, h) for name, seed in model_seeds for tag in tags+["independent_pooled"] for h in horizons}
    assert set(daily[metric_keys+["day"]].itertuples(index=False, name=None)) == {
        (name, seed, tag, h, day) for name, seed in model_seeds for tag in tags for h in horizons for day in range(153, 180)}
    for row in table.itertuples():
        j = horizons.index(row.horizon_steps)
        if row.dataset == "independent_pooled":
            truth = np.concatenate([target_map[tag]["truth"] for tag in independent_tags])
            predicted = np.concatenate([predictions[row.model, row.training_seed, tag] for tag in independent_tags])
        else:
            truth = target_map[row.dataset]["truth"]
            predicted = predictions[row.model, row.training_seed, row.dataset]
        actual = measures(truth[:, :, j], predicted[:, :, j])
        expected = [row.mae, row.rmse, row.wape_pct]
        max_error = max(max_error, max(abs(a-b) for a, b in zip(actual, expected)))
        comparisons += 3
        assert row.observations == truth[:, :, j].size
    for row in daily.itertuples():
        j = horizons.index(row.horizon_steps)
        target = target_map[row.dataset]
        mask = target["day"] == row.day
        assert int(mask.sum()) == 67
        truth = target["truth"][mask, :, j]
        prediction = predictions[row.model, row.training_seed, row.dataset][mask, :, j]
        actual = measures(truth, prediction)
        max_error = max(max_error, max(abs(a-b) for a, b in zip(actual, [row.mae, row.rmse, row.wape_pct])))
        comparisons += 3
        assert row.observations == 67*24
    assert max_error < 1e-10, max_error
    return {"metrics_rows": len(table), "daily_rows": len(daily), "prediction_arrays": len(predictions),
            "raw_metric_comparisons": comparisons, "maximum_absolute_discrepancy": max_error}


def audit(output):
    output = Path(output).resolve()
    assert (output/"evaluation_complete.json").exists()
    assert (output/"matched_ablation/evaluation_complete.json").exists()
    protocol = verify_protocol(output)
    verify_selection(output)
    matched_freeze = verify_matched_selection(output)
    selected = json.loads((output/"selected_models.json").read_text())
    matched_selected = json.loads((output/"matched_ablation/selected_models.json").read_text())
    search = pd.read_csv(output/"hyperparameter_search.csv")
    assert len(search) == 24
    assert set(search.model) == set(NEURAL_MODELS)
    for name in NEURAL_MODELS:
        choices = search[search.model == name]
        assert set(choices.candidate) == {"C0", "C1", "C2"}
        winner = choices.loc[choices.validation_mae.idxmin()].candidate
        entries = [entry for entry in selected if entry["model"] == name]
        assert {entry["seed"] for entry in entries} == {0, 1, 2}
        assert all(entry["candidate"]["id"] == winner for entry in entries)
    assert len(selected) == 27
    assert len(matched_selected) == 9
    assert all(entry["candidate"] == matched_freeze["candidate"] for entry in matched_selected)
    for name in ["edge_stgru", "uniform_neighbor", "no_edge_length"]:
        assert {entry["seed"] for entry in matched_selected if entry["model"] == name} == {0, 1, 2}
    validation_truth = np.load(output/"validation_targets.npz")["truth"]
    validation_checks = 0
    primary_complete = sorted((output/"training").rglob("complete.json"))
    matched_complete = sorted((output/"matched_ablation/training").rglob("complete.json"))
    assert len(primary_complete) == 40
    assert len(matched_complete) <= 4
    for complete_path in primary_complete+matched_complete:
        record = json.loads(complete_path.read_text())
        assert digest(complete_path.parent/"checkpoint.pt") == record["checkpoint_sha256"]
        assert digest(complete_path.parent/"validation_predictions.npz") == record["validation_predictions_sha256"]
        predictions = np.load(complete_path.parent/"validation_predictions.npz")["prediction"]
        assert abs(float(np.abs(predictions-validation_truth).mean())-record["validation_mae"]) < 1e-7
        logs = pd.read_csv(complete_path.parent/"training_log.csv")
        assert logs.epoch.tolist() == list(range(1, record["epochs_run"]+1))
        assert record["epochs_run"] <= 100
        assert abs(logs.validation_mae.min()-record["validation_mae"]) < 1e-7
        assert abs(logs.set_index("epoch").loc[record["best_epoch"], "validation_mae"]-record["validation_mae"]) < 1e-7
        validation_checks += 1
    assert validation_checks == 40+len(matched_complete)
    target_map = {}
    with np.load(protocol["source_dataset"], allow_pickle=False) as saved:
        original = {key: saved[key] for key in saved.files}
    # Recompute normalization directly from training observations only, without
    # calling the implementation that produced the saved statistics.
    observations = np.asarray(original["observations"][:126], dtype=float).reshape(-1, 24, 5)
    train_times = np.tile(original["times"], 126)
    empirical_scale = observations.std(axis=0, ddof=1)
    pooled_scale = observations.reshape(-1, 5).std(axis=0, ddof=1)
    empirical_scale = np.where(empirical_scale > 1e-8, empirical_scale, np.maximum(pooled_scale[None], 1.))
    empirical_medians = np.stack([np.median(observations[train_times % 72 == slot], axis=0) for slot in range(72)])
    with np.load(output/"training_stats.npz") as saved:
        assert np.array_equal(saved["mean"], observations.mean(axis=0))
        assert np.array_equal(saved["scale"], empirical_scale)
        assert np.array_equal(saved["median_by_slot"], empirical_medians)
    dataset_manifest = {item["dataset_seed"]: item for item in json.loads((output/"final_dataset_manifest.json").read_text())}
    assert set(dataset_manifest) == set(protocol["final_dataset_seeds"])
    assert len({item["inflow_sha256"] for item in dataset_manifest.values()}) == 3
    repair = json.loads((output/"matched_ablation/EVALUATION_LOAD_REPAIR.json").read_text())
    assert digest(Path(__file__).with_name("run_forecast_matched_safe_load.py")) == repair["wrapper_sha256"]
    assert digest(Path(__file__).with_name("run_forecast_matched_ablation.py")) == repair["frozen_evaluator_sha256"]
    assert repair["matched_predictions_present_when_repair_recorded"] == 0
    for filename, expected_digest in repair["dataset_sha256"].items():
        assert digest(output/"final_datasets"/filename) == expected_digest
    tests = json.loads((output/"unit_test_results.json").read_text())
    assert tests["status"] == "PASS" and tests["tests_run"] == 9
    assert digest(Path(__file__).resolve().parents[1]/tests["test_file"]) == tests["test_file_sha256"]
    graph_hash = hashlib.sha256(original["adjacency"].tobytes()).hexdigest()
    for tag in ["legacy_test"]+[f"independent_{seed}" for seed in protocol["final_dataset_seeds"]]:
        with np.load(output/"targets"/f"{tag}.npz") as saved:
            targets = {key: saved[key] for key in saved.files}
        assert targets["truth"].shape == (1809, 24, 4)
        assert targets["horizons"].tolist() == [1, 2, 4, 6]
        assert targets["day"].tolist() == np.repeat(np.arange(153, 180), 67).tolist()
        assert targets["origin"].tolist() == np.tile(np.arange(67), 27).tolist()
        if tag == "legacy_test":
            data = original
        else:
            path = output/"final_datasets"/f"dataset_seed{tag.removeprefix('independent_')}.npz"
            assert path.stat().st_mtime >= (output/"selection_freeze.json").stat().st_mtime
            assert path.stat().st_mtime >= (output/"matched_ablation/selection_freeze.json").stat().st_mtime
            with np.load(path, allow_pickle=False) as saved:
                data = {key: saved[key] for key in saved.files if key != "config"}
            manifest_entry = dataset_manifest[int(tag.removeprefix("independent_"))]
            assert digest(path) == manifest_entry["npz_sha256"]
            assert hashlib.sha256(data["inflow"].tobytes()).hexdigest() == manifest_entry["inflow_sha256"]
            assert hashlib.sha256(data["adjacency"].tobytes()).hexdigest() == graph_hash
            assert not np.array_equal(data["inflow"], original["inflow"])
        positions = targets["origin"]-int(data["times"][0])
        for j, horizon in enumerate([1, 2, 4, 6]):
            assert np.array_equal(targets["truth"][:, :, j], data["inflow"][targets["day"], positions+horizon])
        target_map[tag] = targets
    primary = check_metrics(output, output, target_map)
    matched = check_metrics(output/"matched_ablation", output, target_map, matched=True)
    assert (primary["metrics_rows"], primary["daily_rows"], primary["prediction_arrays"]) == (540, 11664, 108)
    assert (matched["metrics_rows"], matched["daily_rows"], matched["prediction_arrays"]) == (144, 2916, 27)
    assert check_summary(output, "independent_summary.csv") == 44
    assert check_summary(output/"matched_ablation", "summary.csv") == 12
    pairs = pd.read_csv(output/"matched_ablation/paired_differences.csv")
    assert len(pairs) == 96
    assert not pairs.duplicated(["ablation", "training_seed", "dataset", "horizon_steps"]).any()
    matched_metrics = pd.read_csv(output/"matched_ablation/metrics.csv")
    pivot = matched_metrics.pivot(index=["training_seed", "dataset", "horizon_steps"], columns="model", values="mae")
    for row in pairs.itertuples():
        values = pivot.loc[row.training_seed, row.dataset, row.horizon_steps]
        assert abs(values[row.ablation]-values.edge_stgru-row.ablation_minus_edge_mae) < 1e-12
    report = {"status": "PASS", "primary": primary, "matched_ablation": matched,
              "validation_training_runs_checked": validation_checks,
              "primary_training_runs_checked": len(primary_complete),
              "additional_matched_training_runs_checked": len(matched_complete),
              "training_only_normalization_recomputed_exactly": True,
              "hyperparameter_candidates_per_neural_family": 3, "neural_families": 8,
              "training_seeds_per_selected_neural_family": 3,
              "independent_final_dataset_seeds": protocol["final_dataset_seeds"],
              "final_test_days_per_dataset": 27, "target_values_per_horizon_per_dataset": 43416,
              "matched_pair_rows_checked": len(pairs),
              "primary_summary_rows_checked": 44, "matched_summary_rows_checked": 12,
              "unit_tests_passed": tests["tests_run"],
              "evaluation_load_repair_sha256": digest(output/"matched_ablation/EVALUATION_LOAD_REPAIR.json"),
              "evaluation_loader_wrapper_sha256": repair["wrapper_sha256"],
              "dataset_archives_unchanged_by_loader_repair": True,
              "protocol_sha256": digest(output/"protocol.json"),
              "selection_freeze_sha256": digest(output/"selection_freeze.json"),
              "matched_protocol_sha256": digest(output/"matched_ablation/protocol.json"),
              "scope": "Checks stored arrays, targets, selection, hashes and protocol timing; does not establish field validity, optimal tuning or methodological superiority."}
    (output/"NUMERIC_AUDIT.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/supplement_v3/forecast")
    audit(parser.parse_args().output)
