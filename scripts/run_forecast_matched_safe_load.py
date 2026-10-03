"""Evaluation-only loader repair; preserves frozen sources and every NPZ byte.

The simulator returns a redundant `config` dictionary alongside its arrays.
NumPy serializes this dictionary as an object member; it is never a predictor
input. Exclude that member from iteration while retaining allow_pickle=False.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import run_forecast_matched_ablation as matched
from iotexp.forecast_supplement import sha256, write_json


def evaluate(output):
    output = Path(output).resolve()
    data_directory = output/"final_datasets"
    record_path = output/"matched_ablation/EVALUATION_LOAD_REPAIR.json"
    record = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Evaluation-only compatibility repair. The frozen matched evaluator rejected a redundant object-valued config NPZ member before producing any predictions. Skip that member when iterating arrays, retain allow_pickle=False, and leave NPZ bytes, source protocol, weights, normalization, targets, metrics and all numerical predictor inputs unchanged. The simulator configuration is already recorded as JSON in the frozen primary protocol.",
        "wrapper_sha256": sha256(__file__),
        "frozen_evaluator_sha256": sha256(matched.__file__),
        "dataset_sha256": {path.name: sha256(path) for path in sorted(data_directory.glob("*.npz"))},
        "matched_predictions_present_when_repair_recorded": len(list((output/"matched_ablation/predictions").glob("*.npz"))),
    }
    if record_path.exists():
        stored = json.loads(record_path.read_text())
        for key in ["wrapper_sha256", "frozen_evaluator_sha256", "dataset_sha256"]:
            if stored[key] != record[key]:
                raise ValueError("Recorded evaluation-only loader repair changed: "+key)
    else:
        write_json(record_path, record, exclusive=True)
    original_load = np.load

    def numeric_dataset_load(file, *args, **kwargs):
        saved = original_load(file, *args, **kwargs)
        if isinstance(file, (str, Path)) and Path(file).resolve().parent == data_directory:
            if kwargs.get("allow_pickle", False):
                raise ValueError("Object loading must remain disabled")
            saved.files = [key for key in saved.files if key != "config"]
        return saved

    np.load = numeric_dataset_load
    try:
        matched.evaluate(output)
    finally:
        np.load = original_load


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/supplement_v3/forecast")
    evaluate(parser.parse_args().output)
