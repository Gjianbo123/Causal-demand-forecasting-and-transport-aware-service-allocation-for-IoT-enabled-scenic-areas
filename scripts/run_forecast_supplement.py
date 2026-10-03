"""Explicit staged runner; final datasets cannot be evaluated before selection freeze."""
from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iotexp.forecast_supplement import (
    freeze_protocol, train_experiment, evaluate_experiment,
    report_experiment, smoke_experiment,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "freeze", "train", "evaluate", "report"])
    parser.add_argument("--output", default="runs/supplement_v3/forecast")
    arguments = parser.parse_args()
    functions = {"smoke": smoke_experiment, "freeze": freeze_protocol, "train": train_experiment,
                 "evaluate": evaluate_experiment, "report": report_experiment}
    functions[arguments.stage](arguments.output)
