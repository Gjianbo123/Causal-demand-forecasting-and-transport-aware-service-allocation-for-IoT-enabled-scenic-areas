"""Read-only verification of the recorded serial timing table; never remeasure."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iotexp.util import digest, write_json
V3 = ROOT/"runs/supplement_v3"


def main():
    out = V3/"serial_latency"
    protocol = json.loads((out/"protocol.json").read_text())
    assert protocol["workers"] == protocol["torch_threads"] == 1
    assert protocol["cached_forecasts"] is False
    assert protocol["source_sha256"] == digest(ROOT/"scripts/measure_supplement_latency.py")
    assert protocol["solver_options"] == json.loads((V3/"control/solver_protocol.json").read_text())["options"]
    data = pd.read_csv(out/"decisions.csv")
    summary = pd.read_csv(out/"summary.csv").set_index(["method", "predictor"])
    expected = {("static", "edge_stgru"), ("reactive", "edge_stgru"), ("lookahead", "edge_stgru"),
                ("F0", "edge_stgru"), ("F1", "edge_stgru"), ("F4", "edge_stgru"), ("F4", "ridge"), ("MPC", "edge_stgru")}
    assert set(summary.index) == expected
    assert set(zip(data.method, data.predictor)) == expected
    assert len(data) == 8*3*72 and not data.duplicated(["method", "predictor", "day", "step"]).any()
    stages = ["state_ms", "forecast_ms", "decision_ms", "repair_ms"]
    assert np.isfinite(data[stages+["pipeline_ms", "ledger_update_ms"]]).all().all()
    assert (data[stages+["pipeline_ms", "ledger_update_ms"]] >= 0).all().all()
    assert np.allclose(data[stages].sum(1), data.pipeline_ms, atol=1e-10, rtol=1e-12)
    assert np.array_equal(data.warmup, data.step < 5)
    assert (data.repair_ms >= data.ledger_update_ms).all()
    assert (data.loc[data.method != "MPC", "ledger_update_ms"] == 0).all()
    quantile_checks = 0
    for (method, predictor), group in data.groupby(["method", "predictor"]):
        assert set(zip(group.day, group.step)) == {(day, step) for day in [0, 13, 26] for step in range(72)}
        if method.startswith("F"):
            checkpoint = V3/"policies"/predictor/"checkpoints"/f"{method}_seed0"/"best.pt"
            assert set(group.checkpoint_sha256) == {digest(checkpoint)}
        else:
            assert group.checkpoint_sha256.isna().all()
        measured = group[~group.warmup]
        assert len(measured) == summary.loc[(method, predictor), "n"] == 201
        for key in stages+["pipeline_ms"]:
            values = measured[key].to_numpy()
            for suffix, actual in [("mean", values.mean()), ("median", np.median(values)), ("p95", np.percentile(values, 95))]:
                assert np.isclose(actual, summary.loc[(method, predictor), key+"_"+suffix], atol=1e-10, rtol=1e-12)
                quantile_checks += 1
    report = dict(status="PASS", methods=8, raw_decisions=len(data), retained_decisions=int((~data.warmup).sum()),
                  retained_per_method=201, aggregate_checks=quantile_checks,
                  checks=["single-worker single-thread uncached measurement protocol", "frozen MPC options and timing script hash",
                          "complete method/day/step matrix", "seed0 checkpoint hashes", "exact first-five-per-day warmup exclusion",
                          "pipeline equals four measured stages; MPC bookkeeping already in repair", "all stage mean/median/P95 values match saved summary"],
                  limits="Arithmetic and provenance verification; no remeasurement, hardware-general latency claim or real network timing.",
                  input_sha256={path.name: digest(path) for path in [out/"protocol.json", out/"decisions.csv", out/"summary.csv"]})
    write_json(V3/"audits/latency_independent_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
