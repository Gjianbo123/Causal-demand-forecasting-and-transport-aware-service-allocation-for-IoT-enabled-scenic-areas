"""Audit a completed run. Usage: python validation/check_outputs.py runs/verified_demo"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iotexp.analysis import check_matrix
from iotexp.util import digest, source_hash, write_json


def audit(path):
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert digest(path / "config.json") == manifest["config_sha256"]
    assert source_hash() == manifest["source_sha256"]
    for file, sha in manifest["asset_sha256"].items():
        assert digest(path / file) == sha
    df = pd.read_csv(path / "episodes.csv")
    check_matrix(df, cfg)
    for (method, seed), rows in df.groupby(["method", "training_seed"]):
        folder = path / "checkpoints" / f"{method}_seed{seed}"
        meta = json.loads((folder / "metadata.json").read_text())
        assert meta["total_steps"] == cfg["ppo"]["total_steps"]
        sha = digest(folder / "best.pt")
        assert meta["checkpoint_sha256"] == sha and (rows.checkpoint_sha256 == sha).all()
        for line in (folder / "training.jsonl").read_text().splitlines():
            record = json.loads(line)
            assert np.isfinite(record["loss"]) and np.isfinite(record["approx_kl"])
    masks, noise = {}, {}
    for row in df.itertuples():
        with np.load(path / row.trace_path, allow_pickle=False) as z:
            times = z["decision_time"]
            length = z["observed"].shape[1]
            grid = times[:, None, None] + np.arange(-length+1, 1)[None, :, None]
            valid = np.isfinite(z["source_event"])
            assert np.all((z["source_event"] <= grid) | ~valid)
            assert np.all((z["source_receipt"] <= times[:, None, None]) | ~valid)
            np.testing.assert_allclose(z["age"][valid], np.broadcast_to(grid, valid.shape)[valid] - z["source_event"][valid])
            waits = z["served_waits_min"]
            assert len(waits) == row.served_count
            assert np.isclose(waits.sum(), row.wait_sum_min)
            assert np.isclose(waits.mean(), row.mean_wait_min)
            over = (z["occupancy"] > z["visitor_capacity"]) & z["overload_nodes"].astype(bool)
            assert int(over.sum()) == row.visitor_overload_count
            assert int(z["overload_nodes"].sum()) == row.visitor_overload_denominator
            if manifest["is_demo"]:
                assert row.served_count + row.unserved_count == row.arrivals_count
                np.testing.assert_array_equal(z["executed_action"].sum(-1), np.tile(
                    cfg["demo"]["resource_totals"], (len(times), 1)))
            key = (row.environment_seed, row.corruption_replicate, row.condition)
            for cache, field in ((masks, "packet_dropped"), (noise, "packet_noise_draw")):
                value = z[field]
                if key in cache:
                    np.testing.assert_array_equal(cache[key], value)
                else:
                    cache[key] = value.copy()
    per_seed = pd.read_csv(path / "per_seed.csv")
    assert per_seed.groupby(["method", "condition"]).size().eq(len(cfg["training_seeds"])).all()
    tests = json.loads((path / "paired_tests.json").read_text())
    if len(cfg["training_seeds"]) < cfg["minimum_test_seeds"]:
        assert all(t["p_value"] is None for t in tests)
    report = {"status": "passed", "is_demo": bool(manifest["is_demo"]), "episode_records": len(df),
              "trained_checkpoints": len(df.groupby(["method", "training_seed"])),
              "per_seed_condition_records": len(per_seed), "paired_comparisons": len(tests),
              "checks": ["complete unique matrix", "config/source/assets/checkpoint hashes",
                         "finite training loss and KL", "causal event and receipt times", "waiting samples",
                         "visitor overload counts", "paired noise and packet-loss schedules",
                         "seed aggregation", "minimum sample size for inference"],
              "max_executable_infeasible_pct": float(df.executable_infeasible_pct.max())}
    write_json(path / "audit.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), ensure_ascii=False, indent=2))
