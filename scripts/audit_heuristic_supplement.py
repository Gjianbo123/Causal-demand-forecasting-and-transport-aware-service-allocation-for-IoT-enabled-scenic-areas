"""Independent read-only audit of all saved supplementary heuristic traces.

Uses generated demand, saved target/ledger/queue/wait arrays and geometry;
does not call the heuristic, repair function, environment step or runner.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iotexp.supplement_common import V3, scenarios, make_scenario_backend, verify_v2
from iotexp.util import digest, write_json


def close(left, right, label):
    if not np.allclose(left, right, rtol=1e-7, atol=1e-5, equal_nan=True):
        raise AssertionError(label)


def main():
    begin = time.perf_counter()
    verify_v2()
    out = V3 / "heuristics"
    hashes = json.loads((out / "scenario_hashes.json").read_text())
    frames = []
    input_hashes = {}
    expected = {(name, day) for name, _, _ in scenarios() for day in range(27)}
    for method in ("static", "reactive", "lookahead"):
        path = out / f"{method}_episodes.csv"
        frame = pd.read_csv(path)
        assert len(frame) == 189 and set(frame.method) == {method}
        assert not frame.duplicated(["scenario", "environment_seed"]).any()
        assert set(zip(frame.scenario, frame.environment_seed)) == expected
        input_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    total_waits = 0
    causal_episodes = 0
    trace_hashes = {}
    pooled = []
    for name, seed, surge in scenarios():
        backend = make_scenario_backend(seed, surge)
        cfg, data = backend.cfg, backend.data
        assert hashes[name] == dict(seed=seed, surge=surge, **backend.dataset_hashes())
        totals = np.asarray(cfg["resource_totals"])
        holding = np.zeros((4, 24), int)
        holding[:, :16] = np.asarray(cfg["holding_per_service_node"])[:, None]
        for row in table[table.scenario == name].itertuples():
            assert row.dataset_seed == seed and row.surge == surge and row.training_seed == -1
            day = 153 + row.environment_seed
            requests = data["demand"][day, 15:]
            assert row.arrivals_count == requests.sum()
            assert row.decision_epochs == 72 and row.served_count + row.unserved_count == row.arrivals_count
            path = ROOT / row.trace_path
            trace_hashes[row.trace_path] = digest(path)
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z["decision_time"], np.arange(72))
                assert np.array_equal(z["arrivals_count"], requests.sum((1, 2)))
                waits = z["served_waits_min"]
                assert len(waits) == row.served_count and (waits >= 0).all() and (waits % 10 == 0).all()
                close(waits.sum(), row.wait_sum_min, "raw served wait sum")
                close(waits.mean(), row.mean_wait_min, "raw served wait mean")
                close(np.percentile(waits, 95), row.p95_wait_min, "raw served P95")
                close(z["occupancy"].sum(1), z["unserved_count"], "post-service queue counts")
                area = 10 * z["unserved_count"][:-1].sum()
                close(z["queue_area_min"], np.r_[0, 10*z["unserved_count"][:-1]], "per-step accrued queue integral")
                close(area, row.waiting_area_min, "total accrued queue integral")
                close(area / row.arrivals_count, row.restricted_mean_wait_min, "restricted waiting")
                terminal = z["terminal_queue_batches"]
                assert terminal.sum() == row.unserved_count == z["unserved_count"][-1]
                close(area, waits.sum() + 10*(terminal*(72-np.arange(73))).sum(), "served plus terminal censored waiting")
                close(100*row.unserved_count/row.arrivals_count, row.unserved_pct, "unserved proportion")
                target = z["executed_action"]
                inventory = z["ledger"][:, :96].reshape(72, 4, 24)
                reserved = z["ledger"][:, 96:192].reshape(72, 4, 24)
                for value in (target, inventory, reserved):
                    assert (value >= 0).all() and np.array_equal(value, np.rint(value))
                assert np.array_equal(target.sum(2), np.tile(totals, (72, 1)))
                assert np.array_equal((inventory+reserved).sum(2), np.tile(totals, (72, 1)))
                assert (target >= reserved).all() and (target <= holding[None]).all()
                assert (inventory+reserved <= holding[None]).all()
                # Independently reconstruct nearest-origin matching from each
                # saved pre-action ledger and the repaired destination target.
                movement = np.zeros(72)
                for t in range(72):
                    for k in range(4):
                        current = inventory[t, k] + reserved[t, k]
                        supply = np.maximum(current-target[t, k], 0).astype(int)
                        need = np.maximum(target[t, k]-current, 0).astype(int)
                        assert (supply <= inventory[t, k]).all()
                        for dest in np.flatnonzero(need):
                            origins = sorted(np.flatnonzero(supply), key=lambda i: (data["shortest_distance_m"][i, dest], i))
                            for origin in origins:
                                count = min(supply[origin], need[dest])
                                movement[t] += count*data["shortest_distance_m"][origin, dest]/1000
                                supply[origin] -= count
                                need[dest] -= count
                            assert need[dest] == 0
                        assert not supply.any()
                close(movement, z["movement_cost"], "independent target-to-transport movement")
                close(movement.sum(), row.movement_cost, "total movement")
                occupancy = z["occupancy"][:, :16]
                reward = -(occupancy.mean(1)/cfg["queue_reward_scale"] + cfg["movement_penalty"]*movement +
                           cfg["overload_penalty"]*np.maximum(occupancy-cfg["queue_capacity_per_service_node"], 0).mean(1))
                close(reward, z["reward"], "physical queue and movement reward")
                close(reward.mean(), row.mean_reward, "mean reward")
                overload = occupancy > cfg["queue_capacity_per_service_node"]
                assert overload.sum() == row.visitor_overload_count and overload.size == row.visitor_overload_denominator
                assert row.executable_infeasible_epochs == 0 and not z["executable_infeasible"].any()
                for key, field in [("candidate_infeasible", "candidate_infeasible_epochs"), ("repaired", "repair_epochs")]:
                    assert z[key].sum() == getattr(row, field)
                assert z["resource_in_transit_units"][-1] == row.final_in_transit_units
                if "source_event" in z.files:
                    assert row.environment_seed == 0
                    event, receipt = z["source_event"], z["source_receipt"]
                    grid = np.arange(72)[:, None, None] - np.arange(11, -1, -1)[None, :, None]
                    assert (np.isnan(event) | (event <= grid)).all()
                    assert (np.isnan(receipt) | (receipt <= np.arange(72)[:, None, None])).all()
                    close(z["observed"][:, -1, :, 1:].sum((1, 2)), np.r_[0, z["unserved_count"][:-1]], "causal observed queue")
                    causal_episodes += 1
                total_waits += len(waits)
        print(f"Audited all heuristics {name}", flush=True)
    for (method, name), group in table.groupby(["method", "scenario"]):
        pooled.append(dict(method=method, scenario=name,
                           restricted_mean_wait_min=float(group.waiting_area_min.sum()/group.arrivals_count.sum()),
                           unserved_pct=float(100*group.unserved_count.sum()/group.arrivals_count.sum()),
                           movement_cost=float(group.movement_cost.mean())))
    report = dict(status="PASS", episodes=len(table), steps=len(table)*72,
                  raw_completed_request_samples=total_waits, detailed_causal_episodes=causal_episodes,
                  checks=["complete three-method seven-scenario 27-day matrix", "frozen v2 source and dataset hashes",
                          "independently generated request denominators", "raw served wait mean/P95/sum",
                          "queue area and terminal censored wait identity", "resource target budgets, holding caps and reservations",
                          "independent nearest-origin movement reconstruction", "queue/movement reward reconstruction",
                          "reported feasibility flags and backlog exceedance counts", "day0 causal event/receipt/queue traces"],
                  limits="No heuristic rerun or full FIFO service replay; causal packet histories exist only for day0 (21 episodes).",
                  input_sha256=input_hashes, trace_sha256=trace_hashes, elapsed_seconds=time.perf_counter()-begin)
    audit_dir = V3 / "audits"
    audit_dir.mkdir(exist_ok=True)
    write_json(audit_dir/"heuristics_independent_audit.json", report)
    pd.DataFrame(pooled).to_csv(audit_dir/"heuristics_per_scenario_independent.csv", index=False)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
