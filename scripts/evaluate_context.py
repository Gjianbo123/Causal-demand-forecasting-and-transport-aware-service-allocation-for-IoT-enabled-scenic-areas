"""Prespecified contextual baselines and frozen-policy travel sensitivity.

Examples (run only after preparation has written manifest.json)::
    python scripts/evaluate_context.py --out runs/scenic_rebuild_v1
    python scripts/evaluate_context.py --out runs/scenic_rebuild_v1 --sensitivity

The baseline stage never reads PPO outputs. The sensitivity stage reads only
the prespecified F1/F4 checkpoints; it neither retrains nor changes saved assets.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from iotexp.engine import Session, evaluate_episode, make_policy
from iotexp.metrics import Metrics
from iotexp.scenic import ScenicBackend
from iotexp.telemetry import CONDITIONS
from iotexp.util import digest, source_hash, write_json


BASELINES = ("static", "reactive", "lookahead")
TRAVEL_MULTIPLIERS = (0.5, 1.5)
POLICY_METHODS = ("F1", "F4")
POLICY_SEEDS = tuple(range(10))
CORRUPTION_REPLICATES = (11,)


def corruption_seed(environment_seed, replicate):
    return int(np.random.SeedSequence([environment_seed, replicate, 500009]).generate_state(1)[0])


def load_frozen_run(output):
    if not (output / "manifest.json").is_file():
        raise RuntimeError("Preparation has not completed: manifest.json is required")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if digest(output / "config.json") != manifest["config_sha256"]:
        raise ValueError("Main run configuration changed after preparation")
    if source_hash() != manifest["source_sha256"]:
        raise ValueError("iotexp source changed after preparation")
    for relative, expected in manifest["asset_sha256"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Frozen asset changed: {relative}")
    if config["test_environment_seeds"] != list(range(27)) or config["corruption_seeds"] != [11]:
        raise ValueError("This context protocol requires all 27 test days and corruption replicate 11")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    backend = ScenicBackend(config)
    backend.load(output / "assets")
    return config, manifest, backend


def baseline_episode(backend, method, env_seed, condition, corruption, trace_path):
    session = Session(backend, "test", env_seed, condition, corruption)
    metrics = Metrics()
    waiting_area = 0.0
    trace = {key: [] for key in (
        "decision_time", "raw_action", "candidate_action", "executed_action", "ledger",
        "candidate_infeasible", "executable_infeasible", "repaired", "reward",
        "arrivals_count", "unserved_count", "movement_cost", "queue_area_min",
        "occupancy", "visitor_capacity", "overload_nodes", "resource_in_transit_units",
        "baseline_logic_latency_ms", "local_pipeline_latency_ms")}
    # A compact causal audit for the same first test day in all nine conditions.
    detailed = env_seed == 0
    if detailed:
        trace.update({key: [] for key in ("observed", "mask", "age", "source_event", "source_receipt")})
    done = False
    while not done:
        pipeline_begin = time.perf_counter()
        window = session.gateway.window(session.frame.time)
        logic_begin = time.perf_counter()
        raw = backend.heuristic_action(window, method)
        logic_ms = 1000 * (time.perf_counter() - logic_begin)
        pipeline_without_repair = 1000 * (time.perf_counter() - pipeline_begin)
        decision_time = session.frame.time
        ledger = session.frame.ledger.copy()
        reward, done, audit = session.step(raw)
        metrics.add(audit)
        if not hasattr(audit, "queue_area_min"):
            raise ValueError("Simulator must expose accrued request waiting")
        waiting_area += float(audit.queue_area_min)
        if audit.executable_infeasible:
            raise RuntimeError("A contextual baseline produced an infeasible execution")
        entry = dict(
            decision_time=decision_time, raw_action=raw,
            candidate_action=audit.candidate_action, executed_action=audit.executed_action,
            ledger=ledger, candidate_infeasible=audit.candidate_infeasible,
            executable_infeasible=audit.executable_infeasible, repaired=audit.repaired,
            reward=reward, arrivals_count=audit.arrivals_count, unserved_count=audit.unserved_count,
            movement_cost=audit.movement_cost, queue_area_min=audit.queue_area_min,
            occupancy=audit.occupancy, visitor_capacity=audit.visitor_capacity,
            overload_nodes=audit.overload_nodes, resource_in_transit_units=audit.resource_in_transit_units,
            baseline_logic_latency_ms=logic_ms,
            local_pipeline_latency_ms=pipeline_without_repair+session.env.last_repair_ms)
        if detailed:
            entry.update(observed=window.values, mask=window.mask, age=window.age,
                         source_event=window.source_event, source_receipt=window.source_receipt)
        for key, value in entry.items():
            trace[key].append(value)
    result = metrics.result()
    if result["served_count"] + result["unserved_count"] != result["arrivals_count"]:
        raise ValueError("Request count conservation failed")
    # This truth is used after rollout for score auditing, never for actions.
    remaining_ages = (session.env.t - np.arange(session.env.queue_batches.shape[-1]))
    censored_waiting = float((session.env.queue_batches * remaining_ages).sum() *
                            session.env.cfg["decision_minutes"])
    if not np.isclose(waiting_area, result["wait_sum_min"] + censored_waiting):
        raise ValueError("Accrued waiting is inconsistent with completed/censored FIFO requests")
    warm = min(5, result["decision_epochs"] - 1)
    result.update(
        waiting_area_min=waiting_area,
        restricted_mean_wait_min=waiting_area / max(result["arrivals_count"], 1),
        unserved_pct=100 * result["unserved_count"] / max(result["arrivals_count"], 1),
        mean_reward=float(np.mean(trace["reward"])),
        baseline_logic_latency_ms=float(np.mean(trace["baseline_logic_latency_ms"][warm:])),
        local_pipeline_latency_ms=float(np.mean(trace["local_pipeline_latency_ms"][warm:])),
        final_in_transit_units=int(session.env.reserved.sum()))
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    saved = {key: np.asarray(value) for key, value in trace.items()}
    saved["served_waits_min"] = np.asarray(metrics.waits, dtype=np.float32)
    saved["terminal_queue_batches"] = session.env.queue_batches.copy()
    if detailed:
        saved.update(session.gateway.arrays())
    np.savez_compressed(trace_path, **saved)
    return result


class AtomicRows:
    def __init__(self, target):
        self.target = target
        self.partial = target.with_suffix(".in_progress.csv")
        if target.exists() or self.partial.exists():
            raise FileExistsError(f"Refusing to overwrite context output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.partial.open("x", newline="", encoding="utf-8")
        self.writer = None
        self.count = 0

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()
        self.count += 1

    def complete(self, expected):
        if self.count != expected:
            raise ValueError(f"Incomplete contextual matrix: {self.count} != {expected}")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.partial, self.target)


def run_baselines(output, config, backend):
    target = output / "context" / "baseline_episodes.csv"
    writer = AtomicRows(target)
    for method in BASELINES:
        for condition in CONDITIONS:
            for env_seed in config["test_environment_seeds"]:
                for replicate in CORRUPTION_REPLICATES:
                    corruption = corruption_seed(env_seed, replicate)
                    relative = Path("context/traces/baselines") / f"{method}_{condition.name}_e{env_seed}_r{replicate}.npz"
                    result = baseline_episode(backend, method, env_seed, condition, corruption, output / relative)
                    writer.add(dict(method=method, training_seed=-1, environment_seed=env_seed,
                                    corruption_replicate=replicate, corruption_seed=corruption,
                                    condition=condition.name, missing_probability=condition.missing,
                                    noise_scale=condition.noise, delay_intervals=condition.delay,
                                    travel_time_multiplier=1.0, is_demo=False, trainable_parameters=0,
                                    trace_path=relative.as_posix(), **result))
            print(f"Baseline {method}: completed {condition.name}", flush=True)
    writer.complete(3 * 9 * 27)
    return writer.count


def run_sensitivity(output, config, backend):
    if config["training_seeds"] != list(POLICY_SEEDS):
        raise ValueError("Sensitivity protocol requires original training seeds 0 through 9")
    checkpoints = {}
    for method in POLICY_METHODS:
        for seed in POLICY_SEEDS:
            path = output / "checkpoints" / f"{method}_seed{seed}" / "best.pt"
            metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
            if digest(path) != metadata["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed: {path}")
            checkpoints[method, seed] = path
    writer = AtomicRows(output / "context" / "sensitivity_episodes.csv")
    original_cfg = copy.deepcopy(backend.cfg)
    original_scale = backend.ledger_scale.copy()
    trial_config = dict(config)
    trial_config["trace_mode"] = "compact"
    try:
        for multiplier in TRAVEL_MULTIPLIERS:
            backend.cfg = {**original_cfg, "travel_time_multiplier": multiplier}
            # Fixed training preprocessing: only actual travel dynamics change.
            np.testing.assert_array_equal(backend.ledger_scale, original_scale)
            for method in POLICY_METHODS:
                for seed in POLICY_SEEDS:
                    checkpoint = checkpoints[method, seed]
                    data = torch.load(checkpoint, map_location=config["device"], weights_only=True)
                    if data["method"] != method or data["training_seed"] != seed or data["is_demo"]:
                        raise ValueError("Wrong sensitivity checkpoint identity")
                    model = make_policy(backend, config, method, seed)
                    model.load_state_dict(data["state_dict"])
                    for env_seed in config["test_environment_seeds"]:
                        for replicate in CORRUPTION_REPLICATES:
                            corruption = corruption_seed(env_seed, replicate)
                            relative = Path("context/traces/sensitivity") / f"travel{multiplier:g}_{method}_s{seed}_e{env_seed}_r{replicate}.npz"
                            result = evaluate_episode(model, backend, trial_config, "test", env_seed,
                                                      CONDITIONS[0], corruption, output / relative)
                            if result["executable_infeasible_pct"] != 0:
                                raise ValueError("Sensitivity execution violated resource constraints")
                            if result["served_count"] + result["unserved_count"] != result["arrivals_count"]:
                                raise ValueError("Sensitivity service request conservation failed")
                            writer.add(dict(method=method, training_seed=seed, environment_seed=env_seed,
                                            corruption_replicate=replicate, corruption_seed=corruption,
                                            condition="nominal", missing_probability=0.0,
                                            noise_scale=0.0, delay_intervals=0,
                                            travel_time_multiplier=multiplier, is_demo=False,
                                            checkpoint_sha256=digest(checkpoint),
                                            trainable_parameters=model.parameter_count(),
                                            trace_path=relative.as_posix(), **result))
                    print(f"Travel x{multiplier:g}: {method} seed {seed} completed", flush=True)
    finally:
        backend.cfg = original_cfg
        np.testing.assert_array_equal(backend.ledger_scale, original_scale)
    writer.complete(2 * 2 * 10 * 27)
    return writer.count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/scenic_rebuild_v1"))
    parser.add_argument("--sensitivity", action="store_true", help="Run frozen F1/F4 travel sensitivity instead of baselines")
    args = parser.parse_args()
    output = args.out.resolve()
    config, manifest, backend = load_frozen_run(output)
    started = time.perf_counter()
    stage = "sensitivity" if args.sensitivity else "baseline"
    count = run_sensitivity(output, config, backend) if args.sensitivity else run_baselines(output, config, backend)
    write_json(output / "context" / f"{stage}_protocol.json", dict(
        stage=stage, completed=True, episodes=count, elapsed_seconds=time.perf_counter()-started,
        primary_run_manifest_sha256=digest(output / "manifest.json"),
        script_sha256=digest(Path(__file__)), iotexp_source_sha256=manifest["source_sha256"],
        methods=list(POLICY_METHODS if args.sensitivity else BASELINES),
        training_seeds=list(POLICY_SEEDS) if args.sensitivity else [],
        training_seed_note="-1 denotes a deterministic untrained baseline, not an independent training seed",
        environment_seeds=config["test_environment_seeds"], corruption_replicates=list(CORRUPTION_REPLICATES),
        travel_multipliers=list(TRAVEL_MULTIPLIERS) if args.sensitivity else [1.0],
        conditions=["nominal"] if args.sensitivity else [c.name for c in CONDITIONS],
        measurement_population="independent service requests, not unique tourists",
        restricted_wait="sum(step-start outstanding requests * 10 min) / all generated requests",
        sensitivity_scope="zero-shot changed travel dynamics; frozen policies/forecaster/normalization; no retraining",
        latency_note="baseline_logic includes its actual forecast call for lookahead; local pipeline includes causal window, logic and repair; excludes queue simulation and physical communications",
        inference_note="deterministic baselines have no policy-seed uncertainty; do not duplicate them across ten seeds as independent trials"))
    print(f"Completed {stage}: {count} episodes in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
