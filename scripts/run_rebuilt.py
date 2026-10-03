"""Run the newly specified study without borrowing any historical paper values."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    from iotexp.scenic import ScenicBackend
    from iotexp.engine import train_one, make_policy, evaluate_episode
    from iotexp.telemetry import CONDITIONS
    from iotexp.util import digest, write_json
    import numpy as np
    output = Path(args.out)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    backend = ScenicBackend(config)
    backend.load(output / "assets")
    if args.stage == "train":
        result = train_one(backend, config, args.method, args.seed, output)
        print(json.dumps(result), flush=True)
        return
    model = make_policy(backend, config, args.method, args.seed)
    cp = output / "checkpoints" / f"{args.method}_seed{args.seed}" / "best.pt"
    model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True)["state_dict"])
    conditions = CONDITIONS if args.method in config["robustness_methods"] else [CONDITIONS[0]]
    rows = []
    for condition in conditions:
        for env_seed in config["test_environment_seeds"]:
            for replicate in config["corruption_seeds"]:
                corruption = int(np.random.SeedSequence([env_seed, replicate, 500009]).generate_state(1)[0])
                relative = Path("traces") / f"{args.method}_s{args.seed}_{condition.name}_e{env_seed}_r{replicate}.npz"
                trial_config = dict(config)
                if args.seed == 0 and env_seed == 0 and condition.name in ("nominal", "missing_0.30", "delay_2"):
                    trial_config["trace_mode"] = "full"
                r = evaluate_episode(model, backend, trial_config, "test", env_seed, condition, corruption,
                                     output / relative)
                rows.append(dict(method=args.method, training_seed=args.seed, environment_seed=env_seed,
                                 corruption_replicate=replicate, corruption_seed=corruption,
                                 condition=condition.name, missing_probability=condition.missing,
                                 noise_scale=condition.noise, delay_intervals=condition.delay,
                                 is_demo=False, trace_path=relative.as_posix(),
                                 checkpoint_sha256=digest(cp), trainable_parameters=model.parameter_count(), **r))
        print(f"{args.method} seed {args.seed}: evaluated {condition.name}", flush=True)
    target = output / "evaluation_shards" / f"{args.method}_{args.seed}.csv"
    target.parent.mkdir(exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate", "parallel-train", "parallel-evaluate", "aggregate"])
    parser.add_argument("--out", default="runs/scenic_rebuild_v1")
    parser.add_argument("--config", default="configs/scenic_rebuild.json")
    parser.add_argument("--method", default="F0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.out)
    if args.stage in ("train", "evaluate"):
        worker(args)
        return
    if args.stage == "prepare":
        import torch
        from iotexp.scenic import ScenicBackend
        from iotexp.util import digest, write_json, environment_info, source_hash
        from iotexp.telemetry import condition_dicts
        if output.exists():
            raise FileExistsError(output)
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        output.mkdir(parents=True)
        torch.set_num_threads(2)
        write_json(output / "config.json", config)
        backend = ScenicBackend(config)
        start = time.perf_counter()
        backend.prepare(output / "assets")
        write_json(output / "manifest.json", dict(
            is_demo=False, data_provenance="new_fully_synthetic_reimplementation",
            reproduces_original=False, source_sha256=source_hash(),
            config_sha256=digest(output / "config.json"),
            asset_sha256={p.relative_to(output).as_posix(): digest(p) for p in sorted((output / "assets").rglob("*")) if p.is_file()},
            software=environment_info(), backend=backend.describe(), horizons=backend.horizons,
            conditions=condition_dicts(), preparation_seconds=time.perf_counter()-start,
            checkpoint_selection="maximum mean validation reward; no test-based selection",
            uncertainty_unit="independently trained policy seed; held-out days paired",
            measurement_unit="service request, not unique visitor",
            warning="New ScenicIoT-Rebuild study; historical tables are not reproduced or combined"))
        return
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if args.stage.startswith("parallel-"):
        stage = args.stage.removeprefix("parallel-")
        logdir = output / "logs"
        logdir.mkdir(exist_ok=True)
        pending = [(m, s) for s in config["training_seeds"] for m in config["methods"]]
        running = []
        done = 0
        while pending or running:
            while pending and len(running) < args.workers:
                if stage == "evaluate":
                    ready = next((i for i, (m, s) in enumerate(pending)
                                  if (output/f"checkpoints/{m}_seed{s}/metadata.json").exists()), None)
                    if ready is None:
                        break
                    pending.insert(0, pending.pop(ready))
                method, seed = pending.pop(0)
                target = output / (f"checkpoints/{method}_seed{seed}/metadata.json" if stage == "train" else f"evaluation_shards/{method}_{seed}.csv")
                if target.exists():
                    done += 1
                    continue
                handle = (logdir / f"{stage}_{method}_{seed}.log").open("w", encoding="utf-8")
                cmd = [sys.executable, str(Path(__file__).resolve()), stage, "--out", str(output.resolve()),
                       "--method", method, "--seed", str(seed)]
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                running.append((process, handle, method, seed))
                print(f"Started {stage} {method} seed {seed}", flush=True)
            for item in list(running):
                process, handle, method, seed = item
                status = process.poll()
                if status is not None:
                    handle.close()
                    running.remove(item)
                    if status:
                        for p, f, _, _ in running:
                            p.terminate()
                            f.close()
                        raise RuntimeError(f"Worker failed {method} seed {seed}: inspect logs")
                    done += 1
                    print(f"Completed {stage} {done}/{len(config['training_seeds'])*len(config['methods'])}: {method} seed {seed}", flush=True)
            time.sleep(1)
        return
    import pandas as pd
    from iotexp.analysis import summarize
    shards = list((output / "evaluation_shards").glob("*.csv"))
    pd.concat([pd.read_csv(p) for p in shards], ignore_index=True).to_csv(output / "episodes.csv", index=False)
    summarize(output)
    print(f"Aggregated {len(shards)} policy shards", flush=True)


if __name__ == "__main__":
    main()
