import argparse
import importlib
import json
from pathlib import Path
import torch
from .analysis import summarize, plot_results
from .engine import train_all, evaluate_all
from .telemetry import condition_dicts
from .util import write_json, digest, environment_info, source_hash


def backend_from_config(config):
    if config["backend"] == "demo":
        from .demo import DemoBackend
        return DemoBackend(config)
    module_name, factory = config["backend"].split(":", 1)
    return getattr(importlib.import_module(module_name), factory)(config)


def validate_config(config):
    if config["methods"] != ["F0", "F1", "F2", "F3", "F4"]:
        raise ValueError("This prespecified suite requires ordered F0-F4")
    if set(config["robustness_methods"]) != {"F0", "F1", "F4"}:
        raise ValueError("Robustness suite requires F0, F1, F4")
    for key in ("training_seeds", "validation_seeds", "test_environment_seeds", "corruption_seeds"):
        values = config[key]
        if not values or len(values) != len(set(values)) or any(not isinstance(v, int) or v < 0 for v in values):
            raise ValueError(f"{key} must be unique nonnegative integers")
    p = config["ppo"]
    for key in ("total_steps", "rollout_steps", "update_epochs", "minibatch_size", "validate_every_steps", "hidden", "embed_dim"):
        if not isinstance(p[key], int) or p[key] <= 0:
            raise ValueError(f"Invalid PPO {key}")
    if not 0 < p["gamma"] <= 1 or not 0 <= p["gae_lambda"] <= 1 or not 0 < p["clip"] < 1:
        raise ValueError("Invalid PPO gamma/lambda/clip")
    if p["learning_rate"] <= 0 or p["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer configuration")


def main():
    parser = argparse.ArgumentParser(description="IoT experiment suite; demo results are not manuscript results")
    parser.add_argument("command", choices=["run", "train", "evaluate", "summarize", "matrix"])
    parser.add_argument("--config", type=Path, default=Path("configs/demo_quick.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/demo_quick"))
    args = parser.parse_args()
    if args.command == "matrix":
        print(json.dumps({"methods": ["F0", "F1", "F2", "F3", "F4"], "conditions": condition_dicts()}, indent=2))
        return
    if args.command == "summarize":
        summarize(args.out)
        plot_results(args.out)
        print(f"Saved summaries and figures: {args.out.resolve()}")
        return
    if args.command == "evaluate":
        config = json.loads((args.out / "config.json").read_text(encoding="utf-8"))
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    torch.set_num_threads(config.get("torch_threads", 1))
    torch.use_deterministic_algorithms(True)
    backend = backend_from_config(config)
    if args.command in ("train", "run"):
        if args.out.exists():
            raise FileExistsError(f"Output exists: {args.out}; choose a new --out (no automatic overwrite)")
        args.out.mkdir(parents=True)
        write_json(args.out / "config.json", config)
        backend.prepare(args.out / "assets")
        assets = {p.relative_to(args.out).as_posix(): digest(p)
                  for p in sorted((args.out / "assets").rglob("*")) if p.is_file()}
        write_json(args.out / "manifest.json", {"is_demo": bool(backend.is_demo), "data_provenance": "fully_synthetic",
                   "source_sha256": source_hash(), "config_sha256": digest(args.out / "config.json"),
                   "asset_sha256": assets, "software": environment_info(), "backend": backend.describe(),
                   "horizons": list(backend.horizons), "conditions": condition_dicts(),
                   "checkpoint_selection": "minimum pooled served-visitor mean wait on validation seeds",
                   "uncertainty_unit": "independently trained policy seed",
                   "aggregation": "pool served waits within seed/condition; sample SD across training seeds",
                   "warning": "DEMO ONLY; not a reproduction" if backend.is_demo else "Inspect adapter and original simulator provenance"})
        train_all(backend, config, args.out)
    else:
        manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
        if digest(args.out / "config.json") != manifest["config_sha256"] or source_hash() != manifest["source_sha256"]:
            raise ValueError("Configuration/source changed since training; create a separate run")
        for path, expected in manifest["asset_sha256"].items():
            if digest(args.out / path) != expected:
                raise ValueError(f"Frozen training asset changed: {path}")
        backend.load(args.out / "assets")
    if args.command in ("run", "evaluate"):
        evaluate_all(backend, config, args.out)
        summarize(args.out)
        plot_results(args.out)
    print(f"Completed: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
