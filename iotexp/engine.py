"""Paired PPO training/evaluation with clean separation of observations and truth."""
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from .fusion import ActorCritic
from .metrics import Metrics, forecast_errors
from .telemetry import TelemetryGateway, Condition, CONDITIONS
from .util import digest, write_json


def tensor(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def make_policy(backend, config, method, seed):
    p = config["ppo"]
    return ActorCritic(method, backend.state_dim, backend.nodes, backend.resources,
                       hidden=p["hidden"], embed_dim=p["embed_dim"],
                       horizons=len(backend.horizons), seed=seed,
                       action_shape=getattr(backend, "action_shape", None)).to(config["device"])


class Session:
    def __init__(self, backend, split, environment_seed, condition, corruption_seed):
        self.backend = backend
        self.env = backend.make_env(split, environment_seed)
        self.gateway = TelemetryGateway(backend.stats, condition, corruption_seed, backend.history)
        frames = self.env.reset()
        if not frames or frames[-1].time != 0:
            raise ValueError("Warm-up frames must end at decision t=0")
        if len(frames) < backend.history + condition.delay:
            raise ValueError("Warm-up does not cover history plus upload delay")
        for f in frames:
            self.gateway.submit(f.time, f.visitor)
        self.frame = frames[-1]
        self.truth = {self.frame.time: self.frame.visitor[:, 0].copy()}
        self._input_cache = None

    def inputs(self, method, device):
        if self._input_cache is None:
            window = self.gateway.window(self.frame.time)
            state = self.backend.encode_state(window, self.frame.ledger.copy())
            forecast = np.asarray(self.backend.forecast(window), dtype=np.float32)
            self._input_cache = (window, state, forecast)
        else:
            window, state, forecast = self._input_cache
        if state.shape != (self.backend.state_dim,) or forecast.shape != (
                self.backend.nodes, len(self.backend.horizons)):
            raise ValueError("Backend state/forecast shape mismatch")
        if not np.isfinite(state).all() or not np.isfinite(forecast).all():
            raise ValueError("Nonfinite policy input")
        scaled = (forecast - self.backend.stats.mean[:, 0, None]) / self.backend.stats.scale[:, 0, None]
        if method == "F0":
            # Forecast may be measured as a shadow diagnostic; it never reaches reactive PPO.
            scaled = np.zeros_like(scaled)
        return tensor(state[None], device), tensor(scaled[None], device), forecast, window

    def step(self, raw):
        previous_time = self.frame.time
        frame, reward, done, audit = self.env.step(raw)
        if frame.time != previous_time + 1:
            raise ValueError("Environment must advance exactly one decision interval")
        if not np.isfinite(reward):
            raise ValueError("Nonfinite reward")
        self.frame = frame
        self._input_cache = None
        self.truth[frame.time] = frame.visitor[:, 0].copy()
        self.gateway.submit(frame.time, frame.visitor)
        return float(reward), bool(done), audit


def generalized_advantage(rewards, values, next_values, dones, gamma, lam):
    rewards, values, next_values = map(lambda a: np.asarray(a, float), (rewards, values, next_values))
    dones = np.asarray(dones, bool)
    advantage = np.empty_like(rewards)
    carry = 0.0
    for t in range(len(rewards)-1, -1, -1):
        alive = float(not dones[t])
        delta = rewards[t] + gamma * next_values[t] * alive - values[t]
        carry = delta + gamma * lam * alive * carry
        advantage[t] = carry
    return advantage, advantage + values


def ppo_update(model, optimizer, buffer, cfg, device, rng):
    state = tensor(np.stack(buffer["state"]), device)
    forecast = tensor(np.stack(buffer["forecast"]), device)
    raw = tensor(np.stack(buffer["raw"]), device)
    old_logp = tensor(buffer["logp"], device)
    adv, returns = generalized_advantage(buffer["reward"], buffer["value"], buffer["next_value"],
                                        buffer["done"], cfg["gamma"], cfg["gae_lambda"])
    advantage = tensor((adv - adv.mean()) / (adv.std() + 1e-8), device)
    returns = tensor(returns, device)
    losses, divergences = [], []
    for _ in range(cfg["update_epochs"]):
        for start in range(0, len(raw), cfg["minibatch_size"]):
            # One independently shuffled permutation per epoch is prepared below.
            if start == 0:
                permutation = rng.permutation(len(raw))
            idx = torch.as_tensor(permutation[start:start+cfg["minibatch_size"]], device=device)
            dist, value, _ = model(state[idx], forecast[idx])
            new_logp = dist.log_prob(raw[idx])  # raw action, NEVER repaired integers
            log_ratio = new_logp - old_logp[idx]
            ratio = log_ratio.exp()
            approx_kl = float(((ratio - 1) - log_ratio).mean().detach())
            if cfg.get("target_kl") and approx_kl > 1.5 * cfg["target_kl"]:
                return {"loss": float(np.mean(losses)) if losses else 0.0,
                        "approx_kl": approx_kl, "kl_early_stop": True}
            unclipped = ratio * advantage[idx]
            clipped = ratio.clamp(1-cfg["clip"], 1+cfg["clip"]) * advantage[idx]
            loss = -torch.minimum(unclipped, clipped).mean() + cfg["value_coef"] * (
                value - returns[idx]).square().mean() - cfg["entropy_coef"] * dist.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO loss is nonfinite; stop and inspect configuration")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            divergences.append(float(((ratio - 1) - log_ratio).mean().detach()))
    return {"loss": float(np.mean(losses)), "approx_kl": float(np.mean(divergences))}


def evaluate_episode(model, backend, config, split, env_seed, condition, corruption_seed, trace_path=None):
    device = config["device"]
    model.eval()
    session = Session(backend, split, env_seed, condition, corruption_seed)
    metrics, forecasts = Metrics(), {}
    rewards, latency_values, pipeline_values = [], [], []
    waiting_area = 0.0
    trace = {k: [] for k in ("decision_time", "raw_action", "candidate_action", "executed_action",
                            "observed", "mask", "age", "source_event", "source_receipt", "ledger",
                            "forecast", "truth_visitor_before", "truth_visitor_after", "occupancy",
                            "visitor_capacity", "overload_nodes", "candidate_infeasible", "executable_infeasible",
                            "repaired", "reward", "unserved_count", "movement_cost",
                            "attention", "policy_latency_ms")}
    done = False
    while not done:
        pipeline_begin = time.perf_counter()
        state, forecast_input, forecast, window = session.inputs(model.method, device)
        t = session.frame.time
        forecasts[t] = forecast.copy()
        ledger, before = session.frame.ledger.copy(), session.frame.visitor.copy()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        begin = time.perf_counter()
        raw, _, _, weights = model.act(state, forecast_input, deterministic=True)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        latency = 1000 * (time.perf_counter() - begin)
        raw_np = raw[0].cpu().numpy()
        pipeline_before_repair = 1000 * (time.perf_counter() - pipeline_begin)
        reward, done, audit = session.step(raw_np)
        metrics.add(audit)
        rewards.append(reward)
        latency_values.append(latency)
        pipeline_values.append(pipeline_before_repair + float(getattr(session.env, "last_repair_ms", 0.0)))
        waiting_area += float(getattr(audit, "queue_area_min", 0.0))
        entries = {"decision_time": t, "raw_action": raw_np, "candidate_action": audit.candidate_action,
                   "executed_action": audit.executed_action, "observed": window.values,
                   "mask": window.mask, "age": window.age, "source_event": window.source_event,
                   "source_receipt": window.source_receipt, "ledger": ledger, "forecast": forecast,
                   "truth_visitor_before": before, "truth_visitor_after": session.frame.visitor.copy(),
                   "occupancy": audit.occupancy, "visitor_capacity": audit.visitor_capacity,
                   "overload_nodes": audit.overload_nodes,
                   "candidate_infeasible": audit.candidate_infeasible,
                   "executable_infeasible": audit.executable_infeasible, "repaired": audit.repaired,
                   "reward": reward, "unserved_count": audit.unserved_count,
                   "movement_cost": audit.movement_cost, "policy_latency_ms": latency,
                   "attention": weights[0].cpu().numpy() if weights is not None else np.full(
                       (backend.resources, backend.nodes, len(backend.horizons)), np.nan)}
        for k, v in entries.items():
            trace[k].append(v)
    result = metrics.result()
    result["mean_reward"] = float(np.mean(rewards))
    result["waiting_area_min"] = waiting_area
    result["restricted_mean_wait_min"] = waiting_area / max(result["arrivals_count"], 1)
    result["unserved_pct"] = 100 * result["unserved_count"] / max(result["arrivals_count"], 1)
    result.update(forecast_errors(forecasts, session.truth, backend.horizons))
    # Discard initial decisions for a descriptive latency estimate. No network/forecast/repair timing.
    result["policy_latency_ms"] = float(np.mean(latency_values[min(5, len(latency_values)-1):]))
    result["local_pipeline_latency_ms"] = float(np.mean(pipeline_values[min(5, len(pipeline_values)-1):]))
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        if config.get("trace_mode") == "compact":
            # Raw served waits permit exact pooling; one causal audit is additionally
            # retained per seed/method at nominal day zero by the runner.
            keep = ("decision_time", "candidate_infeasible", "executable_infeasible", "repaired",
                    "unserved_count", "movement_cost", "reward", "attention", "policy_latency_ms")
            saved = {k: np.asarray(trace[k]) for k in keep}
        else:
            saved = {k: np.asarray(v) for k, v in trace.items()}
            saved.update(session.gateway.arrays())
        np.savez_compressed(trace_path, **saved, served_waits_min=np.asarray(metrics.waits))
    return result


def train_one(backend, config, method, seed, output_dir):
    cfg, device = config["ppo"], config["device"]
    model = make_policy(backend, config, method, seed)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"])
    sample_rng = torch.Generator(device=device).manual_seed(seed + 200003)
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 300007]))
    output = Path(output_dir) / "checkpoints" / f"{method}_seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    log_file = output / "training.jsonl"
    episode = 0
    def new_session():
        # Same seed schedule across methods; policy sampling never affects arrival RNG.
        return Session(backend, "train", seed * 100000 + episode, Condition(), seed + 400009)
    session = new_session()
    total_steps, best_score, best_step = 0, float("inf"), None
    next_validation = cfg["validate_every_steps"]
    while total_steps < cfg["total_steps"]:
        model.train()
        buffer = {k: [] for k in ("state", "forecast", "raw", "logp", "value", "next_value", "reward", "done")}
        rollout = min(cfg["rollout_steps"], cfg["total_steps"] - total_steps)
        for _ in range(rollout):
            state, forecast, _, _ = session.inputs(method, device)
            raw, logp, value, _ = model.act(state, forecast, sample_rng)
            reward, done, _ = session.step(raw[0].cpu().numpy())
            with torch.no_grad():
                if done:
                    next_value = 0.0
                else:
                    ns, nf, _, _ = session.inputs(method, device)
                    next_value = float(model(ns, nf)[1].item())
            entries = {"state": state[0].cpu().numpy(), "forecast": forecast[0].cpu().numpy(),
                       "raw": raw[0].cpu().numpy(), "logp": float(logp.item()), "value": float(value.item()),
                       "next_value": next_value, "reward": reward * cfg.get("reward_scale", 1.0), "done": done}
            for k, v in entries.items():
                buffer[k].append(v)
            total_steps += 1
            if done:
                episode += 1
                session = new_session()
        loss = ppo_update(model, optimizer, buffer, cfg, device, shuffle_rng)
        record = {"step": total_steps, "completed_episodes": episode, **loss}
        if total_steps >= next_validation or total_steps == cfg["total_steps"]:
            results = [evaluate_episode(model, backend, config, "validation", e, Condition(), 0)
                       for e in config["validation_seeds"]]
            served = sum(r["served_count"] for r in results)
            if config.get("checkpoint_selection") == "validation_reward":
                score = -float(np.mean([r["mean_reward"] for r in results]))
            else:
                score = sum(r["wait_sum_min"] for r in results) / served if served else float("inf")
            record["validation_selection_score"] = score
            record["validation_mean_reward"] = float(np.mean([r["mean_reward"] for r in results]))
            pooled_wait = sum(r["wait_sum_min"] for r in results) / served if served else float("inf")
            record["validation_mean_wait_min"] = pooled_wait if np.isfinite(pooled_wait) else None
            record["validation_unserved_count"] = sum(r["unserved_count"] for r in results)
            if score < best_score:
                best_score, best_step = score, total_steps
                torch.save({"state_dict": model.state_dict(), "method": method, "training_seed": seed,
                            "step": total_steps, "validation_score": score, "is_demo": bool(backend.is_demo)},
                           output / "best.pt")
            while next_validation <= total_steps:
                next_validation += cfg["validate_every_steps"]
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
    if best_step is None:
        raise RuntimeError("No finite validation checkpoint; inspect served/unserved population")
    summary = {"method": method, "training_seed": seed, "total_steps": total_steps,
               "best_step": best_step, "validation_selection_score": best_score,
               "selection_rule": config.get("checkpoint_selection", "served_mean_wait"),
               "trainable_parameters": model.parameter_count(), "is_demo": bool(backend.is_demo),
               "checkpoint_sha256": digest(output / "best.pt")}
    write_json(output / "metadata.json", summary)
    return summary


def train_all(backend, config, output_dir):
    summaries = []
    for seed in config["training_seeds"]:
        for method in config["methods"]:
            print(f"Training {method}, seed={seed}, steps={config['ppo']['total_steps']}", flush=True)
            summaries.append(train_one(backend, config, method, seed, output_dir))
    write_json(Path(output_dir) / "training_summary.json", summaries)


def evaluate_all(backend, config, output_dir):
    output = Path(output_dir)
    result_path = output / "episodes.csv"
    if result_path.exists():
        raise FileExistsError("episodes.csv already exists; use a fresh output directory")
    # Each row is a paired (training seed, environment seed, corruption replicate) trial.
    partial_path = output / "episodes.in_progress.csv"
    with partial_path.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for seed in config["training_seeds"]:
            for method in config["methods"]:
                model = make_policy(backend, config, method, seed)
                checkpoint = output / "checkpoints" / f"{method}_seed{seed}" / "best.pt"
                metadata = json.loads((checkpoint.parent / "metadata.json").read_text(encoding="utf-8"))
                if digest(checkpoint) != metadata["checkpoint_sha256"]:
                    raise ValueError("Selected checkpoint changed after training")
                data = torch.load(checkpoint, map_location=config["device"], weights_only=True)
                if data["method"] != method or data["training_seed"] != seed or data["is_demo"] != backend.is_demo:
                    raise ValueError("Checkpoint identity mismatch")
                model.load_state_dict(data["state_dict"])
                conditions = CONDITIONS if method in config["robustness_methods"] else [CONDITIONS[0]]
                print(f"Evaluating {method}, seed={seed}, conditions={len(conditions)}", flush=True)
                for condition in conditions:
                    for env_seed in config["test_environment_seeds"]:
                        for replicate in config["corruption_seeds"]:
                            # Identical draws across models/seeds for a given held-out scenario.
                            corruption_seed = int(np.random.SeedSequence([env_seed, replicate, 500009]).generate_state(1)[0])
                            relative = Path("traces") / f"{method}_s{seed}_{condition.name}_e{env_seed}_r{replicate}.npz"
                            row = evaluate_episode(model, backend, config, "test", env_seed, condition,
                                                   corruption_seed, output / relative)
                            row = {"method": method, "training_seed": seed, "environment_seed": env_seed,
                                   "corruption_replicate": replicate, "corruption_seed": corruption_seed,
                                   "condition": condition.name, "missing_probability": condition.missing,
                                   "noise_scale": condition.noise, "delay_intervals": condition.delay,
                                   "is_demo": backend.is_demo, "trace_path": relative.as_posix(),
                                   "checkpoint_sha256": digest(checkpoint),
                                   "trainable_parameters": model.parameter_count(), **row}
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            f.flush()
        os.fsync(f.fileno())
    os.replace(partial_path, result_path)
