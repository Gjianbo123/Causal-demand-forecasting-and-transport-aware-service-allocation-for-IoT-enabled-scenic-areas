"""NEW simplified synthetic queue example, NOT ScenicFlow-Sim or E-STGNN.

No visitor routing/dwell model or resource travel time is implemented here.
All defaults in this file/config are demonstration choices, not recovered paper settings.
"""
from collections import deque
from pathlib import Path
import numpy as np
from .contracts import Frame, Audit
from .telemetry import TrainingStats


def largest_remainder(weights, total):
    weights = np.asarray(weights, dtype=float)
    if total < 0 or int(total) != total or (weights < 0).any() or not np.isfinite(weights).all():
        raise ValueError("Invalid allocation weights/budget")
    if weights.sum() <= 0:
        raise ValueError("At least one permitted destination is required")
    quotas = weights / weights.sum() * int(total)
    counts = np.floor(quotas).astype(int)
    remainder = int(total) - counts.sum()
    order = np.lexsort((np.arange(len(counts)), -(quotas - counts)))
    counts[order[:remainder]] += 1
    return counts


def allocation_is_feasible(action, totals, allowed, holding):
    a = np.asarray(action)
    return bool(np.isfinite(a).all() and (a >= 0).all() and np.equal(a, np.floor(a)).all()
                and np.array_equal(a.sum(axis=1), totals) and (a <= holding).all()
                and not np.any(a[~allowed]))


def repair_allocation(raw, totals, allowed, holding):
    """Demo-only target allocation repair. Never reads queue truth or forecasts.

    Mask -> per-type softmax -> exact largest remainder -> holding-limit repair.
    Ties use raw score then node index. Relocation is instantaneous in this demo.
    """
    raw = np.asarray(raw, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    holding = np.asarray(holding, dtype=int)
    totals = np.asarray(totals, dtype=int)
    if raw.shape != allowed.shape or raw.shape != holding.shape or not np.isfinite(raw).all():
        raise ValueError("Invalid raw action")
    if (holding < 0).any() or (totals < 0).any() or np.any((holding * allowed).sum(1) < totals):
        raise ValueError("Infeasible configured resource-holding limits")
    candidate = np.zeros_like(holding)
    executed = np.zeros_like(holding)
    for k, total in enumerate(totals):
        permitted = np.flatnonzero(allowed[k])
        weights = np.zeros(raw.shape[1])
        weights[permitted] = np.exp(raw[k, permitted] - raw[k, permitted].max())
        candidate[k] = largest_remainder(weights, total)
        executed[k] = np.minimum(candidate[k], holding[k])
        remaining = total - executed[k].sum()
        order = permitted[np.lexsort((permitted, -raw[k, permitted]))]
        for node in order:
            take = min(remaining, holding[k, node] - executed[k, node])
            executed[k, node] += take
            remaining -= take
        if remaining:
            raise RuntimeError("Repair failed despite sufficient holding capacity")
    if not allocation_is_feasible(executed, totals, allowed, holding):
        raise RuntimeError("Executable action failed final feasibility validation")
    return candidate, executed


class DemoEnvironment:
    def __init__(self, config, split, seed):
        self.cfg, self.split, self.seed = config, split, seed
        self.n = int(config["nodes"])
        self.totals = np.asarray(config["resource_totals"], dtype=int)
        self.k = len(self.totals)
        self.steps = int(config["episode_steps"])
        self.history = int(config["history"])
        self.interval = float(config["decision_minutes"])
        self.allowed = np.ones((self.k, self.n), dtype=bool)
        self.allowed[-1, -1] = False
        self.holding = np.ceil(2 * self.totals[:, None] / (self.n - 1)).astype(int)
        self.holding = np.repeat(self.holding, self.n, axis=1)
        self.capacities = np.full(self.n, float(config["visitor_capacity"]))
        self.rates = np.asarray(config["service_per_resource"], dtype=float)

    def reset(self):
        split_id = {"train": 11, "validation": 23, "test": 37}[self.split]
        rng = np.random.default_rng(np.random.SeedSequence([int(self.seed), split_id, 871]))
        self.warmup = self.history + 2
        times = np.arange(-self.warmup, self.steps + 1)
        phase = 2 * np.pi * times[:, None] / self.steps
        node_phase = np.arange(self.n)[None] * 2 * np.pi / self.n
        # Fully disclosed NEW demonstration generator: independent Poisson external arrivals.
        day_factor = rng.uniform(.85, 1.15)
        intensity = day_factor * (self.cfg["arrival_base"] + self.cfg["arrival_peak"] *
                                 np.maximum(0, np.sin(phase - node_phase)))
        self.arrivals = rng.poisson(intensity).astype(int)
        self.t = 0
        self.queues = [deque() for _ in range(self.n)]
        _, self.allocation = repair_allocation(np.zeros((self.k, self.n)), self.totals,
                                               self.allowed, self.holding)
        # Warm-up supplies historical inflow and zero queue channels; evaluated queues start empty.
        # It is not a hidden simulation of pre-opening visitors or earlier policy decisions.
        frames = []
        for t in range(-self.warmup, 1):
            visitor = np.column_stack((self.arrivals[t + self.warmup], np.zeros(self.n), np.zeros(self.n)))
            frames.append(Frame(t, visitor.astype(float), self.allocation.copy()))
        return frames

    def step(self, raw_action):
        if self.t >= self.steps:
            raise RuntimeError("Episode already ended")
        candidate, executed = repair_allocation(raw_action, self.totals, self.allowed, self.holding)
        pre = not allocation_is_feasible(candidate, self.totals, self.allowed, self.holding)
        movement = float(np.abs(executed - self.allocation).sum() / 2)
        self.allocation = executed.copy()
        self.t += 1
        arriving = self.arrivals[self.t + self.warmup]
        service = np.floor(self.cfg["base_service"] + (executed * self.rates[:, None]).sum(0)).astype(int)
        waits, completed = [], np.zeros(self.n)
        for i in range(self.n):
            self.queues[i].extend([self.t] * int(arriving[i]))
            for _ in range(min(service[i], len(self.queues[i]))):
                waits.append((self.t - self.queues[i].popleft()) * self.interval)
                completed[i] += 1
        queue = np.asarray([len(q) for q in self.queues])
        # In this demo occupancy equals the residual queue; no dwell/service occupancy.
        visitor = np.column_stack((arriving, queue, queue)).astype(float)
        overload = np.maximum(queue - self.capacities, 0).sum()
        reward = -(queue.mean() / 20 + self.cfg["movement_penalty"] * movement +
                   self.cfg["overload_penalty"] * overload / self.n)
        audit = Audit(candidate, executed, pre, False, not np.array_equal(candidate, executed),
                      queue, self.capacities, np.ones(self.n, dtype=bool), np.asarray(waits),
                      int(queue.sum()), int(arriving.sum()), movement)
        return Frame(self.t, visitor, self.allocation.copy()), float(reward), self.t == self.steps, audit


class DemoBackend:
    is_demo = True

    def __init__(self, config):
        self.config = config
        self.cfg = config["demo"]
        self.nodes = self.cfg["nodes"]
        self.resources = len(self.cfg["resource_totals"])
        self.history, self.horizons = self.cfg["history"], self.cfg["horizons"]
        self.state_dim = self.history * self.nodes * 5 + self.resources * self.nodes + 2
        self.stats = None
        self.coef = None

    def make_env(self, split, seed):
        return DemoEnvironment(self.cfg, split, seed)

    def _features(self, history_values, time):
        scaled = self.stats.normalize(history_values)[:, :, 0].ravel()
        phase = 2 * np.pi * time / self.cfg["episode_steps"]
        return np.concatenate((scaled, [np.sin(phase), np.cos(phase), 1.0]))

    def prepare(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        sequences, samples, times = [], [], []
        for seed in range(self.cfg["calibration_days"]):
            env = self.make_env("train", seed)
            frames = env.reset()
            done = False
            while not done:
                frame, _, done, _ = env.step(np.zeros((self.resources, self.nodes)))
                frames.append(frame)
            sequences.append(frames)
            samples.extend([f.visitor for f in frames])
            times.extend([f.time for f in frames])
        self.stats = TrainingStats.fit(samples, times, self.cfg["episode_steps"])
        x, y = [], []
        for frames in sequences:
            for j in range(self.history - 1, len(frames) - max(self.horizons)):
                x.append(self._features(np.stack([f.visitor for f in frames[j-self.history+1:j+1]]), frames[j].time))
                y.append(np.stack([frames[j+h].visitor[:, 0] for h in self.horizons], axis=-1).ravel())
        x, y = np.asarray(x), np.asarray(y)
        penalty = np.eye(x.shape[1]) * self.cfg["ridge_alpha"]
        penalty[-1, -1] = 0
        self.coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        self.stats.save(output / "training_stats.npz")
        np.savez_compressed(output / "frozen_forecaster.npz", coefficients=self.coef)

    def load(self, output_dir):
        self.stats = TrainingStats.load(Path(output_dir) / "training_stats.npz")
        with np.load(Path(output_dir) / "frozen_forecaster.npz", allow_pickle=False) as z:
            self.coef = z["coefficients"]

    def forecast(self, window):
        return np.maximum(0, self._features(window.values, window.decision_time) @ self.coef).reshape(
            self.nodes, len(self.horizons)).astype(np.float32)

    def encode_state(self, window, ledger):
        phase = 2 * np.pi * window.decision_time / self.cfg["episode_steps"]
        scaled_ledger = ledger / np.asarray(self.cfg["resource_totals"])[:, None]
        return np.concatenate((self.stats.normalize(window.values).ravel(), window.mask.ravel(),
                               (window.age / self.history).ravel(), scaled_ledger.ravel(),
                               [np.sin(phase), np.cos(phase)])).astype(np.float32)

    def describe(self):
        return {"backend": "demo", "is_demo": True, "data_provenance": "fully_synthetic_new_demo",
                "forecaster": "frozen training-only ridge regression, NOT E-STGNN",
                "repair": "masked largest remainder and holding-limit repair using raw priority/index",
                "warmup": "history+2 bins; historical arrival channels; zero queues; static ledger",
                "waiting": "served visitors, discrete-bin waits; unserved reported at closing",
                "occupancy": "residual queue at each post-action decision boundary",
                "resource_dynamics": "instantaneous target allocations; no origin/travel constraints",
                "configuration": self.cfg}
