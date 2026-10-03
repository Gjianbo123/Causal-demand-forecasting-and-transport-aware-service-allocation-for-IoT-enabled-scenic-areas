"""Causal event/receipt-time telemetry, independent of the environment RNG."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Condition:
    name: str = "nominal"
    missing: float = 0.0
    noise: float = 0.0
    delay: int = 0

    def __post_init__(self):
        if not 0 <= self.missing <= 1 or self.noise < 0:
            raise ValueError("Invalid missingness/noise level")
        if not isinstance(self.delay, int) or self.delay < 0:
            raise ValueError("delay must be a nonnegative number of decision intervals")


CONDITIONS = [Condition()] + [Condition(f"missing_{p:.2f}", missing=p) for p in (.1, .2, .3)] + [
    Condition(f"noise_{s:.2f}", noise=s) for s in (.05, .1, .2)
] + [Condition(f"delay_{d}", delay=d) for d in (1, 2)]


@dataclass
class TrainingStats:
    mean: np.ndarray                 # [N,C], training only
    scale: np.ndarray                # [N,C], also noise standard-deviation reference
    median_by_slot: np.ndarray       # [period,N,C], training only

    @classmethod
    def fit(cls, samples, times, period):
        x = np.asarray(samples, dtype=float)
        t = np.asarray(times, dtype=int)
        if x.ndim != 3 or len(x) < 2 or len(x) != len(t) or not np.isfinite(x).all():
            raise ValueError("Expected >=2 finite training samples [T,N,C]")
        scale = x.std(axis=0, ddof=1)
        pooled = x.reshape(-1, x.shape[-1]).std(axis=0, ddof=1)
        # Explicit fallback for a constant node/channel: pooled channel SD, then 1.
        scale = np.where(scale > 1e-8, scale, np.maximum(pooled[None, :], 1.0))
        median = np.median(x, axis=0)
        by_slot = np.stack([np.median(x[t % period == s], axis=0)
                            if np.any(t % period == s) else median for s in range(period)])
        return cls(x.mean(axis=0), scale, by_slot)

    def normalize(self, x):
        return (np.asarray(x) - self.mean) / self.scale

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, scale=self.scale, median_by_slot=self.median_by_slot)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z["mean"], z["scale"], z["median_by_slot"])


@dataclass
class Window:
    values: np.ndarray              # [L,N,C], grid aligned to CURRENT decision time
    mask: np.ndarray                # [L,N], 1 if exact event-time packet is available
    age: np.ndarray                 # [L,N], intervals since source measurement
    source_event: np.ndarray        # [L,N], NaN means training-median fallback
    source_receipt: np.ndarray      # [L,N], NaN means training-median fallback
    grid: np.ndarray                # [L]
    decision_time: int


class TelemetryGateway:
    """Packets have whole-node missingness and per-channel independent noise.

    A window at decision t uses packets with receipt<=t and event<=grid_slot.
    Late packets can improve a NEW decision's history; previous traces are immutable.
    No backward fill, future truth, or shifted forecast target times are used.
    """
    def __init__(self, stats: TrainingStats, condition: Condition, seed: int, history: int):
        if history < 1:
            raise ValueError("history must be positive")
        self.stats, self.condition, self.seed, self.history = stats, condition, int(seed), history
        self.records = []
        self.last_submitted = None

    def submit(self, event_time, values):
        event_time = int(event_time)
        x = np.asarray(values, dtype=float)
        if x.shape != self.stats.mean.shape or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("Visitor channels must be finite, nonnegative [N,C]")
        if self.last_submitted is not None and event_time <= self.last_submitted:
            raise ValueError("Submit one complete node batch per strictly increasing event time")
        self.last_submitted = event_time
        # Event-keyed streams pair perturbations even if methods consume other RNG draws.
        key = event_time & 0xffffffff
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, key, 7919]))
        uniforms = rng.random(x.shape[0])
        noise_draw = rng.standard_normal(x.shape)
        noisy = np.maximum(0, x + self.condition.noise * self.stats.scale * noise_draw)
        for node in range(x.shape[0]):
            self.records.append({"event": event_time, "receipt": event_time + self.condition.delay,
                                 "node": node, "dropped": bool(uniforms[node] < self.condition.missing),
                                 "noise_draw": noise_draw[node].copy(), "value": noisy[node].copy()})

    def window(self, decision_time):
        if self.last_submitted is None or decision_time > self.last_submitted:
            raise ValueError("Submit current reports before requesting a window")
        grid = np.arange(decision_time - self.history + 1, decision_time + 1)
        n, c = self.stats.mean.shape
        values = np.empty((self.history, n, c), dtype=np.float32)
        mask = np.zeros((self.history, n), dtype=np.float32)
        age = np.empty_like(mask)
        source = np.full_like(mask, np.nan)
        receipt = np.full_like(mask, np.nan)
        available = [[] for _ in range(n)]
        for p in self.records:
            if not p["dropped"] and p["receipt"] <= decision_time:
                available[p["node"]].append(p)
        first_event = self.records[0]["event"]
        for j, slot in enumerate(grid):
            for node in range(n):
                prior = [p for p in available[node] if p["event"] <= slot]
                if prior:
                    latest = max(prior, key=lambda p: (p["event"], p["receipt"]))
                    values[j, node] = latest["value"]
                    source[j, node], receipt[j, node] = latest["event"], latest["receipt"]
                    mask[j, node] = float(latest["event"] == slot)
                    age[j, node] = slot - latest["event"]
                else:
                    values[j, node] = self.stats.median_by_slot[slot % len(self.stats.median_by_slot), node]
                    age[j, node] = max(1, slot - first_event + 1)
        return Window(values, mask, age, source, receipt, grid, int(decision_time))

    def arrays(self):
        return {"packet_" + k: np.asarray([p[k] for p in self.records])
                for k in ("event", "receipt", "node", "dropped", "value", "noise_draw")}


def condition_dicts():
    return [asdict(c) for c in CONDITIONS]
