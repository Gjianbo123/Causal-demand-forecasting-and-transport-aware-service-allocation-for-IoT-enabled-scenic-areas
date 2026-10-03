"""The original project only needs an environment/backend adapter; no test truth in inputs."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from .telemetry import TrainingStats, Window


@dataclass
class Frame:
    time: int
    visitor: np.ndarray             # [N,C], channel 0 is target inflow
    ledger: np.ndarray              # exact resource state; normalized by backend


@dataclass
class Audit:
    candidate_action: np.ndarray
    executed_action: np.ndarray
    candidate_infeasible: bool
    executable_infeasible: bool
    repaired: bool
    occupancy: np.ndarray
    visitor_capacity: np.ndarray
    overload_nodes: np.ndarray      # bool [N], fixed prespecified evaluation node set
    served_waits_min: np.ndarray
    unserved_count: int
    arrivals_count: int
    movement_cost: float


class Environment(Protocol):
    def reset(self) -> list[Frame]:
        """Return causal warm-up frames with ascending times, ending at t=0.

        All warm-up frames must come from the SAME fixed warm-up rule for all methods.
        Metrics collected by step() concern evaluation arrivals only; document carry-in queues.
        """
        ...

    def step(self, raw_action: np.ndarray) -> tuple[Frame, float, bool, Audit]:
        """Apply identical forecast-independent repair; advance one decision interval.

        done means a true finite-horizon terminal, not a time-limit truncation.
        For a continuing task with truncation, extend the trainer's bootstrap contract.
        """
        ...


class Backend(Protocol):
    nodes: int
    resources: int
    state_dim: int
    stats: TrainingStats
    is_demo: bool
    horizons: list[int]
    history: int

    def prepare(self, output_dir):
        """Fit/load ONLY training data statistics and a frozen forecaster; persist assets."""
        ...

    def load(self, output_dir):
        """Load saved training assets, never refit using validation/test trajectories."""
        ...

    def make_env(self, split: str, seed: int) -> Environment: ...
    def encode_state(self, window: Window, ledger: np.ndarray) -> np.ndarray: ...
    def forecast(self, window: Window) -> np.ndarray:
        """Return physical inflow units [N,H], targets at window.decision_time+h.

        Only window and pre-known calendar/context may be accessed. No simulator truth.
        Keep the original forecaster frozen and put it in eval mode.
        """
        ...

    def describe(self) -> dict:
        """Include simulator/source/config, repair rule and forecaster checkpoint identity."""
        ...
