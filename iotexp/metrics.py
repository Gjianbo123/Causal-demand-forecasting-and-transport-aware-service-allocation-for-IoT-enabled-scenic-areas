"""Resource feasibility and visitor overload have independent numerators/denominators."""
import numpy as np


class Metrics:
    def __init__(self):
        self.steps = self.pre = self.post = self.repairs = 0
        self.overload_count = self.overload_denominator = self.overload_excess = 0
        self.arrivals = self.unserved = 0
        self.movement = 0.0
        self.waits = []

    def add(self, audit):
        mask = np.asarray(audit.overload_nodes, dtype=bool)
        occupancy = np.asarray(audit.occupancy)[mask]
        capacity = np.asarray(audit.visitor_capacity)[mask]
        if (not len(occupancy) or not np.isfinite(occupancy).all() or not np.isfinite(capacity).all()
                or (capacity < 0).any() or (occupancy < 0).any()):
            raise ValueError("Invalid fixed visitor-overload node set")
        self.steps += 1
        self.pre += int(audit.candidate_infeasible)
        self.post += int(audit.executable_infeasible)
        self.repairs += int(audit.repaired)
        self.overload_count += int(np.sum(occupancy > capacity))
        self.overload_denominator += len(occupancy)
        self.overload_excess += float(np.maximum(occupancy - capacity, 0).sum())
        waits = np.asarray(audit.served_waits_min, dtype=float)
        if not np.isfinite(waits).all() or (waits < 0).any():
            raise ValueError("Invalid served visitor waits")
        self.waits.extend(waits.tolist())
        self.arrivals += int(audit.arrivals_count)
        self.unserved = int(audit.unserved_count)  # remaining at last boundary, not summed
        self.movement += float(audit.movement_cost)

    def result(self):
        if not self.steps:
            raise ValueError("No decisions evaluated")
        return {
            "mean_wait_min": float(np.mean(self.waits)) if self.waits else float("nan"),
            "p95_wait_min": float(np.percentile(self.waits, 95)) if self.waits else float("nan"),
            "wait_sum_min": float(sum(self.waits)), "served_count": len(self.waits),
            "unserved_count": self.unserved, "arrivals_count": self.arrivals,
            "visitor_overload_count": self.overload_count,
            "visitor_overload_denominator": self.overload_denominator,
            "visitor_overload_pct": 100 * self.overload_count / self.overload_denominator,
            "mean_excess_visitors": self.overload_excess / self.overload_denominator,
            "movement_cost": self.movement, "decision_epochs": self.steps,
            "candidate_infeasible_epochs": self.pre, "executable_infeasible_epochs": self.post,
            "repair_epochs": self.repairs, "candidate_infeasible_pct": 100 * self.pre / self.steps,
            "executable_infeasible_pct": 100 * self.post / self.steps,
            "repair_pct": 100 * self.repairs / self.steps,
        }


def forecast_errors(predictions, truth, horizons):
    """predictions[t]=[N,H]; truth[t] is the realized inflow at boundary t.

    Compute AFTER rollout. Targets outside the episode are omitted per horizon.
    No future simulator path is exposed to the policy or forecaster.
    """
    result = {}
    for j, h in enumerate(horizons):
        errors = [np.abs(np.asarray(p)[:, j] - truth[t + h])
                  for t, p in predictions.items() if t + h in truth]
        count = sum(x.size for x in errors)
        absolute_sum = float(sum(x.sum() for x in errors))
        result[f"forecast_abs_sum_h{h}"] = absolute_sum
        result[f"forecast_count_h{h}"] = count
        result[f"forecast_mae_h{h}"] = absolute_sum / count if count else float("nan")
    return result
