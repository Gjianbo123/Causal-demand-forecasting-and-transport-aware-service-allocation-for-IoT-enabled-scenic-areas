"""Cross-check publication summaries against separately audited aggregations."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iotexp.util import digest, write_json
V3 = ROOT / "runs/supplement_v3"
OUT = ROOT / "output/supplement_v3/results"


def same(left, right):
    assert np.allclose(left, right, rtol=1e-12, atol=1e-10, equal_nan=True)


def exact_p(delta):
    # Independent rank assignment and complete sign-permutation enumeration.
    values = np.asarray(delta)
    values = values[values != 0]
    if not len(values):
        return 1.0
    absolute = abs(values)
    ranks = np.array([1 + (absolute < value).sum() + ((absolute == value).sum()-1)/2 for value in absolute])
    observed = abs(sum(np.sign(values)*ranks))
    extreme = sum(abs(sum(np.array(signs)*ranks)) >= observed-1e-12 for signs in itertools.product((-1, 1), repeat=len(values)))
    return extreme/(2**len(values))


def main():
    records = json.loads((OUT/"results.json").read_text())
    for path, expected in records["input_sha256"].items():
        assert digest(ROOT/path) == expected
    assert [records[key] for key in ["policy_episodes", "heuristic_episodes", "mpc_episodes", "perfect_information_episodes"]] == [15120, 567, 189, 81]
    keys = ["predictor", "method", "training_seed", "budget_steps", "scenario"]
    independent = pd.read_csv(V3/"audits/policy_per_seed_scenario_independent.csv").set_index(keys).sort_index()
    published = pd.read_csv(OUT/"policy_per_scenario.csv").set_index(keys).sort_index()
    assert independent.index.equals(published.index) and len(published) == 560
    columns = {"restricted_mean_wait_min": "wait", "unserved_pct": "unserved_pct", "movement_cost": "movement"}
    for independent_key, output_key in columns.items():
        same(independent[independent_key], published[output_key])
    det_ind = pd.read_csv(V3/"audits/heuristics_per_scenario_independent.csv")
    deterministic = pd.read_csv(OUT/"deterministic_per_scenario.csv").set_index(["method", "scenario"])
    for row in det_ind.itertuples():
        for a, b in columns.items():
            same(getattr(row, a), deterministic.loc[(row.method, row.scenario), b])
    for method in ("mpc", "oracle"):
        source = pd.read_csv(V3/"control"/f"{method}_episodes.csv")
        for scenario, group in source.groupby("scenario"):
            row = deterministic.loc[(group.method.iloc[0], scenario)]
            same(row["wait"], group.waiting_area_min.sum()/group.arrivals_count.sum())
            same(row["unserved_pct"], 100*group.unserved_count.sum()/group.arrivals_count.sum())
            same(row["movement"], group.movement_cost.mean())
    frame = independent.reset_index()
    frame["regime"] = frame.scenario.map(lambda name: "legacy" if name == "legacy" else "surge" if name.endswith("surge") else "nominal")
    averaged = frame.groupby(["predictor", "method", "budget_steps", "training_seed", "regime"])[list(columns)].mean()
    summary = pd.read_csv(OUT/"policy_summary.csv")
    for (predictor, method, budget, regime), group in averaged.reset_index().groupby(["predictor", "method", "budget_steps", "regime"]):
        for original, metric in columns.items():
            reported = summary[(summary.predictor == predictor) & (summary.method == method) & (summary.budget_steps == budget) & (summary.regime == regime) & (summary.metric == metric)].iloc[0]
            values = group[original].to_numpy()
            half = t.ppf(.975, 9)*values.std(ddof=1)/np.sqrt(10)
            same([reported["mean"], reported.sd, reported.ci_low, reported.ci_high, reported.n],
                 [values.mean(), values.std(ddof=1), values.mean()-half, values.mean()+half, 10])
    def values(predictor, method, budget=43200, regime="nominal"):
        return averaged.xs((predictor, method, budget), level=(0, 1, 2)).xs(regime, level="regime")["restricted_mean_wait_min"].sort_index().to_numpy()
    deltas = [values("edge_stgru", "F4")-values("edge_stgru", "F1"),
              values("edge_stgru", "F4")-values("edge_stgru", "F0"),
              values("ridge", "F4")-values("ridge", "F1"),
              values("ridge", "F4")-values("edge_stgru", "F4"),
              values("edge_stgru", "F4")-values("edge_stgru", "F4", 14400),
              values("edge_stgru", "F4", regime="surge")-values("edge_stgru", "F1", regime="surge")]
    exact = []
    for delta, contrast in zip(deltas, records["contrasts"]):
        p = exact_p(delta)
        exact.append(p)
        half = t.ppf(.975, 9)*delta.std(ddof=1)/np.sqrt(10)
        same([contrast["mean"], contrast["p_value"], contrast["ci_low"], contrast["ci_high"]],
             [delta.mean(), p, delta.mean()-half, delta.mean()+half])
    ordered = sorted(enumerate(exact[1:5]), key=lambda item: item[1])
    previous = 0
    for rank, (index, p) in enumerate(ordered):
        previous = max(previous, min(1, (4-rank)*p))
        same(previous, records["contrasts"][index+1]["p_holm"])
    report = dict(status="PASS", policy_seed_world_rows=560, policy_metrics_per_row=3,
                  deterministic_method_world_rows=len(deterministic), paired_contrasts=6,
                  checks=["published input hashes", "all policy per-seed/per-world waiting, unserved and movement match independent audit",
                          "all deterministic per-world waiting, unserved and movement match independently audited sources",
                          "policy world averaging and conditional t intervals for three audited outcomes",
                          "independent exact sign-enumeration p values for six paired contrasts", "four-comparison Holm family"],
                  limits="Three principal outcomes audited against independent aggregates; no new experimental samples or selection.",
                  results_sha256=digest(OUT/"results.json"))
    write_json(V3/"audits/summary_independent_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
