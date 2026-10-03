"""Aggregate trials WITHIN trained seed before paired statistical tests."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from .telemetry import CONDITIONS
from .util import write_json


def signed_rank(differences, min_seeds=5):
    d = np.asarray(differences, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError("Nonfinite paired differences: cannot silently drop seeds")
    result = {"n_seeds": len(d), "mean_difference": float(d.mean()),
              "median_difference": float(np.median(d)), "p_value": None,
              "zero_method": "wilcox (drop exact zero paired differences)", "method": "not_tested"}
    sd = d.std(ddof=1) if len(d) > 1 else 0
    result["paired_standardized_difference"] = float(d.mean()/sd) if sd > 0 else None
    if len(d) < min_seeds:
        result["method"] = f"descriptive_only_fewer_than_{min_seeds}_seeds"
        return result
    nonzero = d[d != 0]
    if not len(nonzero):
        result.update(p_value=1.0, method="all_zero_differences")
        return result
    ranks = rankdata(np.abs(nonzero), method="average")
    observed = abs(float(np.dot(np.sign(nonzero), ranks)))
    result["positive_rank_sum"] = float(ranks[nonzero > 0].sum())
    result["negative_rank_sum"] = float(ranks[nonzero < 0].sum())
    if len(nonzero) <= 20:
        # Exact conditional sign enumeration handles tied absolute differences.
        hits = 0
        for start in range(0, 1 << len(nonzero), 4096):
            codes = np.arange(start, min(start + 4096, 1 << len(nonzero)), dtype=np.uint64)
            bits = (codes[:, None] >> np.arange(len(nonzero), dtype=np.uint64)) & 1
            scores = (2 * bits.astype(float) - 1) @ ranks
            hits += int(np.sum(np.abs(scores) >= observed - 1e-12))
        result.update(p_value=hits / (1 << len(nonzero)), method="exact_conditional_sign_enumeration")
    else:
        result.update(p_value=float(2 * norm.sf(observed / np.sqrt(np.square(ranks).sum()))),
                      method="normal_approximation_no_continuity_correction")
    return result


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Holm family must contain only complete finite p values")
    order = np.argsort(p, kind="stable")
    corrected = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty_like(p)
    result[order] = np.minimum(1, corrected)
    return result.tolist()


def check_matrix(df, config):
    keys = ["method", "training_seed", "condition", "environment_seed", "corruption_replicate"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate evaluation trial; refusing to inflate sample size")
    expected = set()
    for method in config["methods"]:
        conditions = CONDITIONS if method in config["robustness_methods"] else [CONDITIONS[0]]
        expected.update(itertools.product([method], config["training_seeds"], [c.name for c in conditions],
                                          config["test_environment_seeds"], config["corruption_seeds"]))
    actual = set(map(tuple, df[keys].itertuples(index=False, name=None)))
    if actual != expected:
        raise ValueError(f"Incomplete/unexpected evaluation matrix: missing={len(expected-actual)}, extra={len(actual-expected)}")


def summarize(output_dir):
    output = Path(output_dir)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    df = pd.read_csv(output / "episodes.csv")
    check_matrix(df, config)
    records = []
    for (method, seed, condition), group in df.groupby(["method", "training_seed", "condition"], sort=True):
        waits = []
        for row in group.itertuples():
            with np.load(output / row.trace_path, allow_pickle=False) as z:
                values = z["served_waits_min"]
                if len(values) != row.served_count or not np.isclose(values.sum(), row.wait_sum_min):
                    raise ValueError("CSV waiting statistics disagree with raw trace")
                waits.append(values)
        waits = np.concatenate(waits)
        if not len(waits):
            raise ValueError("No served visitors in a seed/condition; report censoring, do not drop this seed")
        r = {"method": method, "training_seed": int(seed), "condition": condition,
             "is_demo": bool(manifest["is_demo"]), "n_trials": len(group),
             "mean_wait_min": float(waits.mean()), "p95_wait_min": float(np.percentile(waits, 95)),
             "mean_unserved_count": float(group.unserved_count.mean()),
             "mean_served_count": float(group.served_count.mean()),
             "movement_cost": float(group.movement_cost.mean()),
             "policy_latency_ms": float(group.policy_latency_ms.mean()),
             "trainable_parameters": int(group.trainable_parameters.iloc[0])}
        r["visitor_overload_pct"] = 100 * group.visitor_overload_count.sum() / group.visitor_overload_denominator.sum()
        if "waiting_area_min" in group:
            r["restricted_mean_wait_min"] = group.waiting_area_min.sum() / max(group.arrivals_count.sum(), 1)
            r["unserved_pct"] = 100 * group.unserved_count.sum() / max(group.arrivals_count.sum(), 1)
            r["local_pipeline_latency_ms"] = float(group.local_pipeline_latency_ms.mean())
            r["mean_reward"] = float(group.mean_reward.mean())
        for prefix in ("candidate_infeasible", "executable_infeasible", "repair"):
            r[prefix + "_pct"] = 100 * group[prefix + "_epochs"].sum() / group.decision_epochs.sum()
        for h in manifest["horizons"]:
            count = group[f"forecast_count_h{h}"].sum()
            r[f"forecast_mae_h{h}"] = group[f"forecast_abs_sum_h{h}"].sum() / count if count else np.nan
        records.append(r)
    seed_df = pd.DataFrame(records)
    nominal = seed_df[seed_df.condition == "nominal"].set_index(["method", "training_seed"])
    seed_df["wait_degradation_min"] = [row.mean_wait_min - nominal.loc[
        (row.method, row.training_seed), "mean_wait_min"] for row in seed_df.itertuples()]
    seed_df["wait_degradation_pct"] = [100 * row.wait_degradation_min / nominal.loc[
        (row.method, row.training_seed), "mean_wait_min"] if nominal.loc[
        (row.method, row.training_seed), "mean_wait_min"] != 0 else np.nan for row in seed_df.itertuples()]
    seed_df.to_csv(output / "per_seed.csv", index=False)
    metrics = [c for c in seed_df.columns if c not in (
        "method", "training_seed", "condition", "is_demo", "n_trials", "trainable_parameters")]
    aggregate = []
    for (method, condition), group in seed_df.groupby(["method", "condition"]):
        for metric in metrics:
            aggregate.append({"method": method, "condition": condition, "metric": metric,
                              "mean": group[metric].mean(), "sample_sd": group[metric].std(ddof=1),
                              "n_training_seeds": len(group), "is_demo": bool(manifest["is_demo"])})
    pd.DataFrame(aggregate).to_csv(output / "summary.csv", index=False)
    contrasts, differences = [], []
    def compare(other, condition, family):
        a = seed_df[(seed_df.method == "F4") & (seed_df.condition == condition)].set_index("training_seed")
        b = seed_df[(seed_df.method == other) & (seed_df.condition == condition)].set_index("training_seed")
        if set(a.index) != set(b.index) or not len(a):
            raise ValueError("Primary/secondary paired seeds do not match")
        idx = sorted(a.index)
        endpoint = config.get("primary_metric", "mean_wait_min")
        d = a.loc[idx, endpoint].to_numpy() - b.loc[idx, endpoint].to_numpy()
        result = signed_rank(d, config.get("minimum_test_seeds", 5))
        result.update(comparison=f"F4_minus_{other}", condition=condition, family=family,
                      direction="negative favors F4", endpoint=endpoint,
                      p_holm=None, is_demo=bool(manifest["is_demo"]))
        contrasts.append(result)
        differences.extend({"comparison": f"F4_minus_{other}", "condition": condition,
                            "training_seed": int(s), "wait_difference_min": float(v)} for s, v in zip(idx, d))
    compare("F1", "nominal", "primary_prespecified")
    for other in ("F2", "F3"):
        compare(other, "nominal", "secondary_fusion_two")
    for condition in CONDITIONS[1:]:
        compare("F1", condition.name, "robustness_eight")
    for family in ("secondary_fusion_two", "robustness_eight"):
        members = [r for r in contrasts if r["family"] == family]
        if all(r["p_value"] is not None for r in members):
            for r, p in zip(members, holm([r["p_value"] for r in members])):
                r["p_holm"] = p
    write_json(output / "paired_tests.json", contrasts)
    pd.DataFrame(differences).to_csv(output / "paired_differences.csv", index=False)
    return seed_df


def plot_results(output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    df = pd.read_csv(output / "per_seed.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    colors = {"F0": "#7b8595", "F1": "#4477aa", "F4": "#cc6677"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    factors = [("Missing packet probability", [0, .1, .2, .3], ["nominal", "missing_0.10", "missing_0.20", "missing_0.30"]),
               ("Noise / training SD", [0, .05, .1, .2], ["nominal", "noise_0.05", "noise_0.10", "noise_0.20"]),
               ("Upload delay (intervals)", [0, 1, 2], ["nominal", "delay_1", "delay_2"])]
    for ax, (label, levels, names) in zip(axes, factors):
        for method in ("F0", "F1", "F4"):
            groups = [df[(df.method == method) & (df.condition == name)].mean_wait_min for name in names]
            means = np.asarray([g.mean() for g in groups])
            std = np.asarray([g.std(ddof=1) for g in groups])
            ax.plot(levels, means, "o-", label=method, color=colors[method], linewidth=1.6, markersize=4)
            ax.fill_between(levels, means-std, means+std, color=colors[method], alpha=.15)
        ax.set(xlabel=label, ylabel="Served-visitor mean wait (min)", xticks=levels)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    prefix = "DEMO ONLY — new synthetic example; " if manifest["is_demo"] else ""
    fig.suptitle(prefix + "mean ± sample SD across trained seeds", fontsize=10)
    fig.savefig(output / "robustness.png", dpi=180)
    fig.savefig(output / "robustness.pdf")
    plt.close(fig)
