"""Additional publication figures from frozen control summaries, without new trials."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SOURCE = ROOT / "output/improvement_v4/control_world_metrics.csv"
OUT = ROOT / "output/improvement_v4_more_figures"
FIG = OUT / "figures"
METHODS = ["static", "ppo_f0", "ppo_f4", "gwn_mpc6", "combined_mpc6", "rollout"]
LABELS = {
    "static": "Static", "ppo_f0": "PPO-F0", "ppo_f4": "PPO-F4",
    "gwn_mpc6": "G / MPC-6", "combined_mpc6": "C / MPC-6", "rollout": "Rollout",
}
COLORS = {
    "rollout": "#D96D5E", "combined_mpc6": "#456A90", "gwn_mpc6": "#6D969A",
    "ppo_f0": "#8D9CAF", "ppo_f4": "#9DAA90", "static": "#A3A9B0",
}
MARKERS = dict(zip(METHODS, ["s", "v", "^", "D", "o", "P"]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    for ext in ["pdf", "svg", "png", "eps"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=450, facecolor="white")
    plt.close(fig)


def style(ax, axis="both"):
    ax.grid(axis=axis, color="#E4E8EC", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#677580")
        spine.set_linewidth(0.6)


def tradeoffs(full, summary):
    fig, axes = plt.subplots(2, 2, figsize=(165 / 25.4, 140 / 25.4),
                             gridspec_kw={"height_ratios": [1.05, 1]})
    fig.subplots_adjust(left=.135, right=.98, bottom=.08, top=.945, wspace=.28, hspace=.60)
    positions = {
        False: {
            "static": (0.03, .71), "ppo_f0": (.11, .91), "ppo_f4": (.56, .91),
            "gwn_mpc6": (.57, .57), "combined_mpc6": (.14, .25), "rollout": (.74, .09),
        },
        True: {
            "static": (.015, .68), "ppo_f0": (.13, .95), "ppo_f4": (.56, .87),
            "gwn_mpc6": (.10, .40), "combined_mpc6": (.52, .30), "rollout": (.76, .06),
        },
    }
    for col, surge in enumerate([False, True]):
        ax = axes[0, col]
        for name in METHODS:
            values = full[(full.surge == surge) & (full.method == name)]
            x, y = values.movement.mean(), values.wait.mean()
            light_color = tuple(.4 * x + .6 for x in matplotlib.colors.to_rgb(COLORS[name]))
            ax.scatter(values.movement, values.wait, s=12, marker=MARKERS[name],
                       color=light_color, linewidths=0, zorder=2)
            ax.scatter([x], [y], s=55 if name == "rollout" else 38, marker=MARKERS[name],
                       color=COLORS[name], edgecolor="#23333C", lw=.6, zorder=4)
            ax.annotate(LABELS[name], xy=(x, y), xycoords="data", xytext=positions[surge][name],
                        textcoords="axes fraction", color=COLORS[name] if name != "static" else "#5C6670",
                        fontsize=8.5, fontweight="bold" if name == "rollout" else "normal",
                        ha="left", va="center", zorder=5,
                        bbox=dict(fc="white", ec="none", pad=.2),
                        arrowprops=dict(arrowstyle="-", color=COLORS[name], lw=.6,
                                        shrinkA=2, shrinkB=4, connectionstyle="arc3,rad=0"))
        ax.set_title("(b) Increased demand" if surge else "(a) Nominal demand", loc="left", pad=8)
        ax.set_xlabel("Movement (resource-km/day)", labelpad=4)
        if col == 0:
            ax.set_ylabel("Accrued wait (min/request)", labelpad=4)
        ax.set_xlim(-3, 68 if not surge else 43)
        ax.set_ylim(10 if not surge else 50, 52 if not surge else 109)
        style(ax)

        ax = axes[1, col]
        for i, name in enumerate(METHODS):
            g = summary[(summary.surge == surge) & (summary.method == name)].iloc[0]
            if name == "rollout":
                ax.axhspan(i-.46, i+.46, color="#FBF0EF", zorder=0)
            ax.barh(i-.18, g.unserved_pct_mean, xerr=g.unserved_pct_sd, height=.30,
                    color="#6A92B1", edgecolor="white", lw=.4,
                    error_kw=dict(ecolor="#365369", elinewidth=.65, capsize=1.5), zorder=2)
            ax.barh(i+.18, g.overload_pct_mean, xerr=g.overload_pct_sd, height=.30,
                    color="#B7C7CC", edgecolor="white", lw=.4,
                    error_kw=dict(ecolor="#5F777F", elinewidth=.65, capsize=1.5), zorder=2)
        ax.set_yticks(range(len(METHODS)), [LABELS[n] for n in METHODS] if col == 0 else [""]*len(METHODS))
        ax.set_ylim(len(METHODS)-.45, -.65)
        ax.set_xlim(0, 19 if not surge else 58)
        ax.set_xlabel("Percentage (%)", labelpad=4)
        ax.set_title("(d) Terminal and overload outcomes" if surge else "(c) Terminal and overload outcomes", loc="left", pad=8, fontsize=8.5)
        style(ax, "x")
    fig.legend(handles=[Patch(fc="#6A92B1", label="Terminal unserved"),
                        Patch(fc="#B7C7CC", label="Queue overload")],
               loc="center", bbox_to_anchor=(.55, .495), ncol=2, frameon=False,
               fontsize=8.5, handlelength=1.2, columnspacing=2.2)
    save(fig, "Control_tradeoffs")


def matched(paired):
    fig, axes = plt.subplots(1, 2, figsize=(165 / 25.4, 85 / 25.4), sharey=True)
    fig.subplots_adjust(left=.12, right=.98, bottom=.18, top=.73, wspace=.17)
    for col, surge in enumerate([False, True]):
        ax = axes[col]
        for comparator, color, marker, offset in [
            ("combined_mpc6", "#456A90", "o", -.12),
            ("combined_mpc12", "#6D969A", "s", .12),
        ]:
            values = paired[(paired.surge == surge) & (paired.comparator == comparator)].sort_values("dataset_seed")
            x = np.arange(1, 8) + offset
            light_color = tuple(.5 * v + .5 for v in matplotlib.colors.to_rgb(color))
            ax.vlines(x, 0, values.wait_difference, color=light_color, lw=.9, zorder=2)
            ax.scatter(x, values.wait_difference, color=color, marker=marker, s=25, zorder=3,
                       label="Rollout − C / MPC-6" if comparator.endswith("mpc6") else "Rollout − C / MPC-12")
        ax.axhline(0, color="#69737A", linestyle="--", lw=.85)
        ax.set_xticks(range(1, 8), [f"W{i}" for i in range(1, 8)])
        ax.set_xlim(.7, 7.3)
        ax.set_ylim(-7, .65)
        ax.set_xlabel("Independent random world", labelpad=5, fontsize=9)
        ax.set_title("(b) Increased demand" if surge else "(a) Nominal demand", loc="left", pad=8)
        style(ax)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Paired accrued-wait difference\n(min/request)", labelpad=4, fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.55, .99),
               ncol=2, frameon=False, fontsize=9, handlelength=2, columnspacing=1.8)
    fig.text(.55, .83, "Negative values favor rollout; same 21 episodes per condition", ha="center", fontsize=9, color="#45545F")
    save(fig, "Matched_worlds")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.titlesize": 9,
                         "axes.labelsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    source_hash = sha(SOURCE)
    data = pd.read_csv(SOURCE)
    assert not data.duplicated(["dataset_seed", "surge", "method", "subset"]).any()
    full = data[data.subset.eq("full")].copy()
    subset = data[data.subset.eq("matched12")].copy()
    assert set(full.method) == set(METHODS)
    seeds = sorted(full.dataset_seed.unique().tolist())
    assert len(seeds) == 7
    for surge in [False, True]:
        for method in METHODS:
            g = full[(full.surge == surge) & (full.method == method)]
            assert sorted(g.dataset_seed.tolist()) == seeds
            assert g.days.eq(9).all() and g.days.sum() == 63
        for method in ["rollout", "combined_mpc6", "combined_mpc12"]:
            g = subset[(subset.surge == surge) & (subset.method == method)]
            assert sorted(g.dataset_seed.tolist()) == seeds
            assert g.days.eq(3).all() and g.days.sum() == 21
    summary = full.groupby(["surge", "method"])[["wait", "movement", "unserved_pct", "overload_pct"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, col)).replace("_std", "_sd") for col in summary.columns]
    full.to_csv(OUT / "Control_tradeoffs_worlds.csv", index=False)
    summary.to_csv(OUT / "Control_tradeoffs_summary.csv", index=False)
    rows = []
    for surge in [False, True]:
        r = subset[(subset.surge == surge) & subset.method.eq("rollout")].set_index("dataset_seed")
        for comparator in ["combined_mpc6", "combined_mpc12"]:
            c = subset[(subset.surge == surge) & subset.method.eq(comparator)].set_index("dataset_seed")
            for i, seed in enumerate(seeds, 1):
                rows.append(dict(world=f"W{i}", dataset_seed=seed, surge=surge, subset="matched12",
                                 days_per_world=3, comparator=comparator,
                                 rollout_wait=r.loc[seed, "wait"], comparator_wait=c.loc[seed, "wait"],
                                 wait_difference=r.loc[seed, "wait"]-c.loc[seed, "wait"]))
    paired = pd.DataFrame(rows)
    assert paired.wait_difference.between(-7, .65).all(), "Paired values exceed displayed limits"
    for surge, wait_bounds, movement_max in [(False, (10, 52), 68), (True, (50, 109), 43)]:
        g = full[full.surge == surge]
        assert g.wait.between(*wait_bounds).all(), "Waiting values exceed scatter limits"
        assert g.movement.between(0, movement_max).all(), "Movement values exceed scatter limits"
    paired.to_csv(OUT / "Matched_worlds_paired.csv", index=False)
    tradeoffs(full, summary)
    matched(paired)
    assert sha(SOURCE) == source_hash
    audit = {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": source_hash,
        "script_sha256": sha(Path(__file__)), "world_seeds_in_display_order": seeds,
        "new_experiments": False,
        "full": {"worlds": 7, "days_per_world": 9, "days_per_condition_per_method": 63, "methods": METHODS},
        "matched12": {"worlds": 7, "days_per_world": 3, "days_per_condition_per_method": 21,
                      "comparators": ["combined_mpc6", "combined_mpc12"], "paired_on": "dataset_seed and surge; identical saved matched12 subset"},
        "uncertainty": "Scatter: seven world-level points and larger across-world means. Bars: across-world means +/- sample SD (ddof=1); not confidence intervals. Paired plot: seven individual world-level differences, no error bars.",
        "world_index_note": "W1-W7 map to independent random-world seeds 20261011-20261017, not successive test days; paired points are not connected across worlds.",
        "scatter_axes": "Both dimensions display lower-is-better outcomes. Movement starts at zero (small padding for marker). Waiting axis is cropped for legibility and visibly labelled.",
        "output_sha256": {str(p.relative_to(OUT)): sha(p) for p in sorted(FIG.glob('*')) if p.stem in ['Control_tradeoffs', 'Matched_worlds']},
        "data_sha256": {p.name: sha(p) for p in sorted(OUT.glob('*.csv')) if p.name.startswith(('Control_tradeoffs', 'Matched_worlds'))},
    }
    (OUT / "control_plot_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"source_unchanged": True, "full_rows": len(full), "paired_rows": len(paired),
                      "paired_mean_differences": paired.groupby(["surge", "comparator"]).wait_difference.mean().to_dict().__str__(),
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
