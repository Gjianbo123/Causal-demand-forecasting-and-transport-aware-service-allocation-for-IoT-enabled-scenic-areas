"""Plot archived world-level forecasting differences without rerunning experiments."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/improvement_v4/forecast_world_metrics.csv"
RESULTS = ROOT / "output/improvement_v4/results.json"
OUT = ROOT / "output/improvement_v4_more_figures"
FIGURES = OUT / "figures"
WORLDS = list(range(20261011, 20261018))
COLORS = {"graph_wavenet": "#456A90", "residual": "#C38748"}
NAMES = {"graph_wavenet": "Graph WaveNet", "residual": "Residual GRU"}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    records = {}
    for row in rows:
        key = (int(row["dataset_seed"]), row["surge"].lower() == "true", row["model"])
        if key in records:
            raise ValueError(f"Duplicate world/scenario/model row: {key}")
        records[key] = float(row["mae"])

    values, summaries = [], {}
    for surge in (False, True):
        scenario = "increased_demand" if surge else "nominal"
        for baseline in COLORS:
            diffs = []
            for number, world in enumerate(WORLDS, 1):
                combined = records[world, surge, "combined"]
                comparator = records[world, surge, baseline]
                diff = combined - comparator
                diffs.append(diff)
                values.append(dict(scenario=scenario, world_label=f"W{number}",
                    dataset_seed=world, metric="mae", baseline=baseline,
                    combined_mae=combined, baseline_mae=comparator,
                    difference_combined_minus_baseline=diff))
            summaries[f"{scenario}/{baseline}"] = dict(
                differences=diffs, mean_difference=float(np.mean(diffs)),
                lower_worlds=int(np.sum(np.array(diffs) < 0)),
                higher_worlds=int(np.sum(np.array(diffs) > 0)), n_worlds=len(diffs))

    primary = json.loads(RESULTS.read_text(encoding="utf-8"))["primary"]["forecast"]
    nominal_gwn = summaries["nominal/graph_wavenet"]
    assert np.allclose(nominal_gwn["differences"], primary["differences"], rtol=0, atol=1e-12)
    assert abs(nominal_gwn["mean_difference"] - primary["mean_difference"]) < 1e-12
    stress_residual = summaries["increased_demand/residual"]
    assert stress_residual["higher_worlds"] == 2 and stress_residual["lower_worlds"] == 5

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#72787D",
        "axes.linewidth": 0.65, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(165 / 25.4, 91 / 25.4))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.80, wspace=0.28)
    x = np.arange(1, 8)
    for ax, scenario, title in zip(axes, ("nominal", "increased_demand"),
                                    ("(a) Nominal demand", "(b) Increased demand")):
        ax.axhline(0, color="#595F63", linewidth=0.85, zorder=1)
        for baseline, marker in (("graph_wavenet", "o"), ("residual", "s")):
            yy = summaries[f"{scenario}/{baseline}"]["differences"]
            ax.plot(x, yy, marker=marker, markersize=4.0, linewidth=1.1,
                markeredgewidth=0.6, markeredgecolor="white", color=COLORS[baseline],
                label=f"Combined − {NAMES[baseline]}", zorder=3)
        ax.set_title(title, loc="left", pad=9)
        ax.set_xticks(x, [f"W{i}" for i in x])
        ax.set_xlim(0.65, 7.35)
        ax.set_xlabel("Independent world")
        ax.grid(axis="y", color="#DFE5EA", linewidth=0.6, zorder=0)
        ax.tick_params(axis="both", width=0.6, length=3, pad=3)
    axes[0].set_ylabel("MAE difference (visit events)")
    axes[0].set_ylim(-0.0150, 0.0013)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.005))
    axes[1].set_ylim(-0.15, 0.008)
    axes[1].yaxis.set_major_locator(MultipleLocator(0.05))
    # A labelled linear-scale inset makes the two positive residual differences
    # visible. Keep the inset and its external title entirely in the empty gap
    # between the two main curves; the caption explains the sign convention.
    inset = axes[1].inset_axes([0.30, 0.63, 0.66, 0.19])
    inset.set_facecolor("white")
    inset.plot(x, stress_residual["differences"], color=COLORS["residual"],
               marker="s", markersize=3.5, linewidth=1.0, zorder=3)
    inset.axhline(0, color="#595F63", linewidth=0.7, zorder=1)
    inset.set_xlim(0.65, 7.35)
    inset.set_ylim(-0.0051, 0.0020)
    inset.set_xticks(x)
    inset.tick_params(axis="x", labelbottom=False, direction="in")
    inset.set_yticks([-0.004, 0.0], ["−0.004", "0"])
    inset.tick_params(labelsize=8, length=2, pad=2)
    inset.set_title("Residual GRU (zoom)", fontsize=8, loc="left", pad=4)
    inset.grid(axis="y", color="#DFE5EA", linewidth=0.5)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#B4BDC5")
        spine.set_linewidth(0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.985),
               ncol=2, frameon=False, columnspacing=1.5, handlelength=2.0)
    fig.text(0.52, 0.025, "W1–W7: dataset seeds 20261011–20261017; panel scales differ.",
             ha="center", fontsize=8, color="#434B51")

    for ext in ("pdf", "svg", "png", "eps"):
        fig.savefig(FIGURES / f"Forecast_consistency.{ext}", dpi=450)
    plt.close(fig)
    with (OUT / "forecast_consistency_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    audit = dict(
        title="World-level paired forecasting differences", metric="MAE", unit="visit events",
        unit_of_replication="Independent synthetic world (dataset seed); seven per scenario",
        difference_definition="Combined MAE minus comparator MAE; negative favors Combined",
        sources=[dict(path=str(path.relative_to(ROOT)).replace("\\", "/"),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in (SOURCE, RESULTS)],
        summaries=summaries,
        primary_mean_match=True, primary_differences_match=True,
        visual_notes="Panel y-axis scales differ; the residual comparison inset and its external title occupy the gap between the main curves. Sign interpretation is in the caption, outside the data area.",
        inference="Descriptive display only; no new significance tests or claims",
        outputs=[f"figures/Forecast_consistency.{ext}" for ext in ("pdf", "svg", "png", "eps")],
    )
    (OUT / "forecast_consistency_provenance.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"saved": audit["outputs"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
