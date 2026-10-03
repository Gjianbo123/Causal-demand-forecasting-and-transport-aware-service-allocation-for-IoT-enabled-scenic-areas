"""Plot archived controller development and serial local-pipeline timing.

No experiments are rerun and no source records are modified. All six
development candidates and every post-warmup serial timing observation are
included. Outputs contain source hashes and tabular plotting values.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/improvement_v4_more_figures/figures"
STEM = "Development_computation"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def source(relative):
        path = ROOT / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    selection = json.loads(source("runs/improvement_v4/control_development/selection.json").read_text())
    dev_rows = []
    grid = np.zeros((2, 3))
    for item in selection["candidates"]:
        h, tw = item["horizon"], item["terminal_weight"]
        episode_rows = [json.loads(source(
            f"runs/improvement_v4/control_development/H{h}_T{tw}/day{day}/result.json"
        ).read_text()) for day in (0, 6, 12)]
        wait_area = sum(x["waiting_area_min"] for x in episode_rows)
        arrivals = sum(x["arrivals_count"] for x in episode_rows)
        value = wait_area / arrivals
        assert np.isclose(value, item["development_wait"], rtol=0, atol=1e-10)
        selected = h == selection["selected"]["horizon"] and tw == selection["selected"]["terminal_weight"]
        grid[(0, 4).index(tw), (6, 12, 18).index(h)] = value
        dev_rows.append(dict(horizon=h, terminal_weight=tw, waiting_area_min=wait_area,
                             arrivals_count=arrivals, development_min_per_request=value,
                             selected=selected, development_episodes=3))
    assert len(dev_rows) == 6
    summary = json.loads(source("runs/improvement_v4/serial_timing/summary.json").read_text())
    stored = {x["method"]: x for x in summary["results"]}
    timing_rows, timing_summary, latency = [], [], {}
    for method in ("rollout", "combined_mpc6", "combined_mpc12"):
        with np.load(source(f"runs/improvement_v4/serial_timing/{method}/trace.npz")) as archive:
            raw = np.asarray(archive["local_pipeline_ms"], dtype=float)
        assert raw.shape == (72,)
        values = raw[5:]
        assert np.all(np.isfinite(values)) and np.all(values > 0)
        median, p95 = float(np.median(values)), float(np.percentile(values, 95))
        assert len(values) == stored[method]["measured_decisions"] == 67
        assert np.isclose(median, stored[method]["median_ms"], rtol=0, atol=1e-9)
        assert np.isclose(p95, stored[method]["p95_ms"], rtol=0, atol=1e-9)
        latency[method] = values
        timing_summary.append(dict(method=method, median_ms=median, p95_ms=p95, observations=len(values)))
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=int)
        ranks[order] = np.arange(1, len(values) + 1)
        timing_rows.extend(dict(method=method, decision_index=i + 5, local_pipeline_ms=float(v),
                                ecdf_rank=int(ranks[i]), empirical_cdf=float(ranks[i]/len(values)))
                           for i, v in enumerate(values))

    def csv_write(name, rows):
        with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    csv_write(f"{STEM}_development.csv", dev_rows)
    csv_write(f"{STEM}_latency_observations.csv", timing_rows)
    csv_write(f"{STEM}_latency_summary.csv", timing_summary)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7,
                         "axes.edgecolor": "#64707C", "text.color": "#243442",
                         "axes.labelcolor": "#243442", "xtick.color": "#243442",
                         "ytick.color": "#243442", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(165 / 25.4, 92 / 25.4), facecolor="white")
    # Explicit axes retain sufficient printed type size at journal text width.
    ax = fig.add_axes([.085, .37, .365, .46])
    cmap = LinearSegmentedColormap.from_list("wait_cost", ["#FFF5EE", "#D96D5E"])
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=59.8, vmax=63.4)
    ax.set_xticks(range(3), (6, 12, 18))
    ax.set_yticks(range(2), (0, 4))
    ax.set_xlabel("Planning horizon (steps)")
    ax.set_ylabel("Terminal weight")
    ax.set_title("(a) Development search", loc="left", pad=12)
    for i in range(2):
        for j in range(3):
            is_selected = i == 0 and j == 1
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=10,
                    fontweight="bold" if is_selected else "normal", color="#243442")
    ax.add_patch(Rectangle((.53, -.47), .94, .94, fill=False, edgecolor="#9A372F", linewidth=1.8))
    ax.tick_params(length=0)
    cax = fig.add_axes([.085, .195, .365, .035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[60, 61, 62, 63])
    cb.set_label("Accrued waiting (min/request)", labelpad=3)
    cb.outline.set_linewidth(.5)
    fig.text(.2675, .055, "Outlined cell: selected before testing", ha="center", fontsize=8.5)

    bx = fig.add_axes([.60, .37, .37, .46])
    colors = {"rollout": "#D96D5E", "combined_mpc6": "#456A90", "combined_mpc12": "#799B83"}
    labels = {"rollout": "Rollout", "combined_mpc6": "MPC6", "combined_mpc12": "MPC12"}
    for row in timing_summary:
        method = row["method"]
        x = np.sort(latency[method])
        y = np.arange(1, len(x) + 1) / len(x)
        bx.step(np.r_[x[0], x], np.r_[0, y], where="post", linewidth=1.7, color=colors[method],
                label=f"{labels[method]}: {row['median_ms']:.1f} / {row['p95_ms']:.1f}")
    bx.set_xscale("log")
    bx.set_xlim(10, 1500)
    bx.set_xticks([10, 100, 1000])
    bx.xaxis.set_major_formatter(ScalarFormatter())
    bx.set_ylim(0, 1.02)
    bx.set_yticks([0, .25, .5, .75, 1])
    bx.set_ylabel("Empirical cumulative fraction")
    bx.set_xlabel("Local pipeline latency (ms; log scale)")
    bx.set_title("(b) Serial computation", loc="left", pad=12)
    bx.grid(axis="both", which="major", color="#E1E7EB", linewidth=.6)
    bx.set_axisbelow(True)
    leg = bx.legend(loc="upper left", bbox_to_anchor=(-.03, -.29), frameon=False,
                    title="Median / 95th percentile (ms)", title_fontsize=8.5,
                    handlelength=1.4, handletextpad=.5, borderaxespad=0, labelspacing=.3)
    leg._legend_box.align = "left"
    fig.text(.785, .91, "67 post-warmup decisions per method", ha="center", fontsize=8.5)
    for ext in ("pdf", "svg", "eps", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)
    audit = {"figure": STEM, "source_sha256": sources,
             "development": {"selection": selection["selected"], "candidate_count": 6,
                             "score_formula": "sum(waiting_area_min)/sum(arrivals_count) across day0, day6, day12",
                             "verified_against_raw_episode_results": True,
                             "role": "development selection; not held-out test evidence"},
             "timing": {"rows": timing_summary, "raw_field": "local_pipeline_ms", "discard_first": 5,
                        "summary_agreement_absolute_tolerance_ms": 1e-9,
                        "conditions": summary["conditions"], "included": summary["included"],
                        "excluded": summary["excluded"], "platform": summary["platform"]},
             "dimensions_mm": [165, 92], "font": "Arial", "minimum_font_pt": 8.5,
             "png_dpi": 450, "experiments_rerun": False,
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": str(OUT / STEM), "verified_timing": timing_summary,
                      "verified_development_candidates": len(dev_rows)}, indent=2))


if __name__ == "__main__":
    main()
