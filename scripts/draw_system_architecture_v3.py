"""Draw the implemented ScenicIoT pipeline as editable, publication-scale vectors.

This is a diagram of the synthetic evaluation, not a claim of field deployment.
No experimental values are generated or changed by this script.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath


OUT = ROOT / "output" / "supplement_v3" / "figures"
W, H = 165.0, 210.0  # millimetres; all labels are >= 8.2 pt at this width.
LAYOUT_H = 230.0  # Drawing coordinates; text sizes remain native points.
INK = "#213144"
MUTED = "#4B5D70"
BLUE = "#276B93"
TEAL = "#176B66"
ORANGE = "#995520"
LINE = "#BFCBD4"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "scenic-architecture-v3",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, W), ylim=(0, LAYOUT_H))
    ax.axis("off")
    checks = []

    def txt(x, y, label, size=8.4, weight="normal", color=INK, ha="center", **kw):
        return ax.text(x, y, label, fontsize=size, fontweight=weight,
                       color=color, ha=ha, va="center", linespacing=1.28, **kw)

    def panel(y, height, label, fill):
        ax.add_patch(FancyBboxPatch((10, y), 145, height,
                     boxstyle="round,pad=0,rounding_size=2", linewidth=.75,
                     edgecolor=LINE, facecolor=fill, zorder=0))
        txt(15, y + height - 5.2, label, size=9.2, weight="bold", ha="left")

    def box(x, y, width, height, title, body, color=BLUE, face="white", size=8.4,
            title_size=8.8, gap=6.3):
        patch = FancyBboxPatch((x, y), width, height,
                   boxstyle="round,pad=0,rounding_size=1.25", linewidth=.8,
                   edgecolor=color, facecolor=face, zorder=3)
        ax.add_patch(patch)
        heading = txt(x + width/2, y + height - 4.0, title, size=title_size,
                      weight="bold", color=color, zorder=4)
        n = len(body.splitlines())
        center_y = y + height - 4.0 - gap - (n - 1) * size * 25.4/72 * 1.28 / 2
        body_text = txt(x + width/2, center_y, body, size=size, zorder=4)
        checks.append((title, patch, heading, body_text))
        return patch

    def arrow(points, color=INK, lw=.85, dashed=False, zorder=2):
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-1)
        patch = FancyArrowPatch(path=MplPath(points, codes), arrowstyle="-|>",
                    mutation_scale=8.3, lw=lw, color=color,
                    linestyle=(0, (3.4, 2.6)) if dashed else "-",
                    capstyle="round", joinstyle="round", zorder=zorder)
        ax.add_patch(patch)

    txt(82.5, 223.0, "FULLY SYNTHETIC EVALUATION", 9.4, "bold", TEAL)

    # A: sequential development with held-out test days explicitly separated.
    panel(172.5, 41.5, "(a)  OFFLINE DEVELOPMENT", "#F4F7FA")
    width, step = 31.5, 34.5
    offline = [
        ("Scenario generation", "24 nodes; 31 links\nPoisson inflow\nBinomial requests\n180 synthetic days"),
        ("Chronological split", "126 training days\n27 validation days\n27 held-out test days\nTrain-only statistics"),
        ("Forecast fitting", "Fit on training days\nEpoch: min. val. MAE\nFreeze edge-STGRU\n+ preprocessing"),
        ("Policy training", "Nominal telemetry\nPPO on train days\nMax. mean val. reward\nFreeze checkpoint"),
    ]
    for i, (title, body) in enumerate(offline):
        x = 15 + i * step
        box(x, 180, width, 25.7, title, body, size=8.2, title_size=8.3, gap=4.7)
        if i < 3:
            arrow([(x+width+.25, 192.8), (x+step-.45, 192.8)], color=MUTED, lw=.7)
    txt(82.5, 176.4, "Training and selection use training/validation days only", 8.2, color=MUTED)
    arrow([(82.5, 172.3), (82.5, 168.5)], color=MUTED, dashed=True)

    # B: one causal decision; the history bypasses the forecasting branch.
    panel(67, 101, "(b)  CAUSAL DECISION LOOP  |  10 min per step", "#F2F8FA")
    box(15, 136, 63, 22.5, "Causal telemetry gateway",
        "12 frames: inflow + four queues\nMissing / noisy / delayed packets\nReceipt-time filter; causal fallback", color=BLUE)
    box(86, 136, 63, 22.5, "Trusted ledger and context",
        "Stationary and in-transit stock\nReserved destinations and ETA\nClock + known day descriptors", color=TEAL)
    box(15, 106, 63, 22.5, "Four-horizon prediction",
        "Frozen edge-STGRU; inflow + clock\n10 / 20 / 40 / 60 min\nPost-closing leads set to zero", color=BLUE)
    box(86, 106, 63, 22.5, "State encoding",
        "Normalized telemetry history\nExact-slot mask + data age\nLedger, clock and day descriptors", color=TEAL)
    box(15, 72, 63, 26.0, "Forecast fusion variants",
        "F0: none; F1: concatenation\nF2: uniform; F3: shared query\nF4: resource-specific queries", color=BLUE)
    box(86, 72, 63, 26.0, "Actor–critic (PPO)",
        "4 resource types × 16 service nodes\nRaw Gaussian allocation scores\nSample to train; mean to evaluate", color=ORANGE)

    arrow([(46.5, 135.7), (46.5, 128.9)], color=BLUE)
    arrow([(46.5, 105.7), (46.5, 98.4)], color=BLUE)
    arrow([(78.4, 84.8), (85.6, 84.8)], color=BLUE)
    arrow([(117.5, 135.7), (117.5, 128.9)], color=TEAL)
    arrow([(117.5, 105.7), (117.5, 98.4)], color=TEAL)
    # Causal state history feeds the policy independently of the predictor.
    arrow([(78.4, 147.2), (82, 147.2), (82, 117.1), (85.6, 117.1)], color=BLUE)

    # C: same transformation in training and evaluation; transport consumes time.
    panel(10.5, 51, "(c)  ALLOCATION AND SERVICE", "#FBF6F0")
    box(86, 23, 63, 27, "Deterministic integer repair",
        "Softmax → largest remainder\nStock totals and holding bounds\nRespect committed arrivals", color=ORANGE)
    box(15, 23, 63, 27, "Transport and FIFO service",
        "Shortest-distance dispatch\nIn-transit units cannot serve\nRequests wait until service/closing", color=ORANGE)
    arrow([(117.5, 71.6), (117.5, 50.4)], color=ORANGE)
    txt(119.2, 64.9, "raw scores", size=8.2, color=ORANGE, ha="left",
        bbox=dict(facecolor="white", edgecolor="none", pad=.6), zorder=4)
    arrow([(85.6, 36.4), (78.4, 36.4)], color=ORANGE)

    # Feedback is split: corruptible telemetry versus an exact simulation ledger.
    arrow([(14.6, 36.4), (5.5, 36.4), (5.5, 147.2), (14.6, 147.2)],
          color=BLUE, lw=1.0)
    txt(3.1, 91.5, "Next-step count reports", size=8.2, color=BLUE,
        rotation=90, bbox=dict(facecolor="white", edgecolor="none", pad=.4))
    arrow([(46.5, 22.6), (46.5, 16.1), (159.4, 16.1), (159.4, 147.2), (149.4, 147.2)],
          color=TEAL, lw=1.0)
    txt(103, 16.1, "Next-step resource ledger", size=8.2, color=TEAL,
        bbox=dict(facecolor="#FBF6F0", edgecolor="none", pad=1.1), zorder=4)
    txt(82.5, 5.0, "Synthetic inflow is exogenous; service queues respond to allocation.",
        size=8.2, color=MUTED)

    # Verify every box label stays in its own vector boundary before writing.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, patch, *texts in checks:
        pb = patch.get_window_extent(renderer)
        for artist in texts:
            tb = artist.get_window_extent(renderer)
            if tb.x0 < pb.x0 + 1 or tb.x1 > pb.x1 - 1 or tb.y0 < pb.y0 + 1 or tb.y1 > pb.y1 - 1:
                raise RuntimeError(f"Box text extends outside boundary: {label!r}: {artist.get_text()!r}")
    stem = OUT / "System_architecture"
    metadata = {"Title": "ScenicIoT fully synthetic forecasting and allocation architecture",
                "Subject": "Implementation-aligned causal telemetry, forecasting, PPO, and resource service pipeline",
                "Creator": "Matplotlib; scripts/draw_system_architecture_v3.py"}
    fig.savefig(stem.with_suffix(".pdf"), metadata=metadata)
    fig.savefig(stem.with_suffix(".eps"))
    fig.savefig(stem.with_suffix(".svg"), metadata={"Title": metadata["Title"],
                "Description": metadata["Subject"], "Creator": metadata["Creator"]})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    caption = r"""LATEX FIGURE
\begin{figure}[p]
  \centering
  \includegraphics[width=165mm]{figures/System_architecture.pdf}
  \caption{System architecture of the fully synthetic evaluation: (a) offline fitting and checkpoint selection, (b) causal sensing and forecast-informed decisions, and (c) feasible allocation, transport and FIFO service. The policy receives telemetry histories and the trusted ledger independently of forecast fusion. Count-report faults do not affect the ledger}
  \label{fig:system_architecture}
\end{figure}

BODY REFERENCE PARAGRAPH
Figure~\ref{fig:system_architecture} separates offline development from the causal decision loop used for evaluation. Time-ordered training and validation days support fitting and checkpoint selection; test days remain held out and the selected artifacts are frozen. At each 10-min decision, the inflow forecast branch uses only the reconstructed inflow history and clock; the policy's separate state branch also includes queue histories, exact-slot availability masks, measurement ages, and the exact simulated resource ledger. Four lead times feed the F1--F4 fusion alternatives; F0 has no forecast input or trainable fusion component. During PPO training, likelihood ratios are evaluated for the raw Gaussian scores, before deterministic allocation repair; evaluation uses the Gaussian mean. Repair enforces resource conservation, holding limits and committed arrivals. Resources incur transport delay and stationary resources serve requests in FIFO order. Exogenous inflow and request trajectories are fixed before policy evaluation, whereas service queues and resource locations evolve with the allocations. All counts, requests and telemetry faults are simulated; the units served are requests rather than distinct tourists.

IMPLEMENTATION AUDIT NOTES (FOR AUTHORS; NOT PART OF CAPTION)
- Main architecture corresponds to iotexp/scenic.py, scenic_forecast.py, telemetry.py, fusion.py and engine.py; formal frozen configuration is runs/scenic_rebuild_v2/config.json.
- The 31 physical graph links are undirected. Shortest-distance dispatch is deterministic and does not model link congestion or closures.
- Closing-time zeroing masks forecast leads beyond the episode. It is not a graph action mask, a physical closure, or a learned safety mechanism.
- F0's common evaluation code can compute a diagnostic shadow forecast, but the actor receives a zero forecast tensor and its fusion projection is frozen.
- F2 uses uniform horizon weights, F3 a query shared across resource classes, and F4 one query per resource class. Each is trained as a separate policy; these are alternatives, not consecutive blocks.
- The ledger contains stationary inventory, incoming reserved inventory, quantity-weighted remaining travel intervals, and known day descriptors. It is not corrupted with count telemetry.
- Inventory already in transit is committed; repair enforces lower bounds for incoming reservations, holding upper bounds, and resource-class conservation.
- Transport arrivals occur after service at a step and become service-eligible at the next step. Closing unserved requests remain part of the restricted waiting measure.
- The core experiment uses edge-STGRU as the common frozen predictor. Supplementary predictor substitutions should be described separately rather than relabeling this pipeline as jointly trained.
- Figure canvas: 165 x 210 mm. Minimum text size: 8.2 pt at 165-mm placement width. SVG text remains editable; PDF and EPS contain vector paths and embedded fonts. Do not shrink below 165 mm if an 8-pt minimum is required. Panel labels are lowercase, the overall title is omitted, and the synthetic-evaluation annotation states the scope.
"""
    (OUT / "caption.txt").write_text(caption, encoding="utf-8")
    print(f"Wrote PDF, EPS, SVG, PNG and caption to {OUT}")
    print(f"Canvas {W:g} x {H:g} mm; minimum text 8.2 pt; {len(checks)} boxes checked")


if __name__ == "__main__":
    main()
