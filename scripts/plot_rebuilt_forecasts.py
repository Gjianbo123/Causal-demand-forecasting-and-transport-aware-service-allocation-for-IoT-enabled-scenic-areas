"""Reproducible manuscript figures from frozen ScenicIoT-Rebuild assets only.

No model fitting, hyperparameter changes, or policy selection are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

BLUE = "#0072B2"
ORANGE = "#D55E00"
BLACK = "#222222"
GRAY = "#777777"
LIGHT = "#D5D5D5"


def style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": .65, "lines.linewidth": 1.2, "xtick.major.width": .6,
        "ytick.major.width": .6, "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, output, name):
    for extension in ("png", "pdf", "eps"):
        fig.savefig(output/f"{name}.{extension}", dpi=1200)
    plt.close(fig)


def panel(ax, letter, x=-.12, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def architecture(data, output):
    fig = plt.figure(figsize=(6.5, 4.0))
    flow = fig.add_axes([.055, .635, .92, .295])
    flow.set(xlim=(-.12, 6.12), ylim=(-.57, 1.38))
    flow.axis("off")
    panel(flow, "a", x=-.025, y=1.02)
    boxes = ["Synthetic\nnode inflow", "Causal IoT\ntelemetry", "Frozen\nEdgeSTGRU",
             "F0–F4 fusion\n+ PPO", "Integer repair\n+ dispatch", "FIFO service\nqueues"]
    for i, label in enumerate(boxes):
        face = "#E7F1F7" if i in (1, 2) else "#FCF0E9" if i in (3, 4) else "#F1F1F1"
        box = FancyBboxPatch((i+.025, .37), .91, .52, boxstyle="round,pad=.012,rounding_size=.025",
                             edgecolor=BLACK, facecolor=face, linewidth=.7)
        flow.add_patch(box)
        flow.text(i+.48, .63, label, ha="center", va="center", fontsize=8)
        if i < 5:
            flow.annotate("", xy=(i+1.017, .63), xytext=(i+.95, .63),
                          arrowprops=dict(arrowstyle="->", color=BLACK, lw=.7, mutation_scale=7))
    # An observed-history branch enters PPO without using forecast outputs.
    flow.plot([1.48, 1.48, 3.48, 3.48], [.9, 1.13, 1.13, .91], color=BLACK, lw=.7)
    flow.annotate("", xy=(3.48, .9), xytext=(3.48, 1.03),
                  arrowprops=dict(arrowstyle="->", color=BLACK, lw=.7, mutation_scale=7))
    flow.text(2.48, 1.17, "Observed history + resource ledger", ha="center", fontsize=8)
    # Request generation and the feedback observation channel are explicit.
    flow.plot([.48, .48, 5.48, 5.48], [.36, .05, .05, .35], color=GRAY, lw=.7, ls="--")
    flow.annotate("", xy=(5.48, .36), xytext=(5.48, .2),
                  arrowprops=dict(arrowstyle="->", color=GRAY, lw=.7, mutation_scale=7))
    flow.text(3.0, .075, "Exogenous service requests", ha="center", va="bottom", fontsize=8,
              bbox=dict(facecolor="white", edgecolor="none", pad=1))
    flow.plot([5.76, 5.76, 1.18, 1.18], [.37, -.35, -.35, .35], color=BLUE, lw=.8)
    flow.annotate("", xy=(1.18, .36), xytext=(1.18, .17),
                  arrowprops=dict(arrowstyle="->", color=BLUE, lw=.8, mutation_scale=7))
    flow.text(3.3, -.32, "Queue observations at the next decision", ha="center", va="bottom", fontsize=8,
              color=BLUE, bbox=dict(facecolor="white", edgecolor="none", pad=1))

    graph = fig.add_axes([.09, .105, .86, .45])
    graph.set(xlim=(-.7, 5.5), ylim=(-1.5, 2.32))
    graph.axis("off")
    panel(graph, "b", x=-.07, y=1.02)
    positions = {}
    for cluster in range(4):
        x = 1.6*cluster
        positions[16+cluster] = (x, .35)
        positions[20+cluster] = (x, -.6)
        for local, (dx, y) in enumerate(((-.34, 1.14), (.34, 1.14), (-.34, 1.95), (.34, 1.95))):
            positions[4*cluster+local] = (x+dx, y)
    for i, j in data["edge_index"]:
        p, q = positions[int(i)], positions[int(j)]
        if {int(i), int(j)} == {16, 19}:
            graph.plot([0,-.58,-.58,5.38,5.38,4.8],[.35,.35,-1.25,-1.25,.35,.35],
                       color=LIGHT,linewidth=1,zorder=1)
        else:
            graph.plot([p[0], q[0]], [p[1], q[1]], color=LIGHT, lw=1, zorder=1)
    for nodes, color, marker in ((range(16), BLUE, "o"), (range(16,20), ORANGE, "s"), (range(20,24), BLACK, "^")):
        x, y = zip(*(positions[n] for n in nodes))
        graph.scatter(x, y, s=185, c=color, marker=marker, zorder=3, edgecolors="white", linewidths=.45)
        for n in nodes:
            if n >= 20:
                graph.text(positions[n][0],positions[n][1]-.36,str(n),color=BLACK,
                           fontsize=8,va="center",ha="center",zorder=4)
            else:
                graph.text(*positions[n], str(n), color="white", fontsize=8, va="center", ha="center", zorder=4)
    legend = [Line2D([], [], marker=m, linestyle="none", color=c, markersize=6, label=label)
              for c,m,label in ((BLUE,"o","Service node"),(ORANGE,"s","Hub"),(BLACK,"^","Gate"))]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5,.024), ncol=3,
               frameon=False, handletextpad=.4, columnspacing=1.6)
    save(fig, output, "Fig1_architecture")


def demand_figure(data, output):
    times = data["times"]
    operating = times > 0
    total = data["demand"][:, operating].sum(axis=(1,2,3))
    mean_inflow = data["inflow"][:, operating].mean(axis=0).T
    days = np.arange(1, len(total)+1)
    fig = plt.figure(figsize=(6.5, 4.35))
    upper = fig.add_axes([.105, .61, .735, .31])
    upper.axvspan(126.5,153.5, facecolor="#EEF3F7", zorder=0)
    upper.axvspan(153.5,180.5, facecolor="#FCF0E9", zorder=0)
    upper.plot(days,total,color=BLUE,lw=.8)
    upper.scatter(days,total,color=BLUE,s=5,linewidths=0)
    for boundary in (126.5,153.5):
        upper.axvline(boundary,color=GRAY,ls="--",lw=.65)
    ymax = total.max()*1.35
    upper.set(xlim=(.5,180.5),ylim=(0,ymax),xticks=[1,45,90,126,153,180],
              xlabel="Synthetic day",ylabel="Service requests / day")
    for midpoint,label in ((63.5,"Train (126 days)"),(140,"Validation\n(27 days)"),(167,"Test\n(27 days)")):
        upper.text(midpoint,ymax*.94,label,fontsize=8,ha="center",va="top")
    upper.grid(axis="y",color="#E8E8E8",lw=.5)
    upper.set_axisbelow(True)
    panel(upper,"a",x=-.115)
    lower = fig.add_axes([.105,.12,.735,.32])
    im = lower.pcolormesh(np.arange(5,726,10),np.arange(-.5,24,1),mean_inflow,cmap="Blues",
                         shading="flat",edgecolors="none",antialiased=False,
                         vmin=0,vmax=np.ceil(mean_inflow.max()))
    lower.set(xlabel="Time from opening (min)",ylabel="Node ID",xticks=[10,120,240,360,480,600,720],
              yticks=[0,4,8,12,16,20,23])
    for boundary in (15.5,19.5):
        lower.axhline(boundary,color="#BBBBBB",lw=.6)
    lower.spines[["top","right"]].set_visible(True)
    cbax = fig.add_axes([.855,.12,.018,.32])
    cb = fig.colorbar(im,cax=cbax)
    cb.set_label("Mean inflow\n(events / 10 min)",fontsize=8,labelpad=6)
    cb.ax.tick_params(labelsize=8,length=2)
    panel(lower,"b",x=-.115)
    pd.DataFrame({"day":days,"service_requests":total,
                  "split":["train"]*126+["validation"]*27+["test"]*27}).to_csv(output/"Fig2_daily_requests.csv",index=False)
    pd.DataFrame(mean_inflow,index=np.arange(24),columns=(times[operating]*10)).to_csv(output/"Fig2_mean_node_inflow.csv",index_label="node")
    save(fig,output,"Fig2_demand")


def forecasting_figure(folder, output):
    metrics = pd.read_csv(folder/"metrics.csv")
    metrics = metrics[metrics.split=="test"]
    daily = pd.read_csv(folder/"daily_metrics.csv")
    daily = daily[(daily.split=="test") & daily.model.isin(["edge_stgru","gru"])]
    paired = daily.groupby(["model","seed","day"]).mae.mean().unstack("model")
    paired["difference"] = paired.edge_stgru-paired.gru
    pairs = paired.groupby("day").difference.agg(["mean","std","count"])
    if not (pairs["count"]==3).all() or len(pairs)!=27:
        raise ValueError("Expected three matched forecast seeds for each of 27 test days")
    pairs.to_csv(output/"Fig3_daily_paired_differences.csv",index_label="zero_based_dataset_day")
    summary = metrics.groupby(["model","horizon_steps"]).mae.agg(["mean","std","count"]).reset_index()
    summary.to_csv(output/"Fig3_horizon_mae.csv",index=False)
    fig,axes = plt.subplots(1,2,figsize=(6.5,3.5),gridspec_kw={"width_ratios":[1.12,1]})
    fig.subplots_adjust(left=.095,right=.985,bottom=.19,top=.72,wspace=.37)
    descriptors = [
        ("persistence","Persistence",BLACK,"^","--"),
        ("historical_average","Historical mean",GRAY,"x",":"),
        ("ridge","Ridge",BLUE,"D","-"),
        ("lstm","LSTM",GRAY,"s","--"),
        ("gru","GRU",ORANGE,"o","--"),
        ("edge_stgru","EdgeSTGRU",ORANGE,"v","-"),
    ]
    handles = []
    for name,label,color,marker,linestyle in descriptors:
        values = summary[summary.model==name].sort_values("horizon_steps")
        if len(values)!=4:
            raise ValueError(f"Missing horizon for {name}")
        styling = dict(label=label,color=color,marker=marker,markersize=5 if name=="lstm" else 4,
                       linestyle=linestyle,lw=1,markerfacecolor="white" if name in ("gru","lstm") else color)
        if (values["count"]==1).all():
            handle, = axes[0].plot(values.horizon_steps*10,values["mean"],**styling)
        else:
            handle = axes[0].errorbar(values.horizon_steps*10,values["mean"],yerr=values["std"],
                                      capsize=2,elinewidth=.8,capthick=.7,**styling)
        handles.append(handle)
    axes[0].set(xlabel="Forecast horizon (min)",ylabel="MAE (events / 10 min)",
                xlim=(7,63),xticks=[10,20,40,60],ylim=(0,4),yticks=[0,1,2,3,4])
    axes[0].grid(axis="y",color="#E6E6E6",lw=.55)
    axes[0].set_axisbelow(True)
    panel(axes[0],"a",x=-.19,y=1.07)
    x = np.arange(1,28)
    axes[1].axhline(0,color=BLACK,ls="--",lw=.8)
    axes[1].errorbar(x,pairs["mean"],yerr=pairs["std"],fmt="o",color=BLUE,
                    markersize=3,capsize=1.5,elinewidth=.7,capthick=.6)
    average = float(pairs["mean"].mean())
    axes[1].axhline(average,color=ORANGE,lw=.8)
    extent = max(abs(float((pairs["mean"]-pairs["std"]).min())), abs(float((pairs["mean"]+pairs["std"]).max())))
    axes[1].set(xlabel="Held-out day",ylabel="MAE difference (events / 10 min)",
                xlim=(.3,27.7),xticks=[1,7,14,21,27],ylim=(-extent*1.12,max(.025,extent*.1)))
    panel(axes[1],"b",x=-.2,y=1.07)
    fig.legend(handles=handles,labels=[d[1] for d in descriptors],loc="upper center",
               bbox_to_anchor=(.51,.99),ncol=3,frameon=False,columnspacing=1.8,handlelength=2.3)
    save(fig,output,"Fig3_forecasting")
    return summary, pairs


def captions(output_parent, summary, pairs):
    ridge = summary[summary.model=="ridge"]["mean"].mean()
    graph = summary[summary.model=="edge_stgru"]["mean"].mean()
    text = f"""Figure 1. Independently specified synthetic forecasting and control system. (a) The causal telemetry interface supplies observed histories and resource-state information to a frozen forecasting model and PPO policy. F0 excludes forecast features, while F1–F4 use alternative forecast-fusion rules. Integer target repair enforces resource budgets, holding limits and committed incoming transfers; dispatched units are unavailable during travel. Independent service requests enter FIFO queues, whose observations feed the next decision. Future horizons beyond the publicly known closing time are set to zero for control. (b) The fixed 24-node, 31-edge undirected graph contains 16 service nodes (0–15), four hubs (16–19), and four gates (20–23). Node placement is schematic and does not represent edge lengths. These are synthetic visit events and service requests, not tracked individual tourists or measured IoT traffic.

Figure 2. Realized synthetic demand and the fixed data split. (a) Total service requests during each of 180 operating days. Days 1–126 are used for training, 127–153 for validation, and 154–180 for held-out testing. Each visit event can independently generate requests for multiple service types; totals therefore count requests rather than unique tourists. (b) Mean node inflow by ten-minute operating interval, averaged over all 180 generated days for descriptive visualization. Horizontal boundaries separate service nodes, hubs and gates. Every day contains 72 operating intervals; pre-opening warm-up observations are excluded from both panels. No measured site data are used, and the held-out data shown here were not used for fitting or tuning.

Figure 3. Held-out inflow forecasting performance. (a) Mean absolute error (MAE) for all six models at 10-, 20-, 40- and 60-minute horizons. Neural-model points are means over three separately trained seeds; error bars show sample standard deviations across those seeds and can be smaller than the markers. Persistence, historical mean and ridge regression are deterministic under the fixed data split and have no seed-variability bars. Every horizon uses the same 67 prediction origins on each of 27 test days and all 24 nodes (43,416 target values per horizon per seed). The vertical axis starts at zero. Ridge obtains lower MAE than EdgeSTGRU at all four horizons (four-horizon averages: {ridge:.3f} versus {graph:.3f} events per ten-minute node interval), so the graph model is not claimed to be the best predictor. (b) For each test day, EdgeSTGRU minus graph-free GRU MAE is first averaged over the four horizons within each matched training seed, then summarized across three seeds. Points and error bars are the mean and sample standard deviation of these three paired differences; they are not confidence intervals. Negative values favor EdgeSTGRU. The black dashed line denotes no difference, and the orange line denotes the average over 27 test days ({pairs['mean'].mean():.3f} events per ten-minute node interval). Days remain correlated scenario observations and are not treated as independent training replications.
"""
    (output_parent/"figure_captions.md").write_text(text,encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",default="runs/scenic_rebuild_v2/assets")
    parser.add_argument("--output",default="output/revision/figures")
    args = parser.parse_args()
    assets,output = Path(args.assets),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    with np.load(assets/"synthetic_dataset.npz",allow_pickle=False) as loaded:
        data = {k:loaded[k] for k in loaded.files}
    style()
    architecture(data,output)
    demand_figure(data,output)
    summary,pairs = forecasting_figure(assets/"forecast",output)
    captions(output.parent,summary,pairs)
    (output/"figure_provenance.json").write_text(json.dumps({
        "assets":str(assets.resolve()),"fitting_performed":False,
        "width_inches":6.5,"png_dpi":1200,"font":"Arial",
        "dataset_npz_sha256":hashlib.sha256((assets/"synthetic_dataset.npz").read_bytes()).hexdigest(),
        "forecast_metrics_csv_sha256":hashlib.sha256((assets/"forecast"/"metrics.csv").read_bytes()).hexdigest(),
        "uncertainty":"sample standard deviation over 3 trained forecasting seeds, not confidence intervals",
        "files":["Fig1_architecture","Fig2_demand","Fig3_forecasting"],
    },indent=2),encoding="utf-8")
    print(f"Saved three figures as PNG/PDF/EPS and captions in {output.parent}")


if __name__=="__main__":
    main()
