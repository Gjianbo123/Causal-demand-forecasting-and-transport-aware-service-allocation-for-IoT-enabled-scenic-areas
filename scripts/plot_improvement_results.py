"""Scientific result plots, drawn only from complete analyzed tables."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'output/improvement_v4';FIG=OUT/'figures'
LABELS={'combined':'Combined','residual':'Residual GRU','graph_wavenet':'Graph WaveNet','stid':'STID','ridge':'Ridge','edge_stgru':'Edge-STGRU',
        'rollout':'Transport rollout','static':'Static','gwn_mpc6':'GWN / MPC-6','combined_mpc6':'Combined / MPC-6','combined_mpc12':'Combined / MPC-12','ppo_f0':'PPO-F0','ppo_f4':'PPO-F4'}

def save(fig,name,rect=None):
    fig.tight_layout(pad=.8,rect=rect)
    for suffix in ['pdf','svg','png','eps']:fig.savefig(FIG/f'{name}.{suffix}',dpi=220)
    plt.close(fig)

def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    f=pd.read_csv(OUT/'forecast_world_metrics.csv');c=pd.read_csv(OUT/'control_world_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(165/25.4,82/25.4))
    for ax,surge in zip(axes,[False,True]):
        names=['edge_stgru','ridge','stid','graph_wavenet','residual','combined']
        for name,marker,style in zip(names,['o','s','^','D','v','P'],['-','--',':','-.','--','-']):
            g=f[(f.model==name)&(f.surge==surge)]
            mean=g[[f'mae_h{h}' for h in [1,2,4,6]]].mean()
            ax.plot([10,20,40,60],mean,marker=marker,linestyle=style,ms=3,lw=1.4,label=LABELS[name])
        ax.set(xlabel='Forecast lead (min)',ylabel='MAE (visit events)',title='(b) Increased demand' if surge else '(a) Nominal demand',xticks=[10,20,40,60]);ax.grid(color='#E5E5E5')
    fig.legend(*axes[0].get_legend_handles_labels(),fontsize=8.2,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.52,1.01))
    save(fig,'Forecast_accuracy',rect=(0,0,1,.81))
    fig,axes=plt.subplots(1,2,figsize=(165/25.4,85/25.4),sharey=True)
    names=['static','ppo_f0','ppo_f4','gwn_mpc6','combined_mpc6','rollout']
    for ax,surge in zip(axes,[False,True]):
        for i,name in enumerate(names):
            g=c[(c.method==name)&(c.surge==surge)&(c.subset=='full')].wait
            ax.barh(i,g.mean(),color='#F08D82' if name=='rollout' else '#B7CBD9',edgecolor='#56616C',lw=.6)
            ax.errorbar(g.mean(),i,xerr=g.std(),fmt='none',color='#475362',capsize=2,lw=.8)
        ax.set(yticks=range(len(names)),yticklabels=[LABELS[n] for n in names],xlabel='Accrued wait / request (min)',title='(b) Increased demand' if surge else '(a) Nominal demand');ax.grid(axis='x',color='#E8E8E8');ax.set_axisbelow(True)
    axes[0].invert_yaxis();save(fig,'Control_waiting')
    fig,axes=plt.subplots(1,2,figsize=(165/25.4,72/25.4))
    for ax,surge in zip(axes,[False,True]):
        for name,color,marker,style in [('combined_mpc6','#456A90','o','-'),('combined_mpc12','#956534','s','--')]:
            a=c[(c.method=='rollout')&(c.surge==surge)&(c.subset=='matched12')].sort_values('dataset_seed').wait.to_numpy()
            b=c[(c.method==name)&(c.surge==surge)&(c.subset=='matched12')].sort_values('dataset_seed').wait.to_numpy()
            ax.plot(range(1,8),a-b,marker=marker,linestyle=style,label=LABELS[name],color=color,ms=4)
        ax.axhline(0,color='#68717B',lw=.8,ls='--');ax.set(xlabel='Independent random world',ylabel='Rollout minus MPC (min)',xticks=range(1,8),title='(b) Increased demand' if surge else '(a) Nominal demand');ax.grid(color='#E8E8E8')
    axes[0].legend(fontsize=8,frameon=False);save(fig,'Matched_horizon')

if __name__=='__main__':main()
