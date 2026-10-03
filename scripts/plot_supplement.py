"""Publication figures generated only from completed supplementary outputs."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/supplement_v3'
RESULT=ROOT/'output/supplement_v3/results'
OUT=ROOT/'output/supplement_v3/figures'
BLUE='#2166ac';ORANGE='#c45d25';GREEN='#1b8076';GRAY='#7b818b';PURPLE='#8362a5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'axes.labelsize':8.5,
                     'axes.titlesize':9,'legend.fontsize':8,'xtick.labelsize':8,
                     'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)


def forecast():
    raw=pd.read_csv(RUN/'forecast/metrics.csv')
    raw=raw[raw.dataset=='independent_pooled']
    mean=raw.groupby(['model','horizon_steps']).mae.mean()
    fig,ax=plt.subplots(figsize=(6.5,3.05))
    styles=[('ridge','Ridge',BLUE,'s'),('gru','GRU',GRAY,'v'),('edge_stgru','Edge-STGRU',ORANGE,'o'),
            ('graph_wavenet','Graph WaveNet',GREEN,'D'),('stid','STID',PURPLE,'^'),('itransformer','iTransformer','#b94e77','P')]
    # Model IDs are fixed in the forecasting implementation. Fail loudly instead
    # of silently dropping a baseline if its spelling changes.
    available=set(raw.model)
    for model,label,color,marker in styles:
        if model not in available: raise ValueError((model,sorted(available)))
        values=mean.loc[model].reindex([1,2,4,6]).to_numpy()
        ax.plot([10,20,40,60],values,marker=marker,color=color,label=label,lw=1.5,ms=4)
    ax.set(xlabel='Forecast horizon (min)',ylabel='MAE (visit events)',xticks=[10,20,40,60])
    ax.grid(axis='y',alpha=.2)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=3,frameon=False,columnspacing=1.5)
    fig.subplots_adjust(left=.11,right=.985,bottom=.20,top=.78)
    save(fig,'Supplement_forecast')


def control():
    data=pd.read_csv(RESULT/'policy_per_seed.csv')
    data=data[data.budget_steps==43200]
    fixed=pd.read_csv(RESULT/'deterministic_summary.csv')
    methods=[('static','Static',None),('reactive','Reactive rule',None),('lookahead','Forecast rule',None),
             ('F0','F0 / no forecast','edge_stgru'),('F1','F1 / Edge','edge_stgru'),('F4','F4 / Edge','edge_stgru'),
             ('F1','F1 / Ridge','ridge'),('F4','F4 / Ridge','ridge'),
             ('forecast_flow_mpc','Flow MPC',None),('perfect_information_flow_mpc','Perfect-information MPC',None)]
    fig,axes=plt.subplots(1,2,figsize=(6.5,4),sharey=True)
    for ax,regime,title in zip(axes,['nominal','surge'],['(a) New nominal scenarios','(b) Intensity stress (+60%)']):
        for y,(method,label,predictor) in enumerate(methods):
            if predictor:
                v=data[(data.method==method)&(data.predictor==predictor)&(data.regime==regime)].wait.to_numpy()
                assert len(v)==10
                color=ORANGE if method=='F4' else BLUE if method=='F1' else GRAY
                ax.scatter(v,y+np.linspace(-.14,.14,len(v)),s=10,color=color,alpha=.55,zorder=2)
                ax.errorbar(v.mean(),y,xerr=v.std(ddof=1),fmt='D',color=color,ms=4,capsize=2,lw=1.2,zorder=4)
            else:
                v=fixed[(fixed.method==method)&(fixed.regime==regime)]
                if len(v):ax.scatter(v.wait.iloc[0],y,marker='s',s=23,color=GREEN if 'mpc' in method else GRAY,zorder=3)
                else:ax.text(.97,y,'not evaluated',ha='right',va='center',transform=ax.get_yaxis_transform(),fontsize=8,color=GRAY)
        ax.set_title(title,loc='left',pad=10)
        ax.set_xlabel('Accrued wait (min/request)')
        ax.set_xlim(left=0)
        ax.grid(axis='x',alpha=.2)
    axes[0].set(yticks=range(len(methods)),yticklabels=[x[1] for x in methods]);axes[0].invert_yaxis()
    fig.subplots_adjust(left=.27,right=.985,top=.88,bottom=.16,wspace=.16)
    save(fig,'Supplement_control')


def learning():
    training=pd.read_csv(RESULT/'validation_curves.csv')
    data=pd.read_csv(RESULT/'policy_per_seed.csv')
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.15))
    for method,color in [('F0',GRAY),('F1',BLUE),('F4',ORANGE)]:
        subset=training[(training.predictor=='edge_stgru')&(training.method==method)]
        g=subset.groupby('step').validation_mean_reward.agg(['mean','std'])
        x=g.index.to_numpy()/1000
        axes[0].plot(x,g['mean'],color=color,label=method,lw=1.4)
        axes[0].fill_between(x,g['mean']-g['std'],g['mean']+g['std'],color=color,alpha=.15)
    axes[0].set(xlabel='Training steps (thousands)',ylabel='Validation mean reward',title='(a) Training-budget diagnostic')
    axes[0].legend(frameon=False,ncol=3,loc='lower right',handlelength=.8,columnspacing=.8)
    subset=data[(data.predictor=='edge_stgru')&(data.method=='F4')&(data.regime=='nominal')]
    pivot=subset.pivot(index='training_seed',columns='budget_steps',values='wait').reindex(columns=[14400,43200])
    assert pivot.shape==(10,2)
    for values in pivot.to_numpy():axes[1].plot([0,1],values,'o-',color=GRAY,alpha=.55,lw=.8,ms=3)
    axes[1].plot([0,1],pivot.mean().to_numpy(),'D-',color=ORANGE,lw=2,ms=5,label='Mean')
    axes[1].set(xticks=[0,1],xticklabels=['14,400','43,200'],xlabel='Training-step budget',ylabel='Accrued wait (min/request)',title='(b) Paired F4 evaluation')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.subplots_adjust(left=.15,right=.985,bottom=.22,top=.87,wspace=.5)
    save(fig,'Supplement_learning')


def latency():
    data=pd.read_csv(RUN/'serial_latency/summary.csv')
    names=[('static','edge_stgru','Static'),('reactive','edge_stgru','Reactive rule'),('lookahead','edge_stgru','Forecast rule'),
           ('F0','edge_stgru','F0 / no forecast'),('F1','edge_stgru','F1 / Edge'),('F4','edge_stgru','F4 / Edge'),
           ('F4','ridge','F4 / Ridge'),('MPC','edge_stgru','Flow MPC')]
    fig,ax=plt.subplots(figsize=(6.5,3.0))
    for y,(method,predictor,label) in enumerate(names):
        r=data[(data.method==method)&(data.predictor==predictor)].iloc[0]
        median=r.pipeline_ms_median;p95=r.pipeline_ms_p95
        color=ORANGE if method=='F4' else GREEN if method=='MPC' else BLUE
        ax.plot([median,p95],[y,y],color=color,lw=1.2)
        ax.scatter(median,y,color=color,s=24,zorder=3)
        ax.scatter(p95,y,facecolor='white',edgecolor=color,s=24,zorder=3)
    ax.set(xscale='log',xlabel='Local decision pipeline (ms, log scale)',yticks=range(len(names)),yticklabels=[x[2] for x in names])
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.plot([],[],'o',color=GRAY,label='Median');ax.plot([],[],'o',markerfacecolor='white',color=GRAY,label='95th percentile')
    ax.legend(loc='upper right',frameon=False)
    fig.subplots_adjust(left=.24,right=.98,bottom=.19,top=.96)
    save(fig,'Supplement_latency')


if __name__=='__main__':
    forecast();control();learning();latency()
