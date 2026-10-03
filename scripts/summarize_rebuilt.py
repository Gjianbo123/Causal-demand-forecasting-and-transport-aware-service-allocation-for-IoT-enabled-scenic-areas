"""Final auditable summaries and manuscript figures for the new experiments."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iotexp.analysis import signed_rank

OUT = ROOT/'output/revision'
RUN = ROOT/'runs/scenic_rebuild_v2'
FIG = OUT/'figures'
LABELS = {'F0':'Reactive PPO', 'F1':'Concatenation', 'F2':'Uniform fusion',
          'F3':'Shared attention', 'F4':'Resource attention'}
COLORS = {'F0':'#707070','F1':'#0072B2','F2':'#009E73','F3':'#CC79A7','F4':'#D55E00'}


def interval(values):
    x = np.asarray(values, float)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x)>1 else 0.0
    delta = float(student_t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x))) if len(x)>1 else 0.0
    return dict(mean=mean, sd=sd, ci_low=mean-delta, ci_high=mean+delta, n=len(x))


def save(fig, name):
    fig.savefig(FIG/f'{name}.png',dpi=1200,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.eps',bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    per=pd.read_csv(RUN/'per_seed.csv')
    episodes=pd.read_csv(RUN/'episodes.csv')
    assert len(episodes)==7830
    nominal=per[per.condition=='nominal']
    metrics=['restricted_mean_wait_min','mean_wait_min','p95_wait_min','unserved_pct',
             'visitor_overload_pct','movement_cost','repair_pct','executable_infeasible_pct',
             'policy_latency_ms','local_pipeline_latency_ms','candidate_infeasible_pct']
    policies={m:{col:interval(g[col]) for col in metrics} for m,g in nominal.groupby('method')}
    baselines=pd.read_csv(RUN/'context/baseline_episodes.csv')
    print('Baseline columns',list(baselines.columns))
    base_summary={}
    for name,g in baselines[baselines.condition=='nominal'].groupby('method'):
        waits=[]
        for r in g.itertuples():
            trace=RUN/r.trace_path
            if not trace.exists(): trace=RUN/'context'/r.trace_path
            with np.load(trace) as z: waits.append(z['served_waits_min'])
        waits=np.concatenate(waits)
        base_summary[name]=dict(restricted_mean_wait_min=float(g.waiting_area_min.sum()/g.arrivals_count.sum()),
                                mean_wait_min=float(waits.mean()),p95_wait_min=float(np.percentile(waits,95)),
                                unserved_pct=float(100*g.unserved_count.sum()/g.arrivals_count.sum()),
                                visitor_overload_pct=float(100*g.visitor_overload_count.sum()/g.visitor_overload_denominator.sum()),
                                movement_cost=float(g.movement_cost.mean()),
                                executable_infeasible_pct=float(g.executable_infeasible_pct.mean()))
    tests=json.loads((RUN/'paired_tests.json').read_text())
    contrasts=[]
    for test in tests:
        other=test['comparison'].split('_minus_')[1]
        a=per[(per.method=='F4')&(per.condition==test['condition'])].sort_values('training_seed')
        b=per[(per.method==other)&(per.condition==test['condition'])].sort_values('training_seed')
        diff=a.restricted_mean_wait_min.to_numpy()-b.restricted_mean_wait_min.to_numpy()
        contrasts.append({**test,**interval(diff)})
    forecast=pd.read_csv(RUN/'assets/forecast/metrics.csv')
    # Keep exact stored column names, inspected by the manuscript builder.
    forecast.to_csv(OUT/'forecast_results.csv',index=False)
    sens_path=RUN/'context/sensitivity_episodes.csv'
    sensitivity=[]
    if sens_path.exists():
        sdf=pd.read_csv(sens_path)
        for (method,scale,seed),g in sdf.groupby(['method','travel_time_multiplier','training_seed']):
            sensitivity.append(dict(method=method,travel_multiplier=scale,training_seed=int(seed),
                                    restricted_mean_wait_min=float(g.waiting_area_min.sum()/g.arrivals_count.sum()),
                                    unserved_pct=float(100*g.unserved_count.sum()/g.arrivals_count.sum())))
    result=dict(policies=policies,baselines=base_summary,contrasts=contrasts,sensitivity=sensitivity,
                total_policy_episodes=len(episodes),total_baseline_episodes=len(baselines),
                executed_infeasible_epochs=int(episodes.executable_infeasible_epochs.sum()),
                policy_decisions=int(episodes.decision_epochs.sum()))
    (OUT/'manuscript_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    nominal.to_csv(OUT/'nominal_per_seed.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts_with_intervals.csv',index=False)
    # Main outcomes: show every training seed, plus its mean. No truncated bars.
    fig,axes=plt.subplots(1,2,figsize=(6.6,2.9),layout='constrained')
    rng=np.random.default_rng(42)
    for j,(col,ylab) in enumerate([('restricted_mean_wait_min','Accrued wait per request (min)'),
                                  ('unserved_pct','Requests unserved at closing (%)')]):
        for i,m in enumerate(LABELS):
            values=nominal[nominal.method==m][col].to_numpy()
            axes[j].scatter(np.full(len(values),i)+rng.uniform(-.12,.12,len(values)),values,
                            s=14,color=COLORS[m],alpha=.75)
            axes[j].plot([i-.22,i+.22],[values.mean()]*2,color='black',lw=1.4)
        axes[j].set_xticks(range(5),list(LABELS),rotation=0)
        axes[j].set_ylabel(ylab)
        axes[j].text(-.16,1.03,f'({chr(97+j)})',transform=axes[j].transAxes)
        axes[j].grid(axis='y',lw=.4,color='#dddddd')
    save(fig,'Fig4_allocation')
    fig,axes=plt.subplots(1,3,figsize=(6.6,2.6),layout='constrained')
    factors=[([0,.1,.2,.3],['nominal','missing_0.10','missing_0.20','missing_0.30'],'Packet loss probability'),
             ([0,.05,.1,.2],['nominal','noise_0.05','noise_0.10','noise_0.20'],'Noise / training SD'),
             ([0,10,20],['nominal','delay_1','delay_2'],'Upload delay (min)')]
    for j,(levels,names,label) in enumerate(factors):
        for m in ('F0','F1','F4'):
            groups=[per[(per.method==m)&(per.condition==c)].restricted_mean_wait_min for c in names]
            means=np.array([g.mean() for g in groups]); sds=np.array([g.std(ddof=1) for g in groups])
            axes[j].errorbar(levels,means,yerr=sds,fmt='o-',ms=3,lw=1,capsize=2,label=m,color=COLORS[m])
        axes[j].set_xlabel(label)
        axes[j].set_xticks(levels)
        axes[j].grid(axis='y',lw=.4,color='#dddddd')
        axes[j].text(-.2,1.04,f'({chr(97+j)})',transform=axes[j].transAxes)
    axes[0].set_ylabel('Accrued wait per request (min)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='outside upper center',frameon=False,fontsize=8)
    save(fig,'Fig5_sensing')
    print('Saved authoritative manuscript result tables and figures')


if __name__=='__main__': main()
