"""Report all sealed v4 outcomes, requiring the complete declared matrix."""
from pathlib import Path
import sys,json,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy import stats
from iotexp.util import write_json
V4=ROOT/'runs/improvement_v4';OUT=ROOT/'output/improvement_v4'

def paired(a,b):
    d=np.asarray(a)-np.asarray(b);n=len(d);m=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(n))
    null=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1],repeat=n)])
    p=float(np.mean(np.abs(null)>=abs(m)-1e-12))
    t=stats.t.ppf(.975,n-1)
    return dict(n=n,differences=d.tolist(),mean_difference=m,ci95=[m-t*se,m+t*se],p_exact_two_sided=p,
                wins=int((d<0).sum()),ties=int((d==0).sum()),mean_a=float(np.mean(a)),mean_b=float(np.mean(b)),
                relative_reduction_pct=float(100*(np.mean(b)-np.mean(a))/np.mean(b)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((V4/'experiment_plan.json').read_text());seeds=plan['final_seeds']
    forecast=[]
    for path in sorted((V4/'final_forecast').glob('*/complete.json')):
        for r in json.loads(path.read_text())['scores']:
            for j,h in enumerate([1,2,4,6]):r[f'mae_h{h}']=r['horizon_mae'][j]
            del r['horizon_mae'];forecast.append(r)
    f=pd.DataFrame(forecast);assert len(f)==154 and not f.duplicated(['model','dataset_seed','surge']).any()
    f.to_csv(OUT/'forecast_world_metrics.csv',index=False)
    episodes=[];matrix=[]
    for path in sorted((V4/'final_control').glob('*/*/day*/complete.json')):
        record=json.loads(path.read_text());matrix.append((record['dataset_seed'],record['surge'],record['day'],record['method']))
        for r in record['results']:
            episodes.append({k:record[k] for k in ['dataset_seed','surge','day','method']}|r)
    expected={(s,c,d,m) for s,c,d in itertools.product(seeds,[False,True],plan['control_test_offsets'])
              for m in ['static','rollout','gwn_mpc6','combined_mpc6','ppo_f0','ppo_f4']+(['combined_mpc12'] if d in [0,12,24] else [])}
    assert set(matrix)==expected and len(matrix)==len(expected),(len(matrix),len(expected))
    e=pd.DataFrame(episodes);e.to_csv(OUT/'control_episode_metrics.csv',index=False)
    keys=['waiting_area_min','arrivals_count','unserved_count','movement_cost','visitor_overload_count','visitor_overload_denominator','mean_reward','executable_infeasible_epochs']
    days=e.groupby(['dataset_seed','surge','day','method'])[keys].mean().reset_index()
    world=[]
    for subset in ['full','matched12']:
        subset_days=days if subset=='full' else days[days.day.isin([0,12,24])]
        for (seed,surge,method),g in subset_days.groupby(['dataset_seed','surge','method']):
            if subset=='full' and method=='combined_mpc12':continue
            world.append(dict(dataset_seed=seed,surge=surge,method=method,subset=subset,days=len(g),
                wait=float(g.waiting_area_min.sum()/g.arrivals_count.sum()),
                unserved_pct=float(100*g.unserved_count.sum()/g.arrivals_count.sum()),
                overload_pct=float(100*g.visitor_overload_count.sum()/g.visitor_overload_denominator.sum()),
                movement=float(g.movement_cost.mean()),reward=float(g.mean_reward.mean()),
                infeasible=float(g.executable_infeasible_epochs.sum())))
    c=pd.DataFrame(world);c.to_csv(OUT/'control_world_metrics.csv',index=False)
    def series(df,name,column,surge=False,subset=None):
        group=df[(df.get('model',df.get('method'))==name)&(df.surge==surge)]
        if subset is not None:group=group[group.subset==subset]
        return group.set_index('dataset_seed').loc[seeds,column].values
    primary=dict(forecast=paired(series(f,'combined','mae'),series(f,'graph_wavenet','mae')),
                 control=paired(series(c,'rollout','wait',subset='full'),series(c,'combined_mpc6','wait',subset='full')))
    ordered=sorted(primary,key=lambda k:primary[k]['p_exact_two_sided']);last=0
    for rank,key in enumerate(ordered):
        last=max(last,min(1,(2-rank)*primary[key]['p_exact_two_sided']));primary[key]['p_holm']=last
    comparisons={}
    for surge in [False,True]:
        for method in ['static','gwn_mpc6','combined_mpc6','ppo_f0','ppo_f4','combined_mpc12']:
            subset='matched12' if method=='combined_mpc12' else 'full'
            comparisons[f'{method}_{"surge" if surge else "nominal"}']=paired(series(c,'rollout','wait',surge,subset),series(c,method,'wait',surge,subset))
    fsummary=f.groupby(['surge','model']).agg({k:['mean','std'] for k in ['mae','rmse','wape','mae_h1','mae_h2','mae_h4','mae_h6']})
    fsummary.columns=['_'.join(x) for x in fsummary.columns];fsummary.reset_index().to_csv(OUT/'forecast_summary.csv',index=False)
    csummary=c.groupby(['surge','subset','method']).agg({k:['mean','std'] for k in ['wait','unserved_pct','overload_pct','movement','reward']})
    csummary.columns=['_'.join(x) for x in csummary.columns];csummary.reset_index().to_csv(OUT/'control_summary.csv',index=False)
    result=dict(primary=primary,exploratory_control=comparisons,scope='Independent stochastic worlds of one fully synthetic graph and generator; fixed selected checkpoints. No real-data claim.',
        sign_flip_assumption='Independent seed-level paired differences; exact under sign-symmetry/exchangeability at the null. Seven worlds yield coarse p-values; t intervals descriptive.',
        episodes=int(len(e)),flow_episodes=546,ppo_episodes=2520,decision_epochs=int(e.decision_epochs.sum()),
        execution_infeasible=int(e.executable_infeasible_epochs.sum()),solver_fallback_steps=int(e.solver_fallback_steps.sum()))
    write_json(OUT/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
