"""Summarize the complete frozen v3 matrix, retaining adverse and null results."""
from pathlib import Path
import csv
import json
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from iotexp.analysis import signed_rank,holm
from iotexp.supplement_common import V3, scenarios
from iotexp.util import write_json,digest
OUT=ROOT/'output/supplement_v3/results'


def pool(group):
    denominator=float(group.arrivals_count.sum())
    served=float(group.served_count.sum())
    epochs=float(group.decision_epochs.sum())
    return dict(wait=float(group.waiting_area_min.sum()/denominator),
                unserved_pct=float(100*group.unserved_count.sum()/denominator),
                served_wait=float(group.wait_sum_min.sum()/served),
                movement=float(group.movement_cost.mean()),
                reward=float(group.mean_reward.mean()),
                backlog_exceedance_pct=float(100*group.visitor_overload_count.sum()/group.visitor_overload_denominator.sum()),
                candidate_infeasible_pct=float(100*group.candidate_infeasible_epochs.sum()/epochs),
                repair_pct=float(100*group.repair_epochs.sum()/epochs),
                executable_infeasible_pct=float(100*group.executable_infeasible_epochs.sum()/epochs),
                arrivals=int(denominator),episodes=len(group))


def interval(values):
    x=np.asarray(values,dtype=float)
    n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else 0.0
    half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else 0.0
    return dict(mean=mean,sd=sd,ci_low=mean-half,ci_high=mean+half,n=n)


def paired(left,right,label,family):
    if list(left.index)!=list(right.index): raise ValueError('Unmatched policy seeds')
    delta=left.to_numpy()-right.to_numpy()
    if len(delta)!=10:raise ValueError('Expected ten independently trained paired policies')
    return dict(comparison=label,family=family,**signed_rank(delta),**interval(delta),p_holm=None)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[];chunks=[]
    for predictor,methods,budgets in [('edge_stgru',['F0','F1','F4'],['long','legacy']),('ridge',['F1','F4'],['long'])]:
        for method in methods:
            for seed in range(10):
                for budget in budgets:
                    p=V3/'policies'/predictor/'evaluation_shards'/f'{method}_seed{seed}_{budget}.csv'
                    frame=pd.read_csv(p);inputs.append(p)
                    expected={(s,day) for s,_,_ in scenarios() for day in range(27)}
                    got=set(zip(frame.scenario,frame.environment_seed))
                    assert len(frame)==189 and got==expected,(p,len(frame))
                    assert set(frame.predictor)=={predictor} and set(frame.method)=={method}
                    assert set(frame.training_seed)=={seed}
                    chunks.append(frame)
    raw=pd.concat(chunks,ignore_index=True)
    assert len(raw)==15120
    if raw.duplicated(['predictor','method','training_seed','budget_steps','scenario','environment_seed']).any():
        raise ValueError('Duplicated policy trial')
    rows=[]
    for key,g in raw.groupby(['predictor','method','budget_steps','training_seed','scenario'],sort=True):
        predictor,method,budget,seed,scenario=key
        rows.append(dict(predictor=predictor,method=method,budget_steps=int(budget),training_seed=int(seed),
                         scenario=scenario,regime=('legacy' if scenario=='legacy' else 'surge' if scenario.endswith('surge') else 'nominal'),**pool(g)))
    per_scenario=pd.DataFrame(rows)
    per_scenario.to_csv(OUT/'policy_per_scenario.csv',index=False)
    metric_cols=['wait','unserved_pct','served_wait','movement','reward','backlog_exceedance_pct',
                 'candidate_infeasible_pct','repair_pct','executable_infeasible_pct']
    per_seed=per_scenario.groupby(['predictor','method','budget_steps','training_seed','regime'],sort=True)[metric_cols].mean().reset_index()
    per_seed.to_csv(OUT/'policy_per_seed.csv',index=False)
    summaries=[]
    for key,g in per_seed.groupby(['predictor','method','budget_steps','regime'],sort=True):
        for metric in metric_cols:
            predictor,method,budget,regime=key
            summaries.append(dict(predictor=predictor,method=method,budget_steps=int(budget),
                                  regime=regime,metric=metric,**interval(g[metric])))
    pd.DataFrame(summaries).to_csv(OUT/'policy_summary.csv',index=False)
    def series(predictor,method,budget=43200,regime='nominal'):
        return per_seed[(per_seed.predictor==predictor)&(per_seed.method==method)&(per_seed.budget_steps==budget)&(per_seed.regime==regime)].set_index('training_seed').wait.sort_index()
    contrasts=[paired(series('edge_stgru','F4'),series('edge_stgru','F1'),'Edge F4 minus Edge F1','primary')]
    for a,b,label in [
        (series('edge_stgru','F4'),series('edge_stgru','F0'),'Edge F4 minus Edge F0'),
        (series('ridge','F4'),series('ridge','F1'),'Ridge F4 minus Ridge F1'),
        (series('ridge','F4'),series('edge_stgru','F4'),'Ridge F4 minus Edge F4'),
        (series('edge_stgru','F4'),series('edge_stgru','F4',14400),'Long Edge F4 minus original-budget Edge F4')]:
        contrasts.append(paired(a,b,label,'secondary_four'))
    contrasts.append(paired(series('edge_stgru','F4',regime='surge'),series('edge_stgru','F1',regime='surge'),
                            'Edge F4 minus Edge F1 (surge)','stress_one'))
    family=[c for c in contrasts if c['family']=='secondary_four']
    for row,p in zip(family,holm([c['p_value'] for c in family])):row['p_holm']=p
    pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv',index=False)
    # Exploratory crossed resampling: preserves paired methods and does not call
    # the 3x10 cells independent. Only three generator replicates limits inference.
    matrix=per_scenario[(per_scenario.predictor=='edge_stgru')&(per_scenario.budget_steps==43200)&(per_scenario.regime=='nominal')]
    a=matrix[matrix.method=='F4'].pivot(index='training_seed',columns='scenario',values='wait').sort_index()
    b=matrix[matrix.method=='F1'].pivot(index='training_seed',columns='scenario',values='wait').reindex(index=a.index,columns=a.columns)
    differences=(a-b).to_numpy();rng=np.random.default_rng(817263)
    boot=[]
    for _ in range(10000):
        indices=rng.integers(0,10,10);worlds=rng.integers(0,3,3)
        boot.append(float(differences[np.ix_(indices,worlds)].mean()))
    crossed=dict(exploratory=True,iterations=10000,seed=817263,worlds=3,policies=10,
                 mean=float(differences.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                 caveat='Only three generated worlds; resampling is a sensitivity analysis, not proof of cross-site generalization')
    deterministic=[]
    for p in list((V3/'heuristics').glob('*_episodes.csv'))+[V3/'control/mpc_episodes.csv',V3/'control/oracle_episodes.csv']:
        frame=pd.read_csv(p);inputs.append(p)
        expected_count=81 if 'oracle' in p.name else 189
        assert len(frame)==expected_count,(p,len(frame))
        expected_scenarios=[s for s,_,surge in scenarios() if s!='legacy' and not surge] if 'oracle' in p.name else [s for s,_,_ in scenarios()]
        expected={(scenario,day) for scenario in expected_scenarios for day in range(27)}
        assert set(zip(frame.scenario,frame.environment_seed))==expected,(p,'scenario/day mismatch')
        assert not frame.duplicated(['scenario','environment_seed']).any(),(p,'duplicate day')
        assert frame.method.nunique()==1,(p,'mixed methods')
        for (method,scenario),g in frame.groupby(['method','scenario']):
            deterministic.append(dict(method=method,scenario=scenario,regime=('legacy' if scenario=='legacy' else 'surge' if scenario.endswith('surge') else 'nominal'),**pool(g)))
    deterministic=pd.DataFrame(deterministic)
    deterministic.to_csv(OUT/'deterministic_per_scenario.csv',index=False)
    det_summary=deterministic.groupby(['method','regime'])[metric_cols].mean().reset_index()
    det_summary.to_csv(OUT/'deterministic_summary.csv',index=False)
    training=[]
    for p in (V3/'policies').glob('*/checkpoints/*/training.jsonl'):
        method,seed=p.parent.name.split('_seed');predictor=p.parent.parent.parent.name
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if 'validation_mean_reward' in r:
                training.append(dict(predictor=predictor,method=method,training_seed=int(seed),**r))
    pd.DataFrame(training).to_csv(OUT/'validation_curves.csv',index=False)
    result=dict(policy_episodes=len(raw),heuristic_episodes=567,mpc_episodes=189,perfect_information_episodes=81,
                policy_summaries=summaries,contrasts=contrasts,crossed_bootstrap=crossed,
                deterministic_summaries=det_summary.to_dict(orient='records'),
                conditional_uncertainty='Policy-seed t intervals condition on these generated evaluation worlds',
                input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in inputs},
                protocol_sha256=digest(V3/'protocol.json'),source_sha256=digest(Path(__file__)))
    write_json(OUT/'results.json',result)
    print(json.dumps({k:result[k] for k in ['policy_episodes','heuristic_episodes','mpc_episodes','perfect_information_episodes','contrasts','crossed_bootstrap']},indent=2))


if __name__=='__main__':main()
