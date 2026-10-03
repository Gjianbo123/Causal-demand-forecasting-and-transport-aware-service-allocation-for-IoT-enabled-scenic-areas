"""Independent read-only audit of the final locked experiment outputs.

Writes only an audit report/JSON under output/revision. Does not change results.
"""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN = ROOT/'runs/scenic_rebuild_v2'
OUT = ROOT/'output/revision'
CONDITIONS = ['nominal','missing_0.10','missing_0.20','missing_0.30',
              'noise_0.05','noise_0.10','noise_0.20','delay_1','delay_2']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label, atol=1e-7):
    if not np.allclose(actual,expected,rtol=1e-9,atol=atol,equal_nan=False):
        raise AssertionError(f'{label}: {actual} != {expected}')


def exact_signed_rank(differences):
    nonzero=np.asarray(differences)[np.asarray(differences)!=0]
    if not len(nonzero): return 1.0
    ranks=rankdata(abs(nonzero),method='average')
    observed=abs(np.sum(np.sign(nonzero)*ranks))
    samples=[abs(sum(s*r for s,r in zip(signs,ranks)))
             for signs in itertools.product([-1,1],repeat=len(nonzero))]
    return sum(s>=observed-1e-12 for s in samples)/len(samples)


def main():
    started=time.perf_counter()
    config=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
    manifest=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    for file in ('episodes.csv','per_seed.csv','paired_tests.json'):
        if not (RUN/file).is_file():
            raise RuntimeError(f'Final aggregation is not yet available: {file}')
    episodes=pd.read_csv(RUN/'episodes.csv')
    per=pd.read_csv(RUN/'per_seed.csv').set_index(['method','training_seed','condition'])
    tests=json.loads((RUN/'paired_tests.json').read_text())
    assert digest(RUN/'config.json')==manifest['config_sha256']
    source=hashlib.sha256()
    for path in sorted((ROOT/'iotexp').rglob('*.py')):
        source.update(str(path.relative_to(ROOT/'iotexp')).encode())
        source.update(path.read_bytes())
    assert source.hexdigest()==manifest['source_sha256'],'Frozen iotexp source changed'
    for relative,sha in manifest['asset_sha256'].items():
        assert digest(RUN/relative)==sha,f'Changed asset: {relative}'
    forecast_folder=RUN/'assets/forecast'
    forecast_table=pd.read_csv(forecast_folder/'metrics.csv').query("split=='test'")
    with np.load(forecast_folder/'test_targets.npz') as z:
        truth=z['truth']
        assert truth.shape==(1809,24,4)
    forecast_count=0
    for path in forecast_folder.glob('*_test_predictions.npz'):
        model,seed=path.name.removesuffix('_test_predictions.npz').rsplit('_seed',1)
        with np.load(path) as z: predictions=z['prediction']
        for j,horizon in enumerate([1,2,4,6]):
            values=forecast_table[(forecast_table.model==model)&(forecast_table.seed==int(seed))&
                                  (forecast_table.horizon_steps==horizon)].iloc[0]
            error=predictions[:,:,j]-truth[:,:,j]
            close(np.abs(error).mean(),values.mae,'forecast MAE from raw predictions')
            close(np.sqrt(np.square(error).mean()),values.rmse,'forecast RMSE from raw predictions')
            close(100*float(np.abs(error).sum())/float(np.abs(truth[:,:,j]).sum()),values.wape_pct,'forecast WAPE')
            forecast_count+=1
    assert forecast_count==48
    key=['method','training_seed','condition','environment_seed','corruption_replicate']
    expected=set()
    for method in ['F0','F1','F2','F3','F4']:
        conditions=CONDITIONS if method in ['F0','F1','F4'] else ['nominal']
        expected.update(itertools.product([method],range(10),conditions,range(27),[11]))
    assert len(episodes)==7830 and not episodes.duplicated(key).any()
    assert set(episodes[key].itertuples(index=False,name=None))==expected
    assert len(per)==290 and per.index.is_unique and (per.n_trials==27).all()
    assert (episodes.decision_epochs==72).all() and episodes.decision_epochs.sum()==563760
    assert np.isfinite(episodes.select_dtypes(include=[np.number]).to_numpy()).all()
    assert (episodes.served_count+episodes.unserved_count==episodes.arrivals_count).all()
    assert (episodes.executable_infeasible_epochs==0).all()
    expected_corruption={day:int(np.random.SeedSequence([day,11,500009]).generate_state(1)[0]) for day in range(27)}
    assert all(r.corruption_seed==expected_corruption[r.environment_seed] for r in episodes.itertuples())
    with np.load(RUN/'assets/synthetic_dataset.npz') as data:
        arrivals=data['demand'][153:,data['times']>0].sum(axis=(1,2,3))
    assert all(r.arrivals_count==arrivals[r.environment_seed] for r in episodes.itertuples())
    forecast_columns=[c for c in episodes if c.startswith('forecast_abs_sum_') or c.startswith('forecast_count_')]
    assert (episodes.groupby(['condition','environment_seed','corruption_replicate'])[forecast_columns].nunique()==1).all().all(), \
        'Frozen forecast diagnostics differ across paired policies/seeds'
    trained=[]
    for method,seed in itertools.product(['F0','F1','F2','F3','F4'],range(10)):
        folder=RUN/'checkpoints'/f'{method}_seed{seed}'
        metadata=json.loads((folder/'metadata.json').read_text())
        logs=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        val=[v for v in logs if 'validation_selection_score' in v]
        winner=min(val,key=lambda v:v['validation_selection_score'])
        assert metadata['method']==method and metadata['training_seed']==seed
        assert metadata['selection_rule']=='validation_reward'
        assert metadata['total_steps']==14400 and logs[-1]['completed_episodes']==200
        assert metadata['best_step']==winner['step']
        close(metadata['validation_selection_score'],winner['validation_selection_score'],'checkpoint selection')
        assert metadata['checkpoint_sha256']==digest(folder/'best.pt')
        subset=episodes[(episodes.method==method)&(episodes.training_seed==seed)]
        assert set(subset.checkpoint_sha256)=={metadata['checkpoint_sha256']}
        assert set(subset.trainable_parameters)=={metadata['trainable_parameters']}
        trained.append(metadata)
    count=0
    full_paths=[]
    raw_wait_sum=0.0
    for index,group in episodes.groupby(['method','training_seed','condition']):
        wait_vectors=[]
        for row in group.itertuples():
            path=RUN/row.trace_path
            with np.load(path,allow_pickle=False) as trace:
                assert np.array_equal(trace['decision_time'],np.arange(72))
                waits=trace['served_waits_min']
                assert len(waits)==row.served_count and np.isfinite(waits).all() and (waits>=0).all()
                assert (waits%10==0).all()
                close(waits.sum(),row.wait_sum_min,'raw wait sum')
                close(waits.mean(),row.mean_wait_min,'raw wait mean')
                close(np.percentile(waits,95),row.p95_wait_min,'raw wait p95')
                # Compact traces retain all post-interval backlogs. Q(0)=0.
                # Their first 71 values therefore reconstruct the complete
                # pre-interval queue integral independently of the CSV field.
                area=10*trace['unserved_count'][:-1].sum()
                close(area,row.waiting_area_min,'queue integral from raw backlogs')
                close(area/row.arrivals_count,row.restricted_mean_wait_min,'accrued wait')
                assert area>=waits.sum()
                assert int(trace['unserved_count'][-1])==row.unserved_count
                close(trace['movement_cost'].sum(),row.movement_cost,'movement sum')
                close(trace['reward'].mean(),row.mean_reward,'reward mean')
                for trace_key,csv_key in [('candidate_infeasible','candidate_infeasible_epochs'),
                                          ('executable_infeasible','executable_infeasible_epochs'),('repaired','repair_epochs')]:
                    assert int(trace[trace_key].sum())==getattr(row,csv_key)
                close(trace['policy_latency_ms'][5:].mean(),row.policy_latency_ms,'latency aggregation')
                if 'truth_visitor_before' in trace.files:
                    full_paths.append(row.trace_path)
                    close(10*trace['truth_visitor_before'][...,1:].sum(),area,'truth queue area')
                    target=trace['executed_action']
                    close(target.sum(axis=2),np.tile([30,24,16,12],(72,1)),'target resource budgets')
                    assert (target[:,:,16:]==0).all()
                    assert (target[:,:,:16]<=np.array([5,4,3,3])[None,:,None]).all()
                    reserved=trace['ledger'][:,96:192].reshape(72,4,24)
                    assert (target>=reserved).all()
                    receipt=trace['source_receipt']
                    allowed=np.arange(72)[:,None,None]
                    assert (np.isnan(receipt)|(receipt<=allowed)).all()
                    grid=np.arange(72)[:,None,None]-np.arange(11,-1,-1)[None,:,None]
                    event=trace['source_event']
                    assert (np.isnan(event)|(event<=grid)).all()
                    forecast=trace['forecast']
                    closed=np.arange(72)[:,None]+np.array([1,2,4,6])[None]>72
                    assert (forecast.transpose(0,2,1)[closed]==0).all()
                wait_vectors.append(waits.copy())
                raw_wait_sum+=float(waits.sum())
            count+=1
        waits=np.concatenate(wait_vectors)
        row=per.loc[index]
        values={
            'mean_wait_min':waits.mean(),'p95_wait_min':np.percentile(waits,95),
            'restricted_mean_wait_min':group.waiting_area_min.sum()/group.arrivals_count.sum(),
            'unserved_pct':100*group.unserved_count.sum()/group.arrivals_count.sum(),
            'visitor_overload_pct':100*group.visitor_overload_count.sum()/group.visitor_overload_denominator.sum(),
            'movement_cost':group.movement_cost.mean(),
            'repair_pct':100*group.repair_epochs.sum()/group.decision_epochs.sum(),
        }
        for field,value in values.items(): close(row[field],value,f'per-seed {field}')
        if count%1080==0: print(f'Audited {count}/7830 trace files',flush=True)
    contrasts=[]
    for test in tests:
        other=test['comparison'].split('_minus_')[1]
        condition=test['condition']
        difference=np.array([per.loc[('F4',s,condition),'restricted_mean_wait_min']-
                             per.loc[(other,s,condition),'restricted_mean_wait_min'] for s in range(10)])
        assert test['endpoint']=='restricted_mean_wait_min' and test['n_seeds']==10
        close(test['p_value'],exact_signed_rank(difference),'independently enumerated signed rank')
        close(test['mean_difference'],difference.mean(),'paired difference mean')
        delta=student_t.ppf(.975,9)*difference.std(ddof=1)/np.sqrt(10)
        contrasts.append({'comparison':test['comparison'],'condition':condition,'mean':float(difference.mean()),
                          'ci_low':float(difference.mean()-delta),'ci_high':float(difference.mean()+delta),
                          'p':test['p_value'],'p_holm':test['p_holm'],'family':test['family']})
    assert len(contrasts)==11
    for family,count_family in [('secondary_fusion_two',2),('robustness_eight',8)]:
        members=sorted([c for c in contrasts if c['family']==family],key=lambda c:c['p'])
        assert len(members)==count_family
        running=0.0
        for i,member in enumerate(members):
            running=max(running,(count_family-i)*member['p'])
            close(min(1,running),member['p_holm'],'Holm adjustment')
    nominal=per.reset_index().query("condition == 'nominal'")
    nominal_stats={method:{field:{'mean':float(g[field].mean()),'sd':float(g[field].std(ddof=1))}
                          for field in ['restricted_mean_wait_min','unserved_pct','visitor_overload_pct','movement_cost']}
                   for method,g in nominal.groupby('method')}
    manuscript_path=OUT/'manuscript_results.json'
    manuscript_checked=False
    if manuscript_path.exists():
        manuscript=json.loads(manuscript_path.read_text())
        assert manuscript['total_policy_episodes']==7830 and manuscript['policy_decisions']==563760
        for method,fields in nominal_stats.items():
            for field,stats in fields.items():
                for stat,value in stats.items(): close(manuscript['policies'][method][field][stat],value,'manuscript policy value')
        for expected_contrast,actual_contrast in zip(contrasts,manuscript['contrasts']):
            assert expected_contrast['comparison']==actual_contrast['comparison']
            assert expected_contrast['condition']==actual_contrast['condition']
            for field in ['mean','ci_low','ci_high']:
                close(expected_contrast[field],actual_contrast[field],'manuscript paired interval')
        manuscript_checked=True
    baselines=pd.read_csv(RUN/'context/baseline_episodes.csv')
    assert len(baselines)==729 and not baselines.duplicated(['method','condition','environment_seed']).any()
    assert set(baselines[['method','condition','environment_seed']].itertuples(index=False,name=None))==set(
        itertools.product(['static','reactive','lookahead'],CONDITIONS,range(27)))
    base_summary={}
    for row in baselines.itertuples():
        with np.load(RUN/row.trace_path,allow_pickle=False) as trace:
            waits=trace['served_waits_min']
            area=10*trace['unserved_count'][:-1].sum()
            close(area,row.waiting_area_min,'baseline queue integral')
            close(waits.sum(),row.wait_sum_min,'baseline served waiting')
            close(np.percentile(waits,95),row.p95_wait_min,'baseline served P95')
            close(trace['queue_area_min'].sum(),area,'baseline stored queue-area series')
            terminal=10*(trace['terminal_queue_batches']*np.arange(72,-1,-1)[None,None,:]).sum()
            close(area,waits.sum()+terminal,'baseline completed/censored waiting identity')
            assert len(waits)+row.unserved_count==row.arrivals_count==arrivals[row.environment_seed]
    for method,group in baselines.query("condition=='nominal'").groupby('method'):
        base_summary[method]=float(group.waiting_area_min.sum()/group.arrivals_count.sum())
        if manuscript_checked:
            close(base_summary[method],manuscript['baselines'][method]['restricted_mean_wait_min'],'manuscript baseline accrued wait')
    sensitivity_path=RUN/'context/sensitivity_episodes.csv'
    sensitivity_count=None
    if sensitivity_path.exists():
        sensitivity=pd.read_csv(sensitivity_path)
        keys=['method','travel_time_multiplier','training_seed','environment_seed']
        assert len(sensitivity)==1080 and not sensitivity.duplicated(keys).any()
        assert set(sensitivity[keys].itertuples(index=False,name=None))==set(itertools.product(['F1','F4'],[.5,1.5],range(10),range(27)))
        assert (sensitivity.served_count+sensitivity.unserved_count==sensitivity.arrivals_count).all()
        assert (sensitivity.executable_infeasible_epochs==0).all()
        for row in sensitivity.itertuples():
            with np.load(RUN/row.trace_path,allow_pickle=False) as trace:
                area=10*trace['unserved_count'][:-1].sum()
                close(area,row.waiting_area_min,'sensitivity queue integral')
                close(trace['served_waits_min'].sum(),row.wait_sum_min,'sensitivity served waiting')
                assert row.arrivals_count==arrivals[row.environment_seed]
        if manuscript_checked:
            for (method,multiplier,seed),group in sensitivity.groupby(['method','travel_time_multiplier','training_seed']):
                item=next(r for r in manuscript['sensitivity'] if r['method']==method and r['travel_multiplier']==multiplier and r['training_seed']==seed)
                close(group.waiting_area_min.sum()/group.arrivals_count.sum(),item['restricted_mean_wait_min'],'manuscript travel sensitivity')
        sensitivity_count=len(sensitivity)
    report={
        'status':'PASS','policy_episodes':7830,'policy_decisions':563760,'trace_files_checked':count,
        'full_causal_traces_checked':len(full_paths),'trained_checkpoints_checked':len(trained),
        'frozen_assets_checked':len(manifest['asset_sha256']),'per_seed_groups_checked':len(per),
        'forecast_model_horizon_records_checked':forecast_count,
        'paired_contrasts_checked':len(contrasts),'baseline_episodes_checked':729,
        'sensitivity_episodes_checked':sensitivity_count,'manuscript_results_checked':manuscript_checked,
        'zero_executed_violation_epochs':bool((episodes.executable_infeasible_epochs==0).all()),
        'nominal_stats':nominal_stats,'baseline_accrued_wait':base_summary,'contrasts':contrasts,
        'raw_served_wait_sum_minutes':raw_wait_sum,'elapsed_seconds':time.perf_counter()-started,
        'source_sha256':source.hexdigest(),'episodes_csv_sha256':digest(RUN/'episodes.csv'),
        'per_seed_csv_sha256':digest(RUN/'per_seed.csv'),
    }
    OUT.mkdir(exist_ok=True)
    (OUT/'final_numeric_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    primary=contrasts[0]
    lines=['# Independent final numerical audit','',
           'The audit reads frozen outputs and raw traces; it does not modify or select experiments. All checks below passed.',
           '',f'- Complete policy matrix: **7,830 episodes**, **563,760 decisions**, 50 trained policies and 290 within-seed condition aggregates.',
           '- All five methods have nominal evaluations; F0/F1/F4 additionally have eight fault conditions. Each applicable condition uses ten trained seeds, the same 27 held-out days, and corruption replicate 11.',
           f'- Verified {len(manifest["asset_sha256"])} frozen asset hashes, the configuration/source hashes, and all 50 checkpoint hashes. Selected checkpoints agree with the minimum logged negative validation reward; all policies completed 14,400 steps / 200 episodes.',
           '- Every episode has exactly 72 steps, finite numeric fields, the expected corruption seed, and the exact held-out request count from the frozen generator. Served plus terminal-unserved requests equals arrivals.',
           '- Recomputed all 48 model/seed/horizon forecast entries (MAE, RMSE and WAPE) from saved predictions and the 1,809 aligned held-out origin targets.',
           '- Every paired scenario has identical frozen-forecast diagnostics across policy methods and training seeds, confirming a common predictor and matched inflow perturbations.',
           '- Read all 7,830 raw trace files. Independently reconstructed every queue integral from the first 71 post-step backlog counts plus the zero opening backlog; verified mean/P95 served waiting, movement, reward, action flags and inference-time pooling.',
           f'- Verified causal source/receipt bounds, public closing-time forecast masking, target budgets/holding limits/incoming reservations, and truth-based queue integrals in all {len(full_paths)} retained full causal traces.',
           '- Recomputed every within-seed main endpoint and pooled served-request P95. The inferential unit remains the trained policy seed, not the 7,830 episodes or correlated requests.',
           '- Independently enumerated all 1,024 sign assignments for each ten-seed signed-rank comparison and recomputed Holm adjustments separately for the two secondary nominal and eight sensing contrasts.',
           f'- Primary F4 minus F1: **{primary["mean"]:+.6f} min/request**; unadjusted 95% t interval **[{primary["ci_low"]:.6f}, {primary["ci_high"]:.6f}]**; exact signed-rank **p={primary["p"]:.8f}**.',
           f'- Context matrix: 729 deterministic-baseline episodes with raw queue-area and completed-plus-censored-wait identity checks. Travel sensitivity matrix: {sensitivity_count if sensitivity_count is not None else "not yet available at this audit"}, with raw queue-area/served-wait checks when available.',
           f'- Manuscript result JSON cross-check: {"passed" if manuscript_checked else "not yet available; rerun after summary generation"}.','',
           '## Nominal accrued wait per request','',
           '| Method | Mean (min/request) | Sample SD across 10 seeds |','|---|---:|---:|']
    for method,fields in nominal_stats.items():
        stats=fields['restricted_mean_wait_min']
        lines.append(f'| {method} | {stats["mean"]:.6f} | {stats["sd"]:.6f} |')
    for method,value in base_summary.items(): lines.append(f'| {method} | {value:.6f} | Deterministic; no trained-seed SD |')
    lines += ['', '## Interpretation and reporting checks','',
              '- `summarize_rebuilt.py` pools daily numerators/denominators within each seed before reporting seed variation. Its paired t intervals use ten paired differences, not ten marginal-policy means, and do not replace the prespecified exact signed-rank tests.',
              '- `build_submission.py` reads new frozen summaries and new forecasting files. No historical manuscript table is used. Ridge and heuristic comparisons are retained, and claims are conditional on the synthetic generator.',
              '- Full causal traces are retained for a prespecified audit subset. Zero violation flags for all episodes are supported by runtime invariants; they are not an independent deployment-safety certificate.',
              '- Accrued waiting is the finite-day sum of waiting already experienced by all requests, including censored terminal requests. It is not complete eventual waiting or unique-tourist waiting.',
              '- Nine sensing conditions apply to F0/F1/F4; F2/F3 have nominal evaluations only. One corruption realization is paired across methods and training seeds for each day.',
              '- Correlated test days and one generated network limit external validity. Statistical precision across seeds does not establish field effectiveness.',
              '',f'Audit source: `scripts/audit_rebuilt_results.py`. Elapsed time: {report["elapsed_seconds"]:.1f} seconds. Input SHA256 digests and exact values are recorded in `final_numeric_audit.json`.']
    (OUT/'FINAL_NUMERIC_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['nominal_stats','contrasts','baseline_accrued_wait']},indent=2))


if __name__=='__main__': main()
