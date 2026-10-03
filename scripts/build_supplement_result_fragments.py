"""Create numerical LaTeX fragments from the complete extended-study outputs."""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_supplement_manuscript import table,n,pvalue
OUT=ROOT/'output/supplement_v3'
RUN=ROOT/'runs/supplement_v3'
NAMES={'persistence':'Persistence','historical_average':'Historical mean','ridge':'Ridge','lstm':'LSTM','gru':'GRU',
       'edge_stgru':'Edge-STGRU','graph_wavenet':'Graph WaveNet','stid':'STID','itransformer':'iTransformer',
       'uniform_neighbor':'Uniform neighbor','no_edge_length':'No edge length'}


def escaped(value):
    return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def main():
    results=json.loads((OUT/'results/results.json').read_text())
    raw=pd.read_csv(RUN/'forecast/metrics.csv')
    pooled=raw[raw.dataset=='independent_pooled']
    matched=pd.read_csv(RUN/'forecast/matched_ablation/metrics.csv')
    matched=matched[matched.dataset=='independent_pooled']
    assert len(matched)==36
    mean=matched.groupby(['model','horizon_steps']).mae.mean()
    text='The matched spatial ablation fixes the edge-STGRU-selected width and learning rate across all three variants. '
    text+='At 60 minutes, full edge-STGRU, uniform-neighbor aggregation and removal of the edge-length feature obtain MAEs of '
    text+=', '.join(n(mean.loc[m,6],3) for m in ['edge_stgru','uniform_neighbor','no_edge_length'])
    text+=', respectively. These results isolate the specified scalar attention/edge-feature changes under common recurrent settings; they do not remove the physical graph from every model. '
    text+='The improvements at 60 minutes are small, and full edge-STGRU has slightly higher mean error than both matched variants at 10 minutes. With three fits, these descriptive differences do not establish a strong or uniformly beneficial attention/distance effect. '
    text+='The complete matched and independently tuned ablations are reported separately in Supplementary Material.\n\n'
    (OUT/'matched_ablation_results.tex').write_text(text,encoding='utf-8')

    solver_rows=[]
    for method,count in [('mpc',189),('oracle',81)]:
        paths=sorted((RUN/'control/solver_logs'/method).glob('*/*.jsonl'))
        assert len(paths)==count,(method,len(paths))
        for path in paths:
            records=[json.loads(line) for line in path.read_text().splitlines()]
            assert len(records)==72
            regime='legacy' if path.parent.name=='legacy' else 'surge' if path.parent.name.endswith('surge') else 'nominal'
            for row in records:
                solver_rows.append(dict(method=method,regime=regime,**row))
    solvers=pd.DataFrame(solver_rows)
    assert len(solvers)==19440
    solvers.to_csv(OUT/'results/solver_decisions.csv',index=False)
    solver_summary=[]
    for (method,regime),g in solvers.groupby(['method','regime']):
        finite=g.mip_gap[np.isfinite(g.mip_gap)]
        solver_summary.append(dict(method=method,regime=regime,decisions=len(g),accepted=int(g.accepted.sum()),
            fallbacks=int(g.fallback.sum()),status_counts=json.dumps(Counter(map(int,g.status)),sort_keys=True),
            finite_gap_n=len(finite),gap_median=float(finite.median()),gap_p95=float(finite.quantile(.95)),
            gap_max=float(finite.max()),primal_residual_max=float(g.primal_residual.max())))
    pd.DataFrame(solver_summary).to_csv(OUT/'results/solver_summary.csv',index=False)
    latency=pd.read_csv(RUN/'serial_latency/summary.csv')
    assert len(latency)==8 and set(latency.n)=={201}
    def timing(method,predictor='edge_stgru'):
        return latency[(latency.method==method)&(latency.predictor==predictor)].iloc[0]
    f4=timing('F4');mpc=timing('MPC')
    infeasible=[r['mean'] for r in results['policy_summaries'] if r['metric']=='executable_infeasible_pct']
    infeasible += [r['executable_infeasible_pct'] for r in results['deterministic_summaries']]
    if max(infeasible)==0:
        feasibility='Across 1,088,640 learned-policy decisions and 60,264 heuristic/MPC/diagnostic decisions, no executed action violated the implemented resource-invariant checks. '
    else:
        feasibility='Executed-action invariant violations occurred in the extended study and must be interpreted with the archived per-scenario feasibility records. '
    feasibility+='This concerns inventory conservation, commitments and holding/transfer rules; it does not imply zero backlog or physical crowd safety. '
    fallback=int(solvers.fallback.sum()); accepted=int(solvers.accepted.sum())
    feasibility+=f'Of the 19,440 forecast-MPC and perfect-information decisions, {accepted:,} produced an accepted feasible incumbent and {fallback:,} used the defined hold fallback. '
    feasibility+='Supplementary tables report solver status and finite optimality gaps; per-decision logs also retain incumbent residuals. Time-limited solutions are retained and evaluated by their realized FIFO service outcomes.\n\n'
    feasibility+='Uncached serial F4/Edge computation has median '+n(f4.pipeline_ms_median)+' ms and 95th percentile '+n(f4.pipeline_ms_p95)
    feasibility+=' ms per decision, versus '+n(mpc.pipeline_ms_median)+' and '+n(mpc.pipeline_ms_p95)+' ms for flow MPC. '
    feasibility+='These are local processing costs on the stated CPU, rather than sensor-to-actuator field latency. Both must be interpreted relative to the 10-min decision interval, and lower computation time alone does not establish higher service quality.\n\n'
    (OUT/'solver_and_latency_results.tex').write_text(feasibility,encoding='utf-8')

    si=r'\section{Extended forecasting details}'+'\n'
    si+=r'''Table~\ref{tab:supp_configs} reports selected configurations. Table~\ref{tab:supp_forecast_full} gives the complete 60-min comparison, and Table~\ref{tab:supp_matched} isolates the matched spatial ablation. Tables~\ref{tab:supp_forecast_20261001}, \ref{tab:supp_forecast_20261002} and~\ref{tab:supp_forecast_20261003} retain the separate generator-realization results.

'''
    configs=pd.read_csv(RUN/'forecast/selected_configuration_summary.csv')
    assert len(configs)==8
    rows=[]
    for r in configs.itertuples():
        rows.append([NAMES[r.model],str(r.hidden),f'{r.learning_rate:g}',f'{r.parameters:,}',r.best_epochs])
    si+=table('Validation-selected neural configurations. Width is an architecture-specific hidden dimension, not a claim of equal capacity. Best epochs list training seeds 0/1/2.',
              'tab:supp_configs',['Model','Width','Learning rate','Parameters','Selected epochs'],rows)
    selected=json.loads((RUN/'forecast/selected_models.json').read_text())
    ridge=next(row for row in selected if row['model']=='ridge')
    si+='The expanded ridge benchmark selects its penalty from $\\{1,100,10000\\}$ using the common validation targets; the selected value is '+f"${ridge['alpha']:g}$"+'. It uses flattened twelve-frame node inflows and clock features with an unpenalized intercept. This new forecasting-only fit is distinct from the original frozen ridge model used to train the allocation policies.\n\n'
    protocol=json.loads((RUN/'forecast/protocol.json').read_text(encoding='utf-8'))
    si+=r'\subsection{Architecture adaptations and source provenance}'+'\n'
    for model in ['graph_wavenet','stid','itransformer']:
        si+=r'\noindent\textbf{'+NAMES[model]+'.} '+escaped(protocol['architecture_notes'][model])+'\n\n'
    si+=r'''The inspected author repositories are Graph WaveNet (commit \nolinkurl{6b162e80c59a1d494809252eca055cff93dc66b1}), STID (commit \nolinkurl{e8b313bc591bdd0101a1619962c9b503e75127c0}) and iTransformer (commit \nolinkurl{c2426e68ca13f74aaec08045c5c724d8ad328124}). Source URLs, commit-specific files and licenses are retained in the experiment archive. Architecture adaptations use the common twelve-frame input and four non-contiguous direct prediction leads. Their reported performance is conditional on this short-history synthetic task, not a general ranking of the published algorithms.

Graph WaveNet uses a channels-last FP32 memory layout for GPU execution; the network, loss and training budget are unchanged. The runtime records include numerical precision settings. Timing collected during concurrent training is retained as a diagnostic and is not used for the serial controller-latency comparison.

'''
    rows=[]
    for model in NAMES:
        frame=pooled[(pooled.model==model)&(pooled.horizon_steps==6)]
        assert len(frame) in [1,3]
        def stats(metric):
            return n(frame[metric].mean(),3)+(r' $\pm$ '+n(frame[metric].std(ddof=1),3) if len(frame)>1 else '')
        rows.append([NAMES[model],stats('mae'),stats('rmse'),stats('wape_pct'),str(len(frame))])
    si+=table('Complete 60-min forecasting results. Mean $\\pm$ sample SD is over three training seeds for neural models. Deterministic models have one fitted result, without an invented uncertainty interval. The last two rows select their own validation configurations and therefore differ from the matched mechanism ablation.',
              'tab:supp_forecast_full',['Model','MAE','RMSE',r'WAPE \%','Fits'],rows)
    rows=[]
    for model in ['edge_stgru','uniform_neighbor','no_edge_length']:
        f=matched[matched.model==model]
        rows.append([NAMES[model]]+[n(f[f.horizon_steps==h].mae.mean(),3)+r' $\pm$ '+n(f[f.horizon_steps==h].mae.std(ddof=1),3) for h in [1,2,4,6]])
    si+=table('Matched spatial mechanism ablation. All three variants use the validation-selected edge-STGRU width and learning rate, the common optimization protocol and training seeds 0/1/2. Epoch selection remains validation-only. Entries are MAE mean $\\pm$ sample SD.',
              'tab:supp_matched',['Variant','10 min','20 min','40 min','60 min'],rows)
    for dataset_seed in [20261001,20261002,20261003]:
        frame=raw[raw.dataset==f'independent_{dataset_seed}']
        rows=[]
        for model,label in NAMES.items():
            rows.append([label]+[n(frame[(frame.model==model)&(frame.horizon_steps==h)].mae.mean()) for h in [1,2,4,6]])
        si+=table(f'Forecast MAE by realization: generator seed {dataset_seed}. Each model uses the same 1,809 origins; neural entries average three fits.',
                  f'tab:supp_forecast_{dataset_seed}',['Model','10 min','20 min','40 min','60 min'],rows)

    si+=r'\FloatBarrier\section{Extended operational details}'+'\n'
    si+=r'''Tables~\ref{tab:supp_control_nominal} and~\ref{tab:supp_control_surge} separate operational outcomes by generator realization. Tables~\ref{tab:supp_companion_nominal} and~\ref{tab:supp_companion_surge} report served-only waiting, movement and backlog alongside target-repair frequency.

'''
    methods=[('static','Static',None),('reactive','Reactive rule',None),('lookahead','Forecast rule',None),
             ('F0','F0 / no forecast','edge_stgru'),('F1','F1 / Edge','edge_stgru'),('F4','F4 / Edge','edge_stgru'),
             ('F1','F1 / Ridge','ridge'),('F4','F4 / Ridge','ridge'),('forecast_flow_mpc','Flow MPC',None),
             ('perfect_information_flow_mpc','Perfect-information MPC',None)]
    policies=pd.read_csv(OUT/'results/policy_per_scenario.csv')
    policies=policies[policies.budget_steps==43200]
    fixed=pd.read_csv(OUT/'results/deterministic_per_scenario.csv')
    def get(method,pred,scenario,metric):
        frame=policies[(policies.method==method)&(policies.predictor==pred)&(policies.scenario==scenario)] if pred else fixed[(fixed.method==method)&(fixed.scenario==scenario)]
        return '--' if frame.empty else n(frame[metric].mean())
    for regime,label in [('nominal','nominal'),('surge','intensity stress')]:
        rows=[]
        for method,name,pred in methods:
            rows.append([name]+[get(method,pred,f'seed{seed}_{regime}',metric) for seed in [20261001,20261002,20261003] for metric in ['wait','unserved_pct']])
        si+=table('Per-realization operational outcomes under '+label+'. Within each realization, requests are pooled across 27 days. Learned entries average ten policies; columns give wait (min/request) and unserved percentage for each generator seed.',
                  'tab:supp_control_'+regime,['Method',r'1001\newline wait',r'1001\newline \%',r'1002\newline wait',r'1002\newline \%',r'1003\newline wait',r'1003\newline \%'],rows)
    ps=pd.read_csv(OUT/'results/policy_summary.csv');ds=pd.read_csv(OUT/'results/deterministic_summary.csv')
    for regime in ['nominal','surge']:
        rows=[]
        for method,label,pred in methods:
            vals=[]
            for metric in ['served_wait','movement','backlog_exceedance_pct','repair_pct']:
                if pred:
                    frame=ps[(ps.method==method)&(ps.predictor==pred)&(ps.regime==regime)&(ps.budget_steps==43200)&(ps.metric==metric)]
                    vals.append(n(frame['mean'].iloc[0])+r' $\pm$ '+n(frame.sd.iloc[0]))
                else:
                    frame=ds[(ds.method==method)&(ds.regime==regime)]
                    vals.append('--' if frame.empty else n(frame[metric].iloc[0]))
            rows.append([label]+vals)
        si+=table('Companion service and dispatch metrics under '+regime+'. Served-only waiting excludes unserved requests and is not the primary endpoint. Movement is resource-km/day. Backlog exceedance counts service-node intervals above 50 queued requests. Target repair frequency applies to the target-dispatch methods; zero for MPC does not mean the optimizer lacks constraints.',
                  'tab:supp_companion_'+regime,['Method',r'Served wait\newline min',r'Movement\newline km/day',r'Backlog\newline exceedance \%',r'Target\newline repair \%'],rows)
    si+=r'\FloatBarrier\section{Solver and computational diagnostics}'+'\n'
    si+=r'''Table~\ref{tab:supp_solver} summarizes the accepted solver outcomes and recorded gaps. Table~\ref{tab:supp_latency} reports separately measured serial computation times; these exclude concurrent performance-run timings.

'''
    rows=[]
    for r in solver_summary:
        statuses=json.loads(r['status_counts'])
        rows.append([('PI' if r['method']=='oracle' else 'MPC')+' / '+r['regime'],str(r['decisions']),str(statuses.get('0',0)),str(r['fallbacks']),n(100*r['gap_median'],2),n(100*r['gap_p95'],2)])
    si+=table('Solver diagnostics from formal performance episodes. Success means solver status zero under its configured gap tolerance, not a proof of an exact global optimum. Gap summaries include finite recorded values only; per-decision logs retain all statuses, residuals and missing gaps. Formal-run wall times are excluded from latency claims.',
              'tab:supp_solver',['Condition','Decisions','Success','Fallback',r'Gap median\newline \%',r'Gap P95\newline \%'],rows)
    rows=[]
    for r in latency.itertuples():
        label=(r.method+' / '+('Ridge' if r.predictor=='ridge' else 'Edge')) if r.method.startswith('F') and r.method!='F0' else {'static':'Static','reactive':'Reactive','lookahead':'Forecast rule','MPC':'Flow MPC'}.get(r.method,r.method)
        rows.append([label,n(r.forecast_ms_median),n(r.decision_ms_median),n(r.repair_ms_median),n(r.pipeline_ms_median),n(r.pipeline_ms_p95)])
    si+=table('Serial uncached CPU timings (ms), 201 measured decisions per method. Pipeline includes telemetry/state construction as well as prediction, decision and repair. Separate component medians need not sum to the median of their per-decision sum. F0 skips unused forecasts.',
              'tab:supp_latency',['Method',r'Forecast\newline median',r'Decision\newline median',r'Repair\newline median',r'Pipeline\newline median',r'Pipeline\newline P95'],rows)
    si+=r'''The complete release retains compact outcomes and raw served-wait samples for every PPO episode, full causal/action traces for a specified audit subset, complete MPC dispatch traces, source/configuration/checkpoint hashes and the scripts used to derive every table. Independent audits recompute request accounting and MPC flow/FIFO service; a separate deterministic PPO replay checks its stated subset rather than claiming action-level replay of every policy episode. Logged validation scores through the original 14,400-step training prefix are checked against the corresponding scores in the extended runs. Daily P95 served-wait values are available in raw episode records; averaging those values is not presented as a pooled P95.
'''
    (OUT/'extended_supplement_tables.tex').write_text(si,encoding='utf-8')
    print(json.dumps({'solver_decisions':len(solvers),'supplement_tables':si.count('begin{table}'),
                      'latency_methods':len(latency)},indent=2))


if __name__=='__main__':main()
