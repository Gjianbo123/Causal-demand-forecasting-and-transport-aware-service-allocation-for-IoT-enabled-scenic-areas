"""Assemble the Springer manuscript exclusively from complete v4 records."""
from pathlib import Path
import sys,json,shutil,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from iotexp.util import write_json,digest
V4=ROOT/'runs/improvement_v4';OUT=ROOT/'output/improvement_v4';DEST=OUT/'Discover_IoT_LaTeX'
BASE=ROOT/'output/supplement_v3/Discover_IoT_LaTeX'
FN={'persistence':'Persistence','anchor':'Scaled seasonal profile','ridge':'Ridge','lstm':'LSTM','gru':'GRU','edge_stgru':'Edge-STGRU','graph_wavenet':'Graph WaveNet','stid':'STID','itransformer':'iTransformer','residual':'Residual GRU','combined':'Combined (proposed)'}
CN={'static':'Static','ppo_f0':'PPO-F0','ppo_f4':'PPO-F4','gwn_mpc6':'GWN / MPC-6','combined_mpc6':'Combined / MPC-6','combined_mpc12':'Combined / MPC-12','rollout':'Transport rollout'}

def n(v,d=2):return f'{float(v):.{d}f}'
def table(caption,label,headers,rows,spec=None):
    spec=spec or 'L'+'Y'*(len(headers)-1)
    return '\n'.join([r'\begin{table}[!htbp]',r'\caption{'+caption+r'}\label{'+label+'}',r'\centering\normalsize',
        r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}',r'\begin{tabularx}{\textwidth}{@{}'+spec+'@{}}',r'\toprule',
        ' & '.join(headers)+r' \\',r'\midrule']+[' & '.join(map(str,r))+r' \\' for r in rows]+[r'\bottomrule',r'\end{tabularx}',r'\end{table}',''])
def figure(name,caption,label):return '\n'.join([r'\begin{figure}[!htbp]',r'\centering',r'\includegraphics[width=\textwidth]{'+name+'.pdf}',r'\caption{'+caption+r'}\label{'+label+'}',r'\end{figure}',''])
def put(name,value):(DEST/name).write_text(value,encoding='utf-8')

def main():
    result=json.loads((OUT/'results.json').read_text());audit=json.loads((V4/'INDEPENDENT_AUDIT.json').read_text());assert audit['status']=='PASS'
    timing=json.loads((V4/'serial_timing/summary.json').read_text())
    f=pd.read_csv(OUT/'forecast_summary.csv');c=pd.read_csv(OUT/'control_summary.csv')
    def fm(name,surge=False):return f[(f.model==name)&(f.surge==surge)].iloc[0]
    def cm(name,surge=False,subset='full'):return c[(c.method==name)&(c.surge==surge)&(c.subset==subset)].iloc[0]
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'tables').mkdir(exist_ok=True);(DEST/'figures').mkdir(exist_ok=True)
    for name in ['sn-jnl.cls','sn-mathphys-num.bst']:shutil.copy2(BASE/name,DEST/name)
    bib=(BASE/'references.bib').read_text(encoding='utf-8')+r'''
@article{ciocan2012,
 author={Dragos Florin Ciocan and Vivek Farias},
 title={Model Predictive Control for Dynamic Resource Allocation},
 journal={Mathematics of Operations Research}, volume={37}, number={3}, pages={501--525},
 year={2012}, doi={10.1287/moor.1120.0548}
}
'''
    put('references.bib',bib)
    figure_names={'System_architecture':'Fig1','Forecast_accuracy':'Fig2','Control_waiting':'Fig3','Scenic_graph':'ESM_Fig1','Matched_horizon':'ESM_Fig2'}
    for name,number in figure_names.items():
        for suffix in ['pdf','svg','png','eps']:shutil.copy2(OUT/'figures'/f'{name}.{suffix}',DEST/'figures'/f'{number}.{suffix}')
    old=(BASE/'main.tex').read_text(encoding='utf-8');preamble=old.split(r'\begin{document}')[0]
    authors=r'\author*'+old.split(r'\author*',1)[1].split(r'\abstract',1)[0]
    title='Causal demand forecasting and transport-aware service allocation for IoT-enabled scenic areas'
    p=result['primary']['forecast'];pc=result['primary']['control'];roll=cm('rollout');mpc=cm('combined_mpc6')
    matched=result['exploratory_control']['combined_mpc12_nominal'];matched_s=result['exploratory_control']['combined_mpc12_surge']
    dev=json.loads((V4/'control_development/selection.json').read_text());confirm=json.loads((V4/'control_development/confirmation.json').read_text())
    control_sentence=(f"The rollout obtains {n(roll.wait_mean)} min/request under nominal demand, versus {n(mpc.wait_mean)} for six-step MPC with the identical forecast. "
        f"The paired difference is {n(pc['mean_difference'])} min/request (descriptive 95\\% interval [{n(pc['ci95'][0])}, {n(pc['ci95'][1])}]; Holm-adjusted $p={n(pc['p_holm'],3)}$). "
        f"It has lower waiting in {pc['wins']} of seven nominal worlds. "
        f"Static allocation and PPO-F4 obtain {n(cm('static').wait_mean)} and {n(cm('ppo_f4').wait_mean)} min/request. "
        f"Under increased demand, rollout and identical-predictor six-step MPC obtain {n(cm('rollout',True).wait_mean)} and {n(cm('combined_mpc6',True).wait_mean)} min/request. "
        f"Nominal movement is {n(roll.movement_mean)} resource-km/day for rollout and {n(mpc.movement_mean)} for six-step MPC; waiting performance must therefore be considered alongside relocation cost. "
        f"Under increased demand, overload frequency is {n(cm('rollout',True).overload_pct_mean)}\\% for rollout and {n(cm('combined_mpc6',True).overload_pct_mean)}\\% for six-step MPC. Lower total accrued waiting therefore does not imply uniformly lower node-level threshold exceedance.")
    matched_sentence=(f"On the prespecified three-day-per-world subset, rollout and 12-step MPC obtain {n(matched['mean_a'])} and {n(matched['mean_b'])} min/request under nominal demand, "
        f"and {n(matched_s['mean_a'])} and {n(matched_s['mean_b'])} under increased demand. "
        f"The nominal paired difference is {n(matched['mean_difference'])} min/request (95\\% interval [{n(matched['ci95'][0])}, {n(matched['ci95'][1])}]); "
        f"rollout has lower waiting in {matched['wins']} of seven worlds on this subset. These horizon-matched contrasts are exploratory.")
    audit_sentence=(f"Across {result['decision_epochs']:,} final decisions, {result['execution_infeasible']} executed actions violated the implemented resource checks. "
        f"Independent replay verified all {audit['full_flow_episodes_replayed']} flow-controller episodes, including FIFO waits, reservations, transit timing and terminal censoring. "
        f"The {audit['policy_episodes_aggregate_checked']:,} PPO episodes passed aggregate and raw-served-wait checks; their compact traces do not support a fresh independent action replay. "
        f"MPC used the hold fallback on {result['solver_fallback_steps']} decisions. Resource feasibility does not imply zero queue overload.")
    tr={r['method']:r for r in timing['results']}
    timing_sentence=(f"Serial uncached local computation on validation day 126 has median {n(tr['rollout']['median_ms'])} ms for rollout and {n(tr['combined_mpc6']['median_ms'])} ms for the same-predictor six-step MPC. "
        "Each method contributes 67 decisions after discarding five warm-up decisions. These measurements include forecast evaluation and executable-action construction, but exclude model loading, service simulation, physical communication and actuation. They are local CPU measurements, not field latency.")
    macros=dict(ForecastNominal=n(fm('combined').mae_mean,4),ForecastGWN=n(fm('graph_wavenet').mae_mean,4),ForecastGain=n(p['relative_reduction_pct']),
        ForecastDifference=n(p['mean_difference'],4),ForecastLow=n(p['ci95'][0],4),ForecastHigh=n(p['ci95'][1],4),ForecastP=n(p['p_holm'],3),
        ForecastSurge=n(fm('combined',True).mae_mean,4),ForecastGWNSurge=n(fm('graph_wavenet',True).mae_mean,4),
        DevelopmentWait=n(dev['selected']['development_wait']),ConfirmationWait=n(confirm['wait']),ControlInterpretation=control_sentence,
        MatchedInterpretation=matched_sentence,AuditInterpretation=audit_sentence,TimingInterpretation=timing_sentence,
        ConclusionControl=f"Transport rollout obtains {n(roll.wait_mean)} min/request, compared with {n(mpc.wait_mean)} for six-step MPC using the identical forecast.")
    put('results_macros.tex','\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    abstract=("IoT-enabled destinations require forecasts that support executable resource movements without overlooking service lost in transit. "
        "We evaluate a causal forecasting and transport-rollout framework in a disclosed synthetic scenic-area system with 24 nodes, four resource classes and FIFO queues. "
        "A training-derived daily profile and graph residual GRU are combined with Graph WaveNet. A marginal rollout selects transfers by predicted queue-cost reduction while preserving inventory, reservations and travel delays. "
        "Models and evaluation rules are frozen before generating seven new random worlds under nominal and increased demand. Eleven forecasting variants are compared alongside static allocation, two PPO variants, and matched-predictor model predictive control (MPC). "
        f"The proposed forecast combination reduces nominal mean absolute error from {n(fm('graph_wavenet').mae_mean,4)} to {n(fm('combined').mae_mean,4)} visit events. "
        f"Rollout obtains {n(roll.wait_mean)} min of accrued waiting per request, compared with {n(mpc.wait_mean)} for six-step MPC using the same forecast. "
        f"A matched 12-step comparison obtains {n(matched['mean_a'])} and {n(matched['mean_b'])} min/request on its prespecified subset. "
        "Waiting gains accompany higher nominal relocation distance and do not uniformly reduce overload. Terminal backlog, solver status and uncached computation are reported. The results concern independent realizations of one synthetic generator and graph, rather than field deployment or universal superiority.")
    assert len(abstract.split())<250,len(abstract.split())
    body=(ROOT/'scripts/manuscript_v4_body.tex').read_text(encoding='utf-8').replace('System_architecture.pdf','Fig1.pdf')
    text=preamble+r'\input{results_macros.tex}'+'\n'+r'\begin{document}'+'\n'+r'\title[Causal forecasting and service allocation]{'+title+'}\n'+authors+r'\abstract{'+abstract+'}\n'+r'\keywords{Internet of Things; spatiotemporal forecasting; resource allocation; model predictive control; transport delay; synthetic evaluation}'+'\n'+r'\maketitle'+'\n'+body+'\n'+r'\FloatBarrier\bibliography{references}\end{document}'+'\n'
    put('main.tex',text)
    names=['persistence','anchor','ridge','lstm','gru','edge_stgru','graph_wavenet','stid','itransformer','residual','combined']
    put('tables/forecast_main.tex',table('Forecasting across seven new worlds. Values are world-mean errors over all four leads and all nodes; smaller is better. Every standalone neural family averages three training seeds; the combination uses both three-seed components.','tab:forecast',
        ['Predictor','Nominal MAE','Nominal RMSE','Stress MAE','Stress RMSE'],[[FN[k],n(fm(k).mae_mean,3),n(fm(k).rmse_mean,3),n(fm(k,True).mae_mean,3),n(fm(k,True).rmse_mean,3)] for k in names],spec='p{42mm}YYYY'))
    namesc=['static','ppo_f0','ppo_f4','gwn_mpc6','combined_mpc6','rollout']
    put('tables/control_main.tex',table('Full nine-day-per-world control comparison. Waiting is accrued min/request, including terminal censoring. Unserved and overload percentages refer to nominal demand. Entries are means of seven world-level summaries.','tab:control',
        ['Controller','Nominal wait','Stress wait','Unserved (\%)','Overload (\%)'],[[CN[k],n(cm(k).wait_mean),n(cm(k,True).wait_mean),n(cm(k).unserved_pct_mean),n(cm(k).overload_pct_mean)] for k in namesc],spec='p{43mm}YYYY'))
    put('tables/control_matched.tex',table(r'Horizon comparison on the same 21 episodes per condition (test offsets 0, 12 and 24 in each of seven worlds). Values are mean accrued min/request. These values must not be compared as if they came from the nine-day matrix in Table~\ref{tab:control}.','tab:matched',
        ['Controller','Horizon','Nominal wait','Stress wait'],[[CN[k],h,n(cm(k,subset='matched12').wait_mean),n(cm(k,True,'matched12').wait_mean)] for k,h in [('combined_mpc6',6),('combined_mpc12',12),('rollout',12)]],spec='p{55mm}YYY'))
    put('tables/development.tex',table('Control search on the three predeclared development days. All candidates use the same combined predictor and physical constraints. Selection precedes all new test worlds.','tab:development',
        ['Horizon','Terminal weight','Accrued min/request'],[[r['horizon'],r['terminal_weight'],n(r['development_wait'])] for r in dev['candidates']],spec='YYY'))
    put('tables/timing.tex',table('Serial uncached local computation: 67 measured decisions per method on one validation day. All methods use the identical six-network forecast combination.','tab:timing',
        ['Method','Median (ms)','95th percentile (ms)'],[[CN[r['method']],n(r['median_ms']),n(r['p95_ms'])] for r in timing['results']],spec='LYY'))
    put('figures_forecast.tex',figure('Fig2','Prediction error by horizon. Curves show means across seven independent random worlds; these are three-seed prediction ensembles, not averages of separate seed metrics. Per-world errors and standard deviations are supplied in Supplementary Material','fig:forecast'))
    put('figures_control.tex',figure('Fig3','Accrued waiting in the full nine-day matrix. Bars show seven-world means and whiskers show sample standard deviations across worlds, not confidence intervals. PPO training-seed outcomes are averaged within each world','fig:waiting'))
    supplemental(preamble,authors,f,c,result,timing,dev,names)
    put('cover_letter.tex',r'''\documentclass[12pt]{article}
\usepackage[a4paper,margin=25mm]{geometry}
\begin{document}
Dear Editors of Discover Internet of Things,

Please consider our manuscript, ``'''+title+r'''", as a Research article. It studies how causal demand forecasts can inform executable service allocation when resource transport removes service capacity temporarily. The work provides an explicit synthetic IoT observation interface, seasonal-residual forecast combination, transport-aware rollout, and matched-predictor and matched-horizon MPC comparisons.

The study uses newly generated synthetic demand, not field sensor records. Its implementation, selected checkpoints, raw predictions, dispatch traces and independent replay checks accompany the submission package. Seven new random worlds evaluate frozen methods under nominal and increased demand. The manuscript distinguishes resource feasibility from queue performance and limits its conclusions to the stated simulation and computational budgets.

The work is relevant to the journal's interest in connecting IoT observations with operational decisions. The author must verify originality, exclusive consideration, authorship, funding, competing interests and data-access statements before this letter is submitted.

Sincerely,

Jianbo Guo
\end{document}
''')
    put('README.txt','Compile main.tex and supplement.tex with pdfLaTeX + BibTeX (or Tectonic). Include all figures, tables, references.bib, sn-jnl.cls and sn-mathphys-num.bst.\nAuthor declarations and public data/code access require final author confirmation before submission.\n')
    write_json(OUT/'manuscript_build.json',dict(abstract_words=len(abstract.split()),results_sha256=digest(OUT/'results.json'),audit_sha256=digest(V4/'INDEPENDENT_AUDIT.json'),timing_sha256=digest(V4/'serial_timing/summary.json'),main_sha256=digest(DEST/'main.tex')))

def supplemental(preamble,authors,f,c,result,timing,dev,names):
    parts=[preamble,r'\begin{document}',r'\title{Supplementary Material: Causal demand forecasting and transport-aware service allocation}',authors,
        r'\abstract{This supplement specifies the synthetic generator, model development, held-out evaluation matrix, paired statistics and computational diagnostics. It accompanies the new frozen v4 evaluation; previous exploratory results remain in their separate archived v2/v3 records.}',r'\maketitle',
        r'\section{Generator and physical configuration}',
        'All parameters below specify a synthetic process. No real observations were converted into generated data. The same demand realization is replayed for every controller within a world and condition.']
    cfg=json.loads((ROOT/'runs/scenic_rebuild_v2/config.json').read_text())['scenic']
    settings=[['Time grid','72 intervals/day; 10 min/interval; prehistory -14 to 0'],['Graph','24 nodes; 16 service locations, 4 hubs, 4 gates; 31 undirected links'],
        ['Training / validation','126 / 27 original days; old test days not used for the new final test'],['Gate baseline / peak','(5,6,5,6) /(18,21,19,22)'],['Peak times / widths','(20,47) /(8,10); afternoon ratio 0.8'],
        ['Gate phase offsets','(-4,0,4,8) intervals'],['Propagation','0.68 times previous intensity multiplied by row-normalized adjacency'],['Lognormal noise','Intensity log SD 0.08; daily log SD 0.12'],
        ['Weather multipliers','(0.65,0.9,1.0,1.1), probabilities (0.15,0.20,0.45,0.20)'],['Weekend / holiday','Multipliers 1.20 /1.35'],['Holiday indices',', '.join(map(str,cfg['holiday_days']))],
        ['Localized event','Probability 0.30; amplitude 18; Gaussian width 7'],['Resource totals','(30,24,16,12); each unit serves 2 requests/interval'],['Holding limits','(5,4,3,3) per service node; zero at hubs/gates'],
        ['Class probabilities','(0.75,0.55,0.40,0.30) independent conditional on visit count'],['Transport','Speeds (80,55,40,30) m/min; setup (2,4,6,8) min'],['Planning penalties','Queue scale 25; resource-km 0.025; overload 0.02; queue threshold 50']]
    parts.append(table('Synthetic configuration; exact generator and configuration JSON are included in the reproducibility package.','s:generator',['Setting','Value'],settings,spec='p{44mm}L'))
    parts.append('Figure~\\ref{s:graph} shows the fixed physical topology used by every method and random world. Service labels S1--S16 map to indices 0--15, H1--H4 to 16--19, and G1--G4 to 20--23 in the numeric archive. Exact edge lengths and shortest paths are included with the generator.')
    parts.append(figure('ESM_Fig1','Physical topology of the implemented simulator: 16 service locations, four hubs and four gates linked by 31 undirected edges. The outer link closes the hub cycle; drawing geometry is schematic, while resource travel uses the recorded 90--320 m edge lengths','s:graph'))
    parts += [r'\FloatBarrier\section{Forecast development and comparator implementation}',
        'The seven new forecast candidates use development days 126--143. The selected residual family is then fitted with seeds 0, 1, 2. A separate five-weight grid selects its mixture with the frozen GWN ensemble. Validation diagnostics are not independent final evidence.']
    search=json.loads((V4/'forecast/candidate_results.json').read_text())
    parts.append(table('All new forecast-family candidates; neural scores are seed-zero development MAE.','s:search',['Candidate','Description','MAE'],[[r['id'].replace('_',r'\_'),r['kind']+(' / '+r.get('loss','').replace('_',' ') if r.get('loss') else ''),n(r['development_mae'],5)] for r in search],spec='p{28mm}LY'))
    combo=json.loads((V4/'forecast/combination_selection.json').read_text())
    parts.append(table('Combination development scores, using three-seed means for each component. A weight of one retains only the residual component.','s:mixture',['Residual weight','Development MAE'],[[n(r['residual_weight'],2),n(r['development_mae'],5)] for r in combo['candidates']],spec='YY'))
    specs=json.loads((ROOT/'runs/supplement_v3/forecast/selected_models.json').read_text());rows=[]
    for name in ['lstm','gru','edge_stgru','graph_wavenet','stid','itransformer']:
        ss=[r for r in specs if r['model']==name];r=ss[0]
        rows.append([FN[name],r['candidate']['hidden'],r['candidate']['learning_rate'],r['parameters'],', '.join(str(x['best_epoch']) for x in ss)])
    parts.append(table('Frozen comparator settings. Epochs correspond to seeds 0, 1, 2. Parameters are per single model; deployment averages three models.','s:baseline',['Model','Width','Learning rate','Parameters','Best epochs'],rows,spec='p{36mm}YYYY'))
    parts += ['GWN~\\citep{wu2019} has four blocks with two gated dilated temporal layers per block, physical and learned adaptive supports, order-two diffusion, residual/skip paths and dropout 0.1. The 12-frame input is left-padded to its receptive field of 13. STID~\\citep{shao2022} uses a shared history embedding, node and time identities, and three residual MLP blocks with dropout 0.15. iTransformer~\\citep{liu2024} uses two encoder layers and four heads. Each receives the common inflow/time input; standardized unmasked MSE includes valid zeros. Pinned author source versions, licenses and implementation adaptations are retained in the source manifest. These are adaptations rather than claims to reproduce published benchmark scores.',
        'The new residual model has 22,980 parameters per seed, compared with 96,132 for GWN. The six-network combination has 357,336 neural parameters. The residual model uses raw-count SmoothL1 loss with beta 1.0. Its selected checkpoints occur at epochs 71, 48, 63 for seeds 0, 1, 2. The two MLP candidates and ridge/anchor comparisons remain in development records. Float32 GPU and CPU inference can differ slightly; a deployment-adapter check found a maximum combined prediction difference below 0.001 visit events relative to saved GPU development output.']
    parts += [r'\FloatBarrier\section{Forecast results by lead time}']
    for surge in [False,True]:
        g=f[f.surge==surge].set_index('model')
        parts.append(table(('Increased-demand' if surge else 'Nominal')+' forecast MAE by lead time; entries are mean (sample SD) across seven worlds.','s:forecast'+str(int(surge)),['Model','10 min','20 min','40 min','60 min'],
            [[FN[k]]+[n(g.loc[k,f'mae_h{h}_mean'],3)+' ('+n(g.loc[k,f'mae_h{h}_std'],3)+')' for h in [1,2,4,6]] for k in names],spec='p{38mm}YYYY'))
    fn=f[f.surge==False].set_index('model');fs=f[f.surge==True].set_index('model')
    parts.append(table('Weighted absolute percentage error (WAPE), using the sum of absolute errors divided by the sum of observed counts within each world. Entries are world-mean percentages (sample SD); valid zero targets are retained.','s:wape',['Model','Nominal WAPE (\%)','Stress WAPE (\%)'],
        [[FN[k],n(fn.loc[k,'wape_mean'])+' ('+n(fn.loc[k,'wape_std'])+')',n(fs.loc[k,'wape_mean'])+' ('+n(fs.loc[k,'wape_std'])+')'] for k in names],spec='LYY'))
    parts += [r'\FloatBarrier\section{Service performance and paired statistics}']
    for surge in [False,True]:
        g=c[(c.surge==surge)&(c.subset=='full')]
        parts.append(table(('Increased-demand' if surge else 'Nominal')+' service diagnostics in the full matrix. Movement is mean resource-km/day. Percentages are world-level means.','s:service'+str(int(surge)),['Method','Wait mean (SD)','Unserved (\%)','Overload (\%)','Movement'],
            [[CN[r.method],n(r.wait_mean)+' ('+n(r.wait_std)+')',n(r.unserved_pct_mean),n(r.overload_pct_mean),n(r.movement_mean)] for r in g.itertuples()],spec='p{38mm}YYYY'))
    parts.append('Figure~\\ref{s:paired} compares rollout with both planning horizons on the identical three-day subset in every world.')
    parts.append(figure('ESM_Fig2','Paired world-level waiting differences on the common three-day subset. Negative values favor rollout. Lines connect world identifiers for readability and do not imply a temporal trend','s:paired'))
    primary=result['primary']
    parts.append(table('Predeclared primary contrasts. Differences are proposed minus comparator, so negative favors the proposed method. Intervals are descriptive seven-world Student-t intervals. Exact sign-flip p-values assume sign symmetry; Holm correction covers both contrasts.','s:primary',['Endpoint','Difference','95\% interval','Raw p','Holm p'],
        [[k,n(v['mean_difference'],4),'['+n(v['ci95'][0],4)+', '+n(v['ci95'][1],4)+']',n(v['p_exact_two_sided'],4),n(v['p_holm'],4)] for k,v in primary.items()],spec='p{25mm}YYYY'))
    parts += [r'\FloatBarrier\section{Solver returns and execution audit}']
    rows=[]
    for method in ['gwn_mpc6','combined_mpc6','combined_mpc12']:
        logs=[]
        for p in (V4/'final_control').glob(f'*/{method}/day*/solver.jsonl'):logs.extend(json.loads(x) for x in p.read_text().splitlines())
        gaps=[r['mip_gap'] for r in logs if r.get('mip_gap') is not None]
        rows.append([CN[method],len(logs),sum(r['status']==1 for r in logs),sum(r['fallback'] for r in logs),n(np.median(gaps),4),n(np.max(gaps),4)])
    parts.append(table('MPC diagnostics across both conditions. Status 1 denotes the solver time/iteration-limit return. A time-limited feasible incumbent can still be executed. Gap is relative; all per-decision residuals and bounds remain in the raw logs.','s:solver',['Controller','Calls','Status 1','Fallback','Median gap','Max gap'],rows,spec='p{40mm}YYYYY'))
    parts += ['Independent replay reconstructs resource inventory, reservations and travel arrivals from executed flows without calling the controller or the service simulator step. It checks integer availability, holding limits, service timing, all FIFO waits, movement costs, reward, overload and closing-time queue cohorts. Forecast target indices and errors are separately recomputed from saved numeric datasets. The maximum float32-to-float64 horizon-reduction discrepancy was below 0.000035; audit tolerance for that reduction is 0.00005. PPO compact traces support aggregate checks and raw-served-wait checks; its unchanged original engine retains the earlier full-trace audits.',
        r'\section{Computation and reproduction}',r'\input{tables/timing.tex}',
        'The complete evaluation has 546 flow-controller episodes and 2520 PPO episodes, totaling 220,752 decisions. Control jobs run on CPU with one solver thread. Serial latency is collected only after those workers finish, on physical validation day 126, using fresh uncached forecasting calls; five decisions are discarded and 67 are retained per method. The simulator and hardware do not establish an IoT network latency guarantee.',
        'The package README describes environment setup and the dependency order: train/develop, freeze, evaluate forecasts and control, independently audit, analyze, time serially, draw figures, and compile the manuscript. Frozen outputs are protected from overwrite. Re-evaluation on archived test worlds is verification rather than a new independent test. Source hashes identify the exact implementation. No unavailable original code is required.',r'\FloatBarrier\bibliography{references}\end{document}']
    put('supplement.tex','\n\n'.join(parts)+'\n')

if __name__=='__main__':main()
