"""Build the extended Springer LaTeX article from complete audited experiment outputs.

This is a reporting script: it never trains, selects, or alters experimental data.
"""
from pathlib import Path
import json
import re
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/revision/Discover_IoT_LaTeX'
OUT = ROOT / 'output/supplement_v3'
DEST = OUT / 'Discover_IoT_LaTeX'
RUN = ROOT / 'runs/supplement_v3'


def between(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def n(value, digits=2):
    return f'{float(value):.{digits}f}'


def pvalue(value):
    return r'$<0.001$' if float(value) < .001 else n(value, 3)


def table(caption, label, headers, rows, spec=None):
    spec = spec or 'L' + 'Y' * (len(headers)-1)
    lines = [r'\begin{table}[!htbp]', r'\caption{' + caption + r'}\label{' + label + '}',
             r'\centering\normalsize', r'\setlength{\tabcolsep}{3pt}',
             r'\renewcommand{\arraystretch}{1.15}',
             r'\begin{tabularx}{\textwidth}{@{}' + spec + '@{}}', r'\toprule',
             ' & '.join(headers) + r' \\', r'\midrule']
    lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabularx}', r'\end{table}', '']
    return '\n'.join(lines)


def figure(name, caption, label, width=r'\textwidth', placement='!htbp'):
    return '\n'.join([r'\begin{figure}[' + placement + ']', r'\centering',
                      r'\includegraphics[width=' + width + ']{' + name + '.pdf}',
                      r'\caption{' + caption + r'}\label{' + label + '}',
                      r'\end{figure}', ''])


def main():
    # Missing inputs are fatal: an unfinished matrix cannot generate a final paper.
    result = json.loads((OUT/'results/results.json').read_text())
    assert result['policy_episodes'] == 15120 and result['mpc_episodes'] == 189
    forecast_raw = pd.read_csv(RUN/'forecast/metrics.csv')
    forecast = forecast_raw[forecast_raw.dataset == 'independent_pooled']
    assert len(forecast.model.unique()) == 11
    lat = pd.read_csv(RUN/'serial_latency/summary.csv')
    old = (BASE/'main.tex').read_text(encoding='utf-8')
    preamble = old.split(r'\begin{document}', 1)[0]
    preamble = preamble.replace('% Numerical results and scientific content are unchanged from the audited DOCX.',
                                '% Extended study: tables and reported numbers are generated from archived run outputs.')
    preamble += r'\setlength{\bibsep}{5pt}'+'\n'
    authors = between(old, r'\author', r'\abstract')
    authors = r'\author' + authors
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'figures').mkdir(exist_ok=True)
    for name in ['sn-jnl.cls', 'sn-mathphys-num.bst', 'references.bib']:
        shutil.copy2(BASE/name, DEST/name)
    for src in list((BASE/'figures').glob('*.pdf')) + list((OUT/'figures').glob('*.pdf')):
        shutil.copy2(src, DEST/'figures'/src.name)
    shutil.copy2(OUT/'supplement_methods.tex', DEST/'supplement_methods.tex')

    def policy(predictor, method, metric='wait', regime='nominal', budget=43200):
        return next(r for r in result['policy_summaries'] if r['predictor'] == predictor and r['method'] == method
                    and r['metric'] == metric and r['regime'] == regime and r['budget_steps'] == budget)
    def fixed(method, metric='wait', regime='nominal'):
        return next(r[metric] for r in result['deterministic_summaries'] if r['method'] == method and r['regime'] == regime)
    def fm(model, horizon=6, metric='mae'):
        return forecast[(forecast.model == model) & (forecast.horizon_steps == horizon)][metric].mean()
    def pm(pred, method, metric='wait', regime='nominal'):
        row = policy(pred, method, metric, regime)
        return n(row['mean']) + r' $\pm$ ' + n(row['sd'])
    primary = result['contrasts'][0]
    edge_wait = policy('edge_stgru', 'F4')['mean']
    simple_wait = fixed('static')
    mpc_wait = fixed('forecast_flow_mpc')
    significant = primary['p_value'] < .05
    superiority = significant and primary['mean'] < 0
    primary_sentence = ('Resource-specific attention reduces accrued waiting relative to concatenation' if superiority
                        else 'Resource-specific attention increases accrued waiting relative to concatenation' if significant and primary['mean'] > 0
                        else 'Resource-specific attention does not demonstrate lower accrued waiting than concatenation')
    abstract = (
        'IoT-informed resource allocation must translate uncertain forecasts into movements that preserve service availability. '
        'We evaluate a modular sensing-to-action framework in a disclosed scenic-area simulator with 24 nodes, four resource classes, '
        'transport delays and finite operating days. A causal telemetry interface supplies multi-horizon forecasts to proximal policy optimization '
        '(PPO), followed by deterministic allocation repair. The extended study compares eleven forecasting variants, trains 50 additional policies '
        'with ten seeds per condition, and evaluates three new generator realizations under nominal and increased demand. A six-step mixed-integer '
        'flow controller provides a transport-aware comparator. At 60 minutes, edge-STGRU and ridge regression achieve mean absolute errors of '
        + n(fm('edge_stgru')) + ' and ' + n(fm('ridge')) + ' visit events. '
        + primary_sentence + ': the paired difference is ' + n(primary['mean'])
        + r' min/request (95\% confidence interval ' + n(primary['ci_low']) + ' to ' + n(primary['ci_high']) + '). '
        'Resource-specific PPO, static allocation and flow MPC obtain ' + n(edge_wait) + ', ' + n(simple_wait) + ' and ' + n(mpc_wait)
        + ' min/request, respectively. Terminal backlog, execution feasibility, solver status and uncached computation times are reported alongside '
        'prediction errors. These synthetic results identify the operational limits of forecast fusion and distinguish feasible dispatch from improved service; '
        'they do not establish effectiveness at a real destination.')

    text = preamble + r'\begin{document}' + '\n'
    text += r'\title[Forecasts and transport-aware IoT allocation]{Forecast-informed service allocation for IoT-enabled scenic areas: a reproducible evaluation with transport constraints}' + '\n'
    text += authors + r'\abstract{' + abstract + '}\n'
    text += r'\keywords{Internet of Things; spatiotemporal forecasting; resource allocation; proximal policy optimization; model predictive control; synthetic evaluation}' + '\n'
    text += r'\maketitle' + '\n'
    # The framing file contains complete Introduction and Discussion sections.
    framing = (OUT/'framing_additions.tex').read_text(encoding='utf-8')
    introduction = between(framing, '% BEGIN INTRODUCTION', '% END INTRODUCTION')
    discussion = between(framing, '% BEGIN DISCUSSION', '% END DISCUSSION')
    text += introduction
    text += r'\section{Related work}' + between(old, r'\section{Related work}', r'\section{System model}')
    system = between(old, r'\section{System model}', r'\section{Forecasting and allocation methods}')
    system = system.split('Figure~\\ref{fig:1}', 1)[0]
    generator_detail = 'Here d identifies' + between(system, 'Here d identifies', 'At each service node')
    system = system.replace(generator_detail, r'''Here $d$ identifies a day, $i$ a node, $e$ external intensity and $\eta$ a mean-one lognormal perturbation. External intensities combine morning and afternoon gate peaks, weather and calendar factors, and localized events. The full distributions, numerical settings and fixed graph are supplied in Supplementary Material and the machine-readable configuration.

''')
    system = system.replace('FIFO queues evolve as','First-in-first-out (FIFO) queues evolve as')
    text += r'\section{System model and architecture}' + system
    text += r'''Figure~\ref{fig:system_architecture} separates offline development from the online decision loop. Time-ordered training and validation data support fitting and checkpoint selection. At each 10-min decision, the forecasting branch uses reconstructed inflow history and the clock; the policy also receives queue histories, availability masks, observation ages and the trusted resource ledger. Forecast fusion and allocation repair are distinct operations. Resource movements alter future service availability and queues, while exogenous requests remain fixed across controllers. The figure depicts the implemented synthetic experiment, without implying deployment of physical sensor devices.

'''
    text += figure('System_architecture', 'System architecture of the fully synthetic evaluation: (a) offline fitting and checkpoint selection, (b) causal sensing and forecast-informed decisions, and (c) feasible allocation, transport and FIFO service. The policy receives telemetry histories and the trusted ledger independently of forecast fusion. Count-report faults do not affect the ledger.', 'fig:system_architecture', '165mm', 'p')
    core = between(old, r'\section{Forecasting and allocation methods}', r'\section{Experimental protocol}')
    a = core.index('Prediction comparators are persistence')
    b = core.index('The policy state combines', a)
    frozen_predictor_methods = core[a:b]
    core = core[:a] + r'''The original 32-unit edge-STGRU seed-zero model and validation-selected ridge predictor are frozen inputs to the control experiment. All normalization and fallback medians use original training days; queue-channel statistics come from static-allocation training trajectories. The expanded forecasting benchmark refits its models separately, using validation selection described below; these benchmark checkpoints do not replace the controller's predictor. Edge-STGRU is the explicitly implemented edge-aware GRU, rather than an exact reproduction of the earlier E-STGNN architecture whose executable implementation is unavailable.

''' + core[b:]
    core = core.replace('Five alternatives isolate fusion:', 'Five fusion alternatives were evaluated in the formative study; the extended study focuses on F0, F1 and F4:')
    text += r'\section{Forecasting and allocation methods}' + core
    additions = (OUT/'method_additions.tex').read_text(encoding='utf-8')
    mpc = between(additions, r'\subsection{Transport-aware model-based comparator}', r'\subsection{Aggregation and uncertainty}')
    text += r'\subsection{Transport-aware model-based comparator}' + mpc
    text += r'''A separately labelled perfect-information diagnostic supplies exact future request paths to the same bounded-horizon solver on the 81 new nominal days. It is excluded from deployable-method rankings and is not an optimal lower bound: its horizon, time limit and planning objective remain restricted. Full flow equations and solver execution semantics appear in Supplementary Material.

\section{Experimental protocol}
'''
    sequential = between(additions, r'\subsection{Sequential evaluation and new synthetic realizations}', r'\subsection{Transport-aware model-based comparator}')
    text += r'\subsection{Sequential evaluation and new synthetic realizations}' + sequential
    text += r'''The common forecasting search uses three candidates: hidden width/learning rate 32/0.002, 64/0.001 and 32/0.0005. Candidate selection uses seed zero and validation MAE, followed by seeds one and two for the selected configuration. All neural models use unmasked standardized MSE, AdamW, weight decay $10^{-4}$, batch size 128, a 100-epoch ceiling and patience 12. Deterministic predictors and ridge use training-only fitting. Graph WaveNet retains gated dilated temporal convolutions, diffusion and learned adjacency; STID uses node and within-day identities; iTransformer attends across variate tokens. Input and four-lead output adaptations, selected configurations, parameter counts and source commits are disclosed in Supplementary Material. A separate matched ablation fixes the selected edge-STGRU configuration for its uniform-neighbor and no-edge-length variants, avoiding a change in hyperparameters within that mechanism comparison.

PPO uses 720-step rollouts, four update epochs, minibatches of 120, two 96-unit hidden layers, learning rate $3\times10^{-4}$, $\gamma=0.99$, GAE $\lambda=0.95$, clipping 0.2, entropy coefficient 0.001, value coefficient 0.5 and gradient limit 0.5. Training reward is multiplied by 0.01. A minibatch approximate KL exceeding 0.03 stops remaining updates for that rollout. Raw Gaussian scores define PPO likelihood ratios; integer repairs are not treated as samples from a differentiable action distribution. Shared-backbone initialization is paired by seed, while fusion parameter counts can differ.

Static, reactive and forecast-lookahead rules provide transparent references. Static holds the initial uniform allocation. The other rules form targets from observed queue plus expected requests, divided by per-resource capacity and offset by 0.1; reactive expectations use the latest inflow, whereas lookahead uses the forecast lead nearest each class's representative travel time. These three rules and PPO share deterministic nearest-surplus dispatch. The lookahead rule performs no multi-step optimization.

\subsection{Endpoints and statistical analysis}
'''
    old_protocol = between(old, r'\section{Experimental protocol}', r'\section{Results}')
    endpoint = old_protocol[old_protocol.index('The primary endpoint'):old_protocol.index('Outcomes are pooled')]
    text += endpoint
    text += between(additions, r'\subsection{Aggregation and uncertainty}', r'\subsection{Forecast memoization and local computation time}')
    text += r'''Two-sided paired signed-rank tests enumerate all signs, use average ranks for ties and remove exact-zero differences. Pointwise 95\% Student $t$ intervals use the ten paired seed differences and nine degrees of freedom; they are not multiplicity-adjusted. The significance threshold is 0.05. Forecast mean absolute error (MAE), root mean squared error (RMSE) and weighted absolute percentage error (WAPE) use every valid target, including zeros, and are calculated from saved predictions. WAPE is 100 times the sum of absolute errors divided by the sum of absolute targets. Neural prediction summaries average three fitted seeds; these seeds do not represent independent datasets.

\subsection{Implementation and computation}
Repeated performance runs cache frozen predictions by their complete causal input and timestamp; all numerical policy inputs are unchanged. Cached-run timing fields are excluded. Dedicated serial timing disables caching and uses original validation-day offsets 0, 13 and 26. Five initial decisions per day are discarded, leaving 201 measurements per method. Timed stages are telemetry-window/state construction, required forecasting, controller computation and allocation repair. F0 skips its unused diagnostic forecast. Model loading, physical communication, environment service simulation and actuation are excluded. The measurements describe a warmed local CPU process, not deployment or cold-start latency.

The experiments use Windows, 15.73 GiB RAM, an Intel64 Family 6 Model 186 processor and an NVIDIA GeForce RTX 4060 Laptop GPU. Python 3.11.5, PyTorch 2.7.1/CUDA 11.8, NumPy 1.26.4, SciPy 1.11.4, pandas 2.2.3 and Matplotlib 3.10.7 are recorded with source and checkpoint hashes. Forecast fitting uses the GPU; policy simulations and serial timing use one CPU thread per worker. Policy evaluation totals 15,120 daily episodes: 9,450 for the 50 new policies and 5,670 for retained initial-budget policies, with legacy scenarios separately labelled. Heuristics add 567 episodes, forecast MPC 189 and the perfect-information diagnostic 81.

\section{Results}
\subsection{Prediction accuracy and matched ablations}
'''
    names = [('persistence','Persistence'),('historical_average','Historical mean'),('ridge','Ridge'),('lstm','LSTM'),
             ('gru','GRU'),('edge_stgru','Edge-STGRU'),('graph_wavenet','Graph WaveNet'),('stid','STID'),('itransformer','iTransformer')]
    rows = [[label]+[n(fm(model,h)) for h in [1,2,4,6]]+[n(fm(model,6,'rmse'))] for model,label in names]
    text += table('Forecast errors on the three new nominal realizations (5,427 common origins). Neural entries average three prespecified training seeds of each validation-selected configuration. Units are visit events; complete per-seed metrics and ablations are supplied in Supplementary Material.', 'tab:forecast',
                  ['Model',r'10-min\newline MAE',r'20-min\newline MAE',r'40-min\newline MAE',r'60-min\newline MAE',r'60-min\newline RMSE'],rows)
    ranked = sorted(names, key=lambda item: fm(item[0]))
    text += ('At 60 minutes, '+ranked[0][1]+' has the lowest mean MAE among the nine full-model baselines in Table~\\ref{tab:forecast}, at '
             +n(fm(ranked[0][0]))+'. Edge-STGRU obtains '+n(fm('edge_stgru'))+', compared with '+n(fm('ridge'))+' for ridge and '
             +n(fm('graph_wavenet'))+' for Graph WaveNet. Figure~\\ref{fig:forecast} shows the full horizon profile. '
             'Forecast accuracy is assessed separately from allocation quality; the expanded forecasting winners were not selected retrospectively as controller inputs.\n\n')
    # Matched ablation text/table is generated separately after its exact schema is audited.
    text += (OUT/'matched_ablation_results.tex').read_text(encoding='utf-8')
    text += figure('Supplement_forecast','Forecast error across 10, 20, 40 and 60 minutes on the three new nominal realizations. Six representative methods are shown. Neural curves average three prespecified seeds of each validation-selected configuration; ridge is deterministic. All methods use the same targets.', 'fig:forecast')
    text += r'\subsection{Allocation quality under nominal and increased demand}'+'\n'
    text += r'Table~\ref{tab:control} and Figure~\ref{fig:control} compare accrued waiting and terminal service completion across the tested controllers.'+'\n\n'
    control_methods = [('static','Static',None),('reactive','Reactive rule',None),('lookahead','Forecast rule',None),
                       ('F0','F0 / no forecast','edge_stgru'),('F1','F1 / Edge','edge_stgru'),('F4','F4 / Edge','edge_stgru'),
                       ('F1','F1 / Ridge','ridge'),('F4','F4 / Ridge','ridge'),('forecast_flow_mpc','Flow MPC',None)]
    rows=[]
    for method,label,pred in control_methods:
        vals=[pm(pred,method,metric,regime) if pred else n(fixed(method,metric,regime))
              for regime in ['nominal','surge'] for metric in ['wait','unserved_pct']]
        rows.append([label]+vals)
    rows.append(['Perfect-information MPC']+[n(fixed('perfect_information_flow_mpc',m)) for m in ['wait','unserved_pct']]+['--','--'])
    text += table('Operational results on 81 new nominal days and 81 intensity-stress days. Learned methods use 43,200 training steps and show mean $\\pm$ sample SD across ten policies; deterministic methods show equally weighted realization means. Perfect information is a non-deployable diagnostic.', 'tab:control',
                  ['Method',r'Nominal wait\newline min/request',r'Nominal\newline unserved \%',r'Stress wait\newline min/request',r'Stress\newline unserved \%'],rows)
    text += (primary_sentence+' (Table~\\ref{tab:contrasts}). Its mean accrued waiting is '+n(edge_wait)+' min/request, against '
             +n(policy('edge_stgru','F1')['mean'])+' for concatenation. Static allocation and flow MPC obtain '+n(simple_wait)+' and '+n(mpc_wait)
             +' min/request. These comparisons evaluate full dispatch mechanisms; MPC can choose origin--destination matches directly, whereas PPO dispatches its targets through nearest-surplus matching.\n\n')
    text += ('Under the intensity stress, F4/Edge obtains '+n(policy('edge_stgru','F4',regime='surge')['mean'])+' min/request, with '
             +n(policy('edge_stgru','F4','unserved_pct','surge')['mean'])+r'\% of requests unserved at closing. The corresponding static and MPC waits are '
             +n(fixed('static',regime='surge'))+' and '+n(fixed('forecast_flow_mpc',regime='surge'))
             +'. Stress changes the specified generation intensities by 60\\%, while resource and service capacities remain fixed.\n\n')
    text += figure('Supplement_control','Accrued waiting under nominal and increased intensity. Small points represent ten independently trained policies; diamonds and bars denote their mean and sample SD. Squares denote deterministic-controller realization means. Perfect-information MPC is evaluated only under nominal demand and is a diagnostic, not a deployable method or optimal bound.', 'fig:control')
    text += r'\subsection{Paired effects and training-budget sensitivity}'+'\n'
    rows=[]
    labels=['Edge F4--F1','Edge F4--F0','Ridge F4--F1','Ridge F4--Edge F4','Long--initial F4','Edge F4--F1 (stress)']
    for c,label in zip(result['contrasts'],labels):
        rows.append([label,n(c['mean']),n(c['ci_low'])+' to '+n(c['ci_high']),pvalue(c['p_value']),
                     pvalue(c['p_holm']) if c['p_holm'] is not None else ('Primary' if c['family']=='primary' else 'Single')])
    text += table('Paired differences in accrued waiting (min/request). Negative differences favor the first named condition. Intervals are pointwise and condition on the three generated worlds. Four secondary nominal tests use Holm adjustment; the primary and single stress tests remain separate.', 'tab:contrasts',
                  ['Contrast','Difference',r'95\% $t$ interval','Exact $p$','Adjusted $p$'],rows)
    cross=result['crossed_bootstrap']; budget=result['contrasts'][4]
    text += ('The primary paired difference is '+n(primary['mean'])+' min/request (95\\% interval '+n(primary['ci_low'])+' to '+n(primary['ci_high'])
             +'; exact paired $p='+n(primary['p_value'],3)+'$). The exploratory crossed-bootstrap interval is '+n(cross['ci_low'])+' to '+n(cross['ci_high'])
             +' min/request. With only three generator realizations, neither interval establishes cross-site generalization.\n\n')
    text += ('Increasing F4 training from 14,400 to 43,200 steps changes accrued waiting by '+n(budget['mean'])+' min/request (95\\% interval '
             +n(budget['ci_low'])+' to '+n(budget['ci_high'])+'; Holm-adjusted $p='+n(budget['p_holm'],3)+'$). '
             'Figure~\\ref{fig:learning} reports all validation checkpoints and paired budget outcomes. Checkpoints are selected by validation reward, so a longer budget is not a declaration of convergence. '
             'The original five-fusion comparison and sensing/travel sensitivity experiments remain explicitly formative results in Supplementary Material.\n\n')
    text += figure('Supplement_learning','Training-budget assessment. (a) Validation reward trajectories show mean and sample SD across ten seeds using the frozen Edge predictor. (b) Connected points compare the same F4 training seeds at the initial and extended budgets on the three new nominal realizations. Validation selects checkpoints; test outcomes do not.', 'fig:learning')
    text += r'\subsection{Feasibility, solver diagnostics and local computation}'+'\n'
    text += r'Figure~\ref{fig:latency} summarizes local computation time; detailed feasibility and solver records are supplied in Supplementary Material.'+'\n\n'
    text += (OUT/'solver_and_latency_results.tex').read_text(encoding='utf-8')
    text += figure('Supplement_latency','Serial uncached local computation on three validation days, with 201 measured decisions per method after warm-up. Filled and open points show median and 95th percentile. Measurements include the required forecast and executable-action construction, and exclude model loading, physical communication, service simulation and actuation.', 'fig:latency')
    text += discussion
    text += r'\section{Conclusion}'+'\n'
    text += ('This study provides an executable, auditable evaluation of forecasts, resource-specific fusion and transport-aware service allocation in a synthetic IoT setting. '
             +primary_sentence+', with a paired difference of '+n(primary['mean'])+' min/request. Static allocation and flow MPC obtain '
             +n(simple_wait)+' and '+n(mpc_wait)+' min/request compared with '+n(edge_wait)+' for F4/Edge. '
             'The expanded baselines, independent generator draws and longer training clarify where the tested framework does and does not improve service. '
             'A deployment claim requires measured sensor traces, service times, dispatch records and validation at actual destinations.\n\n')
    declarations=between(old,r'\section*{Declarations}',r'\bibliography{references}')
    declarations=declarations.replace('provided in the accompanying reproducibility archive','provided in the accompanying original and extended reproducibility archives')
    declarations=declarations.replace('AI assistance:', 'Code availability: The accompanying reproducibility archive contains the implemented simulator, adapted forecasting models, control algorithms, pinned third-party source references and licenses, and run-specific source hashes. Reviewer access can be provided through this archive. The author must finalize the current public repository URL and the archived release identifier/DOI before submission.\n\nAI assistance:')
    text += r'\section*{Declarations}'+declarations+r'\bibliography{references}'+'\n'+r'\end{document}'+'\n'
    (DEST/'main.tex').write_text(text,encoding='utf-8')

    # Preserve all formative numerical results, labelled as a separate phase.
    si=preamble+r'\begin{document}'+'\n'+r'\title{Supplementary Material: Forecast-informed service allocation for IoT-enabled scenic areas}'+authors+r'\maketitle'+'\n'
    si+=r'''\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\theequation}{S\arabic{equation}}
\section{Formative study and retained results}
The following protocol and results preserve the completed initial study. They informed the later extension and are not an independent confirmation of it. In this phase, policies use 14,400 steps, the original generator seed and its 27-day test partition, and the original forecasting search. These numerical results must not be combined with the extended study's new worlds or 43,200-step policies as if they used one common protocol. This phase also differs from the earlier JASE manuscript: the latter's aggregate plots are not raw evidence for the newly implemented simulator.

Figures~\ref{fig:1} and~\ref{fig:2} document the fixed graph, formative pipeline and original chronological data split.
'''
    si+=figure('Fig1_architecture','Formative study pipeline and fixed 24-node physical graph. The full main-text architecture supersedes this compact pipeline view; the physical graph is unchanged.','fig:1')
    si+=figure('Fig2_demand','Formative synthetic demand and chronological split for seed 20260929. The new independent realizations are generated separately.','fig:2')
    si+=r'\subsection{Generator parameters}'+generator_detail
    si+=r'\subsection{Original frozen predictors}'+frozen_predictor_methods
    si+=r'\subsection{Initial protocol}'+old_protocol.replace(r'\FloatBarrier','')
    si+=between(old,r'\section{Results}',r'\section{Discussion and limitations}').replace(r'\FloatBarrier','')
    si+=r'\clearpage\input{supplement_methods}'+'\n'
    si+=(OUT/'extended_supplement_tables.tex').read_text(encoding='utf-8')
    si+=r'\FloatBarrier\bibliography{references}'+'\n'+r'\end{document}'+'\n'
    (DEST/'supplement.tex').write_text(si,encoding='utf-8')
    letter=r'''\documentclass[12pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\usepackage{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{8pt}
\setlength{\emergencystretch}{3em}
\raggedright
\begin{document}
{\large\bfseries Submission to Discover Internet of Things}

Dear Editors,

Please consider the Research article ``Forecast-informed service allocation for IoT-enabled scenic areas: a reproducible evaluation with transport constraints'' by Jianbo Guo.

The manuscript examines the connection between causal IoT observations, multi-horizon prediction and executable service-resource allocation. Its fully synthetic testbed includes finite operating days, terminal-unserved requests, transport delay and committed arrivals. The extended evaluation adds modern forecasting comparators, matched spatial ablations, 50 longer-budget policy fits, three independently generated assessment realizations, an intensity stress, and a mixed-integer receding-horizon controller.

'''
    letter+=('The primary resource-attention versus concatenation difference is '+n(primary['mean'])+' min/request, with a 95\\% interval from '
             +n(primary['ci_low'])+' to '+n(primary['ci_high'])+'. The article reports all tested directions, including null and adverse results, and separates computational cost, resource feasibility and service quality. Source code, checkpoints, raw traces and independent numeric audits accompany the submission materials.\n\n')
    letter+=r'''An earlier version was rejected by Journal of Applied Science and Engineering. Because its original executable implementation was unavailable, the present version uses newly implemented and executed experiments. It does not claim recovery or exact reproduction of the earlier numerical results. Its relevance to the journal lies in reproducible performance evaluation of an IoT sensing-to-action application, with explicit limits on real-world inference.

OpenAI Codex assisted with implementation, experiments and preparation, as disclosed in the manuscript. The corresponding author must confirm authorship, funding, competing interests and exclusive submission, finalize the public archival identifiers, and approve the manuscript before upload.

Sincerely,\\ Jianbo Guo\\ Yongcheng Vocational College\\ hsng5987123@163.com
\end{document}
'''
    (DEST/'cover_letter.tex').write_text(letter,encoding='utf-8')
    (DEST/'README.txt').write_text(
        'Discover Internet of Things — LaTeX revision\n'
        'Compile main.tex and supplement.tex with pdfLaTeX, BibTeX, pdfLaTeX twice.\n'
        'Springer Nature sn-jnl class and sn-mathphys-num bibliography style are included.\n'
        'main.tex: extended study; supplement.tex: retained formative evidence and additional methods/results.\n'
        'Funding, competing interests and author-contribution statements still require author confirmation.\n'
        'No real-site validation or exact reproduction of missing original E-STGNN code is claimed.\n'
        'Scientific numeric sources and reproducibility instructions accompany the separate experiment archive.\n', encoding='utf-8')
    print(json.dumps({'main':str(DEST/'main.tex'),'supplement':str(DEST/'supplement.tex'),
                      'abstract_words':len(abstract.split()),'main_words_approx':len(text.split())},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
