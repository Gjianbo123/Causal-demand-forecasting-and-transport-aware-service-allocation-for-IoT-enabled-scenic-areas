"""Add four evidence figures to the current manuscript without changing results."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/improvement_v4_more_figures'
DEST=ROOT/'output/improvement_v4_web/Discover_IoT_LaTeX'
BACK=OUT/'previous_manuscript'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def figure(file,number,label,caption):
    (DEST/file).write_text('\\begin{figure}[!htbp]\n\\centering\n'+
        f'\\includegraphics[width=\\textwidth]{{Fig{number}.pdf}}\n'+
        '\\caption{'+caption+'}\\label{'+label+'}\n\\end{figure}\n',encoding='utf8')


def main():
    OUT.mkdir(exist_ok=True)
    if not BACK.exists():
        BACK.mkdir();(BACK/'figures').mkdir()
        for name in ['main.tex','main.pdf','figures_control.tex','results_macros.tex','supplement.tex','supplement.pdf','README.txt']:
            shutil.copy2(DEST/name,BACK/name)
        shutil.copytree(DEST/'tables',BACK/'tables')
        for p in (DEST/'figures').glob('Fig4.*'):shutil.copy2(p,BACK/'figures'/p.name)
    for p in (BACK/'figures').glob('Fig4.*'):shutil.copy2(p,DEST/'figures'/('Fig5'+p.suffix))
    mapping={'Forecast_consistency':4,'Control_tradeoffs':6,'Matched_worlds':7,'Development_computation':8}
    for stem,number in mapping.items():
        for ext in ['pdf','svg','png','eps']:
            p=OUT/'figures'/f'{stem}.{ext}'
            assert p.exists(),p
            shutil.copy2(p,DEST/'figures'/f'Fig{number}.{ext}')
    old=(BACK/'figures_control.tex').read_text(encoding='utf8');assert 'Fig4.pdf' in old
    (DEST/'figures_control.tex').write_text(old.replace('Fig4.pdf','Fig5.pdf'),encoding='utf8')
    figure('figures_consistency.tex',4,'fig:forecast_consistency',
        'Paired forecast MAE differences across seven worlds (W1--W7: seeds 20261011--20261017). Negative values favor the combination. The inset expands the increased-demand comparison with residual GRU. Panel scales differ; connecting lines do not represent temporal changes')
    figure('figures_tradeoffs.tex',6,'fig:control_tradeoffs',
        'Service outcomes and relocation trade-offs in the full nine-day-per-world matrix. Left and right columns show nominal and increased demand. (a,b) Small markers represent individual worlds; large outlined markers show seven-world means. (c,d) Bars show mean terminal unserved requests and queue-overload frequency; whiskers are sample standard deviations across worlds. The two percentages have different denominators: arrivals and service-node/interval pairs. G and C denote GWN and combined forecasts. PPO outcomes are averaged over training seeds within each world')
    figure('figures_matched_worlds.tex',7,'fig:matched_worlds',
        'Paired accrued-wait differences on the same three days in each of seven worlds, or 21 episodes per controller and condition. C denotes the identical combined forecast. Each point is rollout minus its comparator; negative favors rollout. Both comparisons use this common subset, including the six-step MPC comparison. W1--W7 are independent world identifiers, not chronological test days. These horizon comparisons are exploratory')
    figure('figures_development_computation.tex',8,'fig:development_computation',
        'Development sensitivity and local computation. (a) Accrued waiting on the three predeclared development days for all six settings; the outline marks the setting selected before testing. (b) Empirical cumulative latency distributions for rollout and same-forecast MPC-6/MPC-12, with 67 serial CPU decisions per method after five warmup decisions. The legend reports median and 95th percentile. Timings include history preparation, forecasting, planning and executable-flow construction; they exclude model loading, physical networking, actuation and simulator transitions')
    text=(BACK/'main.tex').read_text(encoding='utf8')
    anchor='Horizon-specific scores and all world-level results are retained in Supplementary Material.'
    replacement=anchor+r'''

Figure~\ref{fig:forecast_consistency} shows lower nominal MAE than both components in every world. Under increased demand, the combination stays below GWN but exceeds residual GRU in W2 and W4. The small aggregate gain over the residual component is therefore not uniform across realizations.
\input{figures_consistency.tex}
'''
    assert text.count(anchor)==1;text=text.replace(anchor,replacement)
    anchor=r'\input{figures_control.tex}'
    replacement=anchor+r'''

Figure~\ref{fig:control_tradeoffs} relates waiting to relocation distance and separates terminal backlog from overload. It makes the higher movement requirement explicit: lower waiting does not imply lower resource travel or uniformly lower overload.
\input{figures_tradeoffs.tex}
'''
    assert text.count(anchor)==1;text=text.replace(anchor,replacement)
    anchor=r'\MatchedInterpretation\par'
    replacement=anchor+r'''

Figure~\ref{fig:matched_worlds} retains the individual paired worlds behind Table~\ref{tab:matched}. All plotted waiting differences favor rollout, although the increased-demand difference from 12-step MPC is close to zero in W5. The figure uses only the common three-day subset and does not reuse the larger full-matrix contrast.
\input{figures_matched_worlds.tex}
'''
    assert text.count(anchor)==1;text=text.replace(anchor,replacement)
    anchor=r'\TimingInterpretation\par'
    replacement=anchor+r'''

Figure~\ref{fig:development_computation} visualizes the entire development grid and the measured latency distributions. The heat map concerns parameter selection, while the latency curves describe local computation on one validation day; neither constitutes a new independent test or a field-latency guarantee.
\input{figures_development_computation.tex}
'''
    assert text.count(anchor)==1;text=text.replace(anchor,replacement)
    text=text.replace(r'\FloatBarrier'+'\n\n'+r'\subsection{Service allocation and matched comparisons}', r'\subsection{Service allocation and matched comparisons}')
    text=text.replace(r'\FloatBarrier'+'\n\n'+r'\subsection{Development sensitivity, feasibility and computation}', r'\subsection{Development sensitivity, feasibility and computation}')
    (DEST/'main.tex').write_text(text,encoding='utf8')
    for p in (BACK/'tables').glob('*.tex'):assert sha(p)==sha(DEST/'tables'/p.name)
    for name in ['results_macros.tex','supplement.tex','supplement.pdf']:assert sha(BACK/name)==sha(DEST/name)
    audit=dict(previous_main_sha256=sha(BACK/'main.tex'),main_sha256=sha(DEST/'main.tex'),
        new_figures={f'Fig{n}':dict(source=stem,sha256=sha(DEST/'figures'/f'Fig{n}.pdf')) for stem,n in mapping.items()},
        previous_waiting_figure='Fig4 -> Fig5; pixels and vector contents unchanged',
        numerical_tables_unchanged=True,numerical_macros_unchanged=True,supplement_unchanged=True,
        experimental_figures_before=2,experimental_figures_after=6,total_main_figures=8)
    (OUT/'manuscript_update.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
