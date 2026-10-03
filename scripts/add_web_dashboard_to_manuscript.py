"""Add the trace-backed browser prototype as Fig. 2 in a new manuscript copy."""
from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/improvement_v4_positioned/Discover_IoT_LaTeX'
OUT=ROOT/'output/improvement_v4_web'
DEST=OUT/'Discover_IoT_LaTeX'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    qa=json.loads((OUT/'BROWSER_QA.json').read_text());assert qa['status']=='PASS'
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'figures').mkdir(exist_ok=True)
    for p in BASE.rglob('*'):
        if p.is_file() and p.suffix in {'.tex','.bib','.bbl','.cls','.bst','.pdf','.svg','.png','.eps','.txt'} and 'figures' not in p.relative_to(BASE).parts:
            q=DEST/p.relative_to(BASE);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for p in (BASE/'figures').iterdir():
        stem={'Fig2':'Fig3','Fig3':'Fig4'}.get(p.stem,p.stem)
        shutil.copy2(p,DEST/'figures'/(stem+p.suffix))
    shutil.copy2(OUT/'Web_dashboard.png',DEST/'figures/Fig2.png')
    for name,old,new in [('figures_forecast.tex','Fig2.pdf','Fig3.pdf'),('figures_control.tex','Fig3.pdf','Fig4.pdf')]:
        p=DEST/name;s=p.read_text(encoding='utf8');assert old in s;p.write_text(s.replace(old,new),encoding='utf8')
    text=(DEST/'main.tex').read_text(encoding='utf8')
    section=r'''
\clearpage
\subsection{Offline web-dashboard prototype}
Figure~\ref{fig:web_dashboard} presents an offline browser prototype for inspecting archived observations, forecasts, queues and recorded transfers. The read-only HTML/CSS/SVG interface supports time-step selection, node selection and state export. It reads saved simulation outputs; it neither runs online inference nor connects to field devices.
\begin{figure}[!htbp]
\centering\includegraphics[width=165mm]{Fig2.png}
\caption{Offline browser replay of nominal world 20261011, physical day 153, at $t=36$ with node S1 selected. Counts, queues, resource states, forecasts and transfers come from the archived rollout trace. OpenAI Codex assisted the inspectable HTML/CSS/SVG implementation; Chromium rendered the screenshot. This interface prototype supports record inspection and provides no additional experimental evidence or field-deployment validation}\label{fig:web_dashboard}
\end{figure}
\FloatBarrier
'''
    anchor=r'\section{Forecasting and transport rollout}'
    assert text.count(anchor)==1;text=text.replace(anchor,section+'\n'+anchor)
    text=text.replace('raw predictions, executed flows, analysis and figure scripts.', 'raw predictions, executed flows, analysis and figure scripts, and the offline browser prototype with its data-provenance mapping.')
    old="Code and numerical records are supplied for inspection. Final scientific review, authorship statements and approval remain the author's responsibility."
    new="The HTML/CSS/SVG interface in Fig.~\\ref{fig:web_dashboard} was rendered in Chromium from archived arrays. Code and data provenance are supplied. Scientific review, authorship and final approval remain the author's responsibility."
    assert text.count(old)==1;text=text.replace(old,new)
    text=text.replace('AI assistance: OpenAI Codex assisted','AI assistance: OpenAI Codex (GPT-6) assisted')
    (DEST/'main.tex').write_text(text,encoding='utf8')
    letter=(DEST/'cover_letter.tex').read_text(encoding='utf8')
    old='The work is relevant to the journal\'s interest in connecting IoT observations with operational decisions.'
    new=old+' An offline browser prototype visualizes archived observations, forecasts, resource states and recorded transfers, with inspectable source and data provenance; it is presented as a replay interface rather than a deployed IoT application.'
    assert old in letter;(DEST/'cover_letter.tex').write_text(letter.replace(old,new),encoding='utf8')
    audit=dict(source_main_sha256=sha(BASE/'main.tex'),new_main_sha256=sha(DEST/'main.tex'),web_figure_sha256=sha(DEST/'figures/Fig2.png'),
        figure_mapping={'Fig1':'unchanged system architecture','Fig2':'new offline browser replay','Fig3':'unchanged forecast results, previously Fig2','Fig4':'unchanged control results, previously Fig3'},
        source_tables={p.name:sha(p) for p in (BASE/'tables').glob('*.tex')},data_inputs=json.loads((OUT/'web_dashboard/provenance.json').read_text()))
    for p in (BASE/'tables').glob('*.tex'):assert sha(p)==sha(DEST/'tables'/p.name)
    assert sha(BASE/'results_macros.tex')==sha(DEST/'results_macros.tex')
    (OUT/'manuscript_update.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print('Added trace-backed web dashboard as Figure 2; original numeric results unchanged.')

if __name__=='__main__':main()
