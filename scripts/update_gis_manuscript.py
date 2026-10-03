"""Update the open manuscript in place; retain its previously delivered edition."""
from pathlib import Path
import hashlib, json, re, shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/improvement_v4_gis'
DEST=ROOT/'output/improvement_v4_web/Discover_IoT_LaTeX'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert json.loads((OUT/'BROWSER_QA.json').read_text())['status']=='PASS'
    backup=OUT/'previous_manuscript'
    if not backup.exists():
        backup.mkdir()
        for name in ['main.tex','main.pdf','cover_letter.tex']:
            shutil.copy2(DEST/name,backup/name)
        shutil.copy2(DEST/'figures/Fig2.png',backup/'Fig2.png')
    text=(backup/'main.tex').read_text(encoding='utf8')
    text=text.replace(r'\usepackage{graphicx}',r'\usepackage{graphicx}'+'\n'+r'\usepackage{rotating}')
    section=r'''
\subsection{GIS command-dashboard prototype}
Figure~\ref{fig:web_dashboard} combines recorded inflow, queues, forecasts and transfers with a schematic map. Labelled ticketing, parking, weather, incident and activity examples are excluded from the experiments.
\begin{sidewaysfigure}[p]
\centering\includegraphics[width=238mm]{Fig2.png}
\caption{Command-dashboard prototype with recorded simulation outputs and labelled interface examples. The map is schematic, without surveyed coordinates. The displayed replay is world 20261011, day 153, at $t=36$. OpenAI Codex assisted the inspectable HTML/CSS/SVG implementation; Chromium rendered the screenshot}\label{fig:web_dashboard}
\end{sidewaysfigure}
\clearpage
'''
    text,n=re.subn(r'\n\\clearpage\n\\subsection\{Offline web-dashboard prototype\}.*?\\FloatBarrier\n',lambda _:section,text,count=1,flags=re.S)
    assert n==1
    text=text.replace('was rendered in Chromium from archived arrays.', 'was rendered in Chromium from archived arrays and explicitly labelled interface examples.')
    (DEST/'main.tex').write_text(text,encoding='utf8')
    shutil.copy2(OUT/'Scenic_Command_Overview_EN.png',DEST/'figures/Fig2.png')
    letter=(backup/'cover_letter.tex').read_text(encoding='utf8').replace('An offline browser prototype visualizes archived observations, forecasts, resource states and recorded transfers, with inspectable source and data provenance; it is presented as a replay interface rather than a deployed IoT application.', 'An offline GIS-style command dashboard visualizes archived observations, forecasts, resource states and transfers. Clearly labelled ticketing, parking, weather, incident and activity examples illustrate the interface context and are excluded from experimental evaluation; source and data provenance are supplied.')
    (DEST/'cover_letter.tex').write_text(letter,encoding='utf8')
    audit=dict(previous_main_sha256=sha(backup/'main.tex'),updated_main_sha256=sha(DEST/'main.tex'),figure_sha256=sha(DEST/'figures/Fig2.png'),
               experimental_tables={p.name:sha(p) for p in (DEST/'tables').glob('*.tex')},results_macros_sha256=sha(DEST/'results_macros.tex'),
               figure_number=2,replay_unchanged=True,demonstration_fields_excluded_from_evaluation=True)
    (OUT/'manuscript_update.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(str(DEST/'main.tex'))


if __name__=='__main__':main()
