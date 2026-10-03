"""Package the expanded experiment figures, source records and current paper."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/improvement_v4_more_figures'
DEST=ROOT/'output/improvement_v4_web/Discover_IoT_LaTeX'
BACK=OUT/'previous_manuscript'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    pdf=json.loads((OUT/'PDF_STRUCTURE_QA.json').read_text())
    assert not pdf['compile_errors'] and not pdf['out_of_bounds'] and not pdf['unresolved_references'] and len(pdf['figures'])==8
    freeze=json.loads((ROOT/'runs/improvement_v4/final_freeze.json').read_text());checked=0
    for group in ['source_sha256','asset_sha256']:
        for name,h in freeze[group].items():assert sha(ROOT/name)==h,name;checked+=1
    for p in (BACK/'tables').glob('*.tex'):assert sha(p)==sha(DEST/'tables'/p.name)
    for name in ['results_macros.tex','supplement.tex','supplement.pdf']:assert sha(BACK/name)==sha(DEST/name)
    for p in (BACK/'figures').glob('Fig4.*'):assert sha(p)==sha(DEST/'figures'/('Fig5'+p.suffix))
    forecast=json.loads((OUT/'forecast_consistency_provenance.json').read_text())
    control=json.loads((OUT/'control_plot_audit.json').read_text())
    computation=json.loads((OUT/'figures/Development_computation_audit.json').read_text())
    sources={v['path']:v['sha256'] for v in forecast['sources']}
    sources[control['source']]=control['source_sha256'];sources.update(computation['source_sha256'])
    for name,h in sources.items():assert sha(ROOT/name)==h,name
    evidence=DEST/'figure_evidence';evidence.mkdir(exist_ok=True)
    for name in sources:
        p=ROOT/name;q=evidence/Path(name);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for name in ['plot_forecast_consistency.py','plot_control_evidence.py','plot_development_computation.py']:
        q=evidence/'scripts'/name;q.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'scripts'/name,q)
    export_files=[p for p in OUT.iterdir() if p.is_file() and p.suffix in {'.csv','.json','.txt'} and p.name not in {'FINAL_QA.json','PACKAGE_MANIFEST.json'}]
    export_files += [p for p in (OUT/'figures').iterdir() if p.is_file()]
    for p in export_files:
        q=evidence/'output/improvement_v4_more_figures'/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    notes='''本次在主文实验部分增加4幅图，共10个主子图；实验图由2幅增加到6幅，连同系统架构和网页界面共8幅主文图。
图4：七个独立世界的预测MAE配对差异，分别与Graph WaveNet、Residual GRU比较，保留高负荷下W2/W4不优于Residual GRU的结果。
图6：等待时间与资源移动量的取舍，以及终点未服务率和过载率；包含全部6个调度方法，显示世界级原始点和均值，条形误差为样本标准差。
图7：相同三日子集的逐世界调度差异；严格使用每条件21个配对回合，不与全量九日矩阵混用。图中的C表示相同的组合预测器。
图8：全部六个开发参数组合及原始运行时延经验CDF。开发集不充当独立测试；时延为本机无缓存串行计算，不含实际网络和执行器。
原调度图从图4顺延为图5；图1系统架构、图2网页、图3预测跨度保留。正文已增加图注、引用和简短解释。所有原始实验数值、数值表、宏和补充材料保持不变，没有新跑实验或调整成绩。
新图提供PDF、SVG、EPS和450dpi PNG；figure_evidence保留3个绘图脚本、逐点数据、源记录与SHA审计。进入figure_evidence后，使用安装了numpy、pandas、matplotlib的Python依次运行scripts目录三个plot脚本即可重绘；无需训练模型。这里的记录子集仅用于重绘，完整实验复现包仍为前版ScenicIoT_Improved_Reproducibility.zip。
'''
    (DEST/'EXPERIMENT_FIGURES_ZH.txt').write_text(notes,encoding='utf8')
    (evidence/'README_ZH.txt').write_text(notes,encoding='utf8')
    pages=pdf['pages']
    (DEST/'README.txt').write_text(f'''Compile main.tex and supplement.tex with pdfLaTeX + BibTeX (or Tectonic), retaining all supplied class, bibliography, figures and tables.
The main paper now has {pages} pages and eight figures. Four experiment figures were added: Fig. 4 paired forecast consistency; Fig. 6 service and relocation trade-offs; Fig. 7 world-level horizon-matched waiting comparisons; Fig. 8 development sensitivity and uncached local latency. The earlier waiting figure is now Fig. 5. Architecture, GIS interface and horizon-accuracy graphics remain Figs. 1-3.
All numerical tables, macros, experimental source records and the eight-page supplementary document are unchanged. New graphics visualize already recorded experiments and do not represent new trials. World replicates, matched subsets, sample-SD whiskers and local-only timing are specified in the captions.
The figure_evidence folder contains plotting scripts, raw inputs needed by those scripts, plotted values, provenance and original-format graphics. It is a lightweight figure-reproduction bundle, not the full experimental archive. See EXPERIMENT_FIGURES_ZH.txt.
The offline bilingual GIS prototype is in command_dashboard/index.html. Recorded synthetic fields and illustrative business fields remain separated; the map is schematic and no field deployment is claimed.
Local Tectonic compilation succeeded and all {pages} pages were visually inspected. The app-native preview compiler remains unavailable due to a platform-directory environment error. Author declarations and public data/code access still require author completion before submission.
''',encoding='utf8')
    p=DEST/'RESULTS_AND_LIMITS_ZH.txt';s=p.read_text(encoding='utf8');s=re.sub(r'主文 \d+ 页',f'主文 {pages} 页',s);s=s.replace('Discover_IoT_GIS_Dashboard_LaTeX.zip','Discover_IoT_Expanded_Experiments_LaTeX.zip')
    extra='本次新增实验图4、6、7、8；详见EXPERIMENT_FIGURES_ZH.txt。'
    if extra not in s:s+='\n'+extra+'\n'
    p.write_text(s,encoding='utf8')
    p=DEST/'WEB_DASHBOARD_NOTES_ZH.txt';s=p.read_text(encoding='utf8').replace('主文15页',f'主文{pages}页').replace('图1架构以及图3、图4结果图不变。','图1架构、图3预测图保留；原调度图顺延为图5，新增实验图4、6、7、8。').replace('Discover_IoT_GIS_Dashboard_LaTeX.zip','Discover_IoT_Expanded_Experiments_LaTeX.zip');p.write_text(s,encoding='utf8')
    qa=dict(status='PASS',main_pages=pages,total_figures=8,experimental_figures=6,added_figures=4,added_main_panels=10,
            figure_pages=pdf['figures'],numerical_tables_unchanged=True,supplement_unchanged=True,frozen_files_unchanged=checked,
            additional_source_records_verified=len(sources),minimum_new_plot_font_pt=8,new_experiments=False,
            visual_review=f'All {pages} PDF pages and all four new scientific figures inspected; no clipping, missing glyphs, overlaps or unresolved references.',
            compiler='Local Tectonic success',native_preview='Unavailable: platform-directory environment error',main_pdf_sha256=sha(DEST/'main.pdf'))
    save(OUT/'FINAL_QA.json',qa)
    shutil.copy2(OUT/'FINAL_QA.json',evidence/'output/improvement_v4_more_figures/FINAL_QA.json')
    for name in ['FINAL_QA.json','PDF_STRUCTURE_QA.json','manuscript_update.json']:shutil.copy2(OUT/name,DEST/name)
    files=[p for p in DEST.rglob('*') if p.is_file() and 'web_prototype' not in p.relative_to(DEST).parts and p.suffix not in {'.aux','.log','.blg','.out'} and p.name!='FILE_SHA256.json']
    save(DEST/'FILE_SHA256.json',{p.relative_to(DEST).as_posix():sha(p) for p in sorted(files)})
    files.append(DEST/'FILE_SHA256.json')
    zipmain=OUT/'Discover_IoT_Expanded_Experiments_LaTeX.zip'
    with zipfile.ZipFile(zipmain,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,'Discover_IoT_LaTeX/'+p.relative_to(DEST).as_posix())
    zipfig=OUT/'Experimental_Figures_and_Source.zip'
    with zipfile.ZipFile(zipfig,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(evidence.rglob('*')):
            if p.is_file():z.write(p,'Experimental_Figures/'+p.relative_to(evidence).as_posix())
    packages={}
    for p in [zipmain,zipfig]:
        with zipfile.ZipFile(p) as z:assert z.testzip() is None;packages[p.name]=dict(bytes=p.stat().st_size,sha256=sha(p),files=len(z.namelist()),crc='PASS')
    save(OUT/'PACKAGE_MANIFEST.json',packages)
    print(json.dumps(dict(qa=qa,packages=packages),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
