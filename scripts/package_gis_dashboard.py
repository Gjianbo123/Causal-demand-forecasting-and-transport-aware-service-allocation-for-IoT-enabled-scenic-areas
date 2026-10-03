"""Deliver the verified bilingual command dashboard and updated LaTeX edition."""
from pathlib import Path
import hashlib, json, shutil, zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/improvement_v4_gis'
DEST=ROOT/'output/improvement_v4_web/Discover_IoT_LaTeX'
BASE=ROOT/'output/improvement_v4_positioned/Discover_IoT_LaTeX'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data): p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    browser=json.loads((OUT/'BROWSER_QA.json').read_text())
    pdf=json.loads((OUT/'PDF_STRUCTURE_QA.json').read_text())
    assert browser['status']=='PASS' and browser['views']==12 and browser['externalNetworkRequests']==0
    assert pdf['pages']==15 and not pdf['out_of_bounds'] and not pdf['compile_errors'] and not pdf['unresolved_references']
    frozen=json.loads((ROOT/'runs/improvement_v4/final_freeze.json').read_text())
    count=0
    for key in ['source_sha256','asset_sha256']:
        for name,h in frozen[key].items():assert sha(ROOT/name)==h,name;count+=1
    for p in list((BASE/'tables').glob('*.tex'))+[BASE/'results_macros.tex',BASE/'supplement.tex',BASE/'supplement.pdf']:
        assert sha(p)==sha(DEST/p.relative_to(BASE)),p
    for old,new in [('Fig1','Fig1'),('Fig2','Fig3'),('Fig3','Fig4')]:
        for p in (BASE/'figures').glob(old+'.*'):assert sha(p)==sha(DEST/'figures'/(new+p.suffix))
    web=OUT/'web_dashboard';prov=json.loads((web/'provenance.json').read_text(encoding='utf8'))
    for name,h in prov['sources'].items():assert sha(ROOT/name)==h,name
    for field,file in [('interface_sha256','index.html'),('replay_data_sha256','replay_data.json'),('illustrative_scenario_sha256','illustrative_scenario.json')]:assert sha(web/file)==prov[field],file
    assert sha(OUT/'Scenic_Command_Overview_EN.png')==sha(DEST/'figures/Fig2.png')
    assert sha(web/'replay_data.json')==sha(ROOT/'output/improvement_v4_web/web_dashboard/replay_data.json')

    destweb=DEST/'command_dashboard';shutil.copytree(web,destweb,dirs_exist_ok=True)
    source=destweb/'source';source.mkdir(exist_ok=True)
    for file in ['scenic_command_template.html','scenic_gis_map.js','build_gis_dashboard.py','render_gis_dashboard.cjs']:
        shutil.copy2(ROOT/'scripts'/file,source/file)
    with (destweb/'README_ZH.txt').open('a',encoding='utf8') as f:
        f.write('\nsource目录包含模板、原创地图、构建与截图脚本。脚本按原实验项目结构读取文件；在完整项目的scripts目录运行。直接使用网页无需这些构建依赖。\n闭园后超出营业时段的预测以N/A呈现并从曲线中隐藏；原始记录不被修改。\n')
    (DEST/'README.txt').write_text('''Compile main.tex and supplement.tex using pdfLaTeX + BibTeX, or Tectonic, with the supplied class, bibliography style, figures and tables.
This revision replaces Figure 2 with a dark GIS-style command-centre interface. Section 3.3 is on page 4; Figure 2 is a full-page sideways figure on page 5. The main PDF has 15 pages; the unchanged supplement has 8 pages.
Open command_dashboard/index.html in a browser, without a server or network. It includes Chinese/English navigation, replay, node selection, map layers, zoom, illustrative incident workflows and state export. Add ?lang=en to the file URL for the English interface.
Recorded synthetic fields (inflow, queues, forecasts, inventory and transfers) are separated from illustrative ticketing, parking, weather, incidents and activities. The map is original SVG schematic terrain, not surveyed geography. There is no live sensor integration, backend dispatch or new experimental result.
The English screenshot is supplied as figures/Fig2.png; Chinese and emergency-view screenshots are included in the standalone dashboard archive. Inspectable source and provenance are included in command_dashboard.
Local Tectonic compilation succeeded; all 15 pages were visually checked. The app-native preview compiler remains unavailable due to a platform-directory environment error. Browser checks covered 12 bilingual navigation states plus replay, export, event workflows, layer controls and scaled display.
Experimental tables, numeric macros, prior result figures, supplementary material and the 89-file experiment freeze are unchanged. Author declarations and long-term public data/code access still need final author completion before submission.
''',encoding='utf8')
    note='''本版按参考图改为深色景区指挥大屏，中央为原创SVG山水示意图，顶部为关键指标，两侧为业务面板，底部为活动和回放控制。
新增中文/英文切换、6个业务导航视图、24个节点选择、72个时点回放、图层开关、缩放、示例事件处置和状态导出。事件操作只更改页面中的演示状态，不发送真实通知或派发真实资源。
保留既有合成实验的流入、队列、预测、资源和转移记录。票务、停车、天气、事件、活动为明确标注的界面示例，排除在实验评价之外。地形和景点名称为原创示意，非实测GIS信息。
论文第3.3节位于第4页；英文界面作为图2横向整页放于第5页。主文15页，补充材料8页。图1架构以及图3、图4结果图不变。论文、投稿信及AI辅助说明已同步修改。
网页包Scenic_Command_Dashboard.zip解压后直接打开index.html。论文包Discover_IoT_GIS_Dashboard_LaTeX.zip包含LaTeX、PDF、图表、参考文献、投稿信和网页源码。
所有实验数值、对照基准和冻结证据保持原状。网页增加了应用场景表达，不构成新增实验或实地系统验证。
'''
    (DEST/'WEB_DASHBOARD_NOTES_ZH.txt').write_text(note,encoding='utf8')
    p=DEST/'RESULTS_AND_LIMITS_ZH.txt';s=p.read_text(encoding='utf8').replace('Discover_IoT_With_Web_Dashboard_LaTeX.zip','Discover_IoT_GIS_Dashboard_LaTeX.zip');p.write_text(s,encoding='utf8')
    qa=dict(status='PASS',main_pages=15,supplement_pages=8,web_section_page=4,web_figure_page=5,
            browser_views_tested=12,screenshot_pixels=[5040,3000],frozen_files_unchanged=count,
            numerical_tables_unchanged=True,original_result_figures_unchanged=True,replay_data_unchanged=True,
            illustrative_fields_excluded_from_experiments=True,source_provenance_verified=True,
            screenshots_and_manuscript_figure_identical=True,visual_review='All 15 pages inspected; Figure 2 at full size. No clipped text, unresolved references or missing glyphs.',
            compiler='Local Tectonic success',native_preview='Unavailable: platform-directory environment error',
            main_pdf_sha256=sha(DEST/'main.pdf'),figure_sha256=sha(DEST/'figures/Fig2.png'))
    save(OUT/'FINAL_QA.json',qa)
    for name in ['FINAL_QA.json','BROWSER_QA.json','PDF_STRUCTURE_QA.json','manuscript_update.json']:shutil.copy2(OUT/name,DEST/name)
    files=[p for p in DEST.rglob('*') if p.is_file() and 'web_prototype' not in p.relative_to(DEST).parts and p.suffix not in {'.aux','.log','.blg','.out'} and p.name!='FILE_SHA256.json']
    save(DEST/'FILE_SHA256.json',{p.relative_to(DEST).as_posix():sha(p) for p in sorted(files)})
    files.append(DEST/'FILE_SHA256.json')
    manuscript=OUT/'Discover_IoT_GIS_Dashboard_LaTeX.zip'
    with zipfile.ZipFile(manuscript,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,'Discover_IoT_LaTeX/'+p.relative_to(DEST).as_posix())
    dashboard=OUT/'Scenic_Command_Dashboard.zip'
    with zipfile.ZipFile(dashboard,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(destweb.rglob('*')):
            if p.is_file():z.write(p,'Scenic_Command_Dashboard/'+p.relative_to(destweb).as_posix())
        for name in ['Scenic_Command_Overview_ZH.png','Scenic_Command_Overview_EN.png','Scenic_Command_Emergency_ZH.png']:
            z.write(OUT/name,'Scenic_Command_Dashboard/screenshots/'+name)
    manifest={}
    for p in [manuscript,dashboard]:
        with zipfile.ZipFile(p) as z:assert z.testzip() is None;manifest[p.name]=dict(bytes=p.stat().st_size,sha256=sha(p),files=len(z.namelist()),crc='PASS')
    save(OUT/'PACKAGE_MANIFEST.json',manifest)
    print(json.dumps(dict(qa=qa,packages=manifest),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
