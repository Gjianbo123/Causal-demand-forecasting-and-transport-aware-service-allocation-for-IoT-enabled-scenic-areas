"""Package the inspected manuscript and its trace-backed offline interface."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/improvement_v4_web'
DEST = OUT / 'Discover_IoT_LaTeX'
BASE = ROOT / 'output/improvement_v4_positioned/Discover_IoT_LaTeX'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    browser = json.loads((OUT / 'BROWSER_QA.json').read_text(encoding='utf8'))
    pdf = json.loads((OUT / 'PDF_STRUCTURE_QA.json').read_text(encoding='utf8'))
    assert browser['status'] == 'PASS' and browser['exportVerified']
    assert pdf['pages'] == 15 and pdf['web_section_pages'] == [5]
    assert not pdf['compile_errors'] and not pdf['out_of_bounds']

    freeze = json.loads((ROOT / 'runs/improvement_v4/final_freeze.json').read_text())
    checked = 0
    for group in ['source_sha256', 'asset_sha256']:
        for name, expected in freeze[group].items():
            assert sha(ROOT / name) == expected, name
            checked += 1
    tables = list((BASE / 'tables').glob('*.tex'))
    for path in tables + [BASE / 'results_macros.tex', BASE / 'supplement.tex', BASE / 'supplement.pdf']:
        assert sha(path) == sha(DEST / path.relative_to(BASE)), path
    for old, new in [('Fig1', 'Fig1'), ('Fig2', 'Fig3'), ('Fig3', 'Fig4')]:
        for path in (BASE / 'figures').glob(old + '.*'):
            assert sha(path) == sha(DEST / 'figures' / (new + path.suffix)), path
    provenance = json.loads((OUT / 'web_dashboard/provenance.json').read_text())
    for name, expected in provenance['sources'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(OUT / 'web_dashboard/index.html') == provenance['interface_sha256']
    assert sha(OUT / 'web_dashboard/replay_data.json') == provenance['replay_data_sha256']
    assert sha(OUT / 'Web_dashboard.png') == sha(DEST / 'figures/Fig2.png')

    (DEST / 'README.txt').write_text('''Compile main.tex and supplement.tex with pdfLaTeX + BibTeX (or Tectonic). Keep the supplied figures, tables, references.bib, sn-jnl.cls and sn-mathphys-num.bst together.
This edition adds Section 3.3 and Figure 2: a browser-rendered, read-only replay of an archived synthetic episode. The previous result figures are now Figures 3 and 4. Experimental values, original figures and supplementary material are unchanged.
main.pdf: 15 pages; supplement.pdf: 8 pages. The main PDF was successfully compiled with local Tectonic and visually inspected. The app's built-in preview compiler reported a platform-directory environment error; it was not used to certify compilation.
Open web_prototype/index.html locally to use the interface. It embeds the replay data and requires no server or external network. Select a decision time or node, or export the current state. No live sensors, online inference or field control are implemented.
web_prototype/source contains the inspectable implementation. Rebuilding from raw arrays requires the original experiment project: place the source files under its scripts directory, then run build_web_dashboard.py followed by render_web_dashboard.cjs. Rendering requires Playwright and Chromium. The supplied index.html works without those dependencies.
FILE_SHA256.json records the delivered file hashes. FINAL_QA.json records checks for this edition. See WEB_DASHBOARD_NOTES_ZH.txt for the Chinese change note.
Author declarations and public data/code access still require final author confirmation before submission.
''', encoding='utf8')
    notes = '''本次新增网页系统示意图，已加入主文第3.3节（第5页），编号为图2。原系统架构图仍为图1，原预测结果图、调度结果图顺延为图3、图4。
界面包含客流、队列、驻点/在途资源、景区图、节点预测、资源台账和可执行调度记录。支持72个决策时点回放、24个节点选择和当前状态JSON导出。
截图使用首个常规测试世界20261011、原始第153日的中点t=36、节点S1。快照显示302次流入事件、323个排队请求、80个驻点资源单位、2个在途单位和5条转移记录。选择规则与指标表现无关；数值来自既有保存轨迹。
这是离线只读仿真回放原型，没有连接真实传感器，没有实现在线模型推理或景区现场控制。截图不构成新增实验或部署证据。
采用可检查的HTML/CSS/SVG源代码，通过Chromium渲染。未使用图像生成模型；论文图注和AI辅助声明已说明代码辅助与渲染来源。
已有实验结果、基线、统计表、原始结果图、补充材料及冻结证据未改动。正文成功编译为15页；新增页面与其余页面均已查看。内置预览编译器存在平台目录环境错误，交付PDF使用本地Tectonic编译。
Discover_IoT_With_Web_Dashboard_LaTeX.zip含完整LaTeX、PDF、图表、投稿信和网页原型。Scenic_IoT_Web_Prototype.zip为可单独打开的网页包。
'''
    (DEST / 'WEB_DASHBOARD_NOTES_ZH.txt').write_text(notes, encoding='utf8')
    result_notes = (DEST / 'RESULTS_AND_LIMITS_ZH.txt').read_text(encoding='utf8')
    result_notes = result_notes.replace('主文 14 页', '主文 15 页').replace('Discover_IoT_Results_Emphasized_LaTeX.zip', 'Discover_IoT_With_Web_Dashboard_LaTeX.zip')
    result_notes += '\n本次增加第3.3节的离线网页原型及图2，详见WEB_DASHBOARD_NOTES_ZH.txt。该界面展示保存的合成轨迹，不增加实验结论。\n'
    (DEST / 'RESULTS_AND_LIMITS_ZH.txt').write_text(result_notes, encoding='utf8')

    web = DEST / 'web_prototype'
    shutil.copytree(OUT / 'web_dashboard', web, dirs_exist_ok=True)
    (web / 'source').mkdir(exist_ok=True)
    for name in ['build_web_dashboard.py', 'web_dashboard_template.html', 'render_web_dashboard.cjs']:
        shutil.copy2(ROOT / 'scripts' / name, web / 'source' / name)
    with (web / 'README_ZH.txt').open('a', encoding='utf8') as f:
        f.write('\nsource目录保留抽取、界面及渲染源代码；重新从原始数组生成页面需要原实验项目，运行时将这些文件放入该项目scripts目录。直接打开index.html无需安装这些依赖。\n')

    final = dict(status='PASS', main_pages=15, supplement_pages=8,
                 new_figure_page=5, screenshot_pixels=[3840, 3630],
                 browser_interaction_and_export='PASS', external_network_requests=0,
                 frozen_files_unchanged=checked, unchanged_result_tables=len(tables),
                 unchanged_results_macros=True, unchanged_original_figures=True,
                 unchanged_supplement=True, source_provenance_hashes_verified=4,
                 visual_review='All 15 rendered pages checked; Figure 2 page also inspected at full size. No clipped text, unresolved references, missing glyphs or page-boundary text.',
                 compiler='Local Tectonic: success',
                 native_preview='Unavailable: Unable to find standard directories for platform',
                 main_pdf_sha256=sha(DEST / 'main.pdf'), web_figure_sha256=sha(DEST / 'figures/Fig2.png'))
    save_json(OUT / 'FINAL_QA.json', final)
    for name in ['FINAL_QA.json', 'BROWSER_QA.json', 'PDF_STRUCTURE_QA.json', 'manuscript_update.json']:
        shutil.copy2(OUT / name, DEST / name)
    excluded = {'.aux', '.log', '.blg', '.out'}
    files = [p for p in DEST.rglob('*') if p.is_file() and p.suffix not in excluded and p.name != 'FILE_SHA256.json']
    save_json(DEST / 'FILE_SHA256.json', {p.relative_to(DEST).as_posix(): sha(p) for p in sorted(files)})
    files.append(DEST / 'FILE_SHA256.json')
    packages = {}
    full_path = OUT / 'Discover_IoT_With_Web_Dashboard_LaTeX.zip'
    with zipfile.ZipFile(full_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, 'Discover_IoT_LaTeX/' + p.relative_to(DEST).as_posix())
    web_path = OUT / 'Scenic_IoT_Web_Prototype.zip'
    with zipfile.ZipFile(web_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(web.rglob('*')):
            if p.is_file():
                z.write(p, 'Scenic_IoT_Web_Prototype/' + p.relative_to(web).as_posix())
        z.write(OUT / 'Web_dashboard.png', 'Scenic_IoT_Web_Prototype/Web_dashboard.png')
    for p in [full_path, web_path]:
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            packages[p.name] = dict(bytes=p.stat().st_size, sha256=sha(p), files=len(z.namelist()), crc_check='PASS')
    save_json(OUT / 'PACKAGE_MANIFEST.json', packages)
    print(json.dumps(dict(qa=final, packages=packages), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
