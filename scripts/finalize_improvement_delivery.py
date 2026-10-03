"""Record completed numerical and visual checks; prepare the delivery files."""
from pathlib import Path
from datetime import datetime, timezone
import json, shutil, hashlib, re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/improvement_v4'
V4 = ROOT / 'runs/improvement_v4'
LATEX = OUT / 'Discover_IoT_LaTeX'

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    freeze = read(V4 / 'final_freeze.json')
    for field in ['source_sha256', 'asset_sha256']:
        for relative, expected in freeze[field].items():
            assert digest(ROOT / relative) == expected, relative
    audit = read(V4 / 'INDEPENDENT_AUDIT.json')
    report = read(OUT / 'REPORT_AUDIT.json')
    build = read(OUT / 'manuscript_build.json')
    structure = read(OUT / 'PDF_STRUCTURE_QA.json')
    assert audit['status'] == report['status'] == 'PASS'
    assert digest(OUT / 'results.json') == report['results_sha256'] == build['results_sha256']
    assert digest(LATEX / 'main.tex') == build['main_sha256']
    assert build['abstract_words'] < 250
    tests = (V4 / 'unit_test_results.txt').read_text(encoding='utf8')
    assert 'Ran 50 tests' in tests and tests.rstrip().endswith('OK')
    pdfs = {}
    for name, title in [('main', 'Manuscript'), ('supplement', 'Supplement')]:
        log = (OUT / f'{name}_compile.log').read_text(encoding='utf8')
        assert not re.search(r'Overfull|undefined|Missing character|^error:', log, re.I | re.M)
        assert not structure[name]['outside_page_spans']
        assert not structure[name]['unresolved_references']
        target = OUT / f'Discover_IoT_Improved_{title}.pdf'
        shutil.copy2(LATEX / f'{name}.pdf', target)
        pdfs[target.name] = dict(sha256=digest(target), pages=structure[name]['pages'])
    results = read(OUT / 'results.json')
    notes = '''已完成：v4 方法重构、冻结后的新合成实验、系统架构图、Springer Nature LaTeX 主文与补充材料。

主要结果（常规负荷）
1. 预测组合 MAE：2.2404；Graph WaveNet 三种子均值：2.2512，降低 0.48%。
2. 调度累计等待/请求：22.67 分钟；相同预测器的六步 MPC：27.14 分钟，降低 16.47%。
3. 预先指定的同为十二步子集：24.60 对 26.21 分钟，降低约 6.15%。子集与上一项九天/世界矩阵不可混用。
4. 两项预先声明的主要配对比较使用七个独立需求世界，Holm 校正 p=0.03125；其他比较属于探索性结果。

不能省略的取舍
常规负荷的资源移动量为 51.75 对 30.58 resource-km/day；新调度移动更多。高负荷的过载频率为 43.32% 对 41.48%，新方法并非该指标最佳。高负荷 RMSE 上，单独残差预测器优于组合预测器。
因此可以写成“在本次列出的对照与条件中取得最低平均 MAE 和等待指标”，不能写成所有指标最优、普遍最佳或已在真实景区验证。

实验与审计
11 个预测变体；7 个独立需求世界；常规与高负荷两种条件；3,066 次完整控制回合；220,752 个决策。
546 个流式控制回合经过独立轨迹重放；2,520 个 PPO 回合经过汇总和原始已服务等待样本核查，未宣称其紧凑轨迹支持独立动作重放。
50 项测试通过；1,478 个汇总数据项独立复算通过；冻结的源代码、权重和方案哈希保持不变。
主文 14 页，补充材料 8 页；已逐页查看渲染输出，无未定义引用或越界文本。

数据与方法边界
这是重新构建的、完全合成的可复现实验，不是恢复已丢失原论文代码。原截图中的 23.5、10.9 等旧值未混入新实验结论。
新的核心是季节锚点＋图残差 GRU/GWN 组合预测，以及显式考虑在途服务损失的运输滚动调度。PPO 保留为对照，不再将新方法命名为原 E-STGNN/PF-PPO。
系统架构图已同步重画：Synthetic inflow is exogenous; service queues respond to allocation.

文件使用
Discover_IoT_Improved_LaTeX.zip：主文 main.tex、补充材料 supplement.tex、参考文献、图表、Springer 类文件、cover_letter.tex 和编译 PDF。
ScenicIoT_Improved_Reproducibility.zip：代码、模型权重、开发候选、原始预测、控制轨迹、统计表和审计记录。
正式提交前仍需作者填写真实基金、利益冲突、贡献声明，并核实署名和建立长期可访问的数据代码地址。当前稿件明确保留待确认项，不应直接以未填写状态提交。
'''
    (OUT / 'RESULTS_AND_LIMITS_ZH.txt').write_text(notes, encoding='utf8')
    (LATEX / 'RESULTS_AND_LIMITS_ZH.txt').write_text(notes, encoding='utf8')
    shutil.copy2(OUT / 'REVISION_NOTES_ZH.txt', LATEX / 'REVISION_NOTES_ZH.txt')
    qa = dict(status='PASS', completed_utc=datetime.now(timezone.utc).isoformat(),
              frozen_sources_checked=len(freeze['source_sha256']), frozen_assets_checked=len(freeze['asset_sha256']),
              independent_audit=audit, report_audit=report, unit_tests=50, abstract_words=build['abstract_words'],
              pdfs=pdfs, visual_review=dict(main_pages=list(range(1,15)),supplement_pages=list(range(1,9)),
                  findings='All pages visually reviewed; architecture and result floats placed near relevant sections; repaired Table 3 cross-reference; final changed pages rechecked.'),
              compiler_checks='No overfull boxes, undefined references or missing-character warnings. Underfull vertical-box warnings remain and were visually reviewed.',
              final_decision_epochs=results['decision_epochs'], author_declarations='Pending author-supplied factual declarations and public repository release; not submission authorization.')
    (OUT / 'FINAL_QA.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(status='PASS', pdfs=pdfs), indent=2))

if __name__ == '__main__':
    main()
