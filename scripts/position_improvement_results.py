"""Emphasize verified minimum outcomes without changing evaluated records."""
from pathlib import Path
from decimal import Decimal
import csv, hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/improvement_v4'
OUT = ROOT / 'output/improvement_v4_positioned'
DEST = OUT / 'Discover_IoT_LaTeX'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new, 1)

def emphasize_table(text, label, columns):
    """Select numeric minima from displayed means, retaining every numeric token."""
    pattern = r'\\begin\{table\}.*?\\end\{table\}'
    matches = [m for m in re.finditer(pattern, text, re.S) if r'\label{'+label+'}' in m.group()]
    assert len(matches) == 1, label
    match = matches[0]
    table = match.group()
    a, body = table.split(r'\midrule', 1)
    body, b = body.split(r'\bottomrule', 1)
    lines = body.splitlines()
    rows = [(i, s.split('&')) for i, s in enumerate(lines) if '&' in s]
    winners = {}
    for col in columns:
        values = {}
        for i, cells in rows:
            cell = cells[col].strip().removesuffix(r'\\').strip()
            value = re.match(r'[-+]?\d+(?:\.\d+)?', cell)
            assert value, (label, cell)
            values[i] = Decimal(value.group())
        minimum = min(values.values())
        winner_indices = [i for i, value in values.items() if value == minimum]
        winners[str(col)] = [dict(row=next(c for j, c in rows if j == i)[0].strip(), value=str(minimum)) for i in winner_indices]
        for i, cells in rows:
            if i in winner_indices:
                value = re.search(r'[-+]?\d+(?:\.\d+)?', cells[col])
                cells[col] = cells[col][:value.start()] + r'\textbf{' + value.group() + '}' + cells[col][value.end():]
    for i, cells in rows:
        lines[i] = '&'.join(cells)
    table = a + r'\midrule' + '\n'.join(lines) + '\n' + r'\bottomrule' + b
    note = ' Bold marks column minima.'
    table = table.replace(r'}\label{'+label+'}', note+r'}\label{'+label+'}', 1)
    new = text[:match.start()] + table + text[match.end():]
    assert re.findall(r'[-+]?\d+(?:\.\d+)?', text) == re.findall(r'[-+]?\d+(?:\.\d+)?', new), label
    return new, winners

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = BASE / 'Discover_IoT_LaTeX'
    DEST.mkdir(exist_ok=True)
    for p in source.rglob('*'):
        if p.is_file() and p.suffix in {'.tex','.bib','.bbl','.cls','.bst','.pdf','.svg','.png','.eps','.txt'}:
            q = DEST / p.relative_to(source)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)

    forecast = list(csv.DictReader((BASE / 'forecast_summary.csv').open(encoding='utf8')))
    control = list(csv.DictReader((BASE / 'control_summary.csv').open(encoding='utf8')))
    for surge in ['False', 'True']:
        f = [r for r in forecast if r['surge'] == surge]
        assert len(f) == 11 and min(f, key=lambda r:float(r['mae_mean']))['model'] == 'combined'
        c = [r for r in control if r['surge'] == surge and r['subset'] == 'full']
        assert len(c) == 6 and min(c, key=lambda r:float(r['wait_mean']))['method'] == 'rollout'
    fs = [r for r in forecast if r['surge'] == 'True']
    assert min(fs, key=lambda r:float(r['rmse_mean']))['model'] == 'residual'
    assert min(fs, key=lambda r:float(r['mae_h6_mean']))['model'] == 'residual'

    abstract = (
        'IoT-enabled destinations need demand forecasts that support executable resource allocation while accounting for service lost during transport. '
        'We evaluate a causal forecast-and-allocation framework in a disclosed synthetic scenic-area system with 24 nodes, four resource classes and FIFO queues. '
        'A training-derived daily profile and graph residual GRU are combined with Graph WaveNet; a transport-aware marginal rollout selects feasible resource transfers. '
        'Methods are frozen before evaluation in seven new random worlds under nominal and increased demand. '
        'Across eleven tested forecasting variants, the combination achieves the lowest mean absolute error (MAE) in both conditions: 2.2404 and 2.9339 visit events. '
        'Rollout also achieves the lowest mean accrued waiting among the six controllers in the full comparison: 22.67 and 75.46 min/request. '
        'Under nominal demand, MAE is 0.48\\% lower than Graph WaveNet and accrued waiting is 16.47\\% lower than six-step model predictive control (MPC) using the identical forecast. '
        'Against 12-step MPC on a prespecified matched subset, waiting is 24.60 versus 26.21 min/request. '
        'These gains accompany higher nominal relocation distance and do not uniformly reduce overload. '
        'Terminal backlog, solver status, uncached computation and reproducible records are reported. '
        'The findings establish observed advantages within the tested synthetic conditions and computational budgets; field effectiveness remains unverified.'
    )
    assert len(abstract.split()) < 250
    main_text = (DEST / 'main.tex').read_text(encoding='utf8')
    main_text, count = re.subn(r'\\abstract\{.*?\}\n\\keywords', lambda _: r'\abstract{'+abstract+'}\n'+r'\keywords', main_text, count=1, flags=re.S)
    assert count == 1
    main_text = replace_once(main_text, r'\setlength{\bibsep}{5pt}', r'\setlength{\bibsep}{2pt}')
    main_text = replace_once(main_text,
        'The combined predictor obtains mean MAE \\ForecastNominal{} under nominal demand, compared with \\ForecastGWN{} for the GWN ensemble, a reduction of \\ForecastGain\\%.',
        'The proposed combination achieves the lowest aggregate mean MAE among the eleven tested forecasting variants under both nominal and increased demand (Table~\\ref{tab:forecast}). Under nominal demand, its MAE is \\ForecastNominal{}, compared with \\ForecastGWN{} for the GWN ensemble, a reduction of \\ForecastGain\\%.')
    main_text = replace_once(main_text,
        'This study evaluates causal demand forecasting and transport-aware marginal service allocation in a reproducible synthetic IoT system. The selected forecast combination improves mean nominal MAE from \\ForecastGWN{} to \\ForecastNominal{} across seven new random worlds. \\ConclusionControl{}',
        'The evaluated framework achieves the lowest mean MAE among eleven forecasting variants and the lowest mean accrued waiting among six controllers in the full comparison, under both tested demand conditions. Across seven new synthetic worlds, the selected combination reduces nominal MAE from \\ForecastGWN{} for GWN to \\ForecastNominal{}. \\ConclusionControl{} These outcomes favor prediction accuracy and accrued waiting; relocation cost and overload retain explicit trade-offs.')
    main_text = main_text.replace(r'The paired mean difference is \ForecastDifference{} visit events', r'The paired mean difference is $\ForecastDifference{}$ visit events')
    (DEST / 'main.tex').write_text(main_text, encoding='utf8')
    macros = (DEST / 'results_macros.tex').read_text(encoding='utf8')
    macros = replace_once(macros,
        r'\newcommand{\ControlInterpretation}{The rollout obtains 22.67 min/request under nominal demand, versus 27.14 for six-step MPC with the identical forecast.',
        r'\newcommand{\ControlInterpretation}{Transport rollout achieves the lowest mean accrued waiting among the six controllers in the full nine-day-per-world comparison under both demand conditions. Under nominal demand, it obtains 22.67 min/request, versus 27.14 for six-step MPC with the identical forecast, a 16.47\% reduction.')
    macros = replace_once(macros,
        'On the prespecified three-day-per-world subset, rollout and 12-step MPC obtain',
        'Rollout retains the lowest mean waiting among the controllers displayed in Table~\\ref{tab:matched} on the prespecified common subset. On this three-day-per-world subset, rollout and 12-step MPC obtain')
    (DEST / 'results_macros.tex').write_text(macros, encoding='utf8')

    rankings = {}
    for file, label, cols in [
        ('tables/forecast_main.tex','tab:forecast',[1,2,3,4]),
        ('tables/control_main.tex','tab:control',[1,2,3,4]),
        ('tables/control_matched.tex','tab:matched',[2,3]),
        ('tables/timing.tex','tab:timing',[1,2]),
    ]:
        old = (DEST / file).read_text(encoding='utf8')
        new, rankings[label] = emphasize_table(old, label, cols)
        (DEST / file).write_text(new, encoding='utf8')
    supplement = (DEST / 'supplement.tex').read_text(encoding='utf8')
    for label, cols in [('s:forecast0',[1,2,3,4]),('s:forecast1',[1,2,3,4]),('s:wape',[1,2]),('s:service0',[1,2,3,4]),('s:service1',[1,2,3,4])]:
        supplement, rankings[label] = emphasize_table(supplement, label, cols)
    (DEST / 'supplement.tex').write_text(supplement, encoding='utf8')
    letter = (DEST / 'cover_letter.tex').read_text(encoding='utf8')
    letter = letter.replace('areas", as a Research article.', "areas'', as a Research article.")
    letter = replace_once(letter,
        'Seven new random worlds evaluate frozen methods under nominal and increased demand.',
        'Seven new random worlds evaluate frozen methods under nominal and increased demand. The combination attains the lowest mean MAE among eleven tested forecasting variants, and transport rollout the lowest mean accrued waiting among six controllers in the full comparison. Nominal waiting is 16.47\\% lower than six-step MPC with the identical forecast; the advantage persists in the prespecified horizon-matched subset. The reported gains are accompanied by explicit relocation and overload trade-offs.')
    (DEST / 'cover_letter.tex').write_text(letter, encoding='utf8')
    notes = '''本版只强化有证据支持的最低值表述，不重新实验、不改变任何实验记录。
摘要、结果段、结论和投稿信突出：本次11种预测变体中，两种负荷的平均MAE最低；全量6种调度方法中，两种负荷的累计等待/请求最低。
主文和补充材料中按列自动加粗最低的已显示均值，并列时一同标记。所有表格数字与原v4稿件完全一致。
高负荷RMSE及60分钟MAE的最低值属于单独残差GRU；高负荷过载率的最低值属于同预测器MPC-6；移动量最低为静态分配。这些最低值同样加粗，避免选择性标记。
既有原始实验、权重、冻结记录和复现包保持原状。原始证据包位于上一版 output/improvement_v4/ScenicIoT_Improved_Reproducibility.zip。
本版稿件适合强调“在已比较方法和既定合成条件中最低”，不支持宣称所有指标或真实景区全局最优。作者声明和公开代码地址仍待作者补齐。
'''
    (DEST / 'PRESENTATION_CHANGES_ZH.txt').write_text(notes, encoding='utf8')
    manifest = dict(source_results_sha256=digest(BASE/'results.json'),abstract_words=len(abstract.split()),
        numerical_tables_unchanged=True,highlighted_displayed_minima=rankings,
        source_files={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file() and p.suffix=='.tex'})
    (OUT / 'presentation_audit.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(abstract_words=manifest['abstract_words'],tables=len(rankings),destination=str(DEST)),ensure_ascii=False))

if __name__ == '__main__':
    main()
