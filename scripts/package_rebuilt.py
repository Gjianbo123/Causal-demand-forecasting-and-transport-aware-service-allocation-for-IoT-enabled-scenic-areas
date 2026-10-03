"""Package executed evidence and source, excluding environments and QA renders."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/revision'

README='''# 新实验复现与文件说明

这是一套重新实现、实际执行的全合成实验。原稿的源代码与训练日志缺失，
因此本包不声称重现原 E-STGNN、原 ScenicFlow-Sim 或原论文数值。
新版 Word 的数值全部来自 runs/scenic_rebuild_v2，不与旧图表混用。

## 环境与复核

执行环境是 Windows、Python 3.11.5、PyTorch 2.7.1+cu118；依赖见
requirements-rebuilt.txt。GPU用于神经预测器训练，PPO及评估使用CPU。
先安装适合硬件的官方 PyTorch 2.7.1，再安装其余固定版本。
不同硬件、库版本或并行负载可能改变浮点和计时结果。

在解压根目录运行：

```text
python -m unittest discover -s tests -v
python scripts/run_rebuilt.py aggregate --out runs/scenic_rebuild_v2
python scripts/summarize_rebuilt.py
python scripts/plot_rebuilt_forecasts.py
```

这几步复核已有原始输出并重建表图，不重新训练。MANIFEST_ARCHIVE.sha256
记录包内文件哈希；重新运行会改变部分输出，核对时请使用原解压副本。

## 从头运行

必须使用新的、尚不存在的输出目录，避免覆盖已执行证据：

```text
python scripts/run_rebuilt.py prepare --config configs/scenic_rebuild.json --out runs/reproduction
python scripts/run_rebuilt.py parallel-train --out runs/reproduction --workers 4
python scripts/run_rebuilt.py parallel-evaluate --out runs/reproduction --workers 4
python scripts/evaluate_context.py --out runs/reproduction
python scripts/evaluate_context.py --out runs/reproduction --sensitivity
python scripts/run_rebuilt.py aggregate --out runs/reproduction
```

汇总与作图脚本的 RUN 常量默认指向随包正式 v2 输出，若分析新运行，需改为
runs/reproduction。重新训练比较请保留所有种子，不能按测试表现挑选模型。

## 关键文件

- iotexp/scenic.py：24节点全合成需求、资源调运和FIFO队列。
- iotexp/scenic_forecast.py：六类预测方法与时间划分。
- iotexp/engine.py、fusion.py、telemetry.py：PPO、融合与因果传感处理。
- runs/scenic_rebuild_v2/config.json、manifest.json：锁定配置与来源。
- assets：完整生成数据、预测模型和预测值。
- checkpoints：50个策略的最佳验证检查点、完整训练日志及元数据。
- episodes.csv、per_seed.csv、paired_tests.json：7,830个策略测试的原始/汇总结果。
- traces：每个测试日的等待样本和动作审计；部分指定案例含完整状态轨迹。
- context：729个启发式对照、1,080个运输敏感性实验及轨迹。
- output/revision：新版论文、投稿信、图源、结果表和审计说明。
- preflight_v1：仅训练与验证的早期预检配置/日志，未纳入论文结果。

“visit event”是节点事件计数，“service request”是可重复的服务请求，
二者都不是唯一游客人数。主等待指标包括闭园时尚未服务请求已发生的等待，
不将此后的等待外推。源代码某些继承列名含 visitor/overload，
在新实验中准确含义是服务请求积压，不能解释为实地人群密度或安全认证。

## 文稿再生成

scripts/build_submission.py 使用 python-docx、lxml、latex2mathml 和安装的
Microsoft Office MathML XSL；它是本机文稿构建工具，不是科学实验依赖。
交付的DOCX可直接在Word编辑，9个公式为可编辑原生公式。最终版已用Word
导出并逐页检查。资金、利益冲突及贡献声明仍须作者确认，尚未自动提交期刊。
'''


def main():
    r=json.loads((OUT/'manuscript_results.json').read_text(encoding='utf-8'))
    assert r['total_policy_episodes']==7830 and r['total_baseline_episodes']==729
    assert len(r['sensitivity'])==40
    (OUT/'README_REPRODUCE_ZH.md').write_text(README,encoding='utf-8')
    entries={}
    for folder in ('iotexp','configs','tests','docs','scripts','runs/scenic_rebuild_v2','output/revision'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.zip','.pyc') or p.name.endswith('.zip.sha256'):
                continue
            entries[p.relative_to(ROOT).as_posix()]=p
    for name in ('requirements-rebuilt.txt','pyproject.toml'):
        entries[name]=ROOT/name
    entries['README_REPRODUCE_ZH.md']=OUT/'README_REPRODUCE_ZH.md'
    old=ROOT/'runs/scenic_rebuild_v1'
    for p in old.rglob('*'):
        if p.is_file() and (p.name in ('config.json','manifest.json','training.jsonl','metadata.json') or p.suffix=='.log'):
            entries['preflight_v1/'+p.relative_to(old).as_posix()]=p
    # Staging all selected paths first avoids adding the archive to itself.
    target=OUT/'ScenicIoT_Reproducibility.zip'
    hashes=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for name,p in sorted(entries.items()):
            data=p.read_bytes()
            hashes.append(hashlib.sha256(data).hexdigest()+'  '+name)
            archive.writestr(name,data)
        archive.writestr('MANIFEST_ARCHIVE.sha256','\n'.join(hashes)+'\n')
    with zipfile.ZipFile(target) as archive:
        bad=archive.testzip()
        assert bad is None,bad
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    (OUT/'ScenicIoT_Reproducibility.zip.sha256').write_text(digest+'  '+target.name+'\n',encoding='ascii')
    print(json.dumps({'archive':str(target),'files':len(entries)+1,'bytes':target.stat().st_size,'sha256':digest}))


if __name__=='__main__': main()
