# 景区资源调度：补充实验代码

本代码包对应两个问题：**资源专属时域注意力是否优于直接拼接预测**，以及**客流观测缺失、噪声、上传延迟怎样影响调度**。

包内已经实现可运行的实验流程，包括五种融合模块、PPO 训练、因果观测接口、九种测试条件、配对统计、原始记录和绘图。原完整版附有实际运行的演示结果。本轻量源码版保留全部代码、配置与中文说明，省略检查点和结果文件；运行命令后会自动生成结果。

**当前未提供原 ScenicFlow-Sim、E-STGNN 代码及检查点。包内演示使用新写的六节点合成排队环境和岭回归预测器，不能把演示数值填进原论文，也不表示已复现原论文结果。** 原模型接入说明在 `docs/INTEGRATION_ZH.md`。演示环境的假设见 `docs/PROTOCOL_ZH.md`。

## 1. 安装与运行

建议 Python 3.12。解压后进入包含 `iotexp` 文件夹的目录。Windows PowerShell 示例：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m iotexp run --config configs/demo_quick.json --out runs/my_demo
```

Linux/macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m iotexp run --config configs/demo_quick.json --out runs/my_demo
```

这组默认配置只用于验证运行流程：2 个训练种子、每个策略 192 次环境交互。无需 GPU。它不是论文训练预算，也不用于判断哪个方法更好。程序拒绝覆盖已有输出目录；再次运行时请换一个 `--out` 路径。

如只安装 CPU 版 PyTorch，可先使用 PyTorch 官方 CPU 包源，再安装其余依赖。`requirements.txt` 固定的是本代码验证环境，不是原论文的软件版本。实际运行的软件信息会保存到 `manifest.json`。

也可以分两步运行：

```bash
python -m iotexp train --config configs/demo_quick.json --out runs/train_first
python -m iotexp evaluate --out runs/train_first
```

第二步读取保存的配置、训练统计和预测器，核对其哈希后再评测，不会重训或使用测试集选择检查点。当前没有中途恢复训练或跳过部分失败评测的功能；失败后保留日志，并使用新目录重新运行。

本地运行完成后，可重新汇总对应目录：

```bash
python -m iotexp summarize --out runs/my_demo
python -m iotexp matrix
```

## 2. 已实现的实验

| 编号 | 融合方法 | 实现 |
|---|---|---|
| F0 | 当前观测 PPO | 策略输入不含未来预测；修复器也不访问预测 |
| F1 | 预测直接拼接 | 四个时域嵌入拼接，再投影到共同维度 |
| F2 | 均匀融合 | 四个时域嵌入等权平均 |
| F3 | 共享时域注意力 | 各资源类型共享一个查询向量 |
| F4 | 资源专属时域注意力 | 每类资源拥有独立查询向量 |

F0–F4 均从头训练，使用同一个冻结预测器、相同状态特征、PPO 骨干、交互预算和验证规则。策略骨干与预测嵌入的初始化分别使用固定随机流，避免融合模块参数量变化打乱骨干初始化。程序会报告各模型实际可训练参数量；参数量差异明显时，还应增加容量匹配的 F1 对照。

观测测试使用冻结的 F0、F1、F4，包含正常观测、缺失率 10%/20%/30%、噪声尺度 0.05/0.10/0.20、上传延迟 1/2 个决策间隔。三类扰动分别测试。没有默认加入训练时扰动或组合攻击条件。

## 3. 输出文件

| 文件 | 内容 |
|---|---|
| `config.json` | 该次运行的完整配置 |
| `manifest.json` | 数据属性、软件环境、代码/配置/资产哈希、统计口径 |
| `assets/` | 训练集统计、冻结预测器 |
| `checkpoints/` | 按验证集选择的检查点、训练日志与参数量 |
| `episodes.csv` | 每个训练种子、测试轨迹、扰动重复的指标 |
| `traces/*.npz` | 原始/候选/执行动作、观测与真值、事件/接收时间、缺失掩码、测量年龄、注意力和等待时间 |
| `per_seed.csv` | 先在同一训练种子内聚合后的指标 |
| `summary.csv` | 独立训练种子间的均值和样本标准差 |
| `paired_differences.csv` | F4 相对 F1/F2/F3 的配对等待时间差 |
| `paired_tests.json` | 配对检验、计算方式和多重比较校正 |
| `robustness.png/.pdf` | 三类观测扰动曲线；阴影为训练种子间样本标准差 |

演示结果每行都带 `is_demo=True`，图中也明确标注。两种子运行只输出描述统计，不计算正式 p 值。代码不会预设 F4 必须最好，也不会修改不显著或不利的结果。

## 4. 接入原项目时优先处理的事项

1. 保留原始仿真动力学、资源动作语义与预测器；接入 `HorizonFusion`、`TelemetryGateway` 和统计模块。
2. 对 F0–F4 采用同一个不读取预测的确定性修复规则，并在修改后的共同设置下重新训练。不要沿用旧稿的 10.9/12.4 分钟作为这次对照结果。
3. 训练集统计只能从原训练划分计算。冻结同一 E-STGNN 检查点，所有测试条件使用同一预处理。
4. 确认原动作空间、PPO 原始动作概率、运输中的库存、容量限制、奖励和仿真终止语义。
5. `configs/original_project_template.json` 给出十种子实验模板；其中未核实字段不能当作原实现事实。适配器未完成时，程序会明确停止。

详细步骤、张量维度和代码片段见 `docs/INTEGRATION_ZH.md`。

## 5. 代码位置

```text
iotexp/fusion.py       融合模块与参考策略
iotexp/telemetry.py    训练统计、缺失/噪声/延迟、因果窗口
iotexp/contracts.py   原项目接口定义
iotexp/engine.py      PPO、配对训练与评测、原始日志
iotexp/metrics.py     动作可行性、游客超限、等待和预测误差
iotexp/analysis.py    种子内聚合、配对检验、Holm 校正、绘图
iotexp/demo.py        独立合成演示环境及冻结岭回归预测器
tests/                因果性、资源守恒、概率记录、统计等检查
```

本包没有外部 API 调用、付费服务或实景数据依赖。需要原代码后才能完成原论文实验适配。
