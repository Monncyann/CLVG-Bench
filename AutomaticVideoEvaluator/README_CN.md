# Automatic Video Evaluator（AVE）

简体中文 | [English](README.md)

本项目是 **Adaptive Video Evaluator（AVE）** 的自包含实现，提供评估器训练与评测代码、
三个任务的配置、初始 prompt、可复现的数据集划分以及公开数据集下载脚本。
项目不会从上层 `VideoFactory` 仓库导入任何内容。

## 支持的实验

| `algorithm` | Prompt 选择方式 | Loss |
|---|---|---|
| `textgrad` | 验证集上最优的 prompt | 二值 Loss |
| `textgrad_semantic_loss` | 验证集上最优的 prompt | Semantic Loss |
| `gepa` | 验证集 Pareto 前沿采样 | 二值 Loss |
| `gepa_semantic_loss` | 验证集 Pareto 前沿采样 | Semantic Loss |

也可以直接运行冷启动评估器，不进行 prompt 优化。

TextGrad 始终从验证集得分最高的 prompt 继续优化。GEPA 根据逐样本得分从验证集
Pareto 前沿中采样，并且通常只在新 prompt 的训练 mini-batch 得分不下降时接纳该
prompt。每个被接纳的 prompt 都会在完整验证集上评测，然后才会进入候选 prompt 表。
与原始实验程序一致，测试集会在训练开始前、每完成三次完整验证后，以及最终选出
最佳 prompt 后进行评测。

## 安装

需要 Python 3.10 或更高版本。

```bash
cd AutomaticVideoEvaluator
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

AVE 使用支持视觉输入、兼容 OpenAI Chat Completions 接口的服务：

```bash
export AVE_API_KEY="your-key"
```

`AVE_BASE_URL` 默认使用项目中 Seed 配置对应的火山引擎方舟接口。只有在使用其他
OpenAI-compatible 服务时才需要显式设置该变量。

项目自带配置使用 Seed 2.0 Lite 作为视频 Judge，使用 Seed 2.0 Pro 作为 prompt
Optimizer 和 Semantic Matcher。可以在 `configs/*.json` 中修改模型名称和 token
价格。

## 下载数据集

公开数据集托管在 `JianhuiWei/AVE_data`。

```bash
python data/download.py
```

下载单个任务，或者校验已经下载的数据：

```bash
python data/download.py --task perception
python data/download.py --verify-only
```

下载脚本会将数据放入本项目的 `data/` 目录，并使用发布的 SHA-256 元数据进行完整性
校验。

## 数据集概览

| 任务 | Train / val / test | 数据行数 | 不重复视频数 |
|---|---:|---:|---:|
| Abnormality | 154 / 148 / 148 | 450 | 353 |
| Perception | 60 / 60 / 63 | 183 | 183 |
| Prompt following | 39 / 39 / 39 | 117 | 117 |

Prompt following 另外包含 165 张参考图和 10 个参考视频。你也可以使用
`scripts/create_splits.py` 按指定比例和随机种子自行划分数据集。

每个任务都包含：

- `all.json`：每个生成视频对应一条规范记录，没有重复 ID 或视频路径；
- `train.json`、`val.json`、`test.json`：可以直接运行的评估器数据划分；
- `SPLIT_PLAN.json`：从 `all.json` 精确恢复发布版划分的映射；
- `MANIFEST.json`：媒体和数据划分的校验和。

`label=0` 表示没有标注到 weakness；非零值表示存在 weakness。人工编写的
`feedback.abnormality` 是权威监督信号。完整字段和数据集说明请参阅
[`data/README.md`](data/README.md)。

校验下载的数据：

```bash
python scripts/validate_dataset.py --data-dir data/abnormality
python scripts/validate_dataset.py --data-dir data/perception
python scripts/validate_dataset.py --data-dir data/prompt_following
```

加入 `--checksums` 可以对全部媒体文件执行完整哈希校验。

## 运行 AVE

训练默认的 GEPA + Semantic Loss 评估器：

```bash
python scripts/train.py --config configs/abnormality.json
python scripts/train.py --config configs/perception.json
python scripts/train.py --config configs/prompt_following.json
```

无需修改配置文件即可运行四种 baseline 中的任意一种：

```bash
python scripts/train.py --config configs/abnormality.json --algorithm textgrad
python scripts/train.py --config configs/abnormality.json --algorithm textgrad_semantic_loss
python scripts/train.py --config configs/abnormality.json --algorithm gepa
python scripts/train.py --config configs/abnormality.json --algorithm gepa_semantic_loss
```

提供 `--algorithm` 后，默认输出目录为 `outputs/<task>_<algorithm>`；也可以使用
`--output-dir` 指定其他目录。每个输出目录会绑定一份完整配置，防止不同 baseline
的实验结果相互混合。

训练默认遵循原始实验流程：

1. 使用冷启动 prompt 在测试集和验证集上评测；
2. 使用训练随机种子 47 采样 mini-batch；
3. 当 mini-batch 已经取得满分时跳过 prompt 修改；
4. GEPA 在同一 mini-batch 上比较修改前后的结果；
5. 连续十次回退后，强制让新 prompt 进入完整验证；
6. 每完成三次完整验证，在测试集上评测一次；
7. 选择验证集上表现最佳的 prompt，并执行最终测试。

数据划分独立使用随机种子 42。评测最多并发处理 64 个数据项；单个数据项内部的五次
投票顺序执行，因此在形成不可逆多数票后可以提前停止。视频帧按照配置的采样率在完整
时长上均匀采样，默认最多 32 帧。每个 API 请求的超时时间为 240 秒，Optimizer 的
视觉输入上限为 45 MiB。

已经完成的数据项会增量保存在 `checkpoints/` 下。如果训练被中断，重新运行完全相同
的命令即可复用已经完成的评测和 prompt proposal。修改配置后必须使用新的
`output_dir`。

不训练，直接评测冷启动 prompt：

```bash
python scripts/evaluate.py \
  --config configs/abnormality.json \
  --prompt prompts/abnormality.json \
  --split test \
  --output outputs/cold_start_test.json
```

评测训练得到的 prompt：

```bash
python scripts/evaluate.py \
  --config configs/abnormality.json \
  --algorithm gepa_semantic_loss \
  --prompt outputs/abnormality_gepa_semantic_loss/best_prompt.json \
  --split test \
  --output outputs/abnormality_gepa_semantic_loss/test_repeat.json
```

Semantic Loss 使用模型对齐预测 weakness 集合与人工 weakness 集合，输出
TP/TN/FP/FN 计数，并先在每个样本内归一化，再进行聚合。集合为空的情况直接计算，
无需调用 Matcher。Judge 默认使用五次多数投票，并在多数结果已经不可逆时提前停止。

## 恢复或创建数据划分

使用 `all.json` 和 `SPLIT_PLAN.json` 精确恢复发布版划分：

```bash
python scripts/create_splits.py \
  --data-dir data/abnormality \
  --output-dir /tmp/ave-abnormality-rebuilt \
  --strategy released
```

创建新的、可确定性复现的 1:1:1 划分：

```bash
python scripts/create_splits.py \
  --data-dir data/prompt_following \
  --output-dir /tmp/ave-prompt-following-new \
  --strategy stratified \
  --seed 42 \
  --ratios 1 1 1
```

使用 `--strategy balanced --total-size N --positive-ratio 0.5` 可以创建类别平衡的
新划分。默认通过硬链接准备媒体文件，因此新目录可以直接运行，并且不会重复占用文件
空间。跨文件系统时使用 `--media-mode copy`；只需要 JSON 进行审计时使用
`--media-mode none`。

## 输出文件

训练会在配置的 `output_dir` 下生成以下文件：

```text
prompt_000.json              初始 prompt
prompt_NNN_proposal.json     每次提出的新 prompt
prompt_NNN.json              通过验证后接纳的 prompt
candidate_table.json         验证分数、继承关系和 Pareto 前沿
best_prompt.json             验证集上表现最佳的 prompt
history.json                 优化轨迹
initial_test_results.json    冷启动测试预测
test_step_NNN.json           周期性测试预测
test_results.json            最终冻结 prompt 的测试预测
run_summary.json             指标、rollout、token 和预估成本
run_metadata.json            防止混用实验结果的配置标识
training_state.json          最新的重放/恢复状态
checkpoints/                 按阶段增量保存的逐样本结果
```

默认停止条件是最多修改 prompt 70 次、执行 4,500 个数据项 rollout，或者达到预估
30 USD 的 API 预算。初始测试和周期性测试都会计入历史协议的预算及 rollout 总量。
为了复现旧版停止规则，所有 token 都按照配置的 Judge 费率计入该预算。
`run_summary.json` 还会分别按照 Judge、Optimizer 和 Matcher 的费率报告成本估算，
该数值通常更接近实际账单。停止条件只在完整迭代之间检查，因此最后一次迭代可能会使
实际用量略微超过配置限制。

## 项目结构

```text
AutomaticVideoEvaluator/
├── ave/                 评估器和 Optimizer 的核心实现
├── configs/             三个可运行的任务配置
├── prompts/             冷启动评估器 prompt
├── scripts/             训练、评测、划分和校验入口
├── data/                下载脚本和下载后的数据集
├── tests/               离线回归测试
├── pyproject.toml
└── requirements.txt
```

运行测试：

```bash
python -m pip install -e '.[dev]'
pytest -q
```
