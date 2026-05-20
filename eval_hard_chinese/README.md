# Eval Pipeline

目录结构：

- `run_main.py`：主入口（只负责参数解析、日志初始化、调用生成/评测）
- `run_bench_local.sh`：本地快速运行脚本（封装常用参数）
- `pipeline/generate_pipeline.py`：生成逻辑（VLM 生成 prompt + T2V 生成视频）
- `pipeline/judge_pipeline.py`：评测逻辑（逐轮对齐判断并输出报告）
- `pipeline/system_prompt.py`：系统提示词（生成/评测提示词模板）
- `data/env.json`：样例/正式 case 定义
- `data/test.json`：小规模测试 case
- `results/`：运行输出目录

## 本地运行

```bash
bash /eval_hard/run_bench_local.sh
```

## 直接运行主入口

```bash
python /eval_hard/run_main.py \
  --env /eval_hard/data/env.json \
  --videos /eval_hard/results/videos \
  --output /eval_hard/results/eval_report.json \
  --workers-gen 1 \
  --workers-eval 1 \
  --limit 1
```

### 参数说明

- `--env`：`env.json` 路径，定义 case/state/ground truth。
- `--videos`：生成视频输出根目录，按 `run_id/case_x/turn_y.mp4` 保存。
- `--output`：评测报告输出路径（JSON）。
- `--workers-gen`：生成并发数，越大速度越快但更占资源。
- `--workers-eval`：评测并发数，越大速度越快但更占资源。
- `--model-t2v`：文生视频模型名（T2V）。
- `--model-vlm-gen`：用于生成 prompt 的 VLM 模型名。
- `--model-vlm-eval`：用于评测的 VLM 模型名。
- `--ratio`：视频比例（如 `16:9`）。
- `--duration`：单段视频时长（秒）。
- `--fps`：评测/输入视频帧率（用于 VLM 输入）。
- `--poll`：轮询生成任务状态的间隔（秒）。
- `--dry-run`：仅模拟执行，不调用真实模型。
- `--limit`：只跑前 N 个 case（用于快速测试）。

运行前请确保环境变量 `ARK_API_KEY` 已设置。
