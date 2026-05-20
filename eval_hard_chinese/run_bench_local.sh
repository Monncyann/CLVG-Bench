#!/usr/bin/env bash
set -euo pipefail

# 可按需修改 ARK_API_KEY 或直接在环境中提前设置

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# 示例：/path/to/python "${SCRIPT_DIR}/run_main.py" ...
# 运行主入口脚本
# --env: 使用的 env.json（定义 case/state/ground truth）
# --videos: 视频输出根目录（按 run_id/case_x 保存）
# --output: 评测报告输出路径
# --workers-gen: 生成并发数（越大越快，但更占资源）
# --workers-eval: 评测并发数（越大越快，但更占资源）
# --limit: 只跑前 N 个 case（用于快速测试）
python "${SCRIPT_DIR}/run_main.py" \
  --env "${SCRIPT_DIR}/data/env.json" \
  --videos "${SCRIPT_DIR}/results/videos" \
  --output "${SCRIPT_DIR}/results/env_eval_report_1-5" \
  --workers-gen 10 \
  --workers-eval 10 \
  --limit 50
