import argparse  
import logging  
import sys  
from pathlib import Path 
import os  
from dotenv import load_dotenv  

EVAL_DIR = Path(__file__).resolve().parent  
PROJECT_ROOT = EVAL_DIR.parent 
PIPELINE_DIR = EVAL_DIR / "pipeline"  
load_dotenv()  
API_KEY = os.getenv("ARK_API_KEY")  
if not API_KEY:  
    raise RuntimeError("ARK_API_KEY is not set. Please export ARK_API_KEY before running.")
if str(EVAL_DIR) not in sys.path:  
    sys.path.insert(0, str(EVAL_DIR))
if str(PIPELINE_DIR) not in sys.path:  
    sys.path.insert(0, str(PIPELINE_DIR))

from generate_pipeline import GenerationConfig, run_generation_with_inline_eval  # 生成+评测模块
from judge_pipeline import EvalConfig  # 评测配置


def _init_logging() -> None:
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )


def main() -> None:
   
    _init_logging()  # 初始化日志
    parser = argparse.ArgumentParser(description="Benchmark runner (generate + eval)")  
    parser.add_argument("--env", dest="env_path", default="./eval/data/test.json")  # env.json 路径
    parser.add_argument("--videos", dest="videos_root", default="./eval/results/videos")  # 视频输出根目录
    parser.add_argument("--output", dest="output_path", default="./eval/results/eval_report.json")  # 评测报告路径
    parser.add_argument("--workers-gen", dest="workers_gen", type=int, default=10)  # 生成并发数
    parser.add_argument("--workers-eval", dest="workers_eval", type=int, default=8)  # 评测并发数
    parser.add_argument("--model-t2v", dest="model_t2v", default="doubao-seedance-1-5-pro-251215")  # T2V 模型 doubao-seedance-2-0-260128  doubao-seedance-1-5-pro-251215
    parser.add_argument("--model-vlm-gen", dest="model_vlm_gen", default="doubao-seed-2-0-pro-260215")  # 生成 VLM 模型
    parser.add_argument("--model-vlm-eval", dest="model_vlm_eval", default="doubao-seed-2-0-pro-260215")  # 评测 VLM 模型
    parser.add_argument("--ratio", dest="ratio", default="16:9")  # 视频比例
    parser.add_argument("--duration", dest="duration", type=int, default=7)  # 视频时长
    parser.add_argument("--fps", dest="fps", type=int, default=1)  # 评测 FPS
    parser.add_argument("--poll", dest="poll_interval", type=int, default=10)  # 轮询间隔
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")  # dry-run 开关
    parser.add_argument("--limit", dest="limit", type=int, default=0)  # 限制 case 数量
    args = parser.parse_args()

    logging.info("[CONFIG] env=%s", args.env_path)
    logging.info("[CONFIG] videos=%s output=%s", args.videos_root, args.output_path)  
    logging.info("[CONFIG] dry_run=%s limit=%d", args.dry_run, args.limit)

    gen_config = GenerationConfig(  # 生成配置
        env_path=args.env_path,
        output_dir=args.videos_root,
        max_workers=args.workers_gen,
        model_t2v=args.model_t2v,
        model_vlm=args.model_vlm_gen,
        ratio=args.ratio,
        duration=args.duration,
        fps=args.fps,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
    )

    eval_config = EvalConfig(
        env_path=args.env_path,
        videos_root=args.videos_root,
        output_path=args.output_path,
        max_workers=args.workers_eval,
        model_vlm=args.model_vlm_eval,
        fps=args.fps,
        limit=args.limit,
    )
    run_generation_with_inline_eval(gen_config, eval_config)


if __name__ == "__main__":  
    main()  
