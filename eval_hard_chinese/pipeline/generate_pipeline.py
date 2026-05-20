import json  # 读取/序列化 JSON
import logging  # 统一日志输出
import os  # 读取环境变量与路径
import time  # 轮询等待
import urllib.request  # 下载视频
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from volcenginesdkarkruntime import Ark

from judge_pipeline import EvalConfig, evaluate_video, summarize_results  # inline 评测


@dataclass
class GenerationConfig:
    """生成阶段配置参数。"""
    env_path: str  # env.json 路径
    output_dir: str  # 输出根目录（按 run_id/case_x 组织）
    max_workers: int  # 并发生成线程数
    model_t2v: str  # 文生视频模型名
    model_vlm: str  # 生成提示词的 VLM 模型名
    ratio: str  # 画面比例（如 16:9）
    duration: int  # 视频时长（秒）
    fps: int  # 输入视频帧率（用于 VLM 输入）
    poll_interval: int  # 轮询生成任务状态的间隔（秒）
    dry_run: bool  # 是否仅模拟执行


def _get_api_key() -> str:
    """从环境变量获取 ARK_API_KEY。"""
    api_key = os.environ.get("ARK_API_KEY")  # 读取 Key
    if not api_key:  # 未设置则报错
        raise RuntimeError("ARK_API_KEY is not set. Please export ARK_API_KEY before running.")
    return api_key  # 返回 Key


def _download_output_video(url: str, output_path: str) -> None:
    """下载视频到指定路径。"""
    try:
        urllib.request.urlretrieve(url, output_path)  # 直接下载
    except Exception as e:
        raise RuntimeError(f"Download failed: {url} -> {output_path} : {e}")


def _load_env(path: str) -> List[Dict[str, Any]]:
    """加载 env.json 中的 case 列表。"""
    with open(path, "r", encoding="utf-8") as f:  # 打开文件
        return json.load(f)  # 解析 JSON


def _build_state_map(states: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """构建 state_id -> state 的索引表。"""
    return {s["state_id"]: s for s in states}


def _select_transition(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """选择当前 state 的下一条 transition（默认取第一条）。"""
    transitions = state.get("transitions") or []  # 获取 transition 列表
    if not transitions:  # 无 transition
        return None
    return transitions[0]  # 取第一条


def clean_text_content(text):
    """清洗 JSON 中的换行符、多余空格和 Markdown 符号"""
    if not text:
        return ""
    # 1. 将所有换行符替换为空格
    text = text.replace('\n', ' ').replace('\r', ' ')
    # 2. 去掉开头可能存在的列表符号 '-' 或 '*' 
    text = re.sub(r'^[-\*\s]+', '', text)
    # 3. 将多个连续空格压缩为一个
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _build_structured_prompt(
    role_text: str,
    history_prompts: List[str],
    history_feedbacks: List[str],
    turn_id: int,
) -> str:
    """构造 video model 的结构化输入文本。"""
    payload = {
        "role": role_text,
        "history_prompts": history_prompts,
        "feedbacks": history_feedbacks,
        "turn_id": turn_id,
        "instruction": "你推理并输出下一个视频",
    }
    return json.dumps(payload, ensure_ascii=False)


def _run_t2v(
    client: Ark,
    prompt_text: str,
    ratio: str,
    duration: int,
    poll_interval: int,
    save_path: str,
    model_t2v: str,
    dry_run: bool,
) -> str:
    """调用 T2V 生成视频并等待完成。"""
    if dry_run:  # dry-run 直接生成占位文件
        Path(save_path).write_text("DRY_RUN_VIDEO")
        return save_path

    create_result = client.content_generation.tasks.create(  # 提交生成任务
        model=model_t2v,
        content=[{"type": "text", "text": prompt_text}],
        ratio=ratio,
        duration=duration,
        watermark=False,
    )
    logging.info("[GEN][T2V_PROMPT_LEN] %d", len(prompt_text))  # 记录 prompt 长度

    task_id = create_result.id  # 任务 ID
    while True:  # 轮询任务状态
        get_result = client.content_generation.tasks.get(task_id=task_id)  # 查询任务
        status = get_result.status  # 当前状态
        if status == "succeeded":
            output_video_url = None
            if hasattr(get_result, "content") and get_result.content:
                task_content = get_result.content
                if hasattr(task_content, "video_url"):
                    output_video_url = task_content.video_url
                elif isinstance(task_content, dict) and "video_url" in task_content:
                    output_video_url = task_content["video_url"]
            if not output_video_url:
                raise RuntimeError("任务成功但未返回视频 URL")
            _download_output_video(output_video_url, save_path)  # 下载视频
            return save_path  # 返回路径
        if status == "failed":
            raise RuntimeError(f"T2V 任务失败: {getattr(get_result, 'error', None) or get_result}")
        time.sleep(poll_interval)  # 等待下一次轮询


def _process_case(case: Dict[str, Any], config: GenerationConfig, run_id: str) -> None:
    """生成单个 case 的所有 turn 视频。"""
    case_id = case.get("id", "unknown")  # case id
    case_dir = Path(config.output_dir) / run_id / f"case_{case_id}"  # 输出目录
    case_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
    logging.info("[GEN] case %s -> %s", case_id, case_dir)  # 日志

    role_text = case.get("role", "")  # 角色设定文本
    state_map = _build_state_map(case.get("states", []))  # state 索引
    current_state_id = case.get("initial_state_id")  # 初始 state_id

    if not current_state_id or current_state_id not in state_map:
        raise ValueError(f"Case {case_id} 缺少有效 initial_state_id")

    client = Ark(  # 创建 Ark 客户端
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=_get_api_key(),
    )

    history_prompts: List[str] = []  # 历史视频生成 prompt 列表
    history_feedbacks: List[str] = []  # 历史反馈列表

    turn_idx = 1  # 当前轮次
    while True:  # 逐轮生成
        state = state_map[current_state_id]  # 当前 state
        transition = _select_transition(state)  # 当前 transition
        env_feedback = transition.get("env_feedback") if transition else ""  # 用户反馈

        if turn_idx == 1:  # turn1 直接用 ground truth prompt
            prompt_text = state.get("video_gen_prompt", "")  # 取 prompt
            logging.info("[GEN][TURN1_GT_PROMPT] %s", prompt_text)  # 打印 prompt
        else:
            prompt_text = _build_structured_prompt(
                role_text=role_text,
                history_prompts=history_prompts,
                history_feedbacks=history_feedbacks,
                turn_id=turn_idx,
            )

        # Prompt Cleaning
        prompt_text = clean_text_content(prompt_text)

        video_path = str(case_dir / f"turn_{turn_idx}.mp4")  # 本轮输出路径
        logging.info("[GEN] case %s turn %d -> %s", case_id, turn_idx, video_path)  # 日志
        try:
            _run_t2v(
                client=client,
                prompt_text=prompt_text,
                ratio=config.ratio,
                duration=config.duration,
                poll_interval=config.poll_interval,
                save_path=video_path,
                model_t2v=config.model_t2v,
                dry_run=config.dry_run,
            )
        except Exception as exc:
            if turn_idx == 2:
                logging.error("[GEN][TURN2_ABORT] T2V failed; stop case %s: %s", case_id, exc)
                break
            raise

        history_prompts.append(prompt_text)  # 记录历史 prompt
        if env_feedback:  # 若有反馈
            history_feedbacks.append(env_feedback)  # 记录反馈

        if not transition or not transition.get("next_state_id"):  # 无下一轮
            break
        current_state_id = transition["next_state_id"]  # 更新 state
        turn_idx += 1  # 轮次 +1


def run_generation(config: GenerationConfig) -> str:
    """批量生成所有 case，返回 run_id。"""
    cases = _load_env(config.env_path)  # 加载 case 列表
    run_id = time.strftime("%Y%m%d_%H%M%S")  # 生成 run_id
    logging.info("[GEN] run_id=%s cases=%d output=%s", run_id, len(cases), config.output_dir)  # 日志

    from concurrent.futures import ThreadPoolExecutor, as_completed  # 并发工具

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:  # 线程池
        futures = [executor.submit(_process_case, case, config, run_id) for case in cases]  # 提交任务
        for future in as_completed(futures):  # 等待任务完成
            future.result()  # 触发异常抛出

    return run_id


def _process_case_inline_eval(
    case: Dict[str, Any],
    gen_config: GenerationConfig,
    eval_config: EvalConfig,
    run_id: str,
    jsonl_handle,
) -> Dict[str, Any]:
    """生成单个 case，并在 turn2+ 生成后立即评测；失败仅停止该 case。"""
    case_id = case.get("id", "unknown")
    case_dir = Path(gen_config.output_dir) / run_id / f"case_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    logging.info("[GEN] case %s -> %s", case_id, case_dir)

    role_text = case.get("role", "")
    state_map = _build_state_map(case.get("states", []))
    current_state_id = case.get("initial_state_id")
    if not current_state_id or current_state_id not in state_map:
        # 这里记录错误并返回，不抛出异常，以免影响整体任务
        error_msg = f"Case {case_id} 缺少有效 initial_state_id"
        logging.error(error_msg)
        return {"case_id": case_id, "pass": False, "reason": error_msg}

    gen_client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=_get_api_key(),
    )
    eval_client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=_get_api_key(),
    )

    history_prompts: List[str] = []
    history_feedbacks: List[str] = []
    turn_results: List[Dict[str, Any]] = []

    turn_idx = 1
    MAX_RETRIES = 1  # 最大重试次数
    
    while True:
        state = state_map[current_state_id]
        transition = _select_transition(state)
        env_feedback = transition.get("env_feedback") if transition else ""

        if turn_idx == 1:
            prompt_text = state.get("video_gen_prompt", "")
            logging.info("[GEN][TURN1_GT_PROMPT] %s", prompt_text)
        else:
            prompt_text = _build_structured_prompt(
                role_text=role_text,
                history_prompts=history_prompts,
                history_feedbacks=history_feedbacks,
                turn_id=turn_idx,
            )
        
        # 1. Prompt 清洗
        prompt_text = clean_text_content(prompt_text)

        video_path = str(case_dir / f"turn_{turn_idx}.mp4")
        logging.info("[GEN] case %s turn %d -> %s", case_id, turn_idx, video_path)
        
        # 2. 增加重试机制
        retry_count = 0
        success = False
        last_error = None
        
        while retry_count <= MAX_RETRIES:
            try:
                _run_t2v(
                    client=gen_client,
                    prompt_text=prompt_text,
                    ratio=gen_config.ratio,
                    duration=gen_config.duration,
                    poll_interval=gen_config.poll_interval,
                    save_path=video_path,
                    model_t2v=gen_config.model_t2v,
                    dry_run=gen_config.dry_run,
                )
                success = True
                break
            except Exception as exc:
                last_error = exc
                logging.warning("[GEN] case %s turn %d T2V failed (retry %d/%d): %s", case_id, turn_idx, retry_count, MAX_RETRIES, exc)
                
                # 如果是 InvalidParameter，尝试进一步截断或简化 prompt
                if "InvalidParameter" in str(exc) or "Invalid content.text" in str(exc):
                    logging.info("[GEN] Attempting to truncate prompt due to InvalidParameter...")
                    if len(prompt_text) > 1000:
                        prompt_text = prompt_text[:1000] # 简单截断
                    elif len(prompt_text) > 500:
                        prompt_text = prompt_text[:500]
                    else:
                        # 已经很短了还报错，可能是非法字符，再次强力清洗
                        prompt_text = re.sub(r'[^\w\s,.:"，。；？！!?-]', '', prompt_text)
                
                retry_count += 1
                time.sleep(2) # 稍作等待

        if not success:
            # 最终失败，记录日志并中止当前 case，但不抛出异常中断整个程序
            logging.error("[GEN][TURN_ABORT] T2V final failed; stop case %s: %s", case_id, last_error)
            record = {"case_id": case_id, "turn": turn_idx, "pass": False, "reason": str(last_error)}
            turn_results.append(record)
            if jsonl_handle:
                jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_handle.flush()
            print(json.dumps(record, ensure_ascii=False), flush=True)
            break # 停止当前 case 的后续轮次

        history_prompts.append(prompt_text)
        if env_feedback:
            history_feedbacks.append(env_feedback)

        if turn_idx >= 2:
            gt_prompt = state.get("video_gen_prompt", "")
            try:
                judge = evaluate_video(
                    role_text=role_text,
                    ground_truth_prompt=gt_prompt,
                    video_path=video_path,
                    model_vlm=eval_config.model_vlm,
                    fps=eval_config.fps,
                    client=eval_client,
                )
            except Exception as exc:
                judge = {"pass": False, "reason": str(exc), "raw": str(exc)}
            record = {"case_id": case_id, "turn": turn_idx, **judge}
            turn_results.append(record)
            if jsonl_handle:
                jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_handle.flush()
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if not judge.get("pass", False):
                logging.error("[EVAL][TURN_ABORT] fail; stop case %s turn %d", case_id, turn_idx)
                break

        if not transition or not transition.get("next_state_id"):
            break
        current_state_id = transition["next_state_id"]
        turn_idx += 1

    case_pass = all(r.get("pass", False) for r in turn_results) if turn_results else True
    return {"case_id": case_id, "turn_results": turn_results, "case_pass": case_pass}


def run_generation_with_inline_eval(gen_config: GenerationConfig, eval_config: EvalConfig) -> Dict[str, Any]:
    """生成并在 turn2+ 立即评测，失败只停止当前 case。"""
    cases = _load_env(gen_config.env_path)
    if eval_config.limit and eval_config.limit > 0:
        cases = cases[: eval_config.limit]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    logging.info("[GEN] run_id=%s cases=%d output=%s", run_id, len(cases), gen_config.output_dir)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Dict[str, Any]] = []
    jsonl_path = eval_config.output_path
    if not jsonl_path.endswith(".jsonl"):
        jsonl_path = f"{jsonl_path}.jsonl"
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_handle:
        with ThreadPoolExecutor(max_workers=gen_config.max_workers) as executor:
            futures = [
                executor.submit(
                    _process_case_inline_eval,
                    case,
                    gen_config,
                    eval_config,
                    run_id,
                    jsonl_handle,
                )
                for case in cases
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    logging.error(f"Case execution failed: {exc}")

    summary = summarize_results(results)
    report = {"run_id": run_id, "summary": summary, "cases": results}
    Path(eval_config.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_config.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
