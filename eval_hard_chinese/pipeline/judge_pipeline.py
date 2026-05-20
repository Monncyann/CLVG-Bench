import base64  # 将视频编码为 Base64 供 API 传输
import json  # JSON 解析与序列化
import logging  # 日志输出
import os  # 环境变量读取
import re  # 正则抽取 JSON
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from volcenginesdkarkruntime import Ark

from system_prompt import eval_prompt  # 评测提示词


@dataclass
class EvalConfig:
    """评测阶段配置参数。"""
    env_path: str  # env.json 路径
    videos_root: str  # 生成视频根目录
    output_path: str  # 评测报告输出路径
    max_workers: int  # 评测并发数
    model_vlm: str  # 评测使用的 VLM 模型
    fps: int  # 输入视频 FPS
    limit: int  # 限制评测 case 数量


def _get_api_key() -> str:
    """从环境变量读取 ARK_API_KEY。"""
    api_key = os.environ.get("ARK_API_KEY")  # 读取 Key
    if not api_key:  # 未设置则报错
        raise RuntimeError("ARK_API_KEY is not set. Please export ARK_API_KEY before running.")
    return api_key  # 返回 Key


def _encode_file_to_base64(file_path: str) -> str:
    """将视频文件编码为 Base64 字符串。"""
    with open(file_path, "rb") as read_file:  # 二进制读取
        return base64.b64encode(read_file.read()).decode("utf-8")  # 编码并转字符串


def _extract_text_from_response(response: Any) -> str:
    """从不同格式的响应中提取文本。"""
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                return content
            text = choices[0].get("text")
            if text:
                return text
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    if hasattr(response, "choices") and response.choices:
        first = response.choices[0]
        if hasattr(first, "message") and first.message:
            content = getattr(first.message, "content", None)
            if content:
                return content
        text = getattr(first, "text", None)
        if text:
            return text
    if hasattr(response, "output"):
        for item in response.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if isinstance(c, dict):
                        if c.get("type") in {"output_text", "text"} and "text" in c:
                            return c["text"]
                    elif hasattr(c, "text"):
                        return c.text
    return str(response)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """尝试从模型输出文本中解析 JSON。"""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _load_env(path: str) -> List[Dict[str, Any]]:
    """读取 env.json 中的 case 列表。"""
    with open(path, "r", encoding="utf-8") as f:  # 打开文件
        return json.load(f)  # 解析 JSON


def _build_state_map(states: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """构建 state_id -> state 的索引表。"""
    return {s["state_id"]: s for s in states}


def _select_transition(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """选择下一条 transition（默认第一条）。"""
    transitions = state.get("transitions") or []  # 获取 transitions
    if not transitions:  # 无 transitions
        return None
    return transitions[0]  # 返回第一条


def _walk_states(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按 transitions 链路遍历 state 序列。"""
    state_map = _build_state_map(case.get("states", []))
    current_state_id = case.get("initial_state_id")
    if not current_state_id or current_state_id not in state_map:
        return []
    ordered = []
    while True:
        state = state_map[current_state_id]
        ordered.append(state)
        transition = _select_transition(state)
        if not transition or not transition.get("next_state_id"):
            break
        current_state_id = transition["next_state_id"]
        if current_state_id not in state_map:
            break
    return ordered


def _judge_video(
    client: Ark,
    role_text: str,
    ground_truth_prompt: str,
    video_path: str,
    model_vlm: str,
    fps: int,
) -> Dict[str, Any]:
    """对单个视频进行评测。"""
    video_base64 = _encode_file_to_base64(video_path)  # 视频转 Base64
    user_payload = {
        "role": role_text,
        "ground_truth_prompt": ground_truth_prompt,
        "instruction": "判断视频是否符合目标提示词，输出 JSON: {pass: boolean, reason: string}。",
    }
    response = client.responses.create(  # 调用评测模型
        model=model_vlm,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": eval_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=False)},
                    {
                        "type": "input_video",
                        "video_url": f"data:video/mp4;base64,{video_base64}",
                        "fps": fps,
                    },
                ],
            },
        ],
    )
    text = _extract_text_from_response(response)  # 提取文本
    parsed = _extract_json(text) or {}  # 解析 JSON
    passed = parsed.get("pass")  # 取 pass 字段
    if isinstance(passed, str):
        passed = passed.strip().lower() in {"true", "yes", "1"}
    if passed is None:
        passed = False
    return {  # 统一返回结构
        "pass": bool(passed),
        "reason": parsed.get("reason", text),
        "raw": text,
    }


def evaluate_video(
    role_text: str,
    ground_truth_prompt: str,
    video_path: str,
    model_vlm: str,
    fps: int,
    client: Optional[Ark] = None,
) -> Dict[str, Any]:
    """评测单个视频（可复用外部 client）。"""
    local_client = client or Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=_get_api_key(),
    )
    return _judge_video(
        client=local_client,
        role_text=role_text,
        ground_truth_prompt=ground_truth_prompt,
        video_path=video_path,
        model_vlm=model_vlm,
        fps=fps,
    )


def _eval_case(case: Dict[str, Any], config: EvalConfig, run_id: str) -> Dict[str, Any]:
    """评测单个 case（从 turn2 开始）。"""
    case_id = case.get("id", "unknown")  # case id
    role_text = case.get("role", "")  # 角色设定
    ordered_states = _walk_states(case)  # state 序列
    case_dir = Path(config.videos_root) / run_id / f"case_{case_id}"  # case 目录
    logging.info("[EVAL] case %s -> %s", case_id, case_dir)  # 日志

    client = Ark(  # 构建 Ark 客户端
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=_get_api_key(),
    )

    turn_results = []  # 逐轮评测结果
    for idx, state in enumerate(ordered_states, start=1):  # 遍历每轮
        if idx == 1:  # turn1 不评
            continue
        video_path = case_dir / f"turn_{idx}.mp4"  # 当前轮视频路径
        if not video_path.exists():  # 若不存在视频
            turn_results.append(
                {
                    "turn": idx,
                    "pass": False,
                    "reason": f"missing video: {video_path}",
                }
            )
            continue
        gt_prompt = state.get("video_gen_prompt", "")  # 取 ground truth
        judge = _judge_video(
            client=client,
            role_text=role_text,
            ground_truth_prompt=gt_prompt,
            video_path=str(video_path),
            model_vlm=config.model_vlm,
            fps=config.fps,
        )
        turn_results.append({"turn": idx, **judge})

    case_pass = all(r["pass"] for r in turn_results) if turn_results else True  # case 是否通过
    return {
        "case_id": case_id,
        "turn_results": turn_results,
        "case_pass": case_pass,
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总统计指标。"""
    turn_stats: Dict[int, Dict[str, int]] = {}  # 每轮统计
    case_total = len(results)  # case 总数
    case_passed = sum(1 for r in results if r.get("case_pass"))  # 通过数
    judged_turns = 0  # 被评测轮次
    passed_turns = 0  # 通过轮次

    for case in results:  # 遍历所有 case
        for r in case.get("turn_results", []):  # 遍历每轮结果
            turn = r["turn"]  # 轮次编号
            turn_stats.setdefault(turn, {"total": 0, "passed": 0})  # 初始化
            turn_stats[turn]["total"] += 1  # 统计总数
            judged_turns += 1  # 计数
            if r["pass"]:  # 通过则计入
                turn_stats[turn]["passed"] += 1
                passed_turns += 1

    turn_rates = {
        str(turn): (stats["passed"] / stats["total"] if stats["total"] else 0.0)
        for turn, stats in turn_stats.items()
    }

    overall_rate = passed_turns / judged_turns if judged_turns else 0.0
    case_rate = case_passed / case_total if case_total else 0.0
    return {
        "case_total": case_total,
        "case_passed": case_passed,
        "case_pass_rate": case_rate,
        "judged_turns": judged_turns,
        "passed_turns": passed_turns,
        "overall_turn_pass_rate": overall_rate,
        "turn_pass_rates": turn_rates,
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对外暴露的汇总接口。"""
    return _summarize(results)


def run_evaluation(config: EvalConfig, run_id: str) -> Dict[str, Any]:
    """评测入口：加载 case 并输出报告。"""
    cases = _load_env(config.env_path)  # 加载 case
    if config.limit and config.limit > 0:  # 限制数量
        cases = cases[: config.limit]
    logging.info("[EVAL] run_id=%s cases=%d videos=%s", run_id, len(cases), config.videos_root)  # 日志

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [executor.submit(_eval_case, case, config, run_id) for case in cases]
        for future in as_completed(futures):
            results.append(future.result())

    summary = _summarize(results)
    report = {
        "run_id": run_id,
        "summary": summary,
        "cases": results,
    }

    Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
