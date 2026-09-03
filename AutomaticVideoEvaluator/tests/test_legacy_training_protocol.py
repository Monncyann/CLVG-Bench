import json
import time

import pytest

from ave.config import Config
from ave.evaluator import ItemEvaluationError
from ave.trainer import Trainer


def _metric_result(data, score):
    return {
        "confusion_matrix": {"TP": score, "TN": 0, "FP": 0, "FN": 1 - score},
        "metrics": {
            "accuracy": score,
            "mcc": score,
            "f1": score,
            "recall_minus_fpr": score,
        },
        "items": [{"id": item["id"], "metrics": {"accuracy": score}} for item in data],
    }


def _protocol_trainer(tmp_path, *, algorithm="textgrad", **overrides):
    data_dir = tmp_path / "data" / "task"
    prompt_dir = tmp_path / "prompts"
    data_dir.mkdir(parents=True)
    prompt_dir.mkdir()
    for split in ("train", "val", "test"):
        (data_dir / f"{split}.json").write_text(
            json.dumps([{"id": split}]), encoding="utf-8"
        )
    initial = "Input Format:\nOutput Format:\nreasoning judgement weakness"
    (prompt_dir / "initial.json").write_text(
        json.dumps({"System Prompt": initial}), encoding="utf-8"
    )
    config = Config(
        data_dir="data/task",
        output_dir="outputs/run",
        initial_prompt="prompts/initial.json",
        algorithm=algorithm,
        metric="accuracy",
        votes=1,
        budget_usd=100.0,
        **overrides,
    )
    trainer = Trainer(config, tmp_path, client=object())
    trainer.validate_data = lambda *_args: None
    return trainer, initial


def test_default_configuration_uses_historical_protocol() -> None:
    config = Config()
    assert config.seed == 47
    assert config.temperature == 0.0001
    assert config.max_workers == 64
    assert config.skip_perfect_score is True
    assert config.max_iterations_without_improvement == 10
    assert config.test_every_validations == 3
    assert config.optimizer_max_payload_mb == 45.0


def test_training_budget_uses_historical_single_rate(tmp_path) -> None:
    config = Config(
        output_dir="outputs/budget",
        budget_input_cost_per_million=1.0,
        budget_output_cost_per_million=2.0,
    )
    trainer = Trainer(config, tmp_path, client=object())
    usage = trainer.evaluator.empty_usage()
    usage["judge"] = {"input_tokens": 10, "output_tokens": 20}
    usage["optimizer"] = {"input_tokens": 30, "output_tokens": 40}
    usage["matcher"] = {"input_tokens": 50, "output_tokens": 60}
    trainer.evaluator.add_usage(usage)

    assert trainer.cost() == 330 / 1_000_000


def test_optimizer_weakness_text_matches_legacy_edge_cases() -> None:
    false_positive = {
        "prediction": {"weakness": ["invented defect"]},
        "reference": "",
        "match_reasoning": {"mode": "rough_or_edge_case"},
        "confusion_matrix": {"TP": 0, "TN": 0, "FP": 1, "FN": 0},
    }
    false_negative = {
        "prediction": {"weakness": []},
        "reference": "missing action",
        "match_reasoning": {"mode": "rough_or_edge_case"},
        "confusion_matrix": {"TP": 0, "TN": 0, "FP": 0, "FN": 1},
    }

    assert "invented defect" in Trainer._matching_weakness(false_positive)
    assert "missing action" in Trainer._matching_weakness(false_negative)


def test_initial_periodic_and_final_tests_are_default(tmp_path) -> None:
    trainer, initial = _protocol_trainer(tmp_path, max_steps=3)
    stages = []
    revisions = []

    def fake_evaluate(data, prompt, stage_name=None):
        stages.append(stage_name)
        score = 0.4 + 0.1 * revisions.index(prompt) if prompt in revisions else 0.3
        return _metric_result(data, score)

    def fake_revise(_prompt, _batch_result, _batch):
        revised = f"{initial} revision {len(revisions) + 1}"
        revisions.append(revised)
        return revised, {}, trainer.evaluator.empty_usage()

    trainer.evaluate = fake_evaluate
    trainer.revise = fake_revise
    trainer.run()

    assert stages[:2] == ["000_initial_test", "000_validation"]
    assert "003_periodic_test" in stages
    assert stages[-1] == "final_test_best_1"
    summary = json.loads((trainer.output / "run_summary.json").read_text())
    assert summary["periodic_test_count"] == 1
    assert summary["optimizer_steps"] == 3


def test_perfect_batch_skips_optimizer_without_spending_a_step(tmp_path) -> None:
    trainer, _initial = _protocol_trainer(tmp_path, max_steps=1, max_rollouts=3)
    revisions = []

    def fake_evaluate(data, _prompt, _stage_name=None):
        trainer.rollouts += len(data)
        return _metric_result(data, 1.0)

    def fake_revise(*_args):
        revisions.append(True)
        raise AssertionError("optimizer must not run for a perfect mini-batch")

    trainer.evaluate = fake_evaluate
    trainer.revise = fake_revise
    trainer.run()

    summary = json.loads((trainer.output / "run_summary.json").read_text())
    assert revisions == []
    assert summary["optimizer_steps"] == 0
    assert summary["stopped_reason"] == "optimization_rollouts"


def test_regressions_force_validation_after_configured_limit(tmp_path) -> None:
    trainer, initial = _protocol_trainer(
        tmp_path,
        algorithm="gepa",
        max_steps=2,
        max_iterations_without_improvement=2,
    )
    proposals = []

    def fake_evaluate(data, prompt, stage_name=None):
        if stage_name and stage_name.endswith("before_update"):
            score = 0.6
        elif stage_name and stage_name.endswith("after_update"):
            score = 0.4
        elif stage_name == "002_validation":
            score = 0.7
        else:
            score = 0.5
        return _metric_result(data, score)

    def fake_revise(_prompt, _batch_result, _batch):
        revised = f"{initial} proposal {len(proposals) + 1}"
        proposals.append(revised)
        return revised, {}, trainer.evaluator.empty_usage()

    trainer.evaluate = fake_evaluate
    trainer.revise = fake_revise
    trainer.run()

    history = json.loads((trainer.output / "history.json").read_text())
    assert history[1]["accepted"] is False
    assert history[2]["mini_batch_accepted"] is False
    assert history[2]["forced_validation"] is True
    assert history[2]["accepted"] is True


def test_concurrent_evaluation_preserves_order_and_resumes_from_cache(tmp_path) -> None:
    config = Config(output_dir="outputs/cache", votes=1, max_workers=4)
    trainer = Trainer(config, tmp_path, client=object())
    data = [{"id": str(index)} for index in range(4)]
    calls = []

    def fake_item(item, _prompt):
        calls.append(item["id"])
        time.sleep((3 - int(item["id"])) * 0.005)
        result = {
            "id": item["id"],
            "prediction": {},
            "reference": "",
            "match_reasoning": {},
            "confusion_matrix": {"TP": 0, "TN": 1, "FP": 0, "FN": 0},
            "metrics": {"accuracy": 1.0},
        }
        usage = trainer.evaluator.empty_usage()
        usage["judge"]["input_tokens"] = 1
        return result, usage

    trainer.evaluator.evaluate_item_with_usage = fake_item
    first = trainer.evaluate(data, "prompt", "cached_stage")
    assert [item["id"] for item in first["items"]] == ["0", "1", "2", "3"]
    assert len(calls) == 4

    resumed = Trainer(config, tmp_path, client=object())
    resumed.evaluator.evaluate_item_with_usage = lambda *_args: (_ for _ in ()).throw(
        AssertionError("cached items must not be evaluated again")
    )
    second = resumed.evaluate(data, "prompt", "cached_stage")
    assert [item["id"] for item in second["items"]] == ["0", "1", "2", "3"]
    assert resumed.evaluator.input_tokens == 4


def test_failed_item_usage_is_preserved_across_resume(tmp_path) -> None:
    config = Config(output_dir="outputs/failed-cache", votes=1, max_workers=1)
    trainer = Trainer(config, tmp_path, client=object())
    data = [{"id": "failed"}]
    usage = trainer.evaluator.empty_usage()
    usage["judge"]["input_tokens"] = 7

    def fail_item(*_args):
        raise ItemEvaluationError(ValueError("invalid response"), usage)

    trainer.evaluator.evaluate_item_with_usage = fail_item
    try:
        trainer.evaluate(data, "prompt", "failed_stage")
    except RuntimeError as error:
        assert "All items failed" in str(error)
    else:
        raise AssertionError("an all-item failure must abort the stage")

    resumed = Trainer(config, tmp_path, client=object())

    def succeed_item(item, _prompt):
        result = {
            "id": item["id"],
            "prediction": {},
            "reference": "",
            "match_reasoning": {},
            "confusion_matrix": {"TP": 0, "TN": 1, "FP": 0, "FN": 0},
            "metrics": {"accuracy": 1.0},
        }
        return result, resumed.evaluator.empty_usage()

    resumed.evaluator.evaluate_item_with_usage = succeed_item
    resumed.evaluate(data, "prompt", "failed_stage")
    assert resumed.evaluator.input_tokens == 7


def test_validation_requires_positive_and_negative_examples(tmp_path) -> None:
    data_dir = tmp_path / "data" / "task"
    video_dir = data_dir / "videos"
    video_dir.mkdir(parents=True)

    def make_row(identifier: str, weakness: str) -> dict:
        video = f"videos/{identifier}.mp4"
        (data_dir / video).write_bytes(b"video")
        return {
            "id": identifier,
            "prompt": "",
            "reference_images": [],
            "video": video,
            "feedback": {"abnormality": weakness},
        }

    trainer = Trainer(
        Config(data_dir="data/task", output_dir="outputs/validation"),
        tmp_path,
        client=object(),
    )
    train = [make_row("train", "")]
    validation = [make_row("val-1", ""), make_row("val-2", "")]
    test = [make_row("test", "weakness")]

    with pytest.raises(ValueError, match="both positive and negative"):
        trainer.validate_data(train, validation, test)
