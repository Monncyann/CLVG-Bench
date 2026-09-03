import json

from ave.config import Config
from ave.trainer import Trainer


def test_textgrad_restarts_from_best_validation_prompt(tmp_path) -> None:
    data_dir = tmp_path / "data" / "task"
    prompt_dir = tmp_path / "prompts"
    data_dir.mkdir(parents=True)
    prompt_dir.mkdir()
    for split in ("train", "val", "test"):
        (data_dir / f"{split}.json").write_text(
            json.dumps([{"id": split}]), encoding="utf-8"
        )
    initial_prompt = "Input Format\nOutput Format\nreasoning judgement weakness"
    (prompt_dir / "initial.json").write_text(
        json.dumps({"System Prompt": initial_prompt}), encoding="utf-8"
    )
    config = Config(
        data_dir="data/task",
        output_dir="outputs/run",
        initial_prompt="prompts/initial.json",
        algorithm="textgrad",
        metric="accuracy",
        votes=1,
        max_steps=2,
        budget_usd=100.0,
    )
    trainer = Trainer(config, tmp_path, client=object())
    trainer.validate_data = lambda *_args: None
    parent_prompts: list[str] = []

    def fake_evaluate(data, prompt, _stage_name=None):
        scores = {
            initial_prompt: 0.4,
            initial_prompt + " step1": 0.8,
            initial_prompt + " step1 step2": 0.6,
        }
        score = scores[prompt]
        return {
            "confusion_matrix": {"TP": score, "TN": 0, "FP": 0, "FN": 1 - score},
            "metrics": {
                "accuracy": score,
                "mcc": score,
                "f1": score,
                "recall_minus_fpr": score,
            },
            "items": [
                {"id": item["id"], "metrics": {"accuracy": score}} for item in data
            ],
        }

    def fake_revise(prompt, _batch_result, _batch):
        parent_prompts.append(prompt)
        return (
            f"{prompt} step{len(parent_prompts)}",
            {},
            trainer.evaluator.empty_usage(),
        )

    trainer.evaluate = fake_evaluate
    trainer.revise = fake_revise
    trainer.run()

    assert parent_prompts == [initial_prompt, initial_prompt + " step1"]
    summary = json.loads((tmp_path / "outputs/run/run_summary.json").read_text())
    assert summary["best_step"] == 1
    assert summary["candidate_count"] == 3


def test_gepa_rejection_returns_to_latest_validated_prompt(tmp_path) -> None:
    data_dir = tmp_path / "data" / "task"
    prompt_dir = tmp_path / "prompts"
    data_dir.mkdir(parents=True)
    prompt_dir.mkdir()
    for split in ("train", "val", "test"):
        (data_dir / f"{split}.json").write_text(
            json.dumps([{"id": split}]), encoding="utf-8"
        )
    initial_prompt = "Input Format\nOutput Format\nreasoning judgement weakness"
    (prompt_dir / "initial.json").write_text(
        json.dumps({"System Prompt": initial_prompt}), encoding="utf-8"
    )
    config = Config(
        data_dir="data/task",
        output_dir="outputs/run",
        initial_prompt="prompts/initial.json",
        algorithm="gepa",
        metric="accuracy",
        votes=1,
        max_steps=2,
        budget_usd=100.0,
    )
    trainer = Trainer(config, tmp_path, client=object())
    trainer.validate_data = lambda *_args: None
    parent_prompts: list[str] = []
    proposals = [initial_prompt + " proposal1", initial_prompt + " proposal2"]

    def fake_evaluate(data, prompt, _stage_name=None):
        split = data[0]["id"]
        if split == "train":
            score = {
                initial_prompt: 0.6,
                proposals[0]: 0.4,
                proposals[1]: 0.8,
            }[prompt]
        else:
            score = {initial_prompt: 0.5, proposals[1]: 0.7}[prompt]
        return {
            "confusion_matrix": {"TP": score, "TN": 0, "FP": 0, "FN": 1 - score},
            "metrics": {
                "accuracy": score,
                "mcc": score,
                "f1": score,
                "recall_minus_fpr": score,
            },
            "items": [
                {"id": item["id"], "metrics": {"accuracy": score}} for item in data
            ],
        }

    def fake_revise(prompt, _batch_result, _batch):
        parent_prompts.append(prompt)
        return (
            proposals[len(parent_prompts) - 1],
            {},
            trainer.evaluator.empty_usage(),
        )

    trainer.evaluate = fake_evaluate
    trainer.revise = fake_revise
    trainer.run()

    assert parent_prompts == [initial_prompt, initial_prompt]
    summary = json.loads((tmp_path / "outputs/run/run_summary.json").read_text())
    history = json.loads((tmp_path / "outputs/run/history.json").read_text())
    assert summary["best_step"] == 2
    assert summary["candidate_count"] == 2
    assert history[1]["accepted"] is False
    assert history[2]["accepted"] is True
