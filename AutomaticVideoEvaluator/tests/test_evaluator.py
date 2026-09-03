from ave.client import ModelResult
from ave.config import Config
from ave.evaluator import Evaluator


class FakeClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def json_completion(self, _model, _system, _content):
        self.calls += 1
        return self.results.pop(0)


def test_voting_semantic_matching_and_role_costs(tmp_path) -> None:
    yes = ModelResult(
        {"reasoning": "visible flaw", "judgement": "yes", "weakness": ["flaw"]},
        input_tokens=10,
        output_tokens=2,
    )
    matched = ModelResult(
        {
            "matrix_counts": {"TP": 2, "TN": 0, "FP": 1, "FN": 1},
            "reasoning": {
                "overall_reasoning": "comparison",
                "matched_reference_points": ["flaw"],
                "extra_valid_points": [],
                "missed_points": ["detail"],
                "hallucinated_points": ["claim"],
            },
        },
        input_tokens=20,
        output_tokens=4,
    )
    client = FakeClient([yes, yes, yes, matched])
    config = Config(
        algorithm="gepa_semantic_loss",
        votes=5,
        judge_input_cost_per_million=1.0,
        judge_output_cost_per_million=2.0,
        matcher_input_cost_per_million=3.0,
        matcher_output_cost_per_million=4.0,
        optimizer_input_cost_per_million=5.0,
        optimizer_output_cost_per_million=6.0,
    )
    evaluator = Evaluator(config, tmp_path, client)
    evaluator.content_for_item = lambda _item: []

    judged = evaluator.judge({}, "prompt")
    assert judged["voting_history"] == ["yes", "yes", "yes"]
    assert client.calls == 3

    matrix, reasoning = evaluator.match(["flaw"], "reference flaw")
    assert matrix == {"TP": 0.5, "TN": 0.0, "FP": 0.25, "FN": 0.25}
    assert reasoning["matched_reference_points"] == ["flaw"]

    evaluator.record(ModelResult({}, input_tokens=7, output_tokens=8), "optimizer")
    usage = evaluator.usage_summary()
    assert usage["by_role"]["judge"] == {"input_tokens": 30, "output_tokens": 6}
    assert usage["by_role"]["matcher"] == {"input_tokens": 20, "output_tokens": 4}
    assert usage["by_role"]["optimizer"] == {"input_tokens": 7, "output_tokens": 8}
    assert usage["estimated_cost_usd"] == 201 / 1_000_000
