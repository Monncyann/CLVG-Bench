from ave.candidates import PromptCandidate, PromptTable


def candidate(
    step: int, score: float, instance_scores: dict[str, float]
) -> PromptCandidate:
    return PromptCandidate(
        step=step,
        prompt=f"prompt {step}",
        prompt_file=f"prompt_{step:03d}.json",
        validation_score=score,
        instance_scores=instance_scores,
        parent_step=None if step == 0 else 0,
    )


def test_best_and_pareto_selection_are_validation_backed_and_deterministic() -> None:
    table = PromptTable(seed=42)
    first = candidate(0, 0.5, {"a": 1.0, "b": 0.0})
    second = candidate(1, 0.5, {"a": 0.0, "b": 1.0})
    dominated = candidate(2, 0.25, {"a": 0.0, "b": 0.0})
    for item in (first, second, dominated):
        table.add(item)

    assert table.best() is first
    assert table.latest() is dominated
    assert table.pareto_front() == [first, second]
    assert table.select("pareto_sampling", local_seed=7) == table.select(
        "pareto_sampling", local_seed=7
    )


def test_candidates_with_partial_validation_results_are_kept() -> None:
    table = PromptTable(seed=42)
    table.add(candidate(0, 0.5, {"a": 1.0}))
    table.add(candidate(1, 0.5, {"b": 1.0}))

    assert [item.step for item in table.pareto_front()] == [0, 1]


def test_legacy_filename_order_controls_pareto_sampling_and_final_ties() -> None:
    table = PromptTable(seed=47)
    step_2 = candidate(2, 0.5, {"a": 1.0, "b": 0.0})
    step_10 = candidate(10, 0.5, {"a": 0.0, "b": 1.0})
    table.add(step_2)
    table.add(step_10)

    # The old runner sorted unpadded Params_N.json strings lexicographically.
    assert table.pareto_front() == [step_10, step_2]
    assert table.top(2) == [step_10, step_2]
