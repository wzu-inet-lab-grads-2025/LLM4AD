from __future__ import annotations

import json
import runpy
from tempfile import TemporaryDirectory
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from llm4ad.base import TextFunctionProgramConverter
from llm4ad.method.eoh_rl.args import apply_runtime_defaults
from llm4ad.method.eoh_rl.controlled_eval import build_query_bank, load_checkpoint_parents, load_query_bank, same_valid_curves, same_valid_summaries, save_query_bank
from llm4ad.method.eoh_rl.profiler import EoHProfiler
from llm4ad.method.eoh_rl.population import Population
from llm4ad.method.eoh_rl.eoh_rl import EoH
from llm4ad.method.eoh_rl.rl.grpo_trainer import EoHGRPOReward, EoHReward, ResidentGRPOPolicy, _trainer_seed, event_funnel
from llm4ad.method.eoh_rl.rl.vc_pair import compare_profiles


_TEMPLATE = """def score(x: int) -> int:
    return x
"""
_COMPLETION = """{{The idea of the algorithm is to return the input deterministically.}}

```python
def score(x: int) -> int:
    return x
```
"""


class _Evaluator:
    def evaluate_program_with_profile(self, _program):
        return {"score": 1.0, "performance_profile": [1.0, 1.0, 1.0]}


def test_grpo_math_defaults_are_explicit():
    grpo = apply_runtime_defaults({"inference_ports": [22001]})["grpo"]
    assert grpo["num_generations"] == 4
    assert grpo["generation_batch_size"] == 4
    assert grpo["per_device_train_batch_size"] == 4
    assert grpo["scale_rewards"] == "group"
    assert grpo["loss_type"] == "dapo"
    assert grpo["importance_sampling_level"] == "token"
    assert grpo["num_iterations"] == 1
    assert grpo["epsilon_high"] == 0.28
    assert grpo["mask_truncated_completions"] is True
    assert grpo["vllm_server_port"] == 22001
    reward = apply_runtime_defaults({})["task_rl_common"]
    assert reward["reward_mode"] == "vc_pair"
    assert reward["pair_confidence"] == 0.95
    assert reward["pair_delta"] == 1.0e-4
    assert reward["pair_positive_reward"] > reward["pair_neutral_reward"] > reward["pair_negative_reward"]
    assert reward["pair_negative_reward"] == -0.2
    blocked = [reward[key] for key in (
        "reward_parse_fail", "reward_exec_fail", "reward_none_return", "reward_random",
        "reward_leak", "reward_exact_copy", "reward_archive_duplicate", "reward_profile_duplicate",
    )]
    assert max(blocked) < 2 * reward["pair_negative_reward"]
    assert grpo["trainer_lifecycle"] == "persistent"
    assert apply_runtime_defaults({})["rl"]["save_final_lora"] is False
    assert apply_runtime_defaults({})["rl"]["compress_history"] is True
    assert apply_runtime_defaults({"args": {"enable_grpo": False}})["rl"]["enabled"] is False


def test_initial_population_path_is_exposed():
    path = "/tmp/initial_population.json"
    assert apply_runtime_defaults({"args": {"initial_population_path": path}})["rl"]["initial_population_path"] == path


def test_event_funnel_has_one_definition_of_validity():
    events = [
        {"idea_contract_success": False, "code_parse_success": False},
        {"idea_contract_success": True, "format_contract_success": True, "code_parse_success": True, "full_contract_success": True, "evaluation_attempted": True},
        {
            "idea_contract_success": True,
            "format_contract_success": False,
            "code_parse_success": True,
            "evaluation_attempted": True,
            "exec_success": True,
            "validity": True,
        },
    ]
    funnel = event_funnel(events)
    assert funnel["completion_count"] == 3
    assert funnel["idea_contract_count"] == 2
    assert funnel["code_parse_count"] == 2
    assert funnel["full_contract_count"] == 1
    assert funnel["evaluation_count"] == 2
    assert funnel["exec_success_count"] == 1
    assert funnel["population_eligible_count"] == 1
    assert funnel["population_eligible_rate"] == 1 / 3


def test_vc_pair_statistics_and_reward():
    positive = compare_profiles([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], minimize=False, confidence=0.95, delta=1.0e-4)
    neutral = compare_profiles([0.9, 2.1], [1.0, 2.0], minimize=False, confidence=0.95, delta=0.0)
    negative = compare_profiles([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], minimize=True, confidence=0.95, delta=1.0e-4)
    assert positive["paired_outcome"] == "positive"
    assert positive["paired_mean"] == 1.0
    assert positive["paired_se"] == 0.0
    assert neutral["paired_outcome"] == "neutral"
    assert neutral["paired_se"] == pytest.approx(0.1)
    assert negative["paired_outcome"] == "negative"
    with pytest.raises(ValueError, match="length mismatch"):
        compare_profiles([1.0], [1.0, 2.0], minimize=False, confidence=0.95, delta=0.0)

    record = {
        "prompt_id": "p1",
        "prompt": "improve score",
        "messages": [{"role": "user", "content": "improve score"}],
        "operator_type": "m1",
        "parent_best_score": 0.0,
        "parent_best_profile": [0.0, 0.0, 0.0],
        "parent_best_id": 1,
        "population_best_score": 0.0,
        "population_best_profile": [0.0, 0.0, 0.0],
        "parent_codes": [],
    }
    class Evaluator:
        def __init__(self):
            self.value = 0

        def evaluate_program_with_profile(self, _program):
            self.value += 1
            return {"score": float(self.value), "performance_profile": [float(self.value)] * 3}

    second = _COMPLETION.replace("return x\n", "return x + 1\n")
    callback = EoHGRPOReward(
        records=[record], evaluator=Evaluator(), template_program=_TEMPLATE,
        reward_config=EoHReward(),
    )
    assert callback([record["messages"]] * 2, [_COMPLETION, second]) == [2.5, 2.5]
    event = callback.events()[0]
    assert event["paired_outcome"] == "positive"
    assert event["paired_n"] == 3
    assert event["paired_confidence"] == 0.95
    assert event["paired_ci_low"] > event["paired_delta"]
    assert event["paired_differences"] == [1.0, 1.0, 1.0]
    assert event["frontier_bonus"] == 0.5
    assert event["group_valid_count"] == 2
    assert event["group_eligible_count"] == 2
    assert event["quality_gate_active"] is True
    assert event["total_reward"] == sum(event[key] for key in ("validity_reward", "quality_reward", "frontier_bonus", "copy_penalty", "duplicate_penalty"))
    assert event["parent_best_id"] == 1

    gated = EoHGRPOReward(records=[record], evaluator=_Evaluator(), template_program=_TEMPLATE, reward_config=EoHReward())
    assert gated([record["messages"]], [_COMPLETION]) == [0.0]
    assert gated.events()[0]["quality_gate_active"] is False


def test_contract_stages_and_duplicate_rewards_are_explicit():
    completion = """```python
def score(x: int) -> int:
    return x
```
"""
    record = {
        "prompt_id": "p1", "prompt": "improve", "operator_type": "m1",
        "parent_best_score": 0.0, "parent_best_profile": [0.0, 0.0, 0.0],
        "population_best_score": 0.0, "population_best_profile": [0.0, 0.0, 0.0],
        "parent_codes": [],
    }
    callback = EoHGRPOReward(
        records=[record], evaluator=_Evaluator(), template_program=_TEMPLATE,
        reward_config=EoHReward(), known_profiles=[[1.0, 1.0, 1.0]],
    )
    assert callback([[{"role": "user", "content": "improve"}]], [completion]) == [-0.5]
    event = callback.events()[0]
    assert event["idea_contract_success"] is False
    assert event["code_parse_success"] is True
    assert event["full_contract_success"] is False
    assert event["exec_success"] is True
    assert event["profile_duplicate"] is True
    assert event["group_valid_count"] == 1
    assert event["group_eligible_count"] == 0
    assert event_funnel([event])["population_eligible_count"] == 0

    archived = EoHGRPOReward(
        records=[record], evaluator=_Evaluator(), template_program=_TEMPLATE,
        reward_config=EoHReward(), known_functions=[_TEMPLATE],
    )
    assert archived([[{"role": "user", "content": "improve"}]], [completion]) == [-0.75]
    assert archived.events()[0]["archive_duplicate"] is True


def test_quality_gate_counts_executable_duplicates_but_not_as_eligible():
    record = {
        "prompt_id": "p1", "prompt": "improve", "operator_type": "m1",
        "parent_best_score": 0.0, "parent_best_profile": [0.0, 0.0, 0.0],
        "population_best_score": 0.0, "population_best_profile": [0.0, 0.0, 0.0],
        "parent_codes": [],
    }

    class Evaluator:
        def __init__(self):
            self.value = 0

        def evaluate_program_with_profile(self, _program):
            self.value += 1
            value = float(self.value)
            return {"score": value, "performance_profile": [value] * 3}

    callback = EoHGRPOReward(
        records=[record], evaluator=Evaluator(), template_program=_TEMPLATE,
        reward_config=EoHReward(), known_profiles=[[1.0, 1.0, 1.0]],
    )
    second = _COMPLETION.replace("return x\n", "return x + 1\n")
    rewards = callback([record["prompt"]] * 2, [_COMPLETION, second])
    events = callback.events()
    assert rewards[0] == -0.5
    assert rewards[1] > 0.0
    assert events[0]["group_valid_count"] == events[1]["group_valid_count"] == 2
    assert events[0]["group_eligible_count"] == events[1]["group_eligible_count"] == 1
    assert all(event["quality_gate_active"] for event in events)


def test_reward_tiers_reject_overlapping_valid_and_blocked_ranges():
    with pytest.raises(ValueError, match="blocked rewards"):
        EoHReward(pair_negative_reward=-0.5)


def test_population_rejects_evolved_profile_duplicates():
    first = TextFunctionProgramConverter.text_to_function(_TEMPLATE)
    second = TextFunctionProgramConverter.text_to_function("def score(x: int) -> int:\n    return x + 1\n")
    first.score = second.score = 1.0
    setattr(first, "_eoh_profile", [1.0, 2.0])
    setattr(second, "_eoh_profile", [1.0, 2.0])
    population = Population(2, pop=[first])
    assert population.has_duplicate_function(_TEMPLATE)
    assert population.register_evolved_function(second)[0] is False


def test_required_reward_controls():
    record = {
        "prompt_id": "p1",
        "prompt": "improve score",
        "messages": [{"role": "user", "content": "improve score"}],
        "operator_type": "m1",
        "parent_best_score": 0.0,
        "population_best_score": 0.0,
        "parent_codes": [],
    }

    validity = EoHGRPOReward(
        records=[record], evaluator=_Evaluator(), template_program=_TEMPLATE,
        reward_config=EoHReward(reward_mode="validity_only"),
    )
    assert validity([record["messages"]], [_COMPLETION]) == [1.0]

    shuffled = EoHGRPOReward(
        records=[record], evaluator=_Evaluator(), template_program=_TEMPLATE,
        reward_config=EoHReward(reward_mode="performance_shuffled"),
    )
    rows = [{"prompt_id": "p1", "exec_success": True}] * 4
    rewards = [1.0, 2.0, 3.0, 4.0]
    shuffled._shuffle_performance_rewards(rows, rewards)
    assert rewards == [2.0, 3.0, 4.0, 1.0]


def test_disabled_grpo_rolls_out_without_training():
    policy = ResidentGRPOPolicy.__new__(ResidentGRPOPolicy)
    policy.config = {"num_generations": 4}
    policy._tokenizer = None
    policy.draw_samples = lambda _prompt, n, **_kwargs: [_COMPLETION] * n
    policy._train_with_trl = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("training called"))
    record = {
        "prompt_id": "p1",
        "prompt": "improve score",
        "messages": [{"role": "user", "content": "improve score"}],
        "operator_type": "m1",
        "parent_best_score": 0.0,
        "parent_best_profile": [0.0, 0.0, 0.0],
        "population_best_score": 0.0,
        "population_best_profile": [0.0, 0.0, 0.0],
        "parent_codes": [],
        "group_size": 4,
    }
    with TemporaryDirectory() as output_dir:
        result = policy.run_update(
            prompt_records=[record],
            evaluator=_Evaluator(),
            reward_fn=EoHReward(),
            template_program=_TEMPLATE,
            output_dir=output_dir,
            training_enabled=False,
        )
    assert result["executed"] is True
    assert result["metrics"]["training_enabled"] is False
    assert result["metrics"]["optimizer_steps_this_update"] == 0
    assert result["reward_summary"]["funnel"]["completion_count"] == 4


def test_fixed_bank_and_same_valid_are_search_free():
    parents = [
        TextFunctionProgramConverter.text_to_function(_TEMPLATE),
        TextFunctionProgramConverter.text_to_function("def score(x: int) -> int:\n    return x + 1\n"),
    ]
    for index, parent in enumerate(parents):
        parent.score = float(index)
        parent.algorithm = f"parent {index}"
        setattr(parent, "_eoh_profile", [float(index)])
    bank = build_query_bank(
        parents,
        task_description="Improve score.",
        template_function=parents[0],
        size=4,
    )
    assert [row["operator_type"] for row in bank] == ["e1", "e2", "m1", "m2"]
    assert all(row["parent_best_profile"] for row in bank)

    common = {"code_parse_success": True, "exec_success": True, "validity": True}
    summaries = same_valid_summaries({
        "a": [
            {**common, "prompt_id": "p1", "completion_index": 1, "score": 1.0, "beats_parent": True, "paired_outcome": "positive", "paired_mean": 0.1},
            {**common, "prompt_id": "p1", "completion_index": 2, "score": 2.0, "paired_outcome": "neutral", "paired_mean": 0.0},
        ],
        "b": [
            {**common, "prompt_id": "p1", "completion_index": 1, "score": 1.5, "paired_outcome": "positive", "paired_mean": 0.2},
        ],
    })
    assert summaries["a"]["population_eligible_count"] == summaries["b"]["population_eligible_count"] == 1
    curves = same_valid_curves({
        "a": [{**common, "prompt_id": "p1", "completion_index": 1, "score": 1.0}],
        "b": [{**common, "prompt_id": "p1", "completion_index": 1, "score": 2.0}],
    })
    assert curves["a"]["curve"][0]["best_score"] == 1.0
    with TemporaryDirectory() as directory:
        path = Path(directory, "bank.json")
        payload = save_query_bank(str(path), bank, {"split": "heldout"})
        assert load_query_bank(str(path))["bank_id"] == payload["bank_id"]


def test_checkpoint_parent_profiles_are_loaded():
    with TemporaryDirectory() as directory:
        path = Path(directory, "checkpoint.json")
        path.write_text(json.dumps({"population": [{"function": _TEMPLATE, "score": 1.0, "algorithm": "base", "performance_profile": [1.0]}]}), encoding="utf-8")
        parents = load_checkpoint_parents(str(path))
    assert parents[0].score == 1.0
    assert parents[0]._eoh_profile == [1.0]


def test_population_rng_is_local_and_trainer_seed_changes_by_update():
    parents = [TextFunctionProgramConverter.text_to_function(f"def score(x: int) -> int:\n    return x + {index}\n") for index in range(5)]
    for index, parent in enumerate(parents):
        parent.score = float(index)
    left, right = Population(5, pop=parents, seed=42), Population(5, pop=parents, seed=42)
    first = [str(left.selection()) for _ in range(20)]
    np.random.seed(999)
    np.random.random(1000)
    assert first == [str(right.selection()) for _ in range(20)]
    assert _trainer_seed(42, 1) != _trainer_seed(42, 2)


def test_profile_failure_is_not_evaluated_twice():
    class Evaluator:
        def __init__(self):
            self.profile_calls = self.fallback_calls = 0

        def evaluate_program_with_profile(self, _program):
            self.profile_calls += 1

        def evaluate_program_record_time_with_diag(self, _program):
            self.fallback_calls += 1
            return None, 0.0, None

    evaluator = Evaluator()
    callback = EoHGRPOReward(records=[{"prompt_id": "p", "prompt": "x", "operator_type": "m1"}], evaluator=evaluator, template_program=_TEMPLATE, reward_config=EoHReward())
    callback._evaluate_one(_TEMPLATE)
    assert (evaluator.profile_calls, evaluator.fallback_calls) == (1, 0)


def test_incomplete_initialization_fails_run():
    method = EoH.__new__(EoH)
    method._population, method._pop_size = [], 1
    method._sample_initialize_population = lambda: None
    method._finalize_run_timing = lambda _started: None
    method._write_run_summary = lambda: None
    method._evaluation_executor = SimpleNamespace(shutdown=lambda **_: None)
    method._llm = SimpleNamespace(close=lambda: None)
    with pytest.raises(RuntimeError, match="initialization obtained only"):
        method.run()


def test_summary_rejects_failed_or_incomplete_run():
    read_run = runpy.run_path(str(Path(__file__).parents[4] / "scripts/eoh_local_rl/summarize_comparison.py"))["read_run"]
    with TemporaryDirectory() as directory:
        run = Path(directory)
        Path(run, "checkpoints").mkdir()
        Path(run, "config.json").write_text(json.dumps({"online_update_count": 2, "grpo_enabled": False, "run_id": "x", "seed": 42}), encoding="utf-8")
        Path(run, "run_summary.json").write_text(json.dumps({"online_update_count": 1}), encoding="utf-8")
        Path(run, "run_manifest.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
        Path(run, "checkpoints/latest_finish.json").write_text(json.dumps({"best_curve": []}), encoding="utf-8")
        assert read_run(run) is None


def test_profiler_archives_history():
    with TemporaryDirectory() as log_dir:
        profiler = EoHProfiler(log_dir=log_dir)
        Path(log_dir, "online_grpo").mkdir()
        Path(log_dir, "population", "one.json").write_text("{}", encoding="utf-8")
        Path(log_dir, "samples", "one.json").write_text("[]", encoding="utf-8")
        Path(log_dir, "online_grpo", "one.json").write_text("{}", encoding="utf-8")
        path = profiler.archive_history()
        with tarfile.open(path, "r:gz") as archive:
            assert {"population/one.json", "samples/one.json", "online_grpo/one.json"}.issubset(archive.getnames())
        assert not Path(log_dir, "population").exists()
