from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from llm4ad.base import Evaluation
from llm4ad.posttrain.config import (
    BuilderConfig,
    GateConfig,
    PostTrainConfig,
    TrainerConfig,
    WorkflowConfig,
)
from llm4ad.posttrain.event_store import EventStore
from llm4ad.posttrain.orchestrator import PostTrainOrchestrator
from llm4ad.posttrain.runtime import PostTrainRuntime
from llm4ad.posttrain.workflows import build_online_eoh_workflow
from llm4ad.tools.llm.local_transformers import LocalTransformersLLM


TEMPLATE_PROGRAM = '''
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Return a score for each bin. Higher is better."""
    return bins - item
'''


class ToyPriorityEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program=TEMPLATE_PROGRAM,
            task_description=(
                "Design a simple NumPy-based priority heuristic for online bin packing. "
                "The function should return a score for each bin, with higher scores meaning a better bin."
            ),
            timeout_seconds=10,
            safe_evaluate=True,
            exec_code=True,
        )

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs):
        if callable_func is None:
            return None

        test_cases = [
            (0.2, np.array([0.3, 0.5, 0.9], dtype=float)),
            (0.4, np.array([0.45, 0.7, 0.95], dtype=float)),
        ]

        total_score = 0.0
        for item, bins in test_cases:
            result = callable_func(item, bins.copy())
            if not isinstance(result, np.ndarray):
                return -100.0
            if result.shape != bins.shape:
                return -100.0
            if not np.all(np.isfinite(result)):
                return -100.0

            feasible = bins >= item
            if feasible.any():
                chosen = int(np.argmax(result))
                if feasible[chosen]:
                    total_score += 1.0
                else:
                    total_score -= 1.0
            total_score += float(np.mean(result[feasible])) if feasible.any() else 0.0

        return total_score


def build_local_llm(
    *, model_ref=None, line_name=None, config=None, base_llm=None, context=None
):
    model_path = config.trainer.base_model if config is not None else None
    if isinstance(model_ref, dict):
        candidate_path = model_ref.get("path")
        if candidate_path and (Path(candidate_path) / "config.json").exists():
            model_path = candidate_path
    if model_path is None and isinstance(base_llm, LocalTransformersLLM):
        model_path = base_llm._model_path

    return LocalTransformersLLM(
        model_path=model_path,
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
    )


def main():
    model_path = "models/Qwen2.5-Coder-1.5B-Instruct"
    llm = LocalTransformersLLM(
        model_path=model_path,
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
    )

    config = PostTrainConfig(
        workflow=WorkflowConfig(
            workflow_name="online_eoh",
            task_name="toy_priority",
            method_name="EoH",
            experiment_name="local_qwen_toy_orchestrator_demo",
        ),
        trainer=TrainerConfig(
            backend="dryrun",
            base_model=model_path,
        ),
        builder=BuilderConfig(
            include_init_phase=True,
        ),
        gate=GateConfig(
            use_fixed_validation=False,
            use_smoke_test=False,
        ),
    )

    runtime = PostTrainRuntime(
        config,
        workflow_name=config.workflow.workflow_name,
        task_name=config.workflow.task_name,
        method_name=config.workflow.method_name,
    )
    event_store = EventStore(
        Path("logs") / "posttrain_demo" / runtime.run_id / "events"
    )

    workflow = build_online_eoh_workflow(
        config=config,
        llm=llm,
        evaluation=ToyPriorityEvaluation(),
        profiler=None,
        method_kwargs={
            "max_sample_nums": 1,
            "max_generations": 1,
            "pop_size": 2,
            "num_samplers": 1,
            "num_evaluators": 1,
        },
        runtime=runtime,
        event_store=event_store,
        llm_builder=build_local_llm,
    )

    try:
        orchestrator = PostTrainOrchestrator(config, runtime, event_store)
        report = orchestrator.run_round(workflow)
        print(json.dumps(report, indent=2, ensure_ascii=True, default=str))
    finally:
        llm.close()


if __name__ == "__main__":
    main()
