from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

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
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.local_transformers import LocalTransformersLLM


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="models/Qwen2.5-Coder-1.5B-Instruct",
    )
    parser.add_argument("--experiment-name", default="local_qwen_eoh_orchestrator_demo")
    parser.add_argument("--max-sample-nums", type=int, default=1)
    parser.add_argument("--max-generations", type=int, default=1)
    parser.add_argument("--pop-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-smoke", action="store_true")
    parser.add_argument("--smoke-max-samples", type=int, default=1)
    return parser


def _resolve_model_path(model_ref, *, fallback_model_path: str) -> str:
    if isinstance(model_ref, dict):
        candidate_path = model_ref.get("path")
        if candidate_path:
            candidate_dir = Path(candidate_path)
            if candidate_dir.exists() and (candidate_dir / "config.json").exists():
                return str(candidate_dir)
    return fallback_model_path


def build_local_llm(
    *,
    model_ref=None,
    line_name=None,
    config=None,
    base_llm=None,
    context=None,
    max_new_tokens=64,
    temperature=0.0,
    top_p=1.0,
):
    fallback_model_path = config.trainer.base_model if config is not None else None
    if fallback_model_path is None and isinstance(base_llm, LocalTransformersLLM):
        fallback_model_path = base_llm._model_path
    model_path = _resolve_model_path(model_ref, fallback_model_path=fallback_model_path)
    return LocalTransformersLLM(
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def main():
    args = build_parser().parse_args()

    llm = LocalTransformersLLM(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    config = PostTrainConfig(
        workflow=WorkflowConfig(
            workflow_name="online_eoh",
            task_name="online_bin_packing",
            method_name="EoH",
            experiment_name=args.experiment_name,
        ),
        trainer=TrainerConfig(
            backend="dryrun",
            base_model=args.model_path,
        ),
        builder=BuilderConfig(
            include_init_phase=True,
        ),
        gate=GateConfig(
            use_fixed_validation=False,
            use_smoke_test=args.enable_smoke,
            smoke_max_samples=args.smoke_max_samples,
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

    def local_llm_builder(
        *, model_ref=None, line_name=None, config=None, base_llm=None, context=None
    ):
        return build_local_llm(
            model_ref=model_ref,
            line_name=line_name,
            config=config,
            base_llm=base_llm,
            context=context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    workflow = build_online_eoh_workflow(
        config=config,
        llm=llm,
        evaluation=OBPEvaluation(),
        profiler=None,
        method_kwargs={
            "max_sample_nums": args.max_sample_nums,
            "max_generations": args.max_generations,
            "pop_size": args.pop_size,
            "num_samplers": 1,
            "num_evaluators": 1,
        },
        runtime=runtime,
        event_store=event_store,
        llm_builder=local_llm_builder,
    )

    try:
        orchestrator = PostTrainOrchestrator(config, runtime, event_store)
        report = orchestrator.run_round(workflow)
        print(json.dumps(report, indent=2, ensure_ascii=True, default=str))
    finally:
        llm.close()


if __name__ == "__main__":
    main()
