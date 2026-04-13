from __future__ import annotations

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

from run_fake_online_eoh_workflow import FakeLLM, build_fake_llm


def main():
    data_path = (
        Path(__file__).resolve().parent.parent
        / "llms"
        / "online_bin_packing_fake"
        / "_data"
        / "rand_function.pkl"
    )
    llm = FakeLLM(str(data_path))

    config = PostTrainConfig(
        workflow=WorkflowConfig(
            workflow_name="online_eoh",
            task_name="online_bin_packing",
            method_name="EoH",
            experiment_name="fake_eoh_orchestrator_demo",
        ),
        trainer=TrainerConfig(
            backend="dryrun",
            base_model="fake-local-model",
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
        evaluation=OBPEvaluation(),
        profiler=None,
        method_kwargs={
            "max_sample_nums": 10,
            "max_generations": 2,
            "pop_size": 2,
            "num_samplers": 1,
            "num_evaluators": 1,
        },
        runtime=runtime,
        event_store=event_store,
        llm_builder=build_fake_llm,
    )

    orchestrator = PostTrainOrchestrator(config, runtime, event_store)
    report = orchestrator.run_round(workflow)
    print(json.dumps(report, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
