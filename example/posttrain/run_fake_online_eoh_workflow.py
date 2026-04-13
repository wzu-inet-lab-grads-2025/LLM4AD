from __future__ import annotations

import pickle
import random
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from llm4ad.base import LLM
from llm4ad.posttrain.config import PostTrainConfig, TrainerConfig, WorkflowConfig
from llm4ad.posttrain.event_store import EventStore
from llm4ad.posttrain.runtime import PostTrainRuntime
from llm4ad.posttrain.workflows import build_online_eoh_workflow
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation


class FakeLLM(LLM):
    def __init__(self, data_path: str):
        super().__init__()
        self._data_path = data_path
        with open(data_path, "rb") as fh:
            self._functions = pickle.load(fh)

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        fake_thought = "{This is a fake thought for the code}\n"
        rand_func = random.choice(self._functions)
        return fake_thought + rand_func


def build_fake_llm(
    *, model_ref=None, line_name=None, config=None, base_llm=None, context=None
):
    if isinstance(base_llm, FakeLLM):
        return FakeLLM(base_llm._data_path)
    data_path = (
        Path(__file__).resolve().parent.parent
        / "llms"
        / "online_bin_packing_fake"
        / "_data"
        / "rand_function.pkl"
    )
    return FakeLLM(str(data_path))


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
            experiment_name="fake_eoh_demo",
        ),
        trainer=TrainerConfig(base_model="fake-local-model"),
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

    report = workflow.run_search_round()
    print(report)


if __name__ == "__main__":
    main()
