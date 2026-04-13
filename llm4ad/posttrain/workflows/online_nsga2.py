from __future__ import annotations

from llm4ad.method.nsga2 import NSGA2
from llm4ad.method.nsga2.resume import resume_nsga2

from llm4ad.posttrain.adapters import EvolutionAdapter

from .online_round import OnlineRoundWorkflow


class NSGA2OnlineWorkflow(OnlineRoundWorkflow):
    def build_adapter(self):
        return EvolutionAdapter(
            round_config=self.config.round,
            event_store=self.event_store,
        )

    def build_method(self, llm, evaluation, profiler, adapter):
        return NSGA2(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler,
            posttrain_runtime=self.runtime,
            posttrain_adapter=adapter,
            eval_trace_recorder=self.eval_trace_recorder,
            **self.method_kwargs,
        )

    def get_resume_fn(self):
        return resume_nsga2


def run_online_nsga2(
    *,
    config,
    llm,
    evaluation,
    profiler=None,
    method_kwargs=None,
    runtime,
    event_store,
    llm_builder=None,
    resume_path=None,
):
    workflow = build_online_nsga2_workflow(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
        llm_builder=llm_builder,
    )
    return workflow.run_search_round(resume_path=resume_path)


def build_online_nsga2_workflow(
    *,
    config,
    llm,
    evaluation,
    profiler=None,
    method_kwargs=None,
    runtime,
    event_store,
    llm_builder=None,
):
    return NSGA2OnlineWorkflow(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
        llm_builder=llm_builder,
    )
