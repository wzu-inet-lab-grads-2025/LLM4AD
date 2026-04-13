from __future__ import annotations

from llm4ad.method.reevo import ReEvo
from llm4ad.method.reevo.resume import resume_reevo

from llm4ad.posttrain.adapters import ReEvoAdapter

from .online_round import OnlineRoundWorkflow


class ReEvoOnlineWorkflow(OnlineRoundWorkflow):
    def build_adapter(self):
        return ReEvoAdapter(
            round_config=self.config.round, event_store=self.event_store
        )

    def build_method(self, llm, evaluation, profiler, adapter):
        return ReEvo(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler,
            posttrain_runtime=self.runtime,
            posttrain_adapter=adapter,
            eval_trace_recorder=self.eval_trace_recorder,
            **self.method_kwargs,
        )

    def get_resume_fn(self):
        return resume_reevo


def run_online_reevo(
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
    workflow = build_online_reevo_workflow(
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


def build_online_reevo_workflow(
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
    return ReEvoOnlineWorkflow(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
        llm_builder=llm_builder,
    )
