from __future__ import annotations

from llm4ad.method.meoh import MEoH
from llm4ad.method.meoh.resume import resume_meoh

from llm4ad.posttrain.adapters import EvolutionAdapter

from .online_round import OnlineRoundWorkflow


class MEoHOnlineWorkflow(OnlineRoundWorkflow):
    def build_adapter(self):
        return EvolutionAdapter(
            round_config=self.config.round,
            event_store=self.event_store,
        )

    def build_method(self, llm, evaluation, profiler, adapter):
        return MEoH(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler,
            posttrain_runtime=self.runtime,
            posttrain_adapter=adapter,
            eval_trace_recorder=self.eval_trace_recorder,
            **self.method_kwargs,
        )

    def get_resume_fn(self):
        return resume_meoh


def run_online_meoh(
    *,
    config,
    llm,
    evaluation,
    profiler=None,
    method_kwargs=None,
    runtime,
    event_store,
    resume_path=None,
):
    workflow = build_online_meoh_workflow(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
    )
    return workflow.run_search_round(resume_path=resume_path)


def build_online_meoh_workflow(
    *,
    config,
    llm,
    evaluation,
    profiler=None,
    method_kwargs=None,
    runtime,
    event_store,
):
    return MEoHOnlineWorkflow(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
    )
