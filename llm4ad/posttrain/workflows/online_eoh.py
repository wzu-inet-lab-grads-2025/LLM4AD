from __future__ import annotations

from llm4ad.method.eoh import EoH
from llm4ad.method.eoh.resume import resume_eoh

from llm4ad.posttrain.adapters import EvolutionAdapter

from .online_round import OnlineRoundWorkflow


class EoHOnlineWorkflow(OnlineRoundWorkflow):
    def build_adapter(self):
        return EvolutionAdapter(
            round_config=self.config.round, event_store=self.event_store
        )

    def build_method(self, llm, evaluation, profiler, adapter):
        return EoH(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler,
            posttrain_runtime=self.runtime,
            posttrain_adapter=adapter,
            eval_trace_recorder=self.eval_trace_recorder,
            **self.method_kwargs,
        )

    def get_resume_fn(self):
        return resume_eoh


def run_online_eoh(
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
    workflow = build_online_eoh_workflow(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
    )
    return workflow.run_search_round(resume_path=resume_path)


def build_online_eoh_workflow(
    *,
    config,
    llm,
    evaluation,
    profiler=None,
    method_kwargs=None,
    runtime,
    event_store,
):
    return EoHOnlineWorkflow(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
    )
