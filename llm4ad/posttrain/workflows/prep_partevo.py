from __future__ import annotations

from llm4ad.method.partevo import PartEvo

from llm4ad.posttrain.adapters.partevo import PartEvoAdapter

from .base import BaseWorkflow


class PartEvoPrepWorkflow(BaseWorkflow):
    def build_adapter(self):
        return PartEvoAdapter(
            round_config=self.config.round, event_store=self.event_store
        )

    def build_method(self, llm, evaluation, profiler, adapter):
        method = PartEvo(
            llm=llm,
            evaluation=evaluation,
            profiler=profiler,
            posttrain_runtime=self.runtime,
            posttrain_adapter=adapter,
            eval_trace_recorder=self.eval_trace_recorder,
            **self.method_kwargs,
        )
        adapter.bind(method, self.runtime, self.event_store)
        return method

    def run_collection(self):
        self.runtime.begin_round("search")
        method = self.build_method(
            self.build_wrapped_llm(),
            self.evaluation,
            self.build_profiler(),
            self.build_adapter(),
        )
        try:
            method.run()
        finally:
            self.runtime.end_round()
        return {
            "log_dir": None if method._profiler is None else method._profiler._log_dir,
            "round_id": self.runtime.round_id,
        }


def run_prep_partevo(
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
    workflow = build_prep_partevo_workflow(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
        llm_builder=llm_builder,
    )
    return workflow.run_collection()


def build_prep_partevo_workflow(
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
    return PartEvoPrepWorkflow(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
        llm_builder=llm_builder,
    )
