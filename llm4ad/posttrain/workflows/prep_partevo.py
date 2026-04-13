from __future__ import annotations

from llm4ad.method.partevo import PartEvo

from llm4ad.posttrain.adapters.partevo import PartEvoAdapter
from llm4ad.posttrain.builders import OutcomeBuilder, ProcessBuilder, VerifierBuilder
from llm4ad.posttrain.collector import UnifiedCollector
from llm4ad.posttrain.formatters import (
    ProcessJsonlFormatter,
    TrlChatFormatter,
    VerifierJsonlFormatter,
)
from llm4ad.posttrain.schemas import DatasetManifest

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

    def build_default_datasets(self, search_report):
        collector = UnifiedCollector(self.config.collector)
        records = collector.collect_round_records(
            log_dir=search_report.get("log_dir"),
            event_store=self.event_store,
            run_id=self.runtime.run_id,
            round_id=self.runtime.round_id,
            method_name=self.config.workflow.method_name,
        )
        datasets_dir = self._get_datasets_dir(search_report.get("log_dir"))
        context = self.runtime.snapshot()

        outcome_examples = OutcomeBuilder().build(
            records, config=self.config.builder, context=context
        )
        process_examples = ProcessBuilder().build(
            records, config=self.config.builder, context=context
        )
        verifier_examples = VerifierBuilder().build(
            records, config=self.config.builder, context=context
        )

        return {
            "outcome": DatasetManifest(
                round_id=self.runtime.round_id,
                dataset_type="outcome",
                source_runs=[self.runtime.run_id],
                source_rounds=[self.runtime.round_id],
                record_count=len(outcome_examples),
                output_path=TrlChatFormatter().format(
                    outcome_examples,
                    output_dir=datasets_dir,
                    dataset_name=f"round_{self.runtime.round_id:04d}_partevo_outcome",
                    context=context,
                ),
                config_digest="partevo_outcome_v1",
            ),
            "process": DatasetManifest(
                round_id=self.runtime.round_id,
                dataset_type="process",
                source_runs=[self.runtime.run_id],
                source_rounds=[self.runtime.round_id],
                record_count=len(process_examples),
                output_path=ProcessJsonlFormatter().format(
                    process_examples,
                    output_dir=datasets_dir,
                    dataset_name=f"round_{self.runtime.round_id:04d}_partevo_process",
                    context=context,
                ),
                config_digest="partevo_process_v1",
            ),
            "verifier": DatasetManifest(
                round_id=self.runtime.round_id,
                dataset_type="verifier",
                source_runs=[self.runtime.run_id],
                source_rounds=[self.runtime.round_id],
                record_count=len(verifier_examples),
                output_path=VerifierJsonlFormatter().format(
                    verifier_examples,
                    output_dir=datasets_dir,
                    dataset_name=f"round_{self.runtime.round_id:04d}_partevo_verifier",
                    context=context,
                ),
                config_digest="partevo_verifier_v1",
            ),
        }

    def _get_datasets_dir(self, log_dir: str | None):
        from pathlib import Path

        if log_dir is None:
            base_dir = Path(self.config.registry.artifact_root) / "datasets"
        else:
            base_dir = Path(log_dir) / "posttrain" / "datasets"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir


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


def run_prep_partevo_and_build(
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
    search_report = workflow.run_collection()
    manifests = workflow.build_default_datasets(search_report)
    return {
        "search_report": search_report,
        "dataset_manifests": {
            name: manifest.__dict__ for name, manifest in manifests.items()
        },
    }


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
