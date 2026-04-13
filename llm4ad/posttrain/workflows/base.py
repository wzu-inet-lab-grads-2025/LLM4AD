from __future__ import annotations

from pathlib import Path

from llm4ad.posttrain.eval_trace import EvalTraceRecorder
from llm4ad.posttrain.llm_proxy import PostTrainLLMProxy


class BaseWorkflow:
    def __init__(
        self,
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
        self.config = config
        self.llm = llm
        self.evaluation = evaluation
        self.profiler = profiler
        self.method_kwargs = method_kwargs or {}
        self.runtime = runtime
        self.event_store = event_store
        self.llm_builder = llm_builder
        self.eval_trace_recorder = EvalTraceRecorder(runtime, event_store)

    def build_wrapped_llm(self):
        return PostTrainLLMProxy(self.llm, self.runtime, self.event_store)

    def build_profiler(self, *, log_dir=None, create_random_path=False):
        if self.profiler is None:
            return None
        profiler_cls = self.profiler.__class__
        return profiler_cls(
            log_dir=log_dir or self.profiler._log_dir,
            initial_num_samples=0,
            log_style=getattr(self.profiler, "_log_style", "complex"),
            create_random_path=create_random_path,
        )

    def build_adapter(self):
        raise NotImplementedError

    def build_method(self, llm, evaluation, profiler, adapter):
        raise NotImplementedError

    def get_resume_fn(self):
        raise NotImplementedError

    def build_validation_spec(self):
        return None

    def build_smoke_spec(self):
        if not hasattr(self, "run_search_round"):
            return None
        if self.llm_builder is None:
            return None
        from llm4ad.posttrain.smoke import WorkflowSmokeRunner

        return WorkflowSmokeRunner(self._build_smoke_workflow)

    def build_llm_for_model_ref(self, *, model_ref, line_name, context=None):
        if self.llm_builder is None:
            return None
        return self.llm_builder(
            model_ref=model_ref,
            line_name=line_name,
            config=self.config,
            base_llm=self.llm,
            context=context,
        )

    def _build_smoke_workflow(self, *, model_ref, line_name, context=None):
        llm = self.build_llm_for_model_ref(
            model_ref=model_ref,
            line_name=line_name,
            context=context,
        )
        if llm is None:
            return None

        from llm4ad.posttrain.event_store import EventStore
        from llm4ad.posttrain.runtime import PostTrainRuntime

        smoke_root = self._build_smoke_root(
            model_ref=model_ref, line_name=line_name, context=context
        )
        smoke_runtime = PostTrainRuntime(
            self.config,
            workflow_name=f"{self.config.workflow.workflow_name}_smoke_{line_name}",
            task_name=self.config.workflow.task_name,
            method_name=self.config.workflow.method_name,
        )
        smoke_event_store = EventStore(smoke_root / "events")
        smoke_profiler = self.build_profiler(
            log_dir=str(smoke_root / "logs"),
            create_random_path=True,
        )
        if smoke_profiler is not None:
            smoke_profiler._log_dir = str(smoke_root / "logs")
        return self.__class__(
            config=self.config,
            llm=llm,
            evaluation=self.evaluation,
            profiler=smoke_profiler,
            method_kwargs=self._build_smoke_method_kwargs(),
            runtime=smoke_runtime,
            event_store=smoke_event_store,
            llm_builder=self.llm_builder,
        )

    def _build_smoke_root(self, *, model_ref, line_name, context=None):
        version = "unknown"
        if isinstance(model_ref, dict):
            version = model_ref.get("version_id", version)
        base_dir = (
            Path(self.config.registry.artifact_root)
            / "smoke"
            / self.runtime.experiment_name
            / line_name
            / version
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def _build_smoke_method_kwargs(self):
        smoke_kwargs = dict(self.method_kwargs)
        smoke_max_samples = getattr(self.config.gate, "smoke_max_samples", None)
        smoke_max_generations = getattr(self.config.gate, "smoke_max_generations", None)

        if smoke_max_samples is not None:
            smoke_kwargs["max_sample_nums"] = smoke_max_samples
        if smoke_max_generations is not None:
            smoke_kwargs["max_generations"] = smoke_max_generations

        return smoke_kwargs
