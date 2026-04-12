from __future__ import annotations

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
    ):
        self.config = config
        self.llm = llm
        self.evaluation = evaluation
        self.profiler = profiler
        self.method_kwargs = method_kwargs or {}
        self.runtime = runtime
        self.event_store = event_store
        self.eval_trace_recorder = EvalTraceRecorder(runtime, event_store)

    def build_wrapped_llm(self):
        return PostTrainLLMProxy(self.llm, self.runtime, self.event_store)

    def build_profiler(self):
        if self.profiler is None:
            return None
        profiler_cls = self.profiler.__class__
        return profiler_cls(
            log_dir=self.profiler._log_dir,
            initial_num_samples=0,
            log_style=getattr(self.profiler, "_log_style", "complex"),
            create_random_path=False,
        )

    def build_adapter(self):
        raise NotImplementedError

    def build_method(self, llm, evaluation, profiler, adapter):
        raise NotImplementedError

    def get_resume_fn(self):
        raise NotImplementedError
