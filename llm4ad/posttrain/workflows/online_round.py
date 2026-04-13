from __future__ import annotations

import inspect

from .base import BaseWorkflow


class OnlineRoundWorkflow(BaseWorkflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._method = None
        self._adapter = None

    def build_or_resume_method(self, *, resume_path: str | None = None):
        wrapped_llm = self.build_wrapped_llm()
        profiler = self.build_profiler()
        adapter = self.build_adapter()
        method = self.build_method(wrapped_llm, self.evaluation, profiler, adapter)

        if resume_path is not None:
            resume_fn = self.get_resume_fn()
            param_num = len(inspect.signature(resume_fn).parameters)
            if param_num >= 2:
                resume_fn(method, resume_path)
            else:
                resume_fn(method)

        adapter.bind(method, self.runtime, self.event_store)

        self._adapter = adapter
        self._method = method
        return method

    def run_search_round(self, *, resume_path: str | None = None):
        self.runtime.begin_round("search")
        try:
            method = self.build_or_resume_method(resume_path=resume_path)
            method.run()
        finally:
            self.runtime.end_round()
        return {
            "log_dir": None if method._profiler is None else method._profiler._log_dir,
            "round_id": self.runtime.round_id,
        }
