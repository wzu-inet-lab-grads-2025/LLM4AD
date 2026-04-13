from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

from .schemas import RunContext


class PostTrainRuntime:
    def __init__(self, config, workflow_name: str, task_name: str, method_name: str):
        self._config = config
        self._workflow_name = workflow_name
        self._task_name = task_name
        self._method_name = method_name
        self._experiment_name = self._resolve_experiment_name()
        self._run_id = self.build_run_id()
        self._round_id = 0
        self._sample_counter = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        self._base_context = RunContext(
            task_name=task_name,
            workflow_name=workflow_name,
            experiment_name=self._experiment_name,
            method_name=method_name,
            run_id=self._run_id,
            round_id=0,
            phase="idle",
        )
        self._set_context(self._base_context)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def round_id(self) -> int:
        return self._round_id

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    def _resolve_experiment_name(self) -> str:
        workflow_cfg = getattr(self._config, "workflow", None)
        explicit_name = getattr(workflow_cfg, "experiment_name", None)
        if explicit_name:
            return explicit_name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = f"{self._method_name.lower()}_{getattr(self._config.trainer, 'backend', 'na')}"
        return f"{stamp}_{summary}"

    def build_run_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{self._workflow_name}"

    def next_round(self) -> int:
        with self._lock:
            self._round_id += 1
            return self._round_id

    def begin_round(self, phase: str = "search") -> RunContext:
        round_id = self.next_round()
        ctx = replace(
            self._base_context,
            round_id=round_id,
            phase=phase,
            sample_id=None,
            operator=None,
            parents=None,
            generation=None,
            cluster_id=None,
            thread_id=None,
            extra={},
        )
        self._base_context = ctx
        self._set_context(ctx)
        return self.snapshot()

    def end_round(self) -> None:
        ctx = self.snapshot()
        self._base_context = replace(
            self._base_context,
            round_id=ctx.round_id,
            phase="idle",
            sample_id=None,
            operator=None,
            parents=None,
            generation=None,
            cluster_id=None,
            thread_id=None,
            extra={},
        )
        self._set_context(self._base_context)

    def begin_sample(self) -> str:
        with self._lock:
            self._sample_counter += 1
            sample_id = f"s_{self._sample_counter:08d}"
        ctx = replace(self.snapshot(), sample_id=sample_id)
        self._set_context(ctx)
        return sample_id

    def set_phase(self, phase: str) -> None:
        self._set_context(replace(self.snapshot(), phase=phase))

    def set_operator(self, operator: str | None) -> None:
        self._set_context(replace(self.snapshot(), operator=operator))

    def set_parents(self, parents: list[int] | None) -> None:
        copied = list(parents) if parents is not None else None
        self._set_context(replace(self.snapshot(), parents=copied))

    def set_generation(self, generation: int | None) -> None:
        self._set_context(replace(self.snapshot(), generation=generation))

    def set_cluster(self, cluster_id: int | None) -> None:
        self._set_context(replace(self.snapshot(), cluster_id=cluster_id))

    def set_thread_id(self, thread_id: int | None) -> None:
        self._set_context(replace(self.snapshot(), thread_id=thread_id))

    def set_extra(self, **kwargs) -> None:
        ctx = self.snapshot()
        extra = dict(ctx.extra)
        extra.update(kwargs)
        self._set_context(replace(ctx, extra=extra))

    def snapshot(self) -> RunContext:
        return deepcopy(self._get_context())

    def _get_context(self) -> RunContext:
        ctx = getattr(self._local, "context", None)
        if ctx is None:
            ctx = self._base_context
            self._local.context = ctx
        return ctx

    def _set_context(self, context: RunContext) -> None:
        self._local.context = context
