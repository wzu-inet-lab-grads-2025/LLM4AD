from __future__ import annotations

from .base import MethodAdapterBase


class RichTraceAdapter(MethodAdapterBase):
    def set_context(
        self,
        *,
        phase: str,
        operator: str | None,
        parents: list[int] | None,
        generation: int | None = None,
        cluster_id: int | None = None,
        **extra,
    ):
        if self._runtime is None:
            return
        self._runtime.set_phase(phase)
        self._runtime.set_operator(operator)
        self._runtime.set_parents(parents)
        self._runtime.set_generation(generation)
        self._runtime.set_cluster(cluster_id)
        if extra:
            self._runtime.set_extra(**extra)
