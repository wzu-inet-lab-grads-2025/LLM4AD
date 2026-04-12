from __future__ import annotations

from .rich_trace import RichTraceAdapter


class PartEvoAdapter(RichTraceAdapter):
    def before_cluster_sample(
        self,
        *,
        operator: str,
        parents: list[int] | None,
        cluster_id: int | None,
        generation: int | None,
    ):
        self.set_context(
            phase="search",
            operator=operator,
            parents=parents,
            generation=generation,
            cluster_id=cluster_id,
        )
