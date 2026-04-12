from __future__ import annotations

from .rich_trace import RichTraceAdapter


class MLESAdapter(RichTraceAdapter):
    def before_multimodal_sample(
        self,
        *,
        operator: str,
        parents: list[int] | None,
        generation: int | None,
        modality: str,
        prompt_variant: str,
    ):
        self.set_context(
            phase="search",
            operator=operator,
            parents=parents,
            generation=generation,
            modality=modality,
            prompt_variant=prompt_variant,
        )
