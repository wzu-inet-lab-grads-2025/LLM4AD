from __future__ import annotations

from .base import DatasetBuilderBase


class OutcomeBuilder(DatasetBuilderBase):
    def build(self, records, *, config, context):
        examples = []
        include_init = getattr(config, "include_init_phase", False)
        for record in records:
            if record.phase == "init" and not include_init:
                continue
            completion = record.response or record.function
            if not record.prompt or not completion:
                continue
            examples.append(
                {
                    "prompt": record.prompt,
                    "completion": completion,
                    "score": record.score,
                    "operator": record.operator,
                    "phase": record.phase,
                    "sample_id": record.sample_id,
                }
            )
        return examples
