from __future__ import annotations

from .base import DatasetBuilderBase


class VerifierBuilder(DatasetBuilderBase):
    def build(self, records, *, config, context):
        examples = []
        include_init = getattr(config, "include_init_phase", False)
        for record in records:
            if record.phase == "init" and not include_init:
                continue
            examples.append(
                {
                    "sample_id": record.sample_id,
                    "prompt": record.prompt,
                    "response": record.response,
                    "function": record.function,
                    "program": record.program,
                    "score": record.score,
                    "operator": record.operator,
                    "phase": record.phase,
                    "provenance": record.provenance,
                }
            )
        return examples
