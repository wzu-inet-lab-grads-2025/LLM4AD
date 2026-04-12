from __future__ import annotations

from .base import DatasetBuilderBase


class ProcessBuilder(DatasetBuilderBase):
    def build(self, records, *, config, context):
        examples = []
        include_init = getattr(config, "include_init_phase", False)
        for record in records:
            if record.phase == "init" and not include_init:
                continue
            if not (
                record.prompt or record.messages or record.response or record.algorithm
            ):
                continue
            examples.append(
                {
                    "sample_id": record.sample_id,
                    "phase": record.phase,
                    "operator": record.operator,
                    "prompt": record.prompt,
                    "messages": record.messages,
                    "algorithm": record.algorithm,
                    "response": record.response,
                    "function": record.function,
                    "score": record.score,
                    "observation": record.observation,
                    "provenance": record.provenance,
                }
            )
        return examples
