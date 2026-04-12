from __future__ import annotations

from .base import AugmenterBase


class RejudgeAugmenter(AugmenterBase):
    def __init__(self, evaluator_callable=None):
        self._evaluator_callable = evaluator_callable

    def augment(self, records, *, context=None):
        if self._evaluator_callable is None:
            return records
        enriched = []
        for record in records:
            updated = self._evaluator_callable(record=record, context=context)
            enriched.append(updated if updated is not None else record)
        return enriched
