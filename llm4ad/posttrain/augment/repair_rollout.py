from __future__ import annotations

from .base import AugmenterBase


class RepairRolloutAugmenter(AugmenterBase):
    def __init__(self, repair_callable=None):
        self._repair_callable = repair_callable

    def augment(self, records, *, context=None):
        if self._repair_callable is None:
            return records
        augmented = list(records)
        for record in records:
            repaired = self._repair_callable(record=record, context=context)
            if repaired is None:
                continue
            if isinstance(repaired, list):
                augmented.extend(repaired)
            else:
                augmented.append(repaired)
        return augmented
