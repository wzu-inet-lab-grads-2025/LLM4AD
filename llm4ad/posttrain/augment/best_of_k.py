from __future__ import annotations

from collections import defaultdict

from .base import AugmenterBase


def _scalar(score):
    if isinstance(score, (int, float)):
        return float(score)
    return None


class BestOfKAugmenter(AugmenterBase):
    def __init__(self, k: int = 4):
        self._k = k

    def augment(self, records, *, context=None):
        grouped = defaultdict(list)
        for record in records:
            grouped[record.prompt].append(record)

        selected = []
        for _, items in grouped.items():
            items = [item for item in items if _scalar(item.score) is not None]
            items.sort(key=lambda item: _scalar(item.score), reverse=True)
            selected.extend(items[: self._k])
        return selected
