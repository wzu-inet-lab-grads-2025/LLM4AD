from __future__ import annotations

from .base import AugmenterBase


class DedupAugmenter(AugmenterBase):
    def augment(self, records, *, context=None):
        seen = set()
        deduped = []
        for record in records:
            key = (record.prompt, record.function, record.response, repr(record.score))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped
