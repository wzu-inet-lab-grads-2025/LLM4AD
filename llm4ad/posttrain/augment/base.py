from __future__ import annotations


class AugmenterBase:
    def augment(self, records, *, context=None):
        raise NotImplementedError
