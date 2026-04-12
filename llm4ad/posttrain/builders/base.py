from __future__ import annotations


class DatasetBuilderBase:
    def build(self, records, *, config, context):
        raise NotImplementedError
