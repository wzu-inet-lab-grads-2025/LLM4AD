from __future__ import annotations


class DatasetFormatterBase:
    def format(self, examples, *, output_dir, dataset_name, context=None):
        raise NotImplementedError
