from __future__ import annotations

import json
from pathlib import Path

from .base import DatasetFormatterBase


class ProcessJsonlFormatter(DatasetFormatterBase):
    def format(self, examples, *, output_dir, dataset_name, context=None):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{dataset_name}.jsonl"
        with file_path.open("w", encoding="utf-8") as fh:
            for example in examples:
                fh.write(json.dumps(example, ensure_ascii=True) + "\n")
        return str(file_path)
