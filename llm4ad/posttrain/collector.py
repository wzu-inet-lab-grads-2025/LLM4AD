from __future__ import annotations

import json
from pathlib import Path

from .schemas import SampleRecord


class UnifiedCollector:
    def __init__(self, config):
        self._config = config

    def collect_round_records(
        self,
        *,
        log_dir: str | None,
        event_store,
        run_id: str,
        round_id: int,
        method_name: str = "unknown",
    ) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        if self._config.include_event_store:
            for payload in event_store.iter_events(
                "SampleRecord", run_id=run_id, round_id=round_id
            ):
                records.append(SampleRecord(**payload))

        if self._config.include_history_logs and log_dir:
            history_records = self._read_history_log_dir(
                log_dir=log_dir,
                run_id=run_id,
                round_id=round_id,
                method_name=method_name,
            )
            records = self._merge_records(records, history_records)
        return records

    def import_history_runs(
        self, log_dirs: list[str], *, method_name: str = "unknown"
    ) -> list[SampleRecord]:
        all_records: list[SampleRecord] = []
        for log_dir in log_dirs:
            run_id = f"history::{Path(log_dir).name}"
            records = self._read_history_log_dir(
                log_dir=log_dir,
                run_id=run_id,
                round_id=0,
                method_name=method_name,
            )
            all_records.extend(records)
        return all_records

    def _read_history_log_dir(
        self,
        *,
        log_dir: str,
        run_id: str,
        round_id: int,
        method_name: str,
    ) -> list[SampleRecord]:
        samples_dir = Path(log_dir) / "samples"
        if not samples_dir.exists():
            return []

        records: list[SampleRecord] = []
        for sample_file in sorted(samples_dir.glob("*.json")):
            if sample_file.name == "samples_best.json":
                continue
            try:
                payload = json.loads(sample_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            samples = payload if isinstance(payload, list) else [payload]
            for sample in samples:
                sample_order = sample.get("sample_order", 0)
                records.append(
                    SampleRecord(
                        run_id=run_id,
                        round_id=round_id,
                        sample_id=f"history_{sample_order:08d}",
                        method_name=method_name,
                        phase="history",
                        operator=sample.get("operator"),
                        parents=None,
                        generation=None,
                        prompt=sample.get("prompt"),
                        messages=sample.get("messages"),
                        response=sample.get("response"),
                        function=sample.get("function"),
                        program=sample.get("program"),
                        algorithm=sample.get("algorithm"),
                        score=sample.get("score"),
                        sample_time=sample.get("sample_time"),
                        eval_time=sample.get("evaluate_time"),
                        observation=sample.get("observation"),
                        image64=sample.get("image64"),
                        provenance={"source": "history_log", "file": str(sample_file)},
                    )
                )
        return records

    def _merge_records(
        self, primary: list[SampleRecord], secondary: list[SampleRecord]
    ) -> list[SampleRecord]:
        seen = {
            (record.function, record.program, repr(record.score), record.prompt)
            for record in primary
        }
        merged = list(primary)
        for record in secondary:
            key = (record.function, record.program, repr(record.score), record.prompt)
            if key in seen:
                continue
            merged.append(record)
            seen.add(key)
        return merged
