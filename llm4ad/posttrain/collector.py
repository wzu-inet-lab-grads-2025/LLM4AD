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
            records.extend(
                self._read_event_store(
                    event_store=event_store,
                    run_id=run_id,
                    round_id=round_id,
                    method_name=method_name,
                )
            )

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

    def _read_event_store(
        self,
        *,
        event_store,
        run_id: str,
        round_id: int,
        method_name: str,
    ) -> list[SampleRecord]:
        request_by_sample = {
            payload["sample_id"]: payload
            for payload in event_store.iter_events(
                "LLMRequestEvent", run_id=run_id, round_id=round_id
            )
        }
        response_by_sample = {
            payload["sample_id"]: payload
            for payload in event_store.iter_events(
                "LLMResponseEvent", run_id=run_id, round_id=round_id
            )
        }
        parse_by_sample = {
            payload["sample_id"]: payload
            for payload in event_store.iter_events(
                "ParseEvent", run_id=run_id, round_id=round_id
            )
        }
        eval_by_sample = {
            payload["sample_id"]: payload
            for payload in event_store.iter_events(
                "EvalTraceRecord", run_id=run_id, round_id=round_id
            )
        }

        sample_records: dict[str, SampleRecord] = {}
        for payload in event_store.iter_events(
            "SampleRecord", run_id=run_id, round_id=round_id
        ):
            record = SampleRecord(**payload)
            sample_records[record.sample_id] = self._enrich_sample_record(
                record,
                request_by_sample=request_by_sample,
                response_by_sample=response_by_sample,
                parse_by_sample=parse_by_sample,
                eval_by_sample=eval_by_sample,
            )

        event_only_sample_ids = (
            set(request_by_sample)
            | set(response_by_sample)
            | set(parse_by_sample)
            | set(eval_by_sample)
        ) - set(sample_records)

        for sample_id in sorted(event_only_sample_ids):
            sample_records[sample_id] = self._build_event_only_record(
                sample_id=sample_id,
                run_id=run_id,
                round_id=round_id,
                method_name=method_name,
                request_payload=request_by_sample.get(sample_id),
                response_payload=response_by_sample.get(sample_id),
                parse_payload=parse_by_sample.get(sample_id),
                eval_payload=eval_by_sample.get(sample_id),
            )

        return list(sample_records.values())

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

    def _enrich_sample_record(
        self,
        record: SampleRecord,
        *,
        request_by_sample: dict,
        response_by_sample: dict,
        parse_by_sample: dict,
        eval_by_sample: dict,
    ) -> SampleRecord:
        request_payload = request_by_sample.get(record.sample_id)
        response_payload = response_by_sample.get(record.sample_id)
        parse_payload = parse_by_sample.get(record.sample_id)
        eval_payload = eval_by_sample.get(record.sample_id)

        provenance = dict(record.provenance or {})
        if request_payload is not None:
            provenance["llm_request"] = request_payload
            if record.prompt is None:
                record.prompt = request_payload.get("prompt")
            if record.messages is None:
                record.messages = request_payload.get("messages")
            if record.phase is None:
                record.phase = request_payload.get("phase", record.phase)
        if response_payload is not None:
            provenance["llm_response"] = response_payload
            if record.response is None:
                record.response = response_payload.get("raw_response")
        if parse_payload is not None:
            provenance["parse_event"] = parse_payload
        if eval_payload is not None:
            provenance["eval_trace"] = eval_payload
            if record.eval_time is None:
                record.eval_time = eval_payload.get("eval_time")
            if record.score is None:
                record.score = eval_payload.get("score")
        record.provenance = provenance
        return record

    def _build_event_only_record(
        self,
        *,
        sample_id: str,
        run_id: str,
        round_id: int,
        method_name: str,
        request_payload: dict | None,
        response_payload: dict | None,
        parse_payload: dict | None,
        eval_payload: dict | None,
    ) -> SampleRecord:
        provenance = {"source": "event_only"}
        if request_payload is not None:
            provenance["llm_request"] = request_payload
        if response_payload is not None:
            provenance["llm_response"] = response_payload
        if parse_payload is not None:
            provenance["parse_event"] = parse_payload
        if eval_payload is not None:
            provenance["eval_trace"] = eval_payload

        return SampleRecord(
            run_id=run_id,
            round_id=round_id,
            sample_id=sample_id,
            method_name=method_name,
            phase="search"
            if request_payload is None
            else request_payload.get("phase", "search"),
            operator=None,
            parents=None,
            generation=None,
            prompt=None if request_payload is None else request_payload.get("prompt"),
            messages=None
            if request_payload is None
            else request_payload.get("messages"),
            response=None
            if response_payload is None
            else response_payload.get("raw_response"),
            function=None,
            program=None,
            algorithm=None,
            score=None if eval_payload is None else eval_payload.get("score"),
            sample_time=None,
            eval_time=None if eval_payload is None else eval_payload.get("eval_time"),
            provenance=provenance,
        )
