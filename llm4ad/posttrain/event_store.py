from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


class EventStore:
    _EVENT_FILE_MAP = {
        "LLMRequestEvent": "llm_request.jsonl",
        "LLMResponseEvent": "llm_response.jsonl",
        "ParseEvent": "parse.jsonl",
        "EvalTraceRecord": "eval_trace.jsonl",
        "SampleRecord": "sample.jsonl",
        "PopulationEvent": "population.jsonl",
        "TrainingRoundRecord": "training_round.jsonl",
        "PromotionRecord": "promotion.jsonl",
        "SyntheticRecord": "synthetic.jsonl",
    }

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def append(self, event: Any) -> None:
        event_type = (
            event.__class__.__name__ if is_dataclass(event) else type(event).__name__
        )
        event_file = self.root_dir / self._event_type_to_file(event_type)
        payload = asdict(event) if is_dataclass(event) else dict(event)
        payload = self._make_jsonable(payload)
        with event_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def append_many(self, events: Iterable[Any]) -> None:
        for event in events:
            self.append(event)

    def iter_events(
        self,
        event_type: str,
        run_id: str | None = None,
        round_id: int | None = None,
    ):
        event_file = self.root_dir / self._event_type_to_file(event_type)
        if not event_file.exists():
            return
        with event_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                payload = json.loads(line)
                if run_id is not None and payload.get("run_id") != run_id:
                    continue
                if round_id is not None and payload.get("round_id") != round_id:
                    continue
                yield payload

    def load_all(self, event_type: str) -> list[dict[str, Any]]:
        return list(self.iter_events(event_type))

    def get_event_file(self, event_type: str) -> Path:
        return self.root_dir / self._event_type_to_file(event_type)

    def _event_type_to_file(self, event_type: str) -> str:
        return self._EVENT_FILE_MAP.get(event_type, f"{event_type}.jsonl")

    def _make_jsonable(self, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(key): self._make_jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._make_jsonable(item) for item in value]
        return str(value)
