from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .schemas import ModelArtifactRecord, PromotionRecord


class ModelRegistry:
    def __init__(self, artifact_root: str, workflow_task_key: str):
        self._root = Path(artifact_root) / workflow_task_key
        self._root.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._root / "registry.json"
        self._state = self._load_state()

    def register_candidate(self, artifact: ModelArtifactRecord) -> None:
        self._state["candidate"] = self._serialize_record(artifact)
        self._save_state()

    def promote_candidate(self, round_id: int) -> PromotionRecord:
        candidate = self._state.get("candidate")
        previous_active = self._state.get("active")
        if candidate is None:
            raise ValueError("No candidate model is registered.")
        self._state["previous"] = previous_active
        self._state["active"] = candidate
        self._state["candidate"] = None
        self._save_state()
        return PromotionRecord(
            round_id=round_id,
            candidate_version=candidate["version_id"],
            previous_active_version=None
            if previous_active is None
            else previous_active["version_id"],
            promoted=True,
            reason="Candidate promoted to active model.",
            static_gate_passed=True,
            smoke_gate_passed=True,
        )

    def rollback(self) -> PromotionRecord:
        previous = self._state.get("previous")
        active = self._state.get("active")
        if previous is None:
            raise ValueError("No previous model available for rollback.")
        self._state["active"], self._state["previous"] = previous, active
        self._save_state()
        return PromotionRecord(
            round_id=-1,
            candidate_version=previous["version_id"],
            previous_active_version=None if active is None else active["version_id"],
            promoted=True,
            reason="Rolled back to previous model.",
            static_gate_passed=True,
            smoke_gate_passed=True,
        )

    def get_active(self) -> dict | None:
        return self._state.get("active")

    def get_previous(self) -> dict | None:
        return self._state.get("previous")

    def get_candidate(self) -> dict | None:
        return self._state.get("candidate")

    def _load_state(self) -> dict:
        if not self._registry_path.exists():
            return {"active": None, "previous": None, "candidate": None}
        return json.loads(self._registry_path.read_text(encoding="utf-8"))

    def _save_state(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def _serialize_record(self, artifact):
        if is_dataclass(artifact):
            return asdict(artifact)
        return dict(artifact)
