from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CollapseConfig:
    enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("eoh_rl.collapse.enabled must be bool")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CollapseConfig":
        if not isinstance(data, dict):
            raise ValueError("eoh_rl.collapse must be a mapping")
        unknown = sorted(set(data) - {"enabled"})
        if unknown:
            raise ValueError(f"eoh_rl.collapse 包含未知字段: {unknown}")
        if "enabled" not in data:
            raise ValueError("eoh_rl.collapse missing required field: enabled")
        return cls(enabled=data["enabled"])

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": bool(self.enabled)}


class CollapseManager:
    """基于真实无突破间隔和结构同质化的自适应坍缩控制器。"""

    VERSION = "adaptive_collapse_recovery_v5"

    def __init__(self, config: CollapseConfig):
        self.config = config
        self._best_score: float | None = None
        self._last_best_update_id: int | None = None
        self._last_collapse_update_id: int | None = None
        self._collapse_count = 0
        self._recovery_active = False
        self._recovery_started_update_id: int | None = None
        self._recovery_min_updates = 0
        self._recovery_diversity_target = 0.0
        self._recent_best_intervals: list[int] = []
        self._recent_registered_counts: list[int] = []
        self._recent_diversities: list[float] = []

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def recovery_active(self) -> bool:
        return bool(self.enabled and self._recovery_active)

    def recovery_ready_to_finish(
        self,
        *,
        rl_update_id: int,
        population_diversity: float,
        frontier_improve_count: int = 0,
    ) -> bool:
        if not self._recovery_active:
            return True
        elapsed = 0 if self._recovery_started_update_id is None else max(0, int(rl_update_id) - int(self._recovery_started_update_id))
        if elapsed < int(self._recovery_min_updates) and int(frontier_improve_count or 0) <= 0:
            return False
        target = max(0.0, float(self._recovery_diversity_target or 0.0))
        return bool(target <= 0.0 or float(population_diversity or 0.0) >= target)

    def observe_update(
        self,
        *,
        rl_update_id: int,
        best_after: float | None,
        population_stats: dict[str, Any] | None,
        summary: dict[str, Any] | None,
        population_size: int,
        minimize: bool,
        target_population_size: int | None = None,
        recovery_active: bool | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"should_collapse": False, "trigger": "disabled"}

        stats = dict(population_stats or {})
        summary = dict(summary or {})
        target = max(1, int(target_population_size or population_size or 1))
        current_size = max(0, int(stats.get("size", population_size) or population_size or 0))
        best_value = self._safe_float_or_none(best_after)
        first_baseline = self._best_score is None and best_value is not None
        improved = (not first_baseline) and self._is_better(best_value, self._best_score, minimize=minimize)
        diversity = max(0.0, min(1.0, self._safe_float(summary.get("population_diversity"), default=0.0)))
        registered_count = max(0, int(summary.get("registered_count", 0) or 0))
        search_count = registered_count
        frontier_count = max(0, int(summary.get("frontier_improve_count", 0) or 0))

        previous_best_id = self._last_best_update_id
        if first_baseline:
            self._best_score = best_value
            self._last_best_update_id = int(rl_update_id)
        elif improved:
            if previous_best_id is not None:
                interval = max(1, int(rl_update_id) - int(previous_best_id))
                self._recent_best_intervals.append(interval)
            self._best_score = best_value
            self._last_best_update_id = int(rl_update_id)

        gap = max(0, int(rl_update_id) - int(self._last_best_update_id)) if self._last_best_update_id is not None else 0
        registered_ref = self._median(self._recent_registered_counts)
        diversity_ref = self._median(self._recent_diversities)
        diversity_target = self._diversity_recovery_target()
        operator_count = max(1, int(summary.get("operator_count", 4) or 4))
        threshold = self._adaptive_threshold(target_size=target, operator_count=operator_count)
        collapse_gap = math.inf if self._last_collapse_update_id is None else max(0, int(rl_update_id) - int(self._last_collapse_update_id))
        cooldown_ready = bool(collapse_gap >= threshold)
        mature = len(self._recent_registered_counts) >= max(target, operator_count)
        in_recovery = self._recovery_active if recovery_active is None else bool(recovery_active)
        population_full = current_size >= target
        local_progress_stalled = bool(frontier_count <= 0 and search_count <= registered_ref)
        diversity_low = bool(self._recent_diversities and diversity <= diversity_ref)
        long_stagnation = bool(gap >= threshold)
        should_collapse = bool(
            population_full
            and not in_recovery
            and self._best_score is not None
            and mature
            and cooldown_ready
            and long_stagnation
            and local_progress_stalled
            and diversity_low
        )

        trigger = "adaptive_collapse" if should_collapse else self._blocked_reason(
            population_full=population_full,
            not_in_recovery=not in_recovery,
            mature=mature,
            cooldown_ready=cooldown_ready,
            long_stagnation=long_stagnation,
            local_progress_stalled=local_progress_stalled,
            diversity_low=diversity_low,
        )
        self._remember_window(target, registered_count=search_count, diversity=diversity)
        return {
            "should_collapse": should_collapse,
            "trigger": trigger,
            "rl_update_id": int(rl_update_id),
            "best_after": best_after,
            "best_known": self._best_score,
            "current_no_improve_streak": int(gap),
            "age_stuck": int(gap),
            "effective_age": int(gap),
            "effective_threshold": float(threshold),
            "collapse_threshold": float(threshold),
            "collapse_cooldown_ready": bool(cooldown_ready),
            "collapse_cooldown_gap": None if math.isinf(collapse_gap) else int(collapse_gap),
            "collapse_cooldown_threshold": float(threshold),
            "operator_count": int(operator_count),
            "population_full": bool(population_full),
            "current_population_size": int(current_size),
            "target_population_size": int(target),
            "recovery_active": bool(in_recovery),
            "history_ready": bool(mature),
            "registered_reference": float(registered_ref),
            "registered_count": int(registered_count),
            "search_evidence_count": int(search_count),
            "population_diversity": float(diversity),
            "diversity_reference": float(diversity_ref),
            "recovery_diversity_target": float(diversity_target),
            "long_effective_stagnation": bool(long_stagnation),
            "local_progress_decayed": bool(local_progress_stalled),
            "diversity_low": bool(diversity_low),
            "recent_best_intervals": list(self._recent_best_intervals),
            "last_collapse_update_id": self._last_collapse_update_id,
            "collapse_count": int(self._collapse_count),
        }

    def record_collapse(self, event: dict[str, Any]) -> dict[str, Any]:
        recorded = dict(event or {})
        if not self.enabled:
            return recorded
        self._collapse_count += 1
        self._recovery_active = True
        collapse_id = recorded.get("rl_update_id")
        self._recovery_started_update_id = int(collapse_id) if collapse_id is not None else None
        self._recovery_min_updates = max(1, int(recorded.get("operator_count", 4) or 4))
        self._recovery_diversity_target = max(
            0.0,
            self._safe_float(recorded.get("recovery_diversity_target"), default=0.0),
            self._safe_float(recorded.get("diversity_reference"), default=0.0),
        )
        self._last_collapse_update_id = int(collapse_id) if collapse_id is not None else None
        recorded.update(
            {
                "collapse_count": int(self._collapse_count),
                "recovery_active": True,
                "recovery_min_updates": int(self._recovery_min_updates),
                "recovery_diversity_target": float(self._recovery_diversity_target),
            }
        )
        return recorded

    def finish_recovery(self, event: dict[str, Any] | None = None) -> dict[str, Any]:
        recorded = dict(event or {})
        recorded["was_recovery_active"] = bool(self._recovery_active)
        self._recovery_active = False
        self._recovery_started_update_id = None
        self._recovery_min_updates = 0
        self._recovery_diversity_target = 0.0
        recorded["recovery_active"] = False
        return recorded

    def get_state(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "config": self.config.to_dict(),
            "best_score": self._best_score,
            "last_best_update_id": self._last_best_update_id,
            "last_collapse_update_id": self._last_collapse_update_id,
            "collapse_count": int(self._collapse_count),
            "recovery_active": bool(self._recovery_active),
            "recovery_started_update_id": self._recovery_started_update_id,
            "recovery_min_updates": int(self._recovery_min_updates),
            "recovery_diversity_target": float(self._recovery_diversity_target),
            "recent_best_intervals": list(self._recent_best_intervals),
            "recent_registered_counts": list(self._recent_registered_counts),
            "recent_diversities": list(self._recent_diversities),
        }

    def load_state(self, state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict) or state.get("config") != self.config.to_dict():
            return
        if state.get("version") != self.VERSION:
            return
        self._best_score = self._safe_float_or_none(state.get("best_score"))
        self._last_best_update_id = self._safe_int_or_none(state.get("last_best_update_id"))
        self._last_collapse_update_id = self._safe_int_or_none(state.get("last_collapse_update_id"))
        self._collapse_count = max(0, int(state.get("collapse_count", 0) or 0))
        self._recovery_active = bool(state.get("recovery_active", False))
        self._recovery_started_update_id = self._safe_int_or_none(state.get("recovery_started_update_id"))
        self._recovery_min_updates = max(0, int(state.get("recovery_min_updates", 0) or 0))
        self._recovery_diversity_target = max(0.0, self._safe_float(state.get("recovery_diversity_target"), default=0.0))
        self._recent_best_intervals = [max(1, int(v)) for v in state.get("recent_best_intervals") or [] if v is not None]
        self._recent_registered_counts = [max(0, int(v)) for v in state.get("recent_registered_counts") or [] if v is not None]
        self._recent_diversities = [max(0.0, min(1.0, self._safe_float(v, default=0.0))) for v in state.get("recent_diversities") or []]

    def _adaptive_threshold(self, *, target_size: int, operator_count: int) -> float:
        refresh = self._median(self._recent_registered_counts)
        fill_horizon = math.ceil(float(target_size) / max(1.0 / max(1.0, float(target_size)), refresh))
        intervals = [float(v) for v in self._recent_best_intervals if v > 0]
        interval_ref = self._median(intervals) if intervals else 0.0
        coverage = max(1, int(target_size)) * max(1, int(operator_count))
        return max(float(coverage), float(fill_horizon), float(interval_ref))

    def _remember_window(self, target_size: int, *, registered_count: int, diversity: float) -> None:
        limit = max(1, int(target_size) * 4)
        self._recent_registered_counts.append(max(0, int(registered_count)))
        self._recent_diversities.append(max(0.0, min(1.0, float(diversity))))
        self._recent_registered_counts = self._recent_registered_counts[-limit:]
        self._recent_diversities = self._recent_diversities[-limit:]
        self._recent_best_intervals = self._recent_best_intervals[-limit:]

    def _diversity_recovery_target(self) -> float:
        values = sorted(float(value) for value in self._recent_diversities if value is not None)
        if not values:
            return 0.0
        return max(self._median(values), self._quantile(values, 0.75))

    @staticmethod
    def _blocked_reason(**flags: bool) -> str:
        for name, ok in flags.items():
            if not ok:
                return name
        return "none"

    @staticmethod
    def _median(values) -> float:
        items = sorted(float(value) for value in values if value is not None)
        if not items:
            return 0.0
        mid = len(items) // 2
        return float(items[mid]) if len(items) % 2 else 0.5 * (items[mid - 1] + items[mid])

    @staticmethod
    def _quantile(values, q: float) -> float:
        items = sorted(float(value) for value in values if value is not None)
        if not items:
            return 0.0
        index = min(len(items) - 1, max(0, int(round((len(items) - 1) * float(q)))))
        return float(items[index])

    @staticmethod
    def _is_better(candidate: float | None, baseline: float | None, *, minimize: bool) -> bool:
        candidate_f = CollapseManager._safe_float_or_none(candidate)
        baseline_f = CollapseManager._safe_float_or_none(baseline)
        if candidate_f is None:
            return False
        if baseline_f is None:
            return True
        return candidate_f < baseline_f if minimize else candidate_f > baseline_f

    @staticmethod
    def _safe_float(value: Any, *, default: float) -> float:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        except Exception:
            return default

    @staticmethod
    def _safe_float_or_none(value: Any) -> float | None:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        except Exception:
            return None

    @staticmethod
    def _safe_int_or_none(value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except Exception:
            return None
