from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


ONLINE_OPERATORS = ("m1", "m2", "e1", "e2")
NORMAL_PRIOR = {
    "m1": 0.30,
    "m2": 0.45,
    "e1": 0.15,
    "e2": 0.10,
}
RECOVERY_PRIOR = {
    "m1": 0.10,
    "m2": 0.30,
    "e1": 0.30,
    "e2": 0.30,
}
MAX_OPERATOR_PROB = 0.60
NORMAL_TEMPERATURE = 0.70


@dataclass(frozen=True)
class AdaptiveOperatorConfig:
    enabled: bool
    mode: str = "sgca_v1"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AdaptiveOperatorConfig":
        if not isinstance(data, dict):
            raise ValueError("eoh_rl.adaptive_operator must be a mapping")
        unknown = sorted(set(data) - {"enabled", "mode"})
        if unknown:
            raise ValueError(f"eoh_rl.adaptive_operator 包含未知字段: {unknown}")
        if "enabled" not in data:
            raise ValueError("eoh_rl.adaptive_operator missing required field: enabled")
        enabled = data["enabled"]
        if not isinstance(enabled, bool):
            raise ValueError("eoh_rl.adaptive_operator.enabled must be bool")
        mode = str(data.get("mode", "sgca_v1")).strip().lower()
        if mode != "sgca_v1":
            raise ValueError("adaptive operator mode currently only supports sgca_v1")
        return cls(enabled=enabled, mode=mode)

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": bool(self.enabled), "mode": str(self.mode)}


class AdaptiveOperatorScheduler:
    def __init__(self, config: AdaptiveOperatorConfig):
        self.config = config
        self._online_score = {op: 0.0 for op in ONLINE_OPERATORS}
        self._online_updates = {op: 0 for op in ONLINE_OPERATORS}
        self._operator_debt = {op: 0.0 for op in ONLINE_OPERATORS}
        self._recovery_debt = {op: 0.0 for op in ONLINE_OPERATORS}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def select_ops(self, *, k: int, operator_cycle: list[str], context: dict[str, Any] | None = None) -> list[str]:
        context = dict(context or {})
        k = max(0, int(k))
        cycle_ops = self._allowed_ops(operator_cycle)
        if k <= 0 or not cycle_ops:
            return []
        if not self.enabled:
            feasible_cycle = self._fallback_feasible_cycle(operator_cycle=cycle_ops, context=context)
            return self._fallback_cycle_selection(k=k, operator_cycle=feasible_cycle, context=context)
        probabilities = self.compute_probabilities(operator_cycle=cycle_ops, context=context)
        ops = [op for op, prob in probabilities.items() if float(prob) > 0.0]
        if not ops:
            feasible_cycle = self._fallback_feasible_cycle(operator_cycle=cycle_ops, context=context)
            return self._fallback_cycle_selection(k=k, operator_cycle=feasible_cycle, context=context)
        return self._debt_selection(k=k, probabilities=probabilities, allowed=ops, debt=self._operator_debt)

    def select_recovery_ops(self, *, k: int, operator_cycle: list[str], context: dict[str, Any] | None = None) -> list[str]:
        context = dict(context or {})
        k = max(0, int(k))
        if k <= 0:
            return []
        if not self.enabled:
            feasible_cycle = self._fallback_feasible_cycle(operator_cycle=self._allowed_ops(operator_cycle), context=context)
            return self._fallback_cycle_selection(k=k, operator_cycle=feasible_cycle, context=context)
        probabilities = self.compute_recovery_probabilities(operator_cycle=operator_cycle, context=context)
        ops = [op for op, prob in probabilities.items() if float(prob) > 0.0]
        if not ops:
            feasible_cycle = self._fallback_feasible_cycle(operator_cycle=self._allowed_ops(operator_cycle), context=context)
            return self._fallback_cycle_selection(k=k, operator_cycle=feasible_cycle, context=context)
        return self._debt_selection(k=k, probabilities=probabilities, allowed=ops, debt=self._recovery_debt)

    def update_from_summary(self, summary: dict[str, Any] | None) -> None:
        self._update_scores_from_summary(
            summary,
            operators=ONLINE_OPERATORS,
            score_store=self._online_score,
            update_store=self._online_updates,
        )

    def update_recovery_from_summary(self, summary: dict[str, Any] | None) -> None:
        if int((summary or {}).get("frontier_improve_count", 0) or 0) > 0:
            self.update_from_summary(summary)

    def _update_scores_from_summary(
        self,
        summary: dict[str, Any] | None,
        *,
        operators: tuple[str, ...],
        score_store: dict[str, float],
        update_store: dict[str, int],
    ) -> bool:
        if not self.enabled or not isinstance(summary, dict):
            return False
        observed = self._step_signals(summary, operators=operators)
        updated = False
        for op, signal in observed.items():
            if signal is None:
                continue
            self._update_running_average(score_store, update_store, op, float(signal))
            updated = True
        return updated

    def compute_probabilities(self, *, operator_cycle: list[str], context: dict[str, Any]) -> dict[str, float]:
        allowed = self._allowed_ops(operator_cycle)
        if not allowed:
            return {}
        unique_parent_count = max(0, int((context or {}).get("unique_parent_count", 0) or 0))
        feasible = self._feasible_online_ops(allowed=allowed, unique_parent_count=unique_parent_count)
        feasible_ops = [op for op in allowed if op in feasible]
        if not feasible_ops:
            return {op: 0.0 for op in allowed}
        prior = self._normalize({op: NORMAL_PRIOR.get(op, 0.0) for op in feasible_ops})
        learned = self._learned_distribution(feasible_ops, self._online_score, temperature=NORMAL_TEMPERATURE)
        alpha = self._blend_weight(feasible_ops, self._online_updates)
        mixed = {
            op: (1.0 - alpha) * float(prior.get(op, 0.0)) + alpha * float(learned.get(op, 0.0))
            for op in feasible_ops
        }
        result = self._finalize_distribution(allowed=allowed, feasible=feasible, probabilities=mixed)
        return result

    def compute_recovery_probabilities(self, *, operator_cycle: list[str], context: dict[str, Any]) -> dict[str, float]:
        if not self.enabled:
            return {}
        allowed_online = self._allowed_ops(operator_cycle)
        if not allowed_online:
            return {}
        context = dict(context or {})
        unique_parent_count = max(0, int(context.get("unique_parent_count", 0) or 0))
        feasible = self._feasible_online_ops(allowed=allowed_online, unique_parent_count=unique_parent_count)
        feasible_ops = [op for op in allowed_online if op in feasible]
        if not feasible_ops:
            return {op: 0.0 for op in allowed_online}
        capped = self._finalize_distribution(
            allowed=feasible_ops,
            feasible=set(feasible_ops),
            probabilities={op: RECOVERY_PRIOR.get(op, 0.0) for op in feasible_ops},
        )
        result = {op: 0.0 for op in allowed_online}
        for op in feasible_ops:
            result[op] = float(capped.get(op, 0.0))
        return result

    def get_state(self) -> dict[str, Any]:
        return {
            "version": "operator_mainline_v1",
            "config": self.config.to_dict(),
            "credit_ema": dict(self._online_score),
            "credit_updates": dict(self._online_updates),
            "operator_debt": dict(self._operator_debt),
            "recovery_operator_debt": dict(self._recovery_debt),
        }

    def load_state(self, state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict) or state.get("config") != self.config.to_dict():
            return
        self._online_score = self._clean_float_map(state.get("credit_ema"), ONLINE_OPERATORS)
        self._online_updates = self._clean_int_map(state.get("credit_updates"), ONLINE_OPERATORS)
        self._operator_debt = self._clean_float_map(state.get("operator_debt"), ONLINE_OPERATORS)
        self._recovery_debt = self._clean_float_map(state.get("recovery_operator_debt"), ONLINE_OPERATORS)

    def _debt_selection(self, *, k: int, probabilities: dict[str, float], allowed: list[str], debt: dict[str, float]) -> list[str]:
        allowed = [op for op in allowed if op in ONLINE_OPERATORS and float(probabilities.get(op, 0.0)) > 0.0]
        if not allowed:
            return []
        selected = []
        for _ in range(max(0, int(k))):
            for op in allowed:
                debt[op] = float(debt.get(op, 0.0)) + float(probabilities.get(op, 0.0))
            op = max(allowed, key=lambda name: (debt.get(name, 0.0), float(probabilities.get(name, 0.0)), -ONLINE_OPERATORS.index(name)))
            debt[op] = float(debt.get(op, 0.0)) - 1.0
            selected.append(op)
        return selected

    def _step_signals(self, summary: dict[str, Any], *, operators: tuple[str, ...]) -> dict[str, float | None]:
        grouped = self._group_events_for_ops(summary, operators)
        observed = {op: None for op in operators}
        for op in operators:
            observed[op] = self._operator_signal(grouped.get(op) or [])
        return observed

    def _operator_signal(self, events: list[dict[str, Any]]) -> float | None:
        total = len(events)
        if total <= 0:
            return None
        valid_events = [
            event for event in events
            if bool(event.get("exec_success", True))
            and bool(event.get("validity", True))
            and not bool(event.get("random_algo"))
            and not bool(event.get("score_metadata_leak"))
            and not bool(event.get("exact_parent_copy"))
            and not event.get("failure_level")
            and not event.get("failure_label")
        ]
        frontier = sum(1 for event in valid_events if bool(event.get("beats_frontier")))
        parent = sum(1 for event in valid_events if bool(event.get("beats_parent")))
        positive = sum(1 for event in valid_events if float(event.get("reward") or 0.0) > 0.0)
        failed = sum(
            1
            for event in events
            if not bool(event.get("exec_success", True))
            or not bool(event.get("validity", True))
            or bool(event.get("random_algo"))
            or bool(event.get("score_metadata_leak"))
            or bool(event.get("exact_parent_copy"))
            or bool(event.get("failure_level"))
            or bool(event.get("failure_label"))
        )
        return float((2.0 * frontier + parent + 0.25 * positive - failed) / max(1, total))

    @staticmethod
    def _group_events_for_ops(summary: dict[str, Any], operators: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
        reward_events = summary.get("reward_events") or []
        grouped = {op: [] for op in operators}
        for event in reward_events:
            op = str(event.get("operator_type") or "").strip().lower()
            if op in grouped:
                grouped[op].append(event)
        return grouped

    @staticmethod
    def _update_running_average(store: dict[str, float], counts: dict[str, int], op: str, value: float) -> None:
        previous_count = max(0, int(counts.get(op, 0)))
        new_count = previous_count + 1
        previous_value = float(store.get(op, 0.0))
        store[op] = previous_value + (float(value) - previous_value) / float(new_count)
        counts[op] = new_count

    @staticmethod
    def _learned_distribution(ops: list[str], scores: dict[str, float], *, temperature: float = 1.0) -> dict[str, float]:
        if not ops:
            return {}
        if len(ops) == 1:
            return {ops[0]: 1.0}
        values = [float(scores.get(op, 0.0)) for op in ops]
        max_value = max(values) if values else 0.0
        tau = max(1.0e-6, float(temperature))
        raw = {op: math.exp((float(scores.get(op, 0.0)) - max_value) / tau) for op in ops}
        return AdaptiveOperatorScheduler._normalize(raw)

    @staticmethod
    def _blend_weight(ops: list[str], counts: dict[str, int], *, prior_factor: int = 1) -> float:
        if not ops:
            return 0.0
        observed_total = sum(max(0, int(counts.get(op, 0))) for op in ops)
        return float(observed_total) / float(observed_total + max(1, int(prior_factor)) * len(ops))

    @classmethod
    def _finalize_distribution(
        cls,
        *,
        allowed: list[str],
        feasible: set[str],
        probabilities: dict[str, float],
        max_probability: float = MAX_OPERATOR_PROB,
    ) -> dict[str, float]:
        feasible_ops = [op for op in allowed if op in feasible]
        if not feasible_ops:
            return {op: 0.0 for op in allowed}
        normalized = cls._normalize({op: float(probabilities.get(op, 0.0)) for op in feasible_ops})
        capped = cls._cap_max_probability(normalized, feasible=set(feasible_ops), max_probability=max_probability)
        return {op: (float(capped.get(op, 0.0)) if op in feasible else 0.0) for op in allowed}

    @classmethod
    def _cap_max_probability(cls, probabilities: dict[str, float], *, feasible: set[str], max_probability: float) -> dict[str, float]:
        feasible_ops = [op for op in probabilities if op in feasible]
        if not feasible_ops:
            return dict(probabilities)
        if len(feasible_ops) == 1:
            only = feasible_ops[0]
            return {op: (1.0 if op == only else 0.0) for op in probabilities}
        effective_cap = max(float(max_probability), 1.0 / len(feasible_ops))
        raw = cls._normalize({op: float(probabilities.get(op, 0.0)) for op in feasible_ops})
        capped: dict[str, float] = {}
        remaining_ops = list(feasible_ops)
        remaining_mass = 1.0
        eps = 1.0e-12
        while remaining_ops:
            remaining_total = sum(raw.get(op, 0.0) for op in remaining_ops)
            if remaining_total <= eps:
                share = remaining_mass / len(remaining_ops)
                for op in remaining_ops:
                    capped[op] = share
                break
            proposed = {op: remaining_mass * raw.get(op, 0.0) / remaining_total for op in remaining_ops}
            over_cap = [op for op, value in proposed.items() if value > effective_cap + eps]
            if not over_cap:
                capped.update(proposed)
                break
            for op in over_cap:
                capped[op] = effective_cap
                remaining_mass -= effective_cap
            remaining_ops = [op for op in remaining_ops if op not in set(over_cap)]
        return {op: float(capped.get(op, 0.0)) if op in feasible else 0.0 for op in probabilities}

    @staticmethod
    def _allowed_ops(operator_cycle: list[str] | tuple[str, ...] | None) -> list[str]:
        allowed = []
        for op in operator_cycle or []:
            name = str(op).strip().lower()
            if name in ONLINE_OPERATORS and name not in allowed:
                allowed.append(name)
        return allowed or list(ONLINE_OPERATORS)

    @staticmethod
    def _feasible_online_ops(*, allowed: list[str], unique_parent_count: int) -> set[str]:
        if unique_parent_count >= 2:
            return set(allowed)
        if unique_parent_count == 1:
            return {op for op in allowed if op in {"m1", "m2"}}
        if "m2" in allowed:
            return {"m2"}
        if "m1" in allowed:
            return {"m1"}
        return set(allowed)

    @staticmethod
    def _normalize(probabilities: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, float(value)) for value in probabilities.values())
        if total <= 0.0:
            keys = list(probabilities)
            if not keys:
                return {}
            uniform = 1.0 / len(keys)
            return {key: uniform for key in keys}
        return {key: max(0.0, float(value)) / total for key, value in probabilities.items()}

    @staticmethod
    def _fallback_cycle_selection(*, k: int, operator_cycle: list[str], context: dict[str, Any]) -> list[str]:
        if not operator_cycle or k <= 0:
            return []
        cycle_index = max(0, int(context.get("cycle_index", 0) or 0))
        return [operator_cycle[(cycle_index + i) % len(operator_cycle)] for i in range(k)]

    @staticmethod
    def _fallback_feasible_cycle(*, operator_cycle: list[str], context: dict[str, Any]) -> list[str]:
        unique_parent_count = max(0, int(context.get("unique_parent_count", 0) or 0))
        if unique_parent_count >= 2:
            return list(operator_cycle)
        feasible = [op for op in operator_cycle if op in {"m1", "m2"}]
        return feasible if feasible else list(operator_cycle)

    @staticmethod
    def _safe_float(value: Any, *, default: float) -> float:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        except Exception:
            return default

    @classmethod
    def _clean_float_map(cls, value: Any, operators: tuple[str, ...]) -> dict[str, float]:
        raw = value if isinstance(value, dict) else {}
        return {op: cls._safe_float(raw.get(op), default=0.0) for op in operators}

    @staticmethod
    def _clean_int_map(value: Any, operators: tuple[str, ...]) -> dict[str, int]:
        raw = value if isinstance(value, dict) else {}
        def parse(op: str) -> int:
            try:
                return max(0, int(raw.get(op, 0) or 0))
            except Exception:
                return 0

        return {op: parse(op) for op in operators}
