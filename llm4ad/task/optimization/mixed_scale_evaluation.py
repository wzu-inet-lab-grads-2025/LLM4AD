from __future__ import annotations

import math
from typing import Any

from ...base import Evaluation
from .cvrp_construct import CVRPEvaluation
from .jssp_construct import JSSPEvaluation
from .online_bin_packing import OBPEvaluation
from .tsp_construct import TSPEvaluation


_META_KEYS = {
    "ratio",
    "weight",
    "pop_size",
    "score_normalizer",
    "normalizer",
    "eoh",
}

_TASK_EVALUATIONS = {
    "tsp_construct": TSPEvaluation,
    "cvrp_construct": CVRPEvaluation,
    "online_bin_packing": OBPEvaluation,
    "jssp_construct": JSSPEvaluation,
}


class MixedScaleEvaluation(Evaluation):
    """单任务多规模复合评估器。"""

    def __init__(self, task_name: str, scale_configs: dict[str, dict[str, Any]]):
        if not isinstance(scale_configs, dict) or not scale_configs:
            raise ValueError("mixed scale config must be a non-empty mapping")
        task_name = str(task_name or "").strip()
        if task_name not in _TASK_EVALUATIONS:
            raise ValueError(f"unsupported mixed-scale task: {task_name!r}")

        evaluation_cls = _TASK_EVALUATIONS[task_name]
        self.task_name = task_name
        self.scale_entries: list[dict[str, Any]] = []
        self._last_eval_diag: dict[str, Any] | None = None
        timeout_seconds = 0.0

        for scale_name, raw_cfg in scale_configs.items():
            scale_cfg = dict(raw_cfg or {})
            ratio = float(scale_cfg.get("ratio", scale_cfg.get("weight", 1.0)))
            if ratio <= 0.0:
                continue
            normalizer = float(scale_cfg.get("score_normalizer", scale_cfg.get("normalizer", 1.0)))
            if not math.isfinite(normalizer) or normalizer <= 0.0:
                raise ValueError(f"mixed scale {scale_name!r} has invalid score_normalizer={normalizer!r}")
            params = {key: value for key, value in scale_cfg.items() if key not in _META_KEYS}
            evaluation = evaluation_cls(**params)
            timeout = getattr(evaluation, "timeout_seconds", None)
            if timeout is not None:
                timeout_seconds += float(timeout)
            self.scale_entries.append(
                {
                    "name": str(scale_name),
                    "ratio": ratio,
                    "normalizer": normalizer,
                    "evaluation": evaluation,
                    "params": params,
                }
            )

        if not self.scale_entries:
            raise ValueError("mixed scale config has no active scale with ratio > 0")

        first = self.scale_entries[0]["evaluation"]
        scale_desc = ", ".join(f"{entry['name']}(ratio={entry['ratio']})" for entry in self.scale_entries)
        task_description = (
            f"{first.task_description}\n\n"
            f"This RL run evaluates each candidate on mixed scales: {scale_desc}."
        )
        composite_timeout = timeout_seconds if timeout_seconds > 0 else None
        super().__init__(
            template_program=first.template_program,
            task_description=task_description,
            use_numba_accelerate=first.use_numba_accelerate,
            use_protected_div=first.use_protected_div,
            protected_div_delta=first.protected_div_delta,
            random_seed=first.random_seed,
            timeout_seconds=composite_timeout,
            exec_code=first.exec_code,
            safe_evaluate=first.safe_evaluate,
            daemon_eval_process=first.daemon_eval_process,
            fork_proc=first.fork_proc,
        )

    @property
    def scale_names(self) -> list[str]:
        return [entry["name"] for entry in self.scale_entries]

    @property
    def last_eval_diag(self) -> dict[str, Any] | None:
        return self._last_eval_diag

    def get_last_eval_diag(self) -> dict[str, Any] | None:
        return self._last_eval_diag

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        self._last_eval_diag = None
        scale_results: list[dict[str, Any]] = []
        weighted_score = 0.0
        total_weight = 0.0
        for entry in self.scale_entries:
            score = entry["evaluation"].evaluate_program(program_str, callable_func, **kwargs)
            if score is None:
                scale_results.append(self._scale_result(entry, None, None, None, ok=False))
                self._last_eval_diag = self._build_eval_diag(
                    composite_score=None,
                    total_weight=total_weight,
                    scale_results=scale_results,
                    ok=False,
                    failure_scale=entry["name"],
                    failure_reason="score_none",
                )
                return None
            score_value = float(score)
            if not math.isfinite(score_value):
                scale_results.append(self._scale_result(entry, score_value, None, None, ok=False))
                self._last_eval_diag = self._build_eval_diag(
                    composite_score=None,
                    total_weight=total_weight,
                    scale_results=scale_results,
                    ok=False,
                    failure_scale=entry["name"],
                    failure_reason="non_finite_score",
                )
                return None
            normalized_score = score_value / float(entry["normalizer"])
            weighted_part = float(entry["ratio"]) * normalized_score
            scale_results.append(self._scale_result(entry, score_value, normalized_score, weighted_part, ok=True))
            weighted_score += weighted_part
            total_weight += float(entry["ratio"])
        if total_weight <= 0.0:
            self._last_eval_diag = self._build_eval_diag(
                composite_score=None,
                total_weight=total_weight,
                scale_results=scale_results,
                ok=False,
                failure_scale=None,
                failure_reason="non_positive_total_weight",
            )
            return None
        composite_score = weighted_score / total_weight
        self._last_eval_diag = self._build_eval_diag(
            composite_score=composite_score,
            total_weight=total_weight,
            scale_results=scale_results,
            ok=True,
            failure_scale=None,
            failure_reason=None,
        )
        return composite_score

    def _scale_result(
        self,
        entry: dict[str, Any],
        raw_score: float | None,
        normalized_score: float | None,
        weighted_score: float | None,
        *,
        ok: bool,
    ) -> dict[str, Any]:
        return {
            "name": entry["name"],
            "ok": bool(ok),
            "ratio": float(entry["ratio"]),
            "normalizer": float(entry["normalizer"]),
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "weighted_score": weighted_score,
            "params": dict(entry.get("params") or {}),
        }

    def _build_eval_diag(
        self,
        *,
        composite_score: float | None,
        total_weight: float,
        scale_results: list[dict[str, Any]],
        ok: bool,
        failure_scale: str | None,
        failure_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "score_kind": "mixed_scale_normalized_composite",
            "mixed_scale": {
                "ok": bool(ok),
                "task_name": self.task_name,
                "composite_score": composite_score,
                "total_weight": float(total_weight),
                "failure_scale": failure_scale,
                "failure_reason": failure_reason,
                "scales": list(scale_results),
            },
        }


def create_mixed_scale_evaluation(task_name: str, mixed_scale_config: dict[str, Any]) -> MixedScaleEvaluation:
    cfg = dict(mixed_scale_config or {})
    scales = cfg.get("scales")
    if scales is None:
        raise ValueError(f"mixed_scale_config must contain 'scales' key, got keys: {list(cfg.keys())}")
    if not isinstance(scales, dict):
        raise TypeError(f"mixed_scale_config['scales'] must be a dict, got {type(scales)}")
    return MixedScaleEvaluation(task_name=task_name, scale_configs=dict(scales))


__all__ = ["MixedScaleEvaluation", "create_mixed_scale_evaluation"]
