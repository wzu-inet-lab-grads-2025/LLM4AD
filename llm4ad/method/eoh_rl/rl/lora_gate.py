from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LoraGateDecision:
    accepted: bool
    reason: str | None
    score: float
    components: dict[str, Any]


class SFTAnchorTrustRegionGate:
    """SFT 锚定的 LoRA 信任域控制器。

    部署 LoRA 只看闭环搜索证据：候选进入主种群或突破 frontier 才接受；
    投影只在权重真实越过 anchor/step 信任域时触发。
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.anchor_radius = self._nonnegative_float(cfg, "anchor_radius", 0.50)
        self.step_radius = self._nonnegative_float(cfg, "step_radius", 0.25)
        self._quality_ema: float | None = None

    @classmethod
    def from_grpo_config(cls, grpo_config: dict[str, Any] | None) -> "SFTAnchorTrustRegionGate":
        cfg = dict(grpo_config or {})
        gate_cfg = cfg.get("lora_gate")
        if gate_cfg is None:
            gate_cfg = {"enabled": True}
        if not isinstance(gate_cfg, dict):
            raise ValueError("grpo.lora_gate must be a mapping")
        return cls(gate_cfg)

    def decide(
        self,
        *,
        summary: dict[str, Any],
        minimize: bool,
        population_best_score: float | None,
        candidate_trainable_state: dict[str, torch.Tensor] | None,
        previous_trainable_state: dict[str, torch.Tensor] | None,
        anchor_trainable_state: dict[str, torch.Tensor] | None,
    ) -> LoraGateDecision:
        if not self.enabled:
            return LoraGateDecision(
                accepted=True,
                reason=None,
                score=0.0,
                components={"enabled": False, "mode": "disabled"},
            )

        delta = self._performance_delta(summary, minimize=minimize, population_best_score=population_best_score)
        sample_count = max(1, int(summary.get("candidate_count", summary.get("total", 1)) or 1))
        anchor_distance = self._state_distance(candidate_trainable_state, anchor_trainable_state)
        step_distance = self._state_distance(candidate_trainable_state, previous_trainable_state)
        quality = self._quality_score(summary)
        self._quality_ema = quality if self._quality_ema is None else 0.8 * self._quality_ema + 0.2 * quality

        valid_count = max(0, int(summary.get("valid_count", 0) or 0))
        candidate_count = max(0, int(summary.get("candidate_count", 0) or 0))
        valid_rate = max(0.0, min(1.0, float(summary.get("valid_rate", 0.0) or 0.0)))
        frontier = int(summary.get("frontier_improve_count", 0) or 0)
        parent = int(summary.get("parent_improve_count", 0) or 0)
        registered = int(summary.get("registered_count", 0) or 0)
        positive = int(summary.get("positive_reward_count", 0) or 0)
        trust_region_lambda = self._projection_lambda(
            anchor_distance=anchor_distance,
            step_distance=step_distance,
        )
        projection_lambda = trust_region_lambda
        no_valid_completion = bool(valid_count <= 0 or candidate_count <= 0)
        has_search_evidence = bool(frontier > 0 or parent > 0 or registered > 0 or positive > 0)

        checks = {
            "finite_completion": not no_valid_completion,
            "valid_completion": valid_count > 0,
            "anchor_in_region": anchor_distance <= self.anchor_radius,
            "step_in_region": step_distance <= self.step_radius,
            "search_evidence": has_search_evidence,
        }
        accepted = bool((not no_valid_completion) and has_search_evidence)
        if accepted:
            reason = None
        elif no_valid_completion:
            reason = "gate_no_valid_completion"
        else:
            reason = "gate_no_search_evidence"
        score = float(self._quality_ema)
        projected = bool(accepted and projection_lambda < 1.0)
        deployment_mode = "projected" if projected else "full" if accepted else "rejected"
        components = {
            "enabled": self.enabled,
            "score": float(score),
            "delta_score": float(delta) if math.isfinite(delta) else None,
            "quality_score": float(quality),
            "quality_ema": float(self._quality_ema),
            "sample_count": sample_count,
            "valid_count": int(valid_count),
            "candidate_count": int(candidate_count),
            "valid_rate": float(valid_rate),
            "frontier_improve_count": frontier,
            "parent_improve_count": parent,
            "registered_count": registered,
            "positive_reward_count": positive,
            "search_evidence": float(3.0 * frontier + 1.5 * parent + 1.0 * positive + 0.5 * registered),
            "anchor_distance": float(anchor_distance),
            "step_distance": float(step_distance),
            "anchor_radius": float(self.anchor_radius),
            "step_radius": float(self.step_radius),
            "trust_region_lambda": float(trust_region_lambda),
            "projection_lambda": float(projection_lambda),
            "projected": projected,
            "checks": checks,
            "hard_checks": {"finite_completion": checks["finite_completion"], "valid_completion": checks["valid_completion"]},
            "acceptance_rule": "positive_or_search_evidence_trust_region",
            "deployment_mode": deployment_mode,
            "best_improved": bool(frontier > 0 or delta > 0.0),
        }
        return LoraGateDecision(accepted=accepted, reason=reason, score=float(score), components=components)

    @staticmethod
    def _performance_delta(summary: dict[str, Any], *, minimize: bool, population_best_score: float | None) -> float:
        best = summary.get("best_completion_score")
        if best is None:
            return float("-inf")
        if population_best_score is None:
            return 0.0
        best_f = float(best)
        base_f = float(population_best_score)
        return base_f - best_f if minimize else best_f - base_f

    @staticmethod
    def _event_rate(summary: dict[str, Any], key: str, total: float) -> float:
        if key in summary:
            return max(0.0, min(1.0, float(summary.get(key, 0) or 0) / total))
        count = sum(1 for event in summary.get("reward_events") or [] if bool(event.get(key)))
        return max(0.0, min(1.0, float(count) / total))

    def _quality_score(self, summary: dict[str, Any]) -> float:
        total = max(1.0, float(summary.get("total", summary.get("candidate_count", 0)) or 0))
        frontier = float(summary.get("frontier_improve_count", 0) or 0)
        parent = float(summary.get("parent_improve_count", 0) or 0)
        invalid = max(0.0, 1.0 - float(summary.get("valid_rate", 0.0) or 0.0)) * total
        exact_copy = self._event_rate(summary, "exact_parent_copy", total) * total
        metadata_leak = (
            self._event_rate(summary, "score_metadata_leak", total)
            + self._event_rate(summary, "metadata_leak", total)
        ) * total
        raw = (3.0 * frontier + 1.5 * parent - invalid - exact_copy - metadata_leak) / total
        return max(-1.0, min(2.0, float(raw)))

    def _projection_lambda(self, *, anchor_distance: float, step_distance: float) -> float:
        ratios = [1.0]
        if anchor_distance > self.anchor_radius > 0.0:
            ratios.append(self.anchor_radius / anchor_distance)
        if step_distance > self.step_radius > 0.0:
            ratios.append(self.step_radius / step_distance)
        return max(0.0, min(1.0, min(ratios)))

    @staticmethod
    def project_trainable_state(
        *,
        candidate_state: dict[str, torch.Tensor] | None,
        anchor_state: dict[str, torch.Tensor] | None,
        previous_state: dict[str, torch.Tensor] | None = None,
        projection_lambda: float,
        anchor_radius: float | None = None,
        step_radius: float | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if not candidate_state:
            return None
        lam = max(0.0, min(1.0, float(projection_lambda)))
        projected = SFTAnchorTrustRegionGate._clone_state(candidate_state)
        changed = False
        shrink_center = previous_state or anchor_state
        if shrink_center and lam < 1.0:
            projected = SFTAnchorTrustRegionGate._interpolate_state(shrink_center, projected, lam)
            changed = True
        for _ in range(3):
            projected_anchor = projected_step = False
            if previous_state and step_radius is not None:
                projected, projected_step = SFTAnchorTrustRegionGate._project_to_radius(
                    projected,
                    center_state=previous_state,
                    radius=float(step_radius),
                )
            if anchor_state and anchor_radius is not None:
                projected, projected_anchor = SFTAnchorTrustRegionGate._project_to_radius(
                    projected,
                    center_state=anchor_state,
                    radius=float(anchor_radius),
                )
            changed = changed or projected_anchor or projected_step
            if not projected_anchor and not projected_step:
                break
        return projected if changed else None

    @staticmethod
    def _state_distance(
        left: dict[str, torch.Tensor] | None,
        right: dict[str, torch.Tensor] | None,
    ) -> float:
        if not left or not right:
            return 0.0
        numerator = 0.0
        denominator = 0.0
        common = set(left).intersection(right)
        if not common:
            return 0.0
        for name in common:
            lhs = left[name].detach().float().cpu()
            rhs = right[name].detach().float().cpu()
            numerator += float(torch.sum((lhs - rhs) ** 2).item())
            denominator += float(torch.sum(rhs ** 2).item())
        return math.sqrt(numerator / max(denominator, 1.0e-12))

    @staticmethod
    def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().to(device="cpu", copy=True) for name, tensor in state.items()}

    @staticmethod
    def _interpolate_state(
        center_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        ratio: float,
    ) -> dict[str, torch.Tensor]:
        ratio = max(0.0, min(1.0, float(ratio)))
        result: dict[str, torch.Tensor] = {}
        for name, target in target_state.items():
            center = center_state.get(name)
            if center is None:
                result[name] = target.detach().to(device="cpu", copy=True)
                continue
            target_cpu = target.detach().to(device="cpu", dtype=torch.float32)
            center_cpu = center.detach().to(device="cpu", dtype=torch.float32)
            value = center_cpu + ratio * (target_cpu - center_cpu)
            result[name] = value.to(dtype=target.dtype if target.dtype.is_floating_point else torch.float32)
        return result

    @staticmethod
    def _project_to_radius(
        state: dict[str, torch.Tensor],
        *,
        center_state: dict[str, torch.Tensor],
        radius: float,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        radius = max(0.0, float(radius))
        distance = SFTAnchorTrustRegionGate._state_distance(state, center_state)
        if distance <= radius or distance <= 1.0e-12:
            return state, False
        return SFTAnchorTrustRegionGate._interpolate_state(center_state, state, radius / distance), True

    @staticmethod
    def _nonnegative_float(cfg: dict[str, Any], key: str, default: float) -> float:
        value = default if cfg.get(key) is None else cfg.get(key)
        return max(0.0, float(value))
