from __future__ import annotations

import math


def compare_profiles(
    candidate,
    parent,
    *,
    minimize: bool,
    se_multiplier: float,
    margin: float,
) -> dict:
    se_multiplier, margin = float(se_multiplier), float(margin)
    if not math.isfinite(se_multiplier) or not math.isfinite(margin) or se_multiplier < 0.0 or margin < 0.0:
        raise ValueError("VC-PAIR se multiplier and margin must be finite and non-negative")
    candidate = _profile(candidate, "candidate")
    parent = _profile(parent, "parent")
    if len(candidate) != len(parent):
        raise ValueError(f"VC-PAIR profile length mismatch: candidate={len(candidate)}, parent={len(parent)}")
    direction = -1.0 if minimize else 1.0
    differences = [direction * (child - base) for child, base in zip(candidate, parent)]
    candidate_mean = sum(candidate) / len(candidate)
    parent_mean = sum(parent) / len(parent)
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1) if len(differences) > 1 else 0.0
    std = math.sqrt(variance)
    se = std / math.sqrt(len(differences))
    threshold = margin + se_multiplier * se
    outcome = "positive" if mean > threshold else "negative" if mean < -threshold else "neutral"
    return {
        "paired_outcome": outcome,
        "paired_n": len(differences),
        "paired_candidate_mean": candidate_mean,
        "paired_parent_mean": parent_mean,
        "paired_mean": mean,
        "paired_std": std,
        "paired_se": se,
        "paired_threshold": threshold,
        "paired_win_rate": sum(value > 0.0 for value in differences) / len(differences),
    }


def _profile(values, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"VC-PAIR requires a non-empty {name} performance profile")
    profile = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"VC-PAIR {name} profile contains a non-numeric value") from exc
        if not math.isfinite(value):
            raise ValueError(f"VC-PAIR {name} profile contains a non-finite value")
        profile.append(value)
    return profile


__all__ = ["compare_profiles"]
