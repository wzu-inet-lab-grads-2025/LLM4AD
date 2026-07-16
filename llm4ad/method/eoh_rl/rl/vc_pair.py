from __future__ import annotations

import math

from scipy.stats import t as student_t


def compare_profiles(candidate, parent, *, minimize: bool, confidence: float, delta: float) -> dict:
    confidence, delta = float(confidence), float(delta)
    if not 0.0 < confidence < 1.0 or not math.isfinite(delta) or delta < 0.0:
        raise ValueError("VC-PAIR requires confidence in (0, 1) and a finite non-negative delta")
    candidate, parent = _profile(candidate, "candidate"), _profile(parent, "parent")
    if len(candidate) != len(parent):
        raise ValueError(f"VC-PAIR profile length mismatch: candidate={len(candidate)}, parent={len(parent)}")

    direction = -1.0 if minimize else 1.0
    differences = [direction * (child - base) for child, base in zip(candidate, parent)]
    n = len(differences)
    candidate_mean, parent_mean = sum(candidate) / n, sum(parent) / n
    mean = sum(differences) / n
    variance = sum((value - mean) ** 2 for value in differences) / (n - 1) if n > 1 else 0.0
    std, se = math.sqrt(variance), math.sqrt(variance / n)
    scale = max(abs(parent_mean), 1.0e-12)
    meaningful_delta = delta * scale
    if n < 2:
        critical = ci_low = ci_high = None
        outcome, evidence = "neutral", 0.0
    else:
        critical = float(student_t.ppf((1.0 + confidence) / 2.0, n - 1))
        ci_low, ci_high = mean - critical * se, mean + critical * se
        if ci_low > meaningful_delta:
            outcome, evidence = "positive", ci_low - meaningful_delta
        elif ci_high < -meaningful_delta:
            outcome, evidence = "negative", -ci_high - meaningful_delta
        else:
            outcome, evidence = "neutral", 0.0
    return {
        "paired_outcome": outcome,
        "paired_n": n,
        "paired_candidate_mean": candidate_mean,
        "paired_parent_mean": parent_mean,
        "paired_differences": differences,
        "paired_mean": mean,
        "paired_std": std,
        "paired_se": se,
        "paired_confidence": confidence,
        "paired_critical_value": critical,
        "paired_ci_low": ci_low,
        "paired_ci_high": ci_high,
        "paired_delta": meaningful_delta,
        "paired_delta_rel": delta,
        "paired_evidence": evidence,
        "paired_evidence_strength": min(evidence / scale, 1.0),
        "paired_win_rate": sum(value > meaningful_delta for value in differences) / n,
    }


def _profile(values, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"VC-PAIR requires a non-empty {name} performance profile")
    try:
        profile = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VC-PAIR {name} profile contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in profile):
        raise ValueError(f"VC-PAIR {name} profile contains a non-finite value")
    return profile


__all__ = ["compare_profiles"]
