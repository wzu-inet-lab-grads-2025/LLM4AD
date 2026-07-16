from __future__ import annotations

import json
import hashlib
import math
import os

from ...base import Function, TextFunctionProgramConverter
from .prompt import EoHPrompt
from .rl.grpo_trainer import EoHReward, event_funnel, is_population_eligible_event


def build_query_bank(
    parents: list[Function],
    *,
    task_description: str,
    template_function: Function,
    size: int,
    group_size: int = 4,
    minimize: bool = False,
    seed: int = 42,
) -> list[dict]:
    size, group_size = int(size), int(group_size)
    if size < 1 or group_size < 1:
        raise ValueError("fixed query bank size and group_size must be positive")
    parents = [parent for parent in parents if _finite(getattr(parent, "score", None)) and getattr(parent, "_eoh_profile", None)]
    if len(parents) < 2:
        raise ValueError("fixed query bank requires at least two scored parents with performance profiles")
    parents.sort(key=lambda item: -float(item.score) if minimize else float(item.score), reverse=True)
    specs = []
    for index, parent in enumerate(parents):
        if index + 1 < len(parents):
            specs.extend((("e1", [parents[0], parents[index + 1]]), ("e2", [parents[0], parents[index + 1]])))
        specs.extend((("m1", [parent]), ("m2", [parent])))
    if size > len(specs):
        raise ValueError(f"fixed query bank requested {size} unique queries, but only {len(specs)} are available")
    records = []
    for index, (operator, selected) in enumerate(specs[:size]):
        direct = min(selected, key=lambda item: float(item.score)) if minimize else max(selected, key=lambda item: float(item.score))
        prompt = {
            "e1": EoHPrompt.get_prompt_e1,
            "e2": EoHPrompt.get_prompt_e2,
            "m1": lambda task, items, template: EoHPrompt.get_prompt_m1(task, items[0], template),
            "m2": lambda task, items, template: EoHPrompt.get_prompt_m2(task, items[0], template),
        }[operator](task_description, selected, template_function)
        records.append(EoHPrompt.build_prompt_record(
            prompt_id=f"fixed_{index + 1:05d}_{operator}",
            prompt=prompt,
            operator_type=operator,
            parent_best_score=direct.score,
            parent_best_profile=getattr(direct, "_eoh_profile"),
            parent_best_id=_parent_id(direct, index + 1),
            parent_codes=[str(parent) for parent in selected],
            parent_ids=[_parent_id(parent, offset + 1) for offset, parent in enumerate(selected)],
            population_best_score=parents[0].score,
            population_best_profile=getattr(parents[0], "_eoh_profile"),
            group_size=group_size,
            reward_contract="vc_pair_v2",
            sampling_seed=int(seed) + index,
        ))
    return records


def load_checkpoint_parents(path: str) -> list[Function]:
    with open(path, encoding="utf-8") as file:
        rows = json.load(file).get("population", [])
    parents = []
    for row in rows:
        function = TextFunctionProgramConverter.text_to_function(row.get("function", ""))
        if function is None or not _finite(row.get("score")) or not row.get("performance_profile"):
            continue
        function.score = float(row["score"])
        function.algorithm = str(row.get("algorithm") or "")
        setattr(function, "_eoh_profile", list(row["performance_profile"]))
        if row.get("sample_order") is not None:
            setattr(function, "_eoh_sample_order", int(row["sample_order"]))
        parents.append(function)
    if not parents:
        raise ValueError(f"checkpoint contains no scored parents with performance profiles: {path}")
    return parents


def reevaluate_parents(parents: list[Function], *, evaluator, template_program) -> list[Function]:
    refreshed = []
    for parent in parents:
        program = TextFunctionProgramConverter.function_to_program(parent, template_program)
        packet = evaluator.evaluate_program_with_profile(program) if program is not None else None
        if not isinstance(packet, dict) or not _finite(packet.get("score")) or not packet.get("performance_profile"):
            raise RuntimeError("fixed-bank parent profile evaluation failed")
        parent.score = float(packet["score"])
        setattr(parent, "_eoh_profile", [float(value) for value in packet["performance_profile"]])
        refreshed.append(parent)
    return refreshed


def save_query_bank(path: str, records: list[dict], metadata: dict | None = None) -> dict:
    payload = {"schema": "eoh_rl_fixed_bank_v1", "bank_id": query_bank_id(records), "metadata": dict(metadata or {}), "records": records}
    _write_json(path, payload)
    return payload


def load_query_bank(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records or payload.get("bank_id") != query_bank_id(records):
        raise ValueError(f"invalid or modified fixed query bank: {path}")
    return payload


def evaluate_fixed_bank(
    *,
    policy,
    records: list[dict],
    evaluator,
    template_program,
    output_dir: str,
    reward: EoHReward | None = None,
) -> dict:
    if not records or len({record["prompt_id"] for record in records}) != len(records) or len({record["prompt"] for record in records}) != len(records):
        raise ValueError("fixed query bank must contain unique prompt_id and prompt values")
    generations = int(policy.config["num_generations"])
    if any(int(record.get("group_size", generations)) != generations for record in records):
        raise ValueError("fixed query bank group_size must match policy num_generations")
    reward = reward or EoHReward()
    result = policy.run_update(
        prompt_records=records,
        evaluator=evaluator,
        reward_fn=reward,
        template_program=template_program,
        output_dir=output_dir,
        update_id=0,
        training_enabled=False,
        known_functions=[code for record in records for code in record.get("parent_codes", [])],
        known_profiles=[profile for record in records for profile in (record.get("parent_best_profile"), record.get("population_best_profile")) if profile],
    )
    if not result.get("executed"):
        raise RuntimeError(f"fixed-bank evaluation failed: {result.get('reason')}")
    events = list((result.get("reward_summary") or {}).get("reward_events") or [])
    payload = {"bank_id": query_bank_id(records), "summary": summarize_events(events, minimize=reward.minimize), "events": events}
    _write_json(os.path.join(output_dir, "fixed_bank_results.json"), payload)
    return payload


def same_valid_summaries(event_sets: dict[str, list[dict]], *, minimize: bool = False) -> dict[str, dict]:
    if not event_sets:
        return {}
    grouped = {
        name: _eligible_by_prompt(events)
        for name, events in event_sets.items()
    }
    prompts = set.intersection(*(set(rows) for rows in grouped.values()))
    selected = {name: [] for name in grouped}
    for prompt_id in sorted(prompts):
        count = min(len(grouped[name][prompt_id]) for name in grouped)
        for name in grouped:
            selected[name].extend(grouped[name][prompt_id][:count])
    return {name: summarize_events(events, minimize=minimize) for name, events in selected.items()}


def same_valid_curves(event_sets: dict[str, list[dict]], *, minimize: bool = False) -> dict:
    grouped = {name: _eligible_by_prompt(events) for name, events in event_sets.items()}
    if not grouped:
        return {}
    prompts = set.intersection(*(set(rows) for rows in grouped.values()))
    max_k = max((min(len(grouped[name][prompt]) for name in grouped) for prompt in prompts), default=0)
    curves = {name: [] for name in grouped}
    for k in range(1, max_k + 1):
        comparable = [prompt for prompt in prompts if all(len(grouped[name][prompt]) >= k for name in grouped)]
        for name in grouped:
            best = []
            for prompt in comparable:
                scores = [float(row["score"]) for row in grouped[name][prompt][:k] if _finite(row.get("score"))]
                if scores:
                    best.append(min(scores) if minimize else max(scores))
            curves[name].append({"k": k, "query_count": len(best), "best_score": _mean(best)})
    return {name: {"curve": curve, "auc": _mean([row["best_score"] for row in curve if row["best_score"] is not None])} for name, curve in curves.items()}


def summarize_events(events: list[dict], *, minimize: bool = False) -> dict:
    eligible = [event for event in events if is_population_eligible_event(event)]
    groups = _eligible_by_prompt(eligible)
    total_queries = len({str(event.get("prompt_id") or "") for event in events})
    best_scores = []
    for rows in groups.values():
        scores = [float(row["score"]) for row in rows if _finite(row.get("score"))]
        if scores:
            best_scores.append(min(scores) if minimize else max(scores))
    paired = [float(event["paired_mean"]) for event in eligible if _finite(event.get("paired_mean"))]
    return {
        **event_funnel(events),
        "total_query_count": total_queries,
        "eligible_query_count": len(groups),
        "parent_win_rate": sum(bool(event.get("beats_parent")) for event in eligible) / len(eligible) if eligible else 0.0,
        "paired_outcome_counts": {
            outcome: sum(event.get("paired_outcome") == outcome for event in eligible)
            for outcome in ("positive", "neutral", "negative")
        },
        "paired_mean": _mean(paired),
        "paired_right_tail_p90": _quantile(paired, 0.9),
        "best_of_k_score": _mean(best_scores),
    }


def query_bank_id(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _eligible_by_prompt(events: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for event in events:
        if is_population_eligible_event(event):
            grouped.setdefault(str(event.get("prompt_id") or ""), []).append(event)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("completion_index") or 0))
    return grouped


def _mean(values: list[float]):
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], q: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _parent_id(parent: Function, fallback: int) -> int:
    value = getattr(parent, "_eoh_sample_order", None)
    return int(fallback if value is None else value)


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


__all__ = [
    "build_query_bank", "evaluate_fixed_bank", "load_checkpoint_parents", "load_query_bank",
    "query_bank_id", "reevaluate_parents", "same_valid_curves", "same_valid_summaries",
    "save_query_bank", "summarize_events",
]
