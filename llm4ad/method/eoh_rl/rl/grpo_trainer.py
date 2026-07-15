from __future__ import annotations

import concurrent.futures
import gc
import inspect
import json
import logging
import math
import os
import re
import shutil
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

import torch

from ....base import LLM, SampleTrimmer, TextFunctionProgramConverter
from ..prompt import EoHPrompt
from ..sampler import EoHSampler
from .sft_train import assert_lora_compatible as assert_adapter_compatible
from .sft_train import normalize_lora_config as normalize_adapter_config
from .sft_train import normalize_optional_path
from .vc_pair import compare_profiles

logger = logging.getLogger(__name__)


def _ensure_peft_tensor_parallel_compat() -> None:
    """Bridge PEFT 0.19 with transformers builds that renamed EmbeddingParallel."""
    try:
        import transformers.integrations.tensor_parallel as tensor_parallel
    except Exception:
        return
    if hasattr(tensor_parallel, "EmbeddingParallel"):
        return
    rowwise = getattr(tensor_parallel, "RowwiseParallel", None)
    if rowwise is None:
        return
    tensor_parallel.EmbeddingParallel = rowwise
    logger.warning(
        "Applied PEFT/transformers tensor_parallel compatibility shim: "
        "EmbeddingParallel -> RowwiseParallel"
    )


@contextmanager
def _redirect_training_output(path: str, enabled: bool = True):
    if not enabled:
        yield
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as file, redirect_stdout(file), redirect_stderr(file):
        yield


def _float(value, default=None):
    try: return default if value is None else float(value)
    except (TypeError, ValueError): return default


def _int(value, default=None):
    try: return default if value is None else int(value)
    except (TypeError, ValueError): return default


def _bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)


def _supports_profile(evaluator) -> bool:
    supported = getattr(evaluator, "supports_profile_evaluation", None)
    return bool(supported) if supported is not None else callable(getattr(evaluator, "evaluate_program_with_profile", None))


def _trainer_seed(seed: int, update_id: int) -> int:
    return (int(seed) + 1009 * int(update_id)) % (2**31 - 1)


def _as_list(value) -> list:
    return [] if value is None else value if isinstance(value, list) else list(value) if isinstance(value, tuple) else [value]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _check_randomness(program_str: str) -> bool:
    pattern = (
        r"\brandom\b|\bnp\.random\b|\bnumpy\.random\b|"
        r"\b(randint|choice|choices|shuffle|sample|uniform|gauss)\s*\(|"
        r"\btime\b|\bdatetime\b|\b(perf_counter|monotonic|utcnow|now)\s*\("
    )
    return bool(re.search(pattern, program_str or "", re.IGNORECASE))


def _check_score_metadata_leak(program_str: str) -> bool:
    text = str(program_str or "").lower()
    if not text: return False
    pattern = (
        r"\b(current_frontier_score|population_best_score|frontier_score|parent_scores?|parents_scores?)\b|"
        r"\bparent_score(?:_\d+)?\b|\bimprovement_over_(?:parent|parents|frontier)\b|\bscores?_after_addition\b"
    )
    return bool(re.search(pattern, text))


@dataclass
class EoHReward:
    minimize: bool = False
    detect_randomness: bool = True
    reward_parse_fail: float = -1.00
    reward_exec_fail: float = -0.80
    reward_none_return: float = -0.60
    reward_random: float = -0.70
    reward_leak: float = -0.70
    epsilon: float = 1.0e-4
    q3_scale: float = 0.50
    reward_mode: str = "vc_pair"
    pair_se_multiplier: float = 1.0
    pair_margin: float = 1.0e-4
    pair_positive_reward: float = 1.0
    pair_neutral_reward: float = 0.0
    pair_negative_reward: float = -0.5

    def __post_init__(self):
        self.minimize = _bool(self.minimize)
        self.detect_randomness = _bool(self.detect_randomness)
        self.pair_se_multiplier = float(self.pair_se_multiplier)
        self.pair_margin = float(self.pair_margin)
        self.pair_positive_reward = float(self.pair_positive_reward)
        self.pair_neutral_reward = float(self.pair_neutral_reward)
        self.pair_negative_reward = float(self.pair_negative_reward)
        if self.reward_mode not in {"vc_pair", "aggregate", "validity_only", "performance_shuffled"}:
            raise ValueError(f"unsupported reward_mode: {self.reward_mode}")
        values = (self.pair_se_multiplier, self.pair_margin, self.pair_positive_reward, self.pair_neutral_reward, self.pair_negative_reward)
        if not all(math.isfinite(value) for value in values) or self.pair_se_multiplier < 0.0 or self.pair_margin < 0.0:
            raise ValueError("VC-PAIR configuration must be finite with non-negative threshold terms")
        if not self.pair_positive_reward > self.pair_neutral_reward > self.pair_negative_reward:
            raise ValueError("VC-PAIR rewards must satisfy positive > neutral > negative")

    def invalid_result(self, label: str, reward: float) -> dict:
        return {
            "reward": float(reward), "reward_label": f"invalid_{label}",
            "reward_quadrant": "invalid", "failure_label": str(label),
            "beats_parent": False, "beats_frontier": False,
            "ties_parent": False, "performance_improved": False,
        }


def build_reward_fn_from_task_rl(task_rl: dict, logger=None) -> EoHReward:
    cfg = task_rl or {}
    reward = EoHReward(**{key: cfg.get(key, default) for key, default in {
        "minimize": False, "detect_randomness": True,
        "reward_parse_fail": -1.00, "reward_exec_fail": -0.80,
        "reward_none_return": -0.60, "reward_random": -0.70, "reward_leak": -0.70,
        "epsilon": 1.0e-4, "q3_scale": 0.50, "reward_mode": "vc_pair",
        "pair_se_multiplier": 1.0, "pair_margin": 1.0e-4,
        "pair_positive_reward": 1.0, "pair_neutral_reward": 0.0, "pair_negative_reward": -0.5,
    }.items()})
    if logger is not None:
        logger.info("奖励配置: %s", vars(reward))
    return reward


def _record(raw: dict) -> dict:
    prompt = str(raw.get("prompt") or "").strip()
    prompt_id = str(raw.get("prompt_id") or "").strip()
    op = str(raw.get("operator_type") or "").strip().lower()
    if not prompt or not prompt_id or not op:
        raise RuntimeError("GRPO prompt record requires prompt, prompt_id and operator_type")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        system = str(raw.get("system_prompt") or "")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}] if system.strip() else [{"role": "user", "content": prompt}]
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "messages": messages,
        "operator_type": op,
        "parent_best_score": _float(raw.get("parent_best_score")),
        "parent_best_profile": raw.get("parent_best_profile"),
        "parent_best_id": raw.get("parent_best_id"),
        "population_best_score": _float(raw.get("population_best_score")),
        "parent_codes": _as_list(raw.get("parent_codes")),
        "parent_ids": None if raw.get("parent_ids") is None else _as_list(raw.get("parent_ids")),
        "group_size": int(raw.get("group_size") or 1),
        "reward_contract": str(raw.get("reward_contract") or "vc_pair_v1"),
    }


def _dataset(records: list[dict]):
    from datasets import Dataset

    return Dataset.from_dict({"prompt": [row["messages"] for row in records]})


def _basic_stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "mean": 0.0, "max": None, "std": 0.0}
    mean = sum(values) / len(values)
    return {"min": min(values), "mean": mean, "max": max(values), "std": (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5}


def is_population_eligible_event(event: dict) -> bool:
    return (
        bool(event.get("exec_success"))
        and bool(event.get("validity", True))
        and not bool(event.get("random_algo"))
        and not bool(event.get("score_metadata_leak"))
        and not bool(event.get("exact_parent_copy"))
    )


def event_funnel(events: list[dict]) -> dict:
    total = len(events)
    counts = {
        "completion_count": total,
        "contract_valid_count": sum(bool(row.get("parse_success")) for row in events),
        "evaluation_count": sum(bool(row.get("evaluation_attempted")) for row in events),
        "execution_valid_count": sum(bool(row.get("exec_success")) for row in events),
        "population_eligible_count": sum(is_population_eligible_event(row) for row in events),
        "archive_novel_count": sum(is_population_eligible_event(row) and not bool(row.get("archive_duplicate")) for row in events),
    }
    rates = {
        key.replace("_count", "_rate"): value / total if total else 0.0
        for key, value in counts.items()
        if key != "completion_count"
    }
    return {**counts, **rates}


class EoHGRPOReward:
    """Parse, evaluate, and score GRPO completions for one EoH prompt."""

    def __init__(
        self,
        *,
        records: list[dict],
        evaluator,
        template_program,
        reward_config: EoHReward,
        tokenizer=None,
        reward_eval_workers: int = 1,
        enable_ast_gate: bool = False,
        known_functions: list[str] | None = None,
    ):
        self.records = [_record(record) for record in records]
        self.evaluator = evaluator
        self.template_program = template_program
        self.reward_config = reward_config
        self.tokenizer = tokenizer
        self.reward_eval_workers = max(1, int(reward_eval_workers or 1))
        self.__name__ = "eohrl_graded_reward"
        self._parser = EoHSampler(None, template_program, enable_ast_gate=enable_ast_gate)
        self._seen_codes = {self._normalize_code(code) for code in known_functions or []}
        self._seen_codes.discard("")
        self._prompt_to_meta = {record["prompt"]: record for record in self.records}
        self._rows: list[dict] = []
        self._candidates: list[dict] = []
        self.reward_values: list[float] = []
        self._seen_prompt_token_keys: set[str] = set()
        self._token_usage = {
            "input_tokens_total": 0,
            "unique_input_tokens_total": 0,
            "output_tokens_total": 0,
            "tokens_total": 0,
            "completion_count": 0,
            "unique_prompt_count": 0,
        }
        self._timing = {
            "reward_callback_wall_elapsed": 0.0,
            "reward_parse_wall_elapsed": 0.0,
            "reward_eval_wall_elapsed": 0.0,
            "reward_eval_program_time_total": 0.0,
            "reward_eval_count": 0,
            "reward_call_count": 0,
        }

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        del kwargs
        call_started = time.time()
        rows, pending, rewards = [], [], []
        for idx, (prompt_obj, completion_obj) in enumerate(zip(prompts, completions)):
            parse_started = time.time()
            prompt_text = self._text(prompt_obj, last=True)
            completion_text = self._text(completion_obj)
            meta = self._resolve_meta(prompt_text)
            row, item = self._parse(meta, completion_text)
            row["completion_index"] = len(self._rows) + len(rows) + 1
            token_usage = self._token_usage_for(prompt_obj, prompt_text, completion_text)
            row.update({key: token_usage[key] for key in ("input_tokens", "output_tokens", "tokens_total")})
            self._accumulate_token_usage(token_usage)
            self._timing["reward_parse_wall_elapsed"] += time.time() - parse_started
            rows.append(row)
            rewards.append(row.get("reward"))
            if item is not None:
                pending.append((idx, item))
        for idx, result in self._evaluate_pending(pending):
            rewards[idx] = self._finish(rows[idx], result)
        self._shuffle_performance_rewards(rows, rewards)
        final = [float(value if value is not None else self.reward_config.reward_parse_fail) for value in rewards]
        for row, reward in zip(rows, final):
            row["reward"] = reward
            self.reward_values.append(reward)
        self._rows.extend(rows)
        self._timing["reward_call_count"] += 1
        self._timing["reward_callback_wall_elapsed"] += time.time() - call_started
        logger.info(
            "[EoHRLReward] completions=%d reward_mean=%.4f tokens=%d input=%d output=%d",
            len(rows),
            _basic_stats(final)["mean"],
            self._token_usage["tokens_total"],
            self._token_usage["input_tokens_total"],
            self._token_usage["output_tokens_total"],
        )
        return final

    def candidates(self) -> list[dict]:
        return list(self._candidates)

    def events(self) -> list[dict]:
        return [self._event(row) for row in self._rows]

    def summary(self) -> dict:
        events = self.events()
        valid = [event for event in events if is_population_eligible_event(event)]
        groups: dict[str, list[float]] = {}
        for event in events:
            groups.setdefault(str(event.get("prompt_id") or ""), []).append(float(event.get("reward") or 0.0))
        zero_std = sum(_basic_stats(values)["std"] == 0.0 for values in groups.values())
        reward_labels: dict[str, int] = {}
        for event in events:
            label = str(event.get("reward_label") or "unknown")
            reward_labels[label] = reward_labels.get(label, 0) + 1
        return {
            "funnel": event_funnel(events),
            "positive_reward_count": sum(float(event.get("reward") or 0.0) > 0.0 for event in events),
            "parent_improve_count": sum(bool(event.get("beats_parent")) for event in valid),
            "parent_improve_only_count": sum(bool(event.get("beats_parent")) and not bool(event.get("beats_frontier")) for event in valid),
            "frontier_improve_count": sum(bool(event.get("beats_frontier")) for event in valid),
            "valid_non_improving_count": sum(not bool(event.get("performance_improved")) for event in valid),
            "archive_duplicate_count": sum(bool(event.get("archive_duplicate")) for event in valid),
            "paired_outcome_counts": {
                outcome: sum(event.get("paired_outcome") == outcome for event in valid)
                for outcome in ("positive", "neutral", "negative")
            },
            "paired_mean_stats": _basic_stats([float(event["paired_mean"]) for event in valid if event.get("paired_mean") is not None]),
            "paired_se_stats": _basic_stats([float(event["paired_se"]) for event in valid if event.get("paired_se") is not None]),
            "frac_reward_zero_std": zero_std / max(len(groups), 1),
            "reward_stats": _basic_stats(self.reward_values),
            "reward_label_counts": reward_labels,
            "reward_events": events,
            "token_usage": dict(self._token_usage),
            "timing": dict(self._timing),
        }

    def _parse(self, meta: dict, completion: str) -> tuple[dict, dict | None]:
        row = {
            "prompt_id": meta["prompt_id"],
            "operator_type": meta["operator_type"],
            "parent_ids": meta.get("parent_ids"),
            "parent_best_id": meta.get("parent_best_id"),
            "parent_best_score": meta.get("parent_best_score"),
            "population_best_score": meta.get("population_best_score"),
            "group_size": meta.get("group_size"),
            "reward_contract": meta.get("reward_contract"),
            "registered_to_population": False,
            "survived_main_population": False,
            "blocked_from_population": True,
            "evaluation_attempted": False,
        }
        if not str(completion or "").strip():
            row.update(self._failure("no_output", parse_success=False))
            return row, None
        parsed = self._parser.parse_response_record(completion)
        row.update(strategy=parsed.get("strategy"))
        if parsed.get("failure_label"):
            row.update(self._failure(str(parsed.get("failure_label") or "missing_function"), parse_success=False))
            return row, None
        return row, {"meta": meta, "func": parsed["func"], "program": parsed["program"]}

    def _evaluate_one(self, program) -> dict:
        started = time.time()
        try:
            if _supports_profile(self.evaluator):
                if hasattr(self.evaluator, "evaluate_program_with_profile_record_time_with_diag"):
                    packet, eval_time, diag = self.evaluator.evaluate_program_with_profile_record_time_with_diag(program)
                else:
                    packet = self.evaluator.evaluate_program_with_profile(program)
                    eval_time, diag = time.time() - started, {"ok": packet is not None, "bucket": "ok" if packet is not None else "profile_eval_failed", "error_type": None}
                if isinstance(packet, dict) and packet.get("score") is not None:
                    return {"score": packet.get("score"), "eval_time": eval_time, "diag": {**dict(diag or {}), **packet}}
                return {"score": None, "eval_time": eval_time, "diag": diag}
            if hasattr(self.evaluator, "evaluate_program_record_time_with_diag"):
                score, eval_time, diag = self.evaluator.evaluate_program_record_time_with_diag(program)
            else:
                score, eval_time = self.evaluator.evaluate_program_record_time(program)
                diag = None
            return {"score": score, "eval_time": eval_time if eval_time is not None else time.time() - started, "diag": diag}
        except Exception as exc:
            return {"score": None, "eval_time": time.time() - started, "diag": {"error": str(exc)}}

    def _evaluate_pending(self, pending: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
        started = time.time()
        if self.reward_eval_workers <= 1 or len(pending) <= 1:
            results = [(idx, {**item, **self._evaluate_one(item["program"])}) for idx, item in pending]
            self._record_eval_timing(started, results)
            return results

        results: list[tuple[int, dict]] = []
        max_workers = min(self.reward_eval_workers, len(pending))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(self._evaluate_one, item["program"]): (idx, item) for idx, item in pending}
            for future in concurrent.futures.as_completed(future_to_item):
                idx, item = future_to_item[future]
                try:
                    eval_out = future.result()
                except Exception as exc:
                    eval_out = {"score": None, "eval_time": 0.0, "diag": {"error": str(exc)}}
                results.append((idx, {**item, **eval_out}))
        results.sort(key=lambda item: item[0])
        self._record_eval_timing(started, results)
        return results

    def _record_eval_timing(self, started: float, results: list[tuple[int, dict]]) -> None:
        self._timing["reward_eval_wall_elapsed"] += time.time() - started
        self._timing["reward_eval_count"] += len(results)
        for _idx, result in results:
            try:
                self._timing["reward_eval_program_time_total"] += float(result.get("eval_time") or 0.0)
            except Exception:
                pass

    def _finish(self, row: dict, result: dict) -> float:
        meta, func, program_text = result["meta"], result["func"], str(result["program"])
        row.update(function=str(func), program=program_text, eval_time=result.get("eval_time"), evaluation_attempted=True, exec_success=False)
        score = _float(result.get("score"))
        if score is None or not math.isfinite(score):
            # Distinguish runtime exception (-0.80) from logic error / returns None (-0.60)
            diag = result.get("diag") or {}
            r = self.reward_config.reward_exec_fail if diag.get("error") else self.reward_config.reward_none_return
            row.update(self._failure("non_finite_score", parse_success=True, reward=r))
            self._candidates.append({"row": row, "func": func, "program_str": program_text, "score": None, "eval_time": result.get("eval_time"), "diag": diag, "meta": meta})
            return r
        random_algo = bool(self.reward_config.detect_randomness and _check_randomness(program_text))
        score_leak = bool(_check_score_metadata_leak(program_text))
        parent_copy = self._is_exact_parent_copy(str(func), meta.get("parent_codes") or [])
        if random_algo or score_leak or parent_copy:
            label = "exact_parent_copy" if parent_copy else "metadata_leak" if score_leak else "randomness_detected"
            r = self.reward_config.reward_random if random_algo else self.reward_config.reward_leak
            row.update(self._failure(label, parse_success=True, reward=r))
            row.update(score=score, exec_success=True, random_algo=random_algo, score_metadata_leak=score_leak, exact_parent_copy=parent_copy)
            self._candidates.append({"row": row, "func": func, "program_str": program_text, "score": score, "eval_time": result.get("eval_time"), "diag": result.get("diag"), "meta": meta})
            return r
        perf = self._compute_perf_reward(score, meta.get("parent_best_score"), meta.get("population_best_score"))
        code_key = self._normalize_code(str(func))
        archive_duplicate = bool(code_key and code_key in self._seen_codes)
        if code_key:
            self._seen_codes.add(code_key)
        mode = self.reward_config.reward_mode
        if mode == "vc_pair":
            paired = compare_profiles(
                (result.get("diag") or {}).get("performance_profile"),
                meta.get("parent_best_profile"),
                minimize=self.reward_config.minimize,
                se_multiplier=self.reward_config.pair_se_multiplier,
                margin=self.reward_config.pair_margin,
            )
            if not math.isclose(paired["paired_candidate_mean"], score, rel_tol=1.0e-9, abs_tol=1.0e-9):
                raise ValueError("VC-PAIR candidate profile mean does not match its score")
            if not math.isclose(paired["paired_parent_mean"], float(meta["parent_best_score"]), rel_tol=1.0e-9, abs_tol=1.0e-9):
                raise ValueError("VC-PAIR parent profile mean does not match its score")
            outcome = paired["paired_outcome"]
            perf.update(
                paired,
                reward={
                    "positive": self.reward_config.pair_positive_reward,
                    "neutral": self.reward_config.pair_neutral_reward,
                    "negative": self.reward_config.pair_negative_reward,
                }[outcome],
                reward_label=f"pair_{outcome}",
                reward_quadrant=outcome,
                performance_improved=outcome == "positive",
            )
        elif mode == "validity_only":
            perf.update(reward=1.0, reward_label="validity_only")
        row.update(
            **perf,
            validity=perf.get("failure_label") is None,
            exec_success=True,
            parse_success=True,
            random_algo=False,
            score_metadata_leak=False,
            exact_parent_copy=False,
            archive_duplicate=archive_duplicate,
            blocked_from_population=perf.get("failure_label") is not None,
        )
        self._candidates.append({"row": row, "func": func, "program_str": program_text, "score": score, "eval_time": result.get("eval_time"), "diag": result.get("diag"), "meta": meta})
        return float(perf["reward"])

    def _shuffle_performance_rewards(self, rows: list[dict], rewards: list) -> None:
        if self.reward_config.reward_mode != "performance_shuffled":
            return
        groups: dict[str, list[int]] = {}
        for idx, row in enumerate(rows):
            if is_population_eligible_event(row):
                groups.setdefault(str(row.get("prompt_id") or ""), []).append(idx)
        for indices in groups.values():
            values = [rewards[idx] for idx in indices]
            if len(values) < 2:
                continue
            for idx, value in zip(indices, values[1:] + values[:1]):
                rewards[idx] = row_reward = float(value)
                rows[idx].update(reward=row_reward, reward_label="performance_shuffled")

    def _compute_perf_reward(self, score: float, parent_score, frontier_score) -> dict:
        cfg = self.reward_config
        utility = lambda v: -float(v) if cfg.minimize else float(v)
        parent = _float(parent_score, frontier_score)
        frontier = _float(frontier_score, parent)
        if parent is None:
            return {"reward": 0.0, "reward_label": "no_baseline", "reward_quadrant": "q3",
                    "beats_parent": False, "beats_frontier": False, "ties_parent": True,
                    "performance_improved": False, "failure_label": None,
                    "delta_parent": None, "delta_parent_rel": None,
                    "delta_frontier": None, "delta_frontier_rel": None}
        u, up, uf = utility(score), utility(parent), utility(frontier)
        dp, df = u - up, u - uf
        rp = dp / max(abs(up), cfg.epsilon)
        rf = df / max(abs(uf), cfg.epsilon)
        beats_parent = rp > cfg.epsilon
        beats_frontier = rf > cfg.epsilon
        ties_parent = abs(rp) <= cfg.epsilon
        base = {
            "failure_label": None, "score": score,
            "delta_parent": dp, "delta_parent_rel": rp,
            "delta_frontier": df, "delta_frontier_rel": rf,
            "beats_parent": beats_parent, "beats_frontier": beats_frontier,
            "ties_parent": ties_parent, "performance_improved": beats_parent or beats_frontier,
        }
        if beats_frontier:
            return {**base, "reward": 1.0 + _clip(rf, 0.0, 1.0),
                    "reward_label": "q1_frontier", "reward_quadrant": "q1"}
        if beats_parent:
            return {**base, "reward": 0.3 + _clip(rp, 0.0, 0.7),
                    "reward_label": "q1_parent", "reward_quadrant": "q1"}
        return {**base, "reward": _clip(rp, -1.0, 0.0) * cfg.q3_scale,
                "reward_label": "q3_continuous", "reward_quadrant": "q3"}

    def _failure(self, label: str, *, parse_success: bool, reward: float | None = None) -> dict:
        r = reward if reward is not None else self.reward_config.reward_parse_fail
        result = self.reward_config.invalid_result(label, r)
        return {
            **result,
            "failure_label": label,
            "validity": False,
            "parse_success": bool(parse_success),
            "exec_success": False,
            "beats_parent": False,
            "beats_frontier": False,
            "performance_improved": False,
            "random_algo": False,
            "score_metadata_leak": False,
            "exact_parent_copy": False,
        }

    @staticmethod
    def _event(row: dict) -> dict:
        event = {key: row.get(key) for key in "prompt_id operator_type parent_ids parent_best_id parent_best_score population_best_score completion_index sample_order score reward reward_label reward_quadrant delta_parent delta_parent_rel delta_frontier delta_frontier_rel paired_outcome paired_n paired_candidate_mean paired_parent_mean paired_mean paired_std paired_se paired_threshold paired_win_rate strategy function failure_label input_tokens output_tokens tokens_total".split()}
        event.update({key: bool(row.get(key, False)) for key in "validity parse_success evaluation_attempted exec_success beats_parent beats_frontier ties_parent performance_improved exact_parent_copy archive_duplicate random_algo score_metadata_leak registered_to_population survived_main_population blocked_from_population".split()})
        return event

    def _resolve_meta(self, prompt: str) -> dict:
        if prompt in self._prompt_to_meta:
            return self._prompt_to_meta[prompt]
        matches = [record for record in self.records if record["prompt"] and record["prompt"] in prompt]
        if len(matches) != 1:
            raise RuntimeError("GRPO prompt metadata missing")
        return matches[0]

    def _is_exact_parent_copy(self, function: str, parent_codes: list[str]) -> bool:
        normalized = self._normalize_code(function)
        return bool(normalized) and any(normalized == self._normalize_code(code) for code in parent_codes or [])

    def _normalize_code(self, code: str) -> str:
        text = str(code or "").strip()
        if not text:
            return ""
        try:
            func = TextFunctionProgramConverter.text_to_function(text)
        except Exception:
            try:
                func = SampleTrimmer.sample_to_function(text, self.template_program)
            except Exception:
                func = None
        body = str(func.body if func is not None else text).strip()
        return "\n".join(line.rstrip() for line in body.splitlines()).strip()

    @staticmethod
    def _text(obj: Any, *, last: bool = False) -> str:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[-1 if last else 0], dict):
            item = obj[-1 if last else 0]
            return str(item.get("content") or item.get("message") or item.get("text") or "")
        if isinstance(obj, dict):
            return str(obj.get("content") or obj.get("prompt") or obj.get("text") or "")
        return str(obj or "")

    def _token_usage_for(self, prompt_obj: Any, prompt_text: str, completion_text: str) -> dict:
        input_tokens = self._count_prompt_tokens(prompt_obj, prompt_text)
        output_tokens = self._count_text_tokens(completion_text)
        return {
            "prompt_key": str(prompt_text or ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_total": input_tokens + output_tokens,
        }

    def _accumulate_token_usage(self, usage: dict) -> None:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("tokens_total", input_tokens + output_tokens) or 0)
        self._token_usage["input_tokens_total"] += input_tokens
        self._token_usage["output_tokens_total"] += output_tokens
        self._token_usage["tokens_total"] += total_tokens
        self._token_usage["completion_count"] += 1
        prompt_key = str(usage.get("prompt_key") or "")
        if prompt_key not in self._seen_prompt_token_keys:
            self._seen_prompt_token_keys.add(prompt_key)
            self._token_usage["unique_prompt_count"] += 1
            self._token_usage["unique_input_tokens_total"] += input_tokens

    def _count_prompt_tokens(self, prompt_obj: Any, prompt_text: str) -> int:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return 0
        try:
            if isinstance(prompt_obj, list) and hasattr(tokenizer, "apply_chat_template"):
                token_ids = tokenizer.apply_chat_template(prompt_obj, add_generation_prompt=True)
                return self._token_len(token_ids)
        except Exception:
            pass
        return self._count_text_tokens(prompt_text)

    def _count_text_tokens(self, text: str) -> int:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return 0
        try:
            return len(tokenizer.encode(str(text or ""), add_special_tokens=False))
        except Exception:
            pass
        try:
            encoded = tokenizer(str(text or ""), add_special_tokens=False)
            return self._token_len(encoded.get("input_ids") if isinstance(encoded, dict) else encoded)
        except Exception:
            return 0

    @staticmethod
    def _token_len(token_ids: Any) -> int:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, dict):
            token_ids = token_ids.get("input_ids", [])
        if token_ids is None:
            return 0
        try:
            return len(token_ids)
        except Exception:
            return 0


class ResidentGRPOPolicy:
    _REQUIRED_CONFIG = (
        "max_seq_length", "seed", "gradient_checkpointing", "num_generations",
        "max_prompt_length", "max_completion_length", "temperature", "top_p",
        "beta", "epsilon", "epsilon_high", "scale_rewards", "loss_type",
        "num_iterations", "importance_sampling_level", "mask_truncated_completions",
        "gpu_memory_utilization",
    )
    _REQUIRED_TRL_ARGS = (
        "num_generations", "generation_batch_size", "max_prompt_length",
        "max_completion_length", "temperature", "top_p", "beta", "epsilon",
        "epsilon_high", "scale_rewards", "loss_type", "num_iterations",
        "importance_sampling_level", "mask_truncated_completions",
    )
    def __init__(
        self,
        *,
        model_name_or_path: str,
        grpo_config: dict,
        adapter_config: dict,
        training_gpus: list[int] | None = None,
        initial_adapter_path: str | None = None,
        quantization: str | None = None,
    ):
        self.config = dict(grpo_config or {})
        self._model_name_or_path = model_name_or_path
        self._adapter_config = normalize_adapter_config(adapter_config)
        self._training_gpus = list(training_gpus or [])
        self._quantization = quantization or self.config.get("quantization", "4bit")
        self._adapter_path = self._normalize_adapter_path(initial_adapter_path)
        self._inference_lora_name = "grpo_trainer_lora_model"
        self._inference_lora_request = None
        self._synced_lora_name = None
        self._model = self._tokenizer = self._fast_language_model_cls = None
        self._optimizer_steps = 0
        self._inference_prepared = False
        self._require_config()
        self._load_model()

    def draw_samples(self, prompt: str | Any, n: int, *args, **kwargs) -> list[str]:
        del args, kwargs
        if n <= 0:
            return []
        from vllm import SamplingParams

        request = self.prepare_for_inference()
        sampling = SamplingParams(
            n=min(int(n), 64),
            max_tokens=int(self.config["max_completion_length"]),
            temperature=float(self.config["temperature"]),
            top_p=float(self.config["top_p"]),
        )
        kwargs_fast = {"sampling_params": sampling, "use_tqdm": False}
        if request is not None:
            kwargs_fast["lora_request"] = request
        with torch.inference_mode():
            outputs = self._model.fast_generate(self._format_prompt(prompt), **kwargs_fast)
        return [getattr(item, "text", "") for output in outputs or [] for item in getattr(output, "outputs", []) or []]

    def run_update(
        self,
        *,
        prompt_records: list[dict],
        evaluator,
        reward_fn,
        template_program,
        output_dir: str | None = None,
        update_id: int = 0,
        reward_eval_workers: int = 1,
        enable_ast_gate: bool = False,
        training_enabled: bool = True,
        known_functions: list[str] | None = None,
        **_,
    ) -> dict:
        if not prompt_records:
            return {"executed": False, "reason": "no_prompts"}
        records = [_record(record) for record in prompt_records]
        update_dir = os.path.join(output_dir or self.config.get("output_dir") or "./eoh_rl_runs", "online_grpo", f"update_{int(update_id):03d}")
        os.makedirs(update_dir, exist_ok=True)
        with open(os.path.join(update_dir, "prompts.json"), "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)

        callback = EoHGRPOReward(
            records=records,
            evaluator=evaluator,
            template_program=template_program,
            reward_config=reward_fn,
            tokenizer=self._tokenizer,
            reward_eval_workers=reward_eval_workers,
            enable_ast_gate=enable_ast_gate,
            known_functions=known_functions,
        )
        started = time.time()
        try:
            metrics = (
                self._train_with_trl(_dataset(records), callback, update_dir, update_id)
                if training_enabled
                else self._rollout_without_training(records, callback)
            )
            summary = callback.summary()
            expected = len(records) * int(self.config["num_generations"])
            actual = int(summary["funnel"]["completion_count"])
            if actual != expected:
                raise RuntimeError(f"expected {expected} completions, got {actual}")
            if not self._finite(summary, metrics):
                raise RuntimeError("GRPO produced non-finite loss or reward stats")
            if training_enabled:
                self.prepare_for_inference(force_reload=True)
        except Exception as exc:
            summary = callback.summary()
            logger.warning("[EoHRL] update=%d failed: %s", int(update_id), exc)
            return {
                "executed": False,
                "reason": "trainer_exception",
                "training_enabled": bool(training_enabled),
                "candidates": callback.candidates(),
                "metrics": {
                    "reward_summary": summary,
                    "token_usage": dict(summary.get("token_usage") or {}),
                    "timing": {**dict(summary.get("timing") or {}), "update_total_elapsed": time.time() - started},
                },
            }

        timing = {
            **dict(summary.get("timing") or {}),
            "update_total_elapsed": time.time() - started,
            "trl_trainer_train_elapsed": float(metrics.get("trl_trainer_train_elapsed", 0.0) or 0.0),
            "trl_trainer_init_elapsed": float(metrics.get("trl_trainer_init_elapsed", 0.0) or 0.0),
            "trl_trainer_total_elapsed": float(metrics.get("trl_trainer_total_elapsed", 0.0) or 0.0),
        }
        metrics.update(
            executed=True,
            training_enabled=bool(training_enabled),
            mode="grpo" if training_enabled else "rollout_only",
            reward_summary=summary,
            token_usage=dict(summary.get("token_usage") or {}),
            timing=timing,
            num_prompts=len(records),
            num_generations=int(self.config["num_generations"]),
        )
        return {"executed": True, "metrics": metrics, "reward_summary": summary, "candidates": callback.candidates()}

    def _rollout_without_training(self, records: list[dict], callback: EoHGRPOReward) -> dict:
        group_size = int(self.config["num_generations"])
        prompts, completions = [], []
        for record in records:
            outputs = self.draw_samples(record["messages"], group_size)
            outputs.extend([""] * (group_size - len(outputs)))
            prompts.extend([record["messages"]] * group_size)
            completions.extend(outputs[:group_size])
        callback(prompts, completions)
        return {"loss": None, "optimizer_steps_this_update": 0}

    def prepare_for_inference(self, *, force_reload: bool = False):
        if force_reload:
            self._inference_lora_request = None
            self._synced_lora_name = None
            self._inference_prepared = False
        if not self._inference_prepared and self._fast_language_model_cls is not None:
            self._fast_language_model_cls.for_inference(self._model)
            self._inference_prepared = True
        return self._sync_lora_to_inference()

    def close(self) -> None:
        try:
            engine = getattr(self._model, "vllm_engine", None) or getattr(getattr(self._model, "model", None), "vllm_engine", None)
            if engine is not None and hasattr(engine, "sleep"):
                engine.sleep(level=2)
        except Exception:
            pass
        self._inference_lora_request = None
        self._synced_lora_name = None
        self._model = self._tokenizer = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self) -> None:
        os.environ.setdefault("UNSLOTH_VLLM_NO_FLASHINFER", "1")
        os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
        for name in ("vllm", "trl", "transformers", "unsloth"):
            logging.getLogger(name).setLevel(logging.INFO)
        _ensure_peft_tensor_parallel_compat()
        if torch.cuda.is_available() and len(self._training_gpus) == 1:
            torch.cuda.set_device(self._visible_cuda_index(self._training_gpus[0]))
        from unsloth import FastLanguageModel, PatchFastRL

        try:
            PatchFastRL("GRPO", FastLanguageModel)
        except OSError as exc:
            if "could not get source code" not in str(exc):
                raise
        self._fast_language_model_cls = FastLanguageModel
        rank = int(self._adapter_config.get("r", self.config.get("lora_rank", 32)))
        load_path = self._adapter_path or self._model_name_or_path
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_path,
            max_seq_length=int(self.config["max_seq_length"]),
            load_in_4bit=str(self._quantization).lower() in {"4bit", "bitsandbytes"},
            fast_inference=True,
            max_lora_rank=rank,
            gpu_memory_utilization=float(self.config["gpu_memory_utilization"]),
        )
        has_peft = bool(getattr(self._model, "peft_config", None))
        if self._adapter_path and not has_peft:
            raise RuntimeError(f"SFT LoRA adapter was not loaded by Unsloth: {self._adapter_path}")
        if not has_peft:
            self._model = FastLanguageModel.get_peft_model(
                self._model,
                r=rank,
                target_modules=list(self._adapter_config["target_modules"]),
                lora_alpha=int(self._adapter_config["lora_alpha"]),
                lora_dropout=float(self._adapter_config["lora_dropout"]),
                use_gradient_checkpointing="unsloth",
                random_state=int(self.config["seed"]),
            )
        if self._adapter_path:
            logger.info("[EoHRLGRPO] loaded SFT LoRA via Unsloth adapter path: %s", self._adapter_path)
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info("[EoHRLGRPO] loaded model: %s", load_path)
        self.prepare_for_inference()

    def _train_with_trl(self, dataset, reward_func, output_dir: str, update_id: int) -> dict:
        from trl import GRPOConfig, GRPOTrainer

        if hasattr(self._model, "for_training"):
            self._model.for_training(use_gradient_checkpointing=bool(self.config["gradient_checkpointing"]))
        self._model.train()
        self._inference_prepared = False
        supported = set(inspect.signature(GRPOConfig).parameters)
        missing = sorted(set(self._REQUIRED_TRL_ARGS) - supported)
        if missing:
            raise RuntimeError("TRL GRPOConfig lacks required arguments: " + ", ".join(missing))
        trainer_seed = _trainer_seed(self.config["seed"], update_id)
        effective_args = {key: value for key, value in self._grpo_args(output_dir).items() if key in supported and value is not None}
        effective_args["seed"] = trainer_seed
        prompts_per_update = max(1, len(dataset) if hasattr(dataset, "__len__") else int(self.config.get("prompts_per_update", 1) or 1))
        grouped_batch_size = prompts_per_update * int(self.config["num_generations"])
        effective_args["per_device_train_batch_size"] = grouped_batch_size
        effective_args["generation_batch_size"] = grouped_batch_size
        effective_args["gradient_accumulation_steps"] = 1
        trainer_kwargs = {"model": self._model, "reward_funcs": reward_func, "args": GRPOConfig(**effective_args), "train_dataset": dataset}
        trainer_kwargs["processing_class" if "processing_class" in set(inspect.signature(GRPOTrainer).parameters) else "tokenizer"] = self._tokenizer
        train_result = trainer = None
        started = time.time()
        quiet = _bool(self.config.get("quiet_training_output", False))
        log_path = os.path.join(output_dir, "trainer_output.log")
        try:
            with _redirect_training_output(log_path, enabled=quiet):
                trainer_init_started = time.time()
                trainer = GRPOTrainer(**trainer_kwargs)
                actual_args = {key: getattr(trainer.args, key, value) for key, value in effective_args.items()}
                drift = {
                    key: (effective_args[key], actual_args[key])
                    for key in self._REQUIRED_TRL_ARGS
                    if actual_args[key] != effective_args[key]
                }
                if drift:
                    raise RuntimeError(f"TRL changed explicit GRPO arguments: {drift}")
                adapter_norm_before = self._trainable_norm()
                train_started = time.time()
                train_result = trainer.train()
            metrics = dict(getattr(train_result, "metrics", {}) or {})
            metrics.setdefault("loss", metrics.get("train_loss", 0.0))
            self._optimizer_steps += int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
            step_metrics = next((row for row in reversed(trainer.state.log_history) if "loss" in row), {})
            metrics.update(
                grpo_effective_args=actual_args,
                optimizer_steps_this_update=int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0),
                optimizer_steps_total=self._optimizer_steps,
                optimizer_reinitialized=True,
                optimizer_state_reused=False,
                grpo_step_metrics=step_metrics,
                adapter_norm_before=adapter_norm_before,
                adapter_norm_after=self._trainable_norm(),
                trainer_seed=trainer_seed,
                trainer_output_log=log_path if quiet else None,
                trl_trainer_init_elapsed=train_started - trainer_init_started,
                trl_trainer_train_elapsed=time.time() - train_started,
                trl_trainer_total_elapsed=time.time() - started,
            )
            return metrics
        finally:
            try:
                self._model.zero_grad(set_to_none=True)
            except Exception:
                pass
            del train_result
            del trainer
            gc.collect()

    def _trainable_norm(self) -> float:
        params = [param.detach().float() for param in self._model.parameters() if param.requires_grad]
        if not params:
            return 0.0
        total = sum(param.square().sum() for param in params)
        return float(total.sqrt().item())

    def _grpo_args(self, output_dir: str) -> dict:
        bf16, fp16 = self._precision()
        values = {
            "output_dir": output_dir,
            "bf16": bf16,
            "fp16": fp16,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "remove_unused_columns": False,
            "report_to": self.config.get("report_to", "none"),
            "disable_tqdm": _bool(self.config.get("disable_tqdm", False)),
            "logging_strategy": self.config.get("logging_strategy", "steps"),
            "log_level": self.config.get("log_level", "warning"),
            "log_level_replica": self.config.get("log_level_replica", "warning"),
        }
        for key, default, cast in [
            ("use_vllm", True, bool), ("learning_rate", 5.0e-5, float), ("adam_beta1", 0.9, float), ("adam_beta2", 0.99, float),
            ("weight_decay", 0.1, float), ("warmup_ratio", 0.0, float), ("lr_scheduler_type", "constant", str), ("optim", "adamw_8bit", str),
            ("logging_steps", 1, int), ("num_generations", 4, int), ("max_prompt_length", 4000, int), ("max_completion_length", 1000, int),
            ("num_train_epochs", 1.0, float), ("max_grad_norm", 0.1, float), ("save_strategy", "no", str),
        ]:
            values[key] = cast(self.config.get(key, default))
        values["beta"], values["epsilon"] = _float(self.config.get("beta", 0.0)), _float(self.config.get("epsilon", 0.15))
        for key in "vllm_server_timeout vllm_gpu_memory_utilization temperature top_p epsilon_high".split():
            values[key] = _float(self.config.get(key))
        for key in "vllm_server_port vllm_group_port vllm_tensor_parallel_size vllm_max_model_length top_k generation_batch_size steps_per_generation".split():
            values[key] = _int(self.config.get(key))
        for key in "vllm_mode vllm_model_impl vllm_enable_sleep_mode vllm_server_host loss_type scale_rewards num_iterations importance_sampling_level mask_truncated_completions vllm_importance_sampling_correction".split():
            values[key] = self.config.get(key)
        return values

    def _format_prompt(self, prompt: str | Any) -> dict:
        messages = [{"role": "user", "content": prompt.strip()}] if isinstance(prompt, str) else prompt
        token_ids = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return {"prompt_token_ids": token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids or [])}

    def _normalize_adapter_path(self, path: str | None) -> str | None:
        path = normalize_optional_path(path)
        if path is None:
            return None
        if not os.path.isdir(path):
            raise ValueError(f"invalid adapter path: {path}")
        assert_adapter_compatible(lora_path=path, expected_lora_config=self._adapter_config)
        return os.path.abspath(path)

    def export_current_adapter(self, path: str) -> str:
        path = os.path.abspath(path)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._model.save_pretrained(path)
        logger.info("[EoHRLGRPO] saved adapter path=%s", path)
        return path

    def _sync_lora_to_inference(self):
        name = self._inference_lora_name
        if self._synced_lora_name == name and self._inference_lora_request is not None:
            return self._inference_lora_request
        started = time.time()
        if not hasattr(self._model, "load_lora"):
            raise RuntimeError("Unsloth fast inference model does not expose load_lora; vLLM LoRA sync is unavailable")
        request = self._model.load_lora(name, load_tensors=True)
        if request is None:
            raise RuntimeError("load_lora(load_tensors=True) did not return a LoRARequest")
        self._inference_lora_request = request
        self._synced_lora_name = name
        logger.info("[EoHRLGRPO] sync_lora_to_inference name=%s elapsed=%.3fs", name, time.time() - started)
        return request

    def _require_config(self) -> None:
        missing = [key for key in self._REQUIRED_CONFIG if self.config.get(key) is None]
        if missing:
            raise ValueError("Missing GRPO YAML fields: " + ", ".join(missing))

    def _precision(self) -> tuple[bool, bool]:
        if self.config.get("bf16") is not None or self.config.get("fp16") is not None:
            return bool(self.config.get("bf16")), bool(self.config.get("fp16"))
        try:
            from unsloth import is_bfloat16_supported

            bf16 = bool(is_bfloat16_supported())
        except Exception:
            bf16 = False
        return bf16, not bf16

    @staticmethod
    def _finite(summary: dict, metrics: dict) -> bool:
        loss = metrics.get("loss", metrics.get("train_loss"))
        return (loss is None or math.isfinite(float(loss))) and all(value is None or math.isfinite(float(value)) for value in (summary.get("reward_stats") or {}).values())

    @staticmethod
    def _visible_cuda_index(physical_id) -> int:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible:
            return int(physical_id)
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        target = str(int(physical_id))
        if target not in ids:
            raise ValueError(f"training_gpus contains GPU {physical_id}, but CUDA_VISIBLE_DEVICES={visible!r}")
        return ids.index(target)


class ResidentGRPOLLm(LLM):
    has_batch_draw = True
    thread_safe_batch_draw = False

    def __init__(self, *, policy: ResidentGRPOPolicy, **kwargs):
        super().__init__(**kwargs)
        self.model_manager = policy
        self.last_source = "local"

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        items = self.model_manager.draw_samples(prompt, 1, *args, **kwargs)
        return items[0] if items else ""

    def draw_samples(self, prompt: str | Any, n: int, *args, **kwargs) -> list[str]:
        return self.model_manager.draw_samples(prompt, n, *args, **kwargs)

    def close(self):
        self.model_manager.close()


__all__ = ["EoHReward", "ResidentGRPOPolicy", "ResidentGRPOLLm", "_check_randomness", "_check_score_metadata_leak", "build_reward_fn_from_task_rl", "event_funnel", "is_population_eligible_event"]
