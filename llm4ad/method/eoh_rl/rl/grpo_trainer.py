from __future__ import annotations

import concurrent.futures
import gc
import inspect
import json
import logging
import math
import os
import shutil
import time
from threading import RLock
from typing import Any

import torch

from ....base import LLM, SampleTrimmer, TextFunctionProgramConverter
from ..prompt import EoHPrompt
from ..sampler import EoHSampler
from .lora_gate import LoraGateDecision, SFTAnchorTrustRegionGate
from .reward import _check_randomness, _check_score_metadata_leak
from .sft_train import assert_lora_compatible, normalize_lora_config, normalize_optional_path

logger = logging.getLogger(__name__)
_PARSE_FAILURES = {"no_output", "missing_idea", "missing_function", "multiple_functions", "syntax_error", "parse_error", "wrong_function_name", "wrong_signature", "metadata_leak"}
_BQR_VALUES = "reward_label reward_quadrant baseline_used signed_delta relative_delta delta_parent delta_parent_rel delta_frontier delta_frontier_rel structure_novelty structural_threshold".split()
_BQR_BOOLS = "beats_parent ties_parent beats_frontier performance_improved q2_enabled structure_passes_q2".split()
_EVENT_VALUES = "prompt_id operator_type parent_best_score population_best_score group_size reward_contract input_tokens output_tokens score reward reward_label baseline_used reward_quadrant signed_delta relative_delta delta_parent delta_parent_rel delta_frontier delta_frontier_rel structure_novelty structural_threshold strategy python function deployment_reject_reason diag".split()
_EVENT_BOOLS = "validity beats_parent ties_parent beats_frontier performance_improved q2_enabled structure_passes_q2 exact_parent_copy random_algo score_metadata_leak registered_to_population blocked_from_population survived_main_population".split()


def _as_list(value) -> list:
    return [] if value is None else value if isinstance(value, list) else list(value) if isinstance(value, tuple) else [value]


def _float(value, default=None):
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=None):
    return default if value is None else int(value)


def _record(raw: dict) -> dict:
    prompt = str(raw.get("prompt") or raw.get("user_prompt") or "").strip()
    prompt_id = str(raw.get("prompt_id") or "").strip()
    op = str(raw.get("operator_type") or "").strip().lower()
    if not prompt or not prompt_id or not op:
        raise RuntimeError("online GRPO prompt record missing prompt/prompt_id/operator_type")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        system = str(raw.get("system_prompt") or "")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}] if system.strip() else EoHPrompt.create_instruct_prompt(prompt)
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "messages": messages,
        "operator_type": op,
        "parent_best_score": _float(raw.get("parent_best_score")),
        "parent_codes": _as_list(raw.get("parent_codes")),
        "parent_ids": None if raw.get("parent_ids") is None else _as_list(raw.get("parent_ids")),
        "population_best_score": _float(raw.get("population_best_score")),
        "group_size": int(raw.get("group_size") or 1),
        "reward_contract": str(raw.get("reward_contract") or "bqr_v1"),
        "input_tokens": int(raw.get("input_tokens") or 0),
    }


def _dataset(records: list[dict]):
    from datasets import Dataset

    return Dataset.from_dict({"prompt": [row["messages"] for row in records]})


def is_population_eligible_event(event: dict) -> bool:
    return (
        bool(event.get("exec_success"))
        and bool(event.get("validity", True))
        and not bool(event.get("blocked_from_population"))
        and not bool(event.get("random_algo"))
        and not bool(event.get("score_metadata_leak"))
        and not bool(event.get("exact_parent_copy"))
        and not event.get("failure_level")
        and not event.get("failure_label")
    )


def is_search_evidence_event(event: dict) -> bool:
    return (
        bool(event.get("exec_success"))
        and bool(event.get("validity", True))
        and not bool(event.get("random_algo"))
        and not bool(event.get("score_metadata_leak"))
        and not bool(event.get("exact_parent_copy"))
        and not event.get("failure_level")
        and not event.get("failure_label")
    )


def _basic_stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "mean": 0.0, "max": None, "std": 0.0}
    mean = sum(values) / len(values)
    return {"min": min(values), "mean": mean, "max": max(values), "std": (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5}


def operator_stats(events: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for event in events or []:
        grouped.setdefault(str(event.get("operator_type") or "unknown"), []).append(event)
    result = {}
    for op, rows in grouped.items():
        total = len(rows)
        valid = [row for row in rows if is_search_evidence_event(row)]
        pairs = [
            ("parse_success", "parse_success", rows), ("exec_success", "exec_success", rows),
            ("frontier_improve", "beats_frontier", valid), ("parent_improve", "beats_parent", valid),
            ("exact_parent_copy", "exact_parent_copy", rows), ("random_algo", "random_algo", rows),
            ("score_metadata_leak", "score_metadata_leak", rows), ("blocked_from_population", "blocked_from_population", rows),
            ("registered_to_population", "registered_to_population", rows), ("main_population_survival", "survived_main_population", rows),
        ]
        counts = {name: sum(bool(row.get(key)) for row in source) for name, key, source in pairs}
        result[op] = {"total": total, "parse_failure_count": total - counts["parse_success"]}
        for name, count in counts.items():
            result[op][f"{name}_count"] = count
            result[op][f"{name}_rate"] = count / total if total else 0.0
    return result


class DirectRewardCallback:
    """直接对 TRL completion 做解析、评估、奖励计算，并把有效候选交给 EoH 主循环入池。"""

    def __init__(self, *, records, evaluator, reward_fn, template_program, register_candidate, tokenizer=None, log_path=None, reward_eval_workers=1, enable_ast_gate=False):
        self.records = [_record(record) for record in records]
        self.evaluator = evaluator
        self.reward_fn = reward_fn
        self.template_program = template_program
        self.register_candidate = register_candidate
        self.log_path = log_path
        self.reward_eval_workers = max(1, int(reward_eval_workers or 1))
        self.__name__ = "eohrl_reward_callback"
        self._parser = EoHSampler(None, template_program, enable_ast_gate=bool(enable_ast_gate))
        self._tokenizer = tokenizer
        self._prompt_to_meta = {record["prompt"]: record for record in self.records}
        self.total = self.exec_failures = self.valid_count = self.positive_reward_count = 0
        self.reward_values: list[float] = []
        self.reward_events: list[dict] = []
        self._rows: list[dict] = []
        self._pending_candidates: list[dict] = []
        if not callable(register_candidate):
            raise ValueError("DirectRewardCallback requires register_candidate")
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        del kwargs
        started = time.time()
        rows, pending, rewards = [], [], []
        for idx, (prompt_obj, completion_obj) in enumerate(zip(prompts, completions)):
            row, item, reward = self._parse(self._resolve_meta(self._text(prompt_obj, last=True)), self._text(completion_obj))
            rows.append(row)
            rewards.append(reward)
            if item is not None:
                pending.append((idx, item))
        eval_started = time.time()
        for idx, result in self._evaluate_many(pending):
            rewards[idx] = self._finish(rows[idx], result)
        final = [float(value if value is not None else self.reward_fn.invalid()) for value in rewards]
        for row, reward in zip(rows, final):
            row["reward"] = reward
            self.reward_values.append(reward)
            self.positive_reward_count += int(reward > 0.0)
            self.reward_events.append(self._event(row))
        self._rows.extend(rows)
        logger.info("[Timing][EoHRLReward] prompts=%d evaluate=%.3fs total=%.3fs", len(rows), time.time() - eval_started, time.time() - started)
        return final

    def _parse(self, meta: dict, completion: str):
        self.total += 1
        row = {key: meta[key] for key in ("prompt_id", "operator_type", "parent_best_score", "population_best_score", "parent_ids", "group_size", "reward_contract", "input_tokens")}
        row.update(response=completion, output_tokens=self._count_text_tokens(completion))
        failure = "no_output" if not completion.strip() else None
        if failure is None:
            reason = self._parser._ast_gate_failure_reason(completion)
            failure = f"ast_gate:{reason}" if reason else None
        if failure:
            row.update(self._reject(failure))
            return row, None, row["reward"]
        parsed = self._parser.parse_response_record(completion, strict_contract=True)
        row.update(strategy=parsed.get("strategy"), python=parsed.get("python"))
        if parsed.get("failure_label"):
            row.update(self._reject(parsed["failure_label"]))
            return row, None, row["reward"]
        if not str(parsed.get("strategy") or "").strip():
            row.update(self._reject("missing_idea"))
            return row, None, row["reward"]
        return row, {"meta": meta, "func": parsed["func"], "program": parsed["program"], "strategy": parsed.get("strategy"), "python": parsed.get("python")}, None

    def _evaluate_many(self, pending: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
        if not pending:
            return []
        if self.reward_eval_workers <= 1 or len(pending) <= 1:
            return [(idx, {**item, **self._evaluate_one(item["program"])}) for idx, item in pending]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.reward_eval_workers, len(pending))) as pool:
            futures = {pool.submit(self._evaluate_one, item["program"]): (idx, item) for idx, item in pending}
            for future in concurrent.futures.as_completed(futures):
                idx, item = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = {"score": None, "eval_time": 0.0, "diag": {"error": str(exc)}}
                results.append((idx, {**item, **outcome}))
        return sorted(results, key=lambda item: item[0])

    def _evaluate_one(self, program) -> dict:
        started = time.time()
        try:
            if hasattr(self.evaluator, "evaluate_program_with_profile"):
                packet = self.evaluator.evaluate_program_with_profile(program)
                if isinstance(packet, dict) and packet.get("score") is not None:
                    diag = {"ok": True, "bucket": "ok", "error_type": None}
                    diag.update(packet)
                    return {"score": packet.get("score"), "eval_time": time.time() - started, "diag": diag}
            if hasattr(self.evaluator, "evaluate_program_record_time_with_diag"):
                score, eval_time, diag = self.evaluator.evaluate_program_record_time_with_diag(program)
            else:
                score, eval_time = self.evaluator.evaluate_program_record_time(program)
                diag = None
            return {"score": score, "eval_time": eval_time if eval_time is not None else time.time() - started, "diag": diag}
        except Exception as exc:
            return {"score": None, "eval_time": time.time() - started, "diag": {"error": str(exc)}}

    def _finish(self, row: dict, result: dict) -> float:
        meta, func, program_text, score = result["meta"], result["func"], str(result["program"]), result["score"]
        full_diag = result.get("diag")
        row.update(function=str(func), program=program_text, eval_time=result.get("eval_time"), diag=self._compact_diag(full_diag), strategy=result.get("strategy"), python=result.get("python"))
        if score is None or not math.isfinite(float(score)):
            self.exec_failures += 1
            row.update(self._reject("exec_error"), score=None, exec_success=False)
            return float(row["reward"])

        random_algo = bool(getattr(self.reward_fn, "detect_randomness", False) and _check_randomness(program_text))
        score_leak = bool(_check_score_metadata_leak(program_text))
        parent_copy = self._is_exact_parent_copy(str(func), meta["parent_codes"])
        failure = "exact_parent_copy" if parent_copy else "randomness_detected" if random_algo else "metadata_leak" if score_leak else None
        if failure:
            row.update(self._reject(failure), score=float(score), exec_success=True, exact_parent_copy=parent_copy, random_algo=random_algo, score_metadata_leak=score_leak)
            return float(row["reward"])

        bqr = self.reward_fn.compute_bqr_result(program_str=program_text, score=float(score), eval_time=result.get("eval_time"), parent_codes=meta["parent_codes"], parent_best_score=meta["parent_best_score"], population_best_score=meta["population_best_score"])
        reward = float(bqr["reward"])
        row.update(score=float(score), base_reward=reward, failure_label=bqr.get("failure_label"), failure_level=bqr.get("failure_label"), validity=bqr.get("failure_label") is None, exec_success=True, exact_parent_copy=False, random_algo=False, score_metadata_leak=False, registered_to_population=False, survived_main_population=False, blocked_from_population=bqr.get("failure_label") is not None)
        row.update({key: bqr.get(key) for key in _BQR_VALUES})
        row.update({key: bool(bqr.get(key, False)) for key in _BQR_BOOLS})
        if row["validity"]:
            self.valid_count += 1
            reg_meta = {key: meta[key] for key in ("operator_type", "parent_ids", "parent_codes", "parent_best_score", "population_best_score")}
            reg_meta.update(strategy=row.get("strategy") or "", reward=reward, validity=True, exec_success=True, failure_level=None, failure_label=None, exact_parent_copy=False, random_algo=False, score_metadata_leak=False, blocked_from_population=False, population_gate_reasons=[])
            self._pending_candidates.append({"row": row, "func": func, "program_str": program_text, "score": float(score), "eval_time": result.get("eval_time"), "diag": full_diag, "meta": reg_meta})
        return reward

    @staticmethod
    def _compact_diag(diag):
        if not isinstance(diag, dict):
            return diag
        compact = {key: value for key, value in diag.items() if key not in {"performance_profile", "profile_score"}}
        profile = diag.get("performance_profile")
        if isinstance(profile, (list, tuple)):
            compact["performance_profile_len"] = len(profile)
        return compact

    def commit_candidates(self) -> int:
        committed = 0
        for item in self._pending_candidates:
            if item.get("committed"):
                continue
            reg_meta = item["meta"]
            try:
                inserted = bool(self.register_candidate(func=item["func"], program_str=item["program_str"], score=item["score"], eval_time=item["eval_time"], diag=item["diag"], meta=reg_meta))
            except Exception as exc:
                logger.warning("register_candidate failed: %s", exc)
                inserted = False
                reg_meta.update(blocked_from_population=True, population_gate_reasons=["registration_exception"])
            if not inserted:
                reg_meta["blocked_from_population"] = True
                reg_meta["population_gate_reasons"] = list(dict.fromkeys(list(reg_meta.get("population_gate_reasons") or []) + ["not_selected_for_population"]))
            item["row"].update(registered_to_population=inserted, survived_main_population=inserted, blocked_from_population=bool(reg_meta.get("blocked_from_population")), population_gate_reasons=list(reg_meta.get("population_gate_reasons") or []))
            item["committed"] = True
            committed += int(inserted)
        self.reward_events = [self._event(row) for row in self._rows]
        self._write_rows()
        return committed

    def flush(self) -> None:
        self.reward_events = [self._event(row) for row in self._rows]
        self._write_rows()

    def summary(self) -> dict:
        total = max(1, self.total)
        events = list(self.reward_events)
        valid = [event for event in events if is_search_evidence_event(event)]
        scored = [float(event["score"]) for event in valid if event.get("score") is not None and math.isfinite(float(event["score"]))]
        count = lambda pred, source=events: sum(1 for event in source if pred(event))
        parse_failures, exec_success = count(lambda e: not bool(e.get("parse_success"))), count(lambda e: bool(e.get("exec_success")))
        registered_count = count(lambda e: bool(e.get("registered_to_population")))
        survived_count = count(lambda e: bool(e.get("survived_main_population")))
        token_usage = self._token_usage(events)
        return {
            "total": self.total,
            "parse_failures": parse_failures,
            "exec_failures": self.exec_failures,
            "exec_success": exec_success,
            "valid_count": self.valid_count,
            "positive_reward_count": self.positive_reward_count,
            "score_metadata_leak": count(lambda e: bool(e.get("score_metadata_leak"))),
            "metadata_leak": count(lambda e: e.get("failure_label") == "metadata_leak"),
            "exact_parent_copy": count(lambda e: bool(e.get("exact_parent_copy"))),
            "parse_success_rate": 1.0 - parse_failures / total,
            "exec_success_rate": exec_success / total,
            "valid_rate": self.valid_count / total,
            "reward_stats": _basic_stats(self.reward_values),
            "operator_stats": operator_stats(events),
            "frontier_improve_count": count(lambda e: bool(e.get("beats_frontier")), valid),
            "parent_improve_count": count(lambda e: bool(e.get("beats_parent")), valid),
            "registered_count": registered_count,
            "survived_count": survived_count,
            "best_completion_score": (min(scored) if getattr(self.reward_fn, "minimize", False) else max(scored)) if scored else None,
            "reward_events": events,
            "recent_reward_events": events,
            "candidate_count": sum(event.get("score") is not None for event in events),
            "token_usage": token_usage,
            **token_usage,
        }

    def _resolve_meta(self, prompt: str) -> dict:
        if prompt in self._prompt_to_meta:
            return self._prompt_to_meta[prompt]
        matches = [record for record in self.records if record["prompt"] and record["prompt"] in prompt]
        if len(matches) != 1:
            raise RuntimeError("online GRPO prompt metadata missing")
        return matches[0]

    def _reject(self, label: str) -> dict:
        reward = float(self.reward_fn.invalid_result(label)["reward"])
        return {"reward": reward, "base_reward": reward, "reward_label": "bqr_q4_invalid", "reward_quadrant": "q4", "failure_label": label, "failure_level": label, "validity": False, "exec_success": False, "registered_to_population": False, "blocked_from_population": True, "population_gate_reasons": [str(label)]}

    @staticmethod
    def _event(row: dict) -> dict:
        failure = row.get("failure_label") or row.get("failure_level")
        parent_ids = row.get("parent_ids")
        event = {key: row.get(key) for key in _EVENT_VALUES}
        event.update(parent_ids=parent_ids if parent_ids is None or isinstance(parent_ids, list) else [parent_ids], base_reward=row.get("base_reward", row.get("reward")), failure_label=failure, failure_level=row.get("failure_level"), parse_success=not (str(failure or "") in _PARSE_FAILURES or str(failure or "").startswith("ast_gate:")), exec_success=bool(row.get("exec_success", row.get("score") is not None)), population_gate_reasons=list(row.get("population_gate_reasons") or []))
        event.update({key: bool(row.get(key, False)) for key in _EVENT_BOOLS})
        return event

    def _write_rows(self) -> None:
        if self.log_path:
            with open(self.log_path, "w", encoding="utf-8") as file:
                for row in self._rows:
                    file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _count_text_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return 0
        try:
            return len(self._tokenizer.encode(str(text or ""), add_special_tokens=False))
        except Exception:
            return 0

    @staticmethod
    def _token_usage(events: list[dict]) -> dict:
        input_total = sum(int(event.get("input_tokens") or 0) for event in events)
        output_total = sum(int(event.get("output_tokens") or 0) for event in events)
        unique_inputs: dict[str, int] = {}
        for event in events:
            prompt_id = str(event.get("prompt_id") or "")
            if prompt_id and prompt_id not in unique_inputs:
                unique_inputs[prompt_id] = int(event.get("input_tokens") or 0)
        total = max(1, len(events))
        return {
            "input_tokens_total": int(input_total),
            "unique_input_tokens_total": int(sum(unique_inputs.values())),
            "output_tokens_total": int(output_total),
            "tokens_total": int(input_total + output_total),
            "input_tokens_mean": float(input_total) / total,
            "output_tokens_mean": float(output_total) / total,
        }

    @staticmethod
    def _text(obj: Any, *, last: bool = False) -> str:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[-1 if last else 0], dict):
            item = obj[-1 if last else 0]
            return str(item.get("content") or item.get("message") or item.get("text") or "")
        return str(obj.get("content") or obj.get("prompt") or obj.get("text") or "") if isinstance(obj, dict) else str(obj or "")

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


class ResidentGRPOPolicy:
    _REQUIRED_GRPO = ("max_seq_length", "seed", "gradient_checkpointing")
    _REQUIRED_VLLM = ("max_tokens", "gpu_memory_utilization", "temperature", "top_p")

    def __init__(self, *, model_name_or_path: str, grpo_config: dict, lora_config: dict, vllm_config: dict, lora_checkpoints_dir: str, training_gpus: list[int] | None = None, logger_name: str | None = None, current_lora_path: str | None = None, quantization: str | None = None):
        self.config = dict(grpo_config or {})
        self._model_name_or_path = model_name_or_path
        self._vllm_config = dict(vllm_config or {})
        self._lora_config = normalize_lora_config(lora_config)
        self._expected_lora_config = dict(self._lora_config)
        self._training_gpus = list(training_gpus or [])
        self._logger_name = logger_name or __name__
        self._quantization = quantization or self.config.get("quantization", "4bit")
        self._lora_checkpoints_dir = lora_checkpoints_dir
        self._vllm_lora_cache_dir = os.path.join(lora_checkpoints_dir, "vllm_lora_cache")
        initial = self._existing_lora_path(current_lora_path)
        self._paths = dict.fromkeys(("current", "latest", "deployed", "anchor", "best"), initial)
        self._lora_version = self._infer_lora_version(initial)
        self._model = self._tokenizer = self._fast_language_model_cls = None
        self._anchor_trainable_state = self._inference_lora_request = self._synced_lora_name = None
        self._inference_prepared = False
        self._draw_lock = RLock()
        self.lora_gate = SFTAnchorTrustRegionGate.from_grpo_config(self.config)
        self._require_config()
        os.makedirs(self._lora_checkpoints_dir, exist_ok=True)
        os.makedirs(self._vllm_lora_cache_dir, exist_ok=True)
        self._load_model()

    def draw_samples(self, prompt: str | Any, n: int, *args, **kwargs) -> list[str]:
        del args
        forbidden = {"max_tokens", "temperature", "top_p"}.intersection(kwargs or {})
        if forbidden:
            raise ValueError("采样参数只能来自 eoh_rl.vllm YAML: " + ", ".join(sorted(forbidden)))
        if n <= 0:
            return []
        from vllm import SamplingParams

        with self._draw_lock:
            started = time.time()
            sampling = {"n": min(int(n), 64), "max_tokens": int(self._vllm_config["max_tokens"]), "temperature": float(self._vllm_config["temperature"]), "top_p": float(self._vllm_config["top_p"])}
            if self._vllm_config.get("stop") is not None:
                sampling["stop"] = [str(item) for item in _as_list(self._vllm_config.get("stop")) if str(item)]
                if "include_stop_str_in_output" in self._vllm_config:
                    sampling["include_stop_str_in_output"] = bool(self._vllm_config["include_stop_str_in_output"])
            request = self.prepare_for_inference()
            kwargs_fast = {"sampling_params": SamplingParams(**sampling), "use_tqdm": False}
            if request is not None:
                kwargs_fast["lora_request"] = request
            with torch.inference_mode():
                outputs = self._model.fast_generate(self._format_prompt(prompt), **kwargs_fast)
            texts = [getattr(item, "text", "") for output in outputs or [] for item in getattr(output, "outputs", []) or []]
            logging.getLogger(self._logger_name).info("[Timing][ResidentGRPO] draw_samples n=%d total=%.3fs", len(texts), time.time() - started)
            return texts

    def train_once(self, *, prompt_records: list[dict], evaluator, reward_fn, template_program, output_dir: str | None = None, update_id: int = 0, reward_eval_workers: int = 1, register_candidate=None, enable_ast_gate: bool = False, recovery_active: bool = False, **_) -> dict:
        if not prompt_records:
            return {"executed": False, "reason": "no_online_prompts"}
        records = [_record(record) for record in prompt_records]
        for record in records:
            record["input_tokens"] = self._count_prompt_tokens(record["messages"])
        update_dir = os.path.join(output_dir or self.config.get("output_dir") or "./rl_training", "online_grpo", f"update_{int(update_id):03d}")
        os.makedirs(update_dir, exist_ok=True)
        dump = int(update_id) % max(1, int(self.config.get("prompt_dump_interval", 10) or 10)) == 0
        if dump:
            with open(os.path.join(update_dir, "prompts.json"), "w", encoding="utf-8") as file:
                json.dump(records, file, ensure_ascii=False, indent=2)
        previous = self._snapshot_policy() if self.lora_gate.enabled else None
        reward = DirectRewardCallback(records=records, evaluator=evaluator, reward_fn=reward_fn, template_program=template_program, register_candidate=register_candidate, tokenizer=self._tokenizer, log_path=os.path.join(update_dir, "completions.jsonl") if dump else None, reward_eval_workers=reward_eval_workers, enable_ast_gate=enable_ast_gate)
        started = time.time()
        try:
            metrics = self._train_with_trl(_dataset(records), reward, update_dir, update_id)
            summary = reward.summary()
            if not self._finite(summary, metrics):
                raise RuntimeError("GRPO 训练产生非有限 loss 或 reward 统计")
            commit_started = time.time()
            committed_count = reward.commit_candidates()
            summary = reward.summary()
            commit_elapsed = time.time() - commit_started
            gate_started = time.time()
            gate, lora_paths = self._deploy_lora(
                summary=summary,
                reward_fn=reward_fn,
                population_best_score=records[0].get("population_best_score"),
                previous=previous,
            )
            gate_elapsed = time.time() - gate_started
            timing = {"candidate_commit_elapsed": commit_elapsed, "lora_gate_decision_elapsed": gate_elapsed}
        except Exception as exc:
            if previous is not None:
                self._restore_policy(previous)
            reward.flush()
            summary = reward.summary()
            logger.warning("[EoHRLGRPO] update=%d failed: %s", int(update_id), exc)
            return {"executed": False, "reason": "trainer_exception", "samples_consumed": bool(summary.get("total")), "total": int(summary.get("total", 0) or 0), "recent_reward_events": summary.get("recent_reward_events", []), "metrics": {"reward_summary": summary, "accepted_lora": False, "reject_reason": "trainer_exception", "timing": {"total_elapsed": time.time() - started}}}
        total_elapsed = time.time() - started
        timing.update(total_elapsed=total_elapsed, full_backend_elapsed=total_elapsed, trainer_elapsed=float(metrics.get("trl_trainer_train_elapsed", 0.0) or 0.0), trl_init_trainer_elapsed=float(metrics.get("trl_init_trainer_elapsed", 0.0) or 0.0), trl_trainer_train_elapsed=float(metrics.get("trl_trainer_train_elapsed", 0.0) or 0.0), trl_cleanup_elapsed=float(metrics.get("trl_cleanup_elapsed", 0.0) or 0.0), trl_cuda_empty_cache_ran=bool(metrics.get("trl_cuda_empty_cache_ran", False)))
        metrics.update(executed=True, backend="resident_unsloth_trl", reward_summary=summary, timing=timing, accepted_lora=bool(gate.accepted), reject_reason=gate.reason, search_state_committed=bool(committed_count > 0), committed_candidate_count=int(committed_count), lora_gate_enabled=bool(self.lora_gate.enabled), lora_gate_score=float(gate.score), lora_gate_components=dict(gate.components), num_prompts=len(records), num_generations=int(self.config.get("num_generations", 4) or 4), recovery_active=bool(recovery_active), **lora_paths)
        return {"executed": True, "metrics": metrics, "reward_summary": summary, "recent_reward_events": summary.get("recent_reward_events", [])}

    def prepare_for_inference(self, *, force_reload: bool = False):
        if force_reload:
            self._invalidate_inference_lora(bump=True)
        if not self._inference_prepared and self._fast_language_model_cls is not None:
            self._fast_language_model_cls.for_inference(self._model)
            self._inference_prepared = True
        return self._sync_lora_to_inference()

    def close(self) -> None:
        try:
            engine = getattr(self._model, "vllm_engine", None) or getattr(getattr(self._model, "model", None), "vllm_engine", None)
            if engine is not None:
                if hasattr(engine, "sleep"):
                    engine.sleep(level=2)
                if hasattr(engine, "shutdown"):
                    engine.shutdown()
        except Exception as exc:
            logging.getLogger(self._logger_name).debug("关闭 resident vLLM engine 时异常: %s", exc)
        self._model = self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_current_lora_path(self): return self._paths["current"]
    def get_latest_lora_path(self): return self._paths["latest"]
    def get_deployed_lora_path(self): return self._paths["deployed"]
    def get_anchor_lora_path(self): return self._paths["anchor"]
    def get_best_lora_path(self): return self._paths["best"]
    def set_latest_lora_path(self, path): self._paths["latest"] = self._existing_lora_path(path)
    def set_deployed_lora_path(self, path): self._paths["deployed"] = self._existing_lora_path(path)
    def set_best_lora_path(self, path): self._paths["best"] = self._existing_lora_path(path)

    def set_current_lora_path(self, path: str | None):
        path = self._existing_lora_path(path)
        if path is not None and not self._same_path(self._paths["current"], path):
            self.close()
            self._paths["current"] = path
            self._lora_version = self._infer_lora_version(path)
            self._load_model()
        self._paths["current"] = path

    def set_anchor_lora_path(self, path: str | None):
        self._paths["anchor"] = self._existing_lora_path(path)
        self._anchor_trainable_state = self._clone_state(self._snapshot_trainable_state()) if self._same_path(self._paths["current"], self._paths["anchor"]) else None

    def _load_model(self) -> None:
        os.environ.setdefault("UNSLOTH_VLLM_NO_FLASHINFER", "1")
        os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
        if torch.cuda.is_available() and len(self._training_gpus) == 1:
            torch.cuda.set_device(self._visible_cuda_index(self._training_gpus[0]))
        from unsloth import FastLanguageModel, PatchFastRL

        self._patch_fast_rl_once(PatchFastRL, FastLanguageModel)
        self._fast_language_model_cls = FastLanguageModel
        adapter_path = self._paths["current"] if self._paths["current"] and os.path.isdir(self._paths["current"]) else None
        if adapter_path:
            assert_lora_compatible(lora_path=adapter_path, expected_lora_config=self._expected_lora_config)
        rank = int(self._lora_config.get("r", self.config.get("lora_rank", 32)))
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(model_name=self._model_name_or_path, max_seq_length=int(self.config["max_seq_length"]), load_in_4bit=str(self._quantization).lower() in {"4bit", "bitsandbytes"}, fast_inference=True, max_lora_rank=rank, gpu_memory_utilization=float(self.config.get("gpu_memory_utilization", self._vllm_config["gpu_memory_utilization"])))
        self._model = FastLanguageModel.get_peft_model(self._model, r=rank, target_modules=list(self._lora_config.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]), lora_alpha=int(self._lora_config.get("lora_alpha", self._lora_config.get("alpha", 2 * rank))), lora_dropout=float(self._lora_config.get("lora_dropout", self._lora_config.get("dropout", 0.0))), use_gradient_checkpointing="unsloth", random_state=int(self.config["seed"]))
        if adapter_path:
            self._load_adapter(adapter_path)
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self._anchor_trainable_state is None:
            self._anchor_trainable_state = self._clone_state(self._snapshot_trainable_state())
        self.prepare_for_inference()

    def _patch_fast_rl_once(self, patch_fast_rl, fast_language_model) -> None:
        try:
            patch_fast_rl("GRPO", fast_language_model)
        except OSError as exc:
            if "could not get source code" not in str(exc):
                raise
            logging.getLogger(self._logger_name).warning(
                "Unsloth PatchFastRL 跳过重复 OpenEnv patch: %s",
                exc,
            )

    def _train_with_trl(self, dataset, reward_func, output_dir: str, update_id: int) -> dict:
        from trl import GRPOConfig, GRPOTrainer

        if hasattr(self._model, "for_training"):
            self._model.for_training(use_gradient_checkpointing=bool(self.config["gradient_checkpointing"]))
        self._model.train()
        self._inference_prepared = False
        supported = set(inspect.signature(GRPOConfig).parameters)
        effective_args = {key: value for key, value in self._grpo_args(output_dir).items() if key in supported and value is not None}
        trainer_kwargs = {"model": self._model, "reward_funcs": reward_func, "args": GRPOConfig(**effective_args), "train_dataset": dataset}
        trainer_kwargs["processing_class" if "processing_class" in set(inspect.signature(GRPOTrainer).parameters) else "tokenizer"] = self._tokenizer
        cwd, metrics, started = os.getcwd(), {}, time.time()
        cleanup_interval = int(self.config.get("cuda_empty_cache_interval", 10) or 0)
        should_empty = cleanup_interval > 0 and (int(update_id or 0) <= 1 or int(update_id or 0) % cleanup_interval == 0)
        trainer = train_result = None
        try:
            init_started = time.time()
            trainer = GRPOTrainer(**trainer_kwargs)
            metrics["trl_init_trainer_elapsed"] = time.time() - init_started
            os.makedirs(os.path.join(output_dir, "vllm_lora_cache"), exist_ok=True)
            os.chdir(os.path.join(output_dir, "vllm_lora_cache"))
            train_started = time.time()
            train_result = trainer.train()
            metrics.update(dict(getattr(train_result, "metrics", {}) or {}))
            metrics.setdefault("loss", metrics.get("train_loss", 0.0))
            metrics.update(grpo_effective_args=effective_args, trl_trainer_train_elapsed=time.time() - train_started, trl_trainer_total_elapsed=time.time() - started)
            return metrics
        finally:
            os.chdir(cwd)
            try:
                self._model.zero_grad(set_to_none=True)
            except Exception:
                pass
            del train_result, trainer
            gc_started = time.time()
            gc.collect() if should_empty else gc.collect(0)
            if should_empty and torch.cuda.is_available():
                torch.cuda.empty_cache()
            metrics.update(trl_gc_elapsed=time.time() - gc_started, trl_cleanup_elapsed=time.time() - gc_started, trl_cuda_empty_cache_ran=bool(should_empty))

    def _deploy_lora(self, *, summary: dict, reward_fn, population_best_score, previous: dict) -> tuple[LoraGateDecision, dict]:
        previous = previous or {"paths": dict(self._paths), "lora_version": self._lora_version}
        old = dict(previous.get("paths") or {})
        latest, deployed, best = old.get("latest") or old.get("current"), old.get("deployed") or old.get("current"), old.get("best")
        saved = best_saved = projected = False
        candidate_state = self._snapshot_trainable_state() if self.lora_gate.enabled else None
        anchor_state = previous.get("anchor_trainable_state") or previous.get("trainable_state")
        gate = self.lora_gate.decide(summary=summary, minimize=bool(getattr(reward_fn, "minimize", False)), population_best_score=population_best_score, candidate_trainable_state=candidate_state, previous_trainable_state=previous.get("trainable_state"), anchor_trainable_state=anchor_state)
        if gate.accepted:
            projected_state = self.lora_gate.project_trainable_state(
                candidate_state=candidate_state,
                anchor_state=anchor_state,
                previous_state=previous.get("trainable_state"),
                projection_lambda=float(gate.components.get("projection_lambda", 1.0) or 1.0),
                anchor_radius=float(gate.components.get("anchor_radius", 0.0) or 0.0),
                step_radius=float(gate.components.get("step_radius", 0.0) or 0.0),
            ) if self.lora_gate.enabled else None
            if projected_state:
                self._restore_trainable_state(projected_state, lora_path=old.get("current"), version=previous.get("lora_version"))
                projected = True
            latest = self._save_lora(tag="latest")
            deployed = self._accept_lora(latest) or latest
            saved = True
            if self._best_improves(summary=summary, reward_fn=reward_fn, population_best_score=population_best_score) and str(self.config.get("lora_checkpoint_policy", "latest_and_best")).lower() == "latest_and_best":
                best = self._save_lora(tag="best")
                best_saved = True
        else:
            self._restore_policy(previous)
        return gate, {"previous_lora_path": old.get("deployed") or old.get("current"), "previous_latest_lora_path": old.get("latest") or old.get("current"), "lora_path": latest, "current_lora_path": self._paths["current"], "deployed_lora_path": deployed, "latest_lora_path": latest, "best_lora_path": best, "lora_saved": saved, "best_lora_saved": best_saved, "used_lora_projection": projected, "used_trainable_state_restore": bool((not gate.accepted) and previous.get("trainable_state")), "in_memory_lora_updated": bool(gate.accepted), "lora_version": int(self._lora_version)}

    def _grpo_args(self, output_dir: str) -> dict:
        bf16, fp16 = self._precision()
        prompts_per_update = max(1, int(self.config.get("prompts_per_update", 1) or 1))
        values = {"output_dir": output_dir, "bf16": bf16, "fp16": fp16, "per_device_train_batch_size": 1 if prompts_per_update == 1 else int(self.config.get("per_device_train_batch_size", 1)), "gradient_accumulation_steps": 1 if prompts_per_update == 1 else int(self.config.get("gradient_accumulation_steps", 1)), "remove_unused_columns": False, "report_to": self.config.get("report_to")}
        for key, default, cast in [
            ("use_vllm", True, bool), ("learning_rate", 1.0e-5, float), ("adam_beta1", 0.9, float), ("adam_beta2", 0.99, float), ("weight_decay", 0.1, float), ("warmup_ratio", 0.0, float),
            ("lr_scheduler_type", "constant", str), ("optim", "adamw_8bit", str), ("logging_steps", 1, int), ("num_generations", 4, int), ("max_prompt_length", 4000, int),
            ("max_completion_length", 1000, int), ("num_train_epochs", 1.0, float), ("max_grad_norm", 0.1, float), ("save_strategy", "no", str), ("save_steps", 0, int),
        ]:
            values[key] = cast(self.config.get(key, default))
        values["beta"], values["epsilon"] = _float(self.config.get("beta", 0.1)), _float(self.config.get("epsilon", 0.2))
        for key in "vllm_server_timeout vllm_gpu_memory_utilization temperature top_p epsilon_high".split():
            values[key] = _float(self.config.get(key))
        for key in "vllm_server_port vllm_group_port vllm_tensor_parallel_size vllm_max_model_length top_k generation_batch_size steps_per_generation".split():
            values[key] = _int(self.config.get(key))
        for key in "vllm_mode vllm_model_impl vllm_enable_sleep_mode vllm_server_host loss_type scale_rewards num_iterations mask_truncated_completions".split():
            values[key] = self.config.get(key)
        return values

    def _snapshot_policy(self) -> dict:
        return {"trainable_state": self._snapshot_trainable_state(), "anchor_trainable_state": self._clone_state(self._anchor_trainable_state), "paths": dict(self._paths), "lora_version": self._lora_version}

    def _restore_policy(self, snapshot: dict | None) -> None:
        snapshot = dict(snapshot or {})
        if snapshot.get("trainable_state"):
            self._restore_trainable_state(snapshot["trainable_state"], lora_path=(snapshot.get("paths") or {}).get("current"), version=snapshot.get("lora_version"))
        self._paths.update({key: self._existing_lora_path(path) for key, path in dict(snapshot.get("paths") or {}).items()})

    def _snapshot_trainable_state(self) -> dict[str, torch.Tensor]:
        if self._model is None or not hasattr(self._model, "named_parameters"):
            return {}
        return {name: param.detach().to(device="cpu", copy=True) for name, param in self._model.named_parameters() if getattr(param, "requires_grad", False)}

    def _restore_trainable_state(self, state: dict[str, torch.Tensor], *, lora_path: str | None = None, version: int | None = None) -> bool:
        params, restored = dict(self._model.named_parameters()), 0
        for name, tensor in (state or {}).items():
            param = params.get(name)
            if param is not None:
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                restored += 1
        current = self._existing_lora_path(lora_path)
        self._paths["current"] = self._paths["deployed"] = current
        self._lora_version = int(version) if version is not None else self._infer_lora_version(current)
        self._invalidate_inference_lora()
        if restored != len(state or {}):
            logging.getLogger(self._logger_name).warning("resident LoRA 权重恢复不完整: restored=%d expected=%d", restored, len(state or {}))
        return restored == len(state or {})

    def _save_lora(self, *, tag: str) -> str:
        self._lora_version += 1
        path = os.path.join(self._lora_checkpoints_dir, str(tag))
        self._remove_path(path)
        os.makedirs(path, exist_ok=True)
        self._model.save_pretrained(path)
        if hasattr(self._tokenizer, "save_pretrained"):
            self._tokenizer.save_pretrained(path)
        if str(tag).lower() in {"latest", "best"}:
            self._paths[str(tag).lower()] = path
        return path

    def _accept_lora(self, path: str | None = None) -> str | None:
        latest = self._assert_lora(path or self._paths["latest"])
        self._paths.update(current=latest, latest=latest, deployed=latest)
        self._invalidate_inference_lora()
        return latest

    def _sync_lora_to_inference(self):
        current = self._paths["current"]
        if current is None:
            self._inference_lora_request = self._synced_lora_name = None
            return None
        name = self._inference_lora_name()
        if self._synced_lora_name == name:
            return self._inference_lora_request
        self._stage_lora(name, current)
        self._inference_lora_request = self._model.load_lora(name, load_tensors=True)
        if self._inference_lora_request is None:
            raise RuntimeError("load_lora(load_tensors=True) 未返回有效 LoRARequest")
        self._synced_lora_name = name
        return self._inference_lora_request

    def _stage_lora(self, name: str, current: str) -> None:
        self._remove_path(name)
        try:
            os.symlink(current, name, target_is_directory=True)
        except OSError:
            shutil.copytree(current, name)

    def _load_adapter(self, lora_path: str) -> None:
        state = self._load_adapter_tensors(lora_path)
        params = dict(self._model.named_parameters()) if hasattr(self._model, "named_parameters") else {}
        missing, loaded = [], 0
        for name, tensor in state.items():
            names = [name]
            for marker in (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B."):
                if marker in name and f"{marker}default." not in name:
                    names.append(name.replace(marker, f"{marker}default.", 1))
            target = next((params.get(key) for key in dict.fromkeys(names) if params.get(key) is not None and tuple(params[key].shape) == tuple(tensor.shape)), None)
            if target is None:
                missing.append(name)
                continue
            target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded += 1
        if missing:
            raise RuntimeError(f"LoRA 权重加载不完整: loaded={loaded} missing={len(missing)} first_missing={missing[:8]}")

    @staticmethod
    def _load_adapter_tensors(lora_path: str) -> dict[str, torch.Tensor]:
        safetensors_path = os.path.join(lora_path, "adapter_model.safetensors")
        bin_path = os.path.join(lora_path, "adapter_model.bin")
        if os.path.isfile(safetensors_path):
            from safetensors.torch import load_file
            return load_file(safetensors_path, device="cpu")
        if os.path.isfile(bin_path):
            payload = torch.load(bin_path, map_location="cpu")
            payload = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            if isinstance(payload, dict):
                return {str(key): value for key, value in payload.items() if torch.is_tensor(value)}
        raise FileNotFoundError(f"LoRA checkpoint missing adapter weights: {lora_path}")

    def _format_prompt(self, prompt: str | Any) -> dict:
        messages = [{"role": "user", "content": prompt.strip()}] if isinstance(prompt, str) else prompt
        token_ids = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return {"prompt_token_ids": token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids or [])}

    def _count_prompt_tokens(self, messages: list[dict]) -> int:
        try:
            token_ids = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids or []))
        except Exception:
            try:
                return len(self._tokenizer.encode(str(messages or ""), add_special_tokens=False))
            except Exception:
                return 0

    def _inference_lora_name(self) -> str:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "_")
        return os.path.join(self._vllm_lora_cache_dir, f"eohrl_resident_lora_v{int(self._lora_version):04d}_{visible}")

    def _invalidate_inference_lora(self, *, bump: bool = False) -> None:
        self._lora_version += int(bool(bump))
        self._inference_lora_request = self._synced_lora_name = None
        self._inference_prepared = False

    def _existing_lora_path(self, path: str | None) -> str | None:
        path = normalize_optional_path(path)
        return None if path is None else self._assert_lora(path)

    def _assert_lora(self, path: str | None) -> str:
        path = normalize_optional_path(path)
        if path is None or not os.path.isdir(path):
            raise ValueError(f"invalid lora path: {path}")
        assert_lora_compatible(lora_path=path, expected_lora_config=self._expected_lora_config)
        return path

    def _require_config(self) -> None:
        missing = [key for key in self._REQUIRED_GRPO if self.config.get(key) is None]
        missing.extend(f"vllm.{key}" for key in self._REQUIRED_VLLM if self._vllm_config.get(key) is None)
        if missing:
            raise ValueError("常驻 Unsloth 缺少必填 YAML 参数: " + ", ".join(missing))

    def _infer_lora_version(self, path: str | None) -> int:
        versions = [0]
        if path:
            base = os.path.basename(os.path.normpath(path))
            if len(base) == 5 and base.startswith("v") and base[1:].isdigit():
                versions.append(int(base[1:]))
        if os.path.isdir(self._lora_checkpoints_dir):
            versions.extend(int(name[1:]) for name in os.listdir(self._lora_checkpoints_dir) if len(name) == 5 and name.startswith("v") and name[1:].isdigit())
        return max(versions)

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
    def _clone_state(state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
        return None if state is None else {name: tensor.detach().to(device="cpu", copy=True) for name, tensor in state.items()}

    @staticmethod
    def _remove_path(path: str) -> None:
        if os.path.lexists(path):
            os.unlink(path) if os.path.islink(path) or os.path.isfile(path) else shutil.rmtree(path)

    @staticmethod
    def _same_path(left: str | None, right: str | None) -> bool:
        return (not left and not right) or bool(left and right and os.path.realpath(left) == os.path.realpath(right))

    @staticmethod
    def _finite(summary: dict, metrics: dict) -> bool:
        loss = metrics.get("loss", metrics.get("train_loss"))
        return (loss is None or math.isfinite(float(loss))) and all(value is None or math.isfinite(float(value)) for value in (summary.get("reward_stats") or {}).values())

    @staticmethod
    def _best_improves(*, summary: dict, reward_fn, population_best_score: float | None) -> bool:
        score = summary.get("best_completion_score")
        if score is None:
            return False
        if population_best_score is None:
            return True
        return float(score) < float(population_best_score) if getattr(reward_fn, "minimize", False) else float(score) > float(population_best_score)

    @staticmethod
    def _visible_cuda_index(physical_id) -> int:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible:
            return int(physical_id)
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        target = str(int(physical_id))
        if target not in ids:
            raise ValueError(f"training_gpus 包含 GPU {physical_id}, 但 CUDA_VISIBLE_DEVICES={visible!r} 不可见")
        return ids.index(target)


class ResidentGRPOLLm(LLM):
    has_batch_draw = True
    thread_safe_batch_draw = False

    def __init__(self, *, policy: ResidentGRPOPolicy, **kwargs):
        super().__init__(**kwargs)
        self.model_manager = policy
        self.last_source = "local"

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        self.last_source = "local"
        items = self.model_manager.draw_samples(prompt, 1, *args, **kwargs)
        return items[0] if items else ""

    def draw_samples(self, prompt: str | Any, n: int, *args, **kwargs) -> list[str]:
        self.last_source = "local"
        return self.model_manager.draw_samples(prompt, n, *args, **kwargs)

    def close(self):
        self.model_manager.close()


__all__ = ["ResidentGRPOPolicy", "ResidentGRPOLLm", "is_population_eligible_event", "operator_stats"]
