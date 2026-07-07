from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import time
import traceback
from typing import Optional

from .population import Population
from .prompt import EoHPrompt
from .rl.grpo_trainer import _check_randomness, _check_score_metadata_leak, build_reward_fn_from_task_rl, is_population_eligible_event
from .sampler import EoHSampler
from ...base import Evaluation, Function, LLM, Program, SecureEvaluator, TextFunctionProgramConverter

logger = logging.getLogger(__name__)


_CHECKPOINT_META = {
    "_eoh_sample_order": "sample_order",
    "_eoh_parent_id": "parent_id",
    "_eoh_op": "op",
    "_eoh_source": "source",
    "_eoh_birth_sample_order": "birth_sample_order",
    "_eoh_birth_generation": "birth_generation",
}


def operator_stats(events: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for event in events or []:
        grouped.setdefault(str(event.get("operator_type") or "unknown"), []).append(event)
    stats = {}
    for op, rows in grouped.items():
        valid = [row for row in rows if is_population_eligible_event(row)]
        count = lambda key, source=rows: sum(bool(row.get(key)) for row in source)
        rewards = [float(row.get("reward") or 0.0) for row in rows]
        stats[op] = {
            "total": len(rows),
            "valid_count": len(valid),
            "valid_rate": len(valid) / len(rows) if rows else 0.0,
            "exec_success_count": count("exec_success"),
            "parent_improve_count": count("beats_parent", valid),
            "parent_improve_only_count": sum(bool(row.get("beats_parent")) and not bool(row.get("beats_frontier")) for row in valid),
            "frontier_improve_count": count("beats_frontier", valid),
            "valid_non_improving_count": sum(row.get("reward_quadrant") == "q3" for row in valid),
            "frac_reward_zero_std": float(len(rewards) > 1 and len(set(rewards)) == 1),
            "registered_to_population_count": count("registered_to_population"),
        }
    return stats


def _write_json_atomic(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _fmt_score(value) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return "None"


def _serialize_function(func: Function) -> dict:
    payload = {
        "algorithm": getattr(func, "algorithm", ""),
        "function": str(func),
        "score": getattr(func, "score", None),
        "evaluate_time": getattr(func, "evaluate_time", None),
        "sample_time": getattr(func, "sample_time", None),
        "operator": getattr(func, "operator", None),
    }
    for attr, key in _CHECKPOINT_META.items():
        value = getattr(func, attr, None)
        if value is not None:
            payload[key] = value
    profile = getattr(func, "_eoh_profile", None)
    if profile is not None:
        payload["performance_profile"] = list(profile)
    diag = getattr(func, "_eval_diag", None)
    if isinstance(diag, dict):
        payload["eval_diag"] = {k: v for k, v in diag.items() if k not in {"performance_profile", "profile_score"}}
    return payload


def _deserialize_functions(rows: list[dict]) -> list[Function]:
    funcs = []
    for row in rows or []:
        func = TextFunctionProgramConverter.text_to_function(row.get("function", ""))
        if func is None:
            continue
        for attr in ("algorithm", "score", "evaluate_time", "sample_time", "operator"):
            setattr(func, attr, row.get(attr))
        for attr, key in _CHECKPOINT_META.items():
            if key in row:
                setattr(func, attr, row[key])
        if isinstance(row.get("performance_profile"), list):
            setattr(func, "_eoh_profile", row["performance_profile"])
        funcs.append(func)
    return funcs


class EoH:
    """EoH search with GRPO embedded as the trainable sampler.

    EoH owns population, parents, operator order, and evaluation.
    GRPO owns only one update over one EoH prompt and returns candidate events.
    """

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler=None,
        max_generations: Optional[int] = 10,
        max_sample_nums: Optional[int] = 100,
        pop_size: Optional[int] = 5,
        use_e2_operator: bool = True,
        use_m1_operator: bool = True,
        use_m2_operator: bool = True,
        num_samplers: int = 1,
        num_evaluators: int = 1,
        *,
        resume_mode: bool = False,
        debug_mode: bool = False,
        multi_thread_or_process_eval: str = "thread",
        **kwargs,
    ):
        del multi_thread_or_process_eval
        self._llm = llm
        self._profiler = profiler
        self._debug_mode = bool(debug_mode)
        self._resume_mode = bool(resume_mode)
        self._num_evaluators = int(num_evaluators)
        self._num_samplers = max(1, int(num_samplers))

        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        self._max_grpo_updates = None if max_generations is None else int(max_generations)
        self._max_sample_nums = None if max_sample_nums is None else int(max_sample_nums)
        self._pop_size = 10 if pop_size is None else int(pop_size)
        self._operator_cycle = self._build_operator_cycle(use_e2_operator, use_m1_operator, use_m2_operator)

        cfg = dict(kwargs)
        rl_config = dict(cfg.pop("rl_config", {}) or {})
        self._grpo_config = dict(cfg.pop("grpo_config", rl_config.get("grpo", {})) or {})
        reward_fn = cfg.pop("reward_fn", None)
        reward_config = dict(cfg.pop("reward_config", rl_config.get("task_rl", {})) or {})
        self._reward_fn = reward_fn or build_reward_fn_from_task_rl(reward_config, logger=logger)
        self._samples_per_prompt = max(1, int(cfg.pop("samples_per_prompt", self._grpo_config.get("num_generations", 4)) or 4))
        configured_prompts_per_update = max(1, int(self._grpo_config.get("prompts_per_update", 1) or 1))
        self._prompts_per_update = max(configured_prompts_per_update, self._num_samplers)
        self._grpo_config["prompts_per_update"] = self._prompts_per_update
        self._reward_eval_workers = max(1, int(self._grpo_config.get("reward_eval_workers", self._num_evaluators) or 1))
        self._enable_ast_gate = bool(cfg.pop("enable_ast_gate", rl_config.get("enable_ast_gate", False)))
        self._checkpoint_dir = cfg.pop("checkpoint_dir", None) or os.path.join(getattr(profiler, "_log_dir", "") or ".", "checkpoints")
        self._checkpoint_auto_resume = bool(cfg.pop("checkpoint_auto_resume", rl_config.get("checkpoint_auto_resume", False)))
        cfg.pop("task_name", None)

        if self._samples_per_prompt != int(self._grpo_config.get("num_generations", self._samples_per_prompt) or self._samples_per_prompt):
            raise ValueError("samples_per_prompt must match grpo.num_generations for the minimal GRPO loop")
        if not getattr(llm, "has_batch_draw", False):
            raise ValueError("EoH-RL requires an LLM with batch draw support")

        llm.debug_mode = debug_mode
        self._population = Population(pop_size=self._pop_size, minimize=self._minimize())
        self._sampler = EoHSampler(llm, self._template_program_str, enable_ast_gate=self._enable_ast_gate)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **cfg)
        self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._num_evaluators)

        self._tot_sample_nums = 0
        self._grpo_update_count = 0
        # Keep sampling during initialization until the active population is full
        # or the overall sample budget is exhausted.
        self._initial_sample_nums_max = self._max_sample_nums if self._max_sample_nums is not None else 2 * self._pop_size
        self._reward_events: list[dict] = []
        self._best_curve: list[dict] = []
        self._latest_saved_lora_path: str | None = None
        self._final_saved_lora_path: str | None = None
        self._token_usage_totals = {
            "input_tokens_total": 0,
            "unique_input_tokens_total": 0,
            "output_tokens_total": 0,
            "tokens_total": 0,
            "completion_count": 0,
            "unique_prompt_count": 0,
        }
        self._timing_totals = {
            "initialization_wall_elapsed": 0.0,
            "grpo_update_wall_elapsed": 0.0,
            "prompt_build_wall_elapsed": 0.0,
            "train_call_wall_elapsed": 0.0,
            "candidate_registration_wall_elapsed": 0.0,
            "train_once_total_elapsed": 0.0,
            "trl_trainer_init_elapsed": 0.0,
            "trl_trainer_train_elapsed": 0.0,
            "trl_trainer_total_elapsed": 0.0,
            "reward_callback_wall_elapsed": 0.0,
            "reward_parse_wall_elapsed": 0.0,
            "reward_eval_wall_elapsed": 0.0,
            "reward_eval_program_time_total": 0.0,
            "run_wall_elapsed": 0.0,
        }

        os.makedirs(self._checkpoint_dir, exist_ok=True)
        if self._profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)
        if self._checkpoint_auto_resume:
            self._restore_checkpoint()

    @staticmethod
    def _build_operator_cycle(use_e2: bool, use_m1: bool, use_m2: bool) -> list[str]:
        ops = ["e1"]
        if use_e2:
            ops.append("e2")
        if use_m1:
            ops.append("m1")
        if use_m2:
            ops.append("m2")
        return ops

    def _minimize(self) -> bool:
        return bool(getattr(self._reward_fn, "minimize", False))

    def _utility(self, score) -> float:
        return -float(score) if self._minimize() else float(score)

    def _best_population_score(self):
        candidates = [
            func
            for func in self._population.population
            if getattr(func, "score", None) is not None and math.isfinite(float(func.score))
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda func: self._utility(func.score))
        return best.score

    def _continue_updates(self) -> bool:
        if self._max_grpo_updates is not None and self._grpo_update_count >= self._max_grpo_updates:
            return False
        return True

    def _operator_for_update(self, update_id: int, prompt_offset: int = 0) -> str:
        return self._operator_cycle[(int(update_id) - 1 + int(prompt_offset)) % len(self._operator_cycle)]

    def _select_parents(self, op: str) -> list[Function]:
        if op in {"e1", "e2"}:
            if self._population.archive_size < 2:
                raise RuntimeError(f"not enough parents for {op}: need=2, have={self._population.archive_size}")
            return self._population.crossover_selection()
        if len(self._population) < 1:
            raise RuntimeError(f"not enough active parents for {op}: need=1, have={len(self._population)}")
        return [self._population.selection()]

    def _parent_ids_for_samples(self, parents: list[Function]) -> list[int]:
        ids = []
        for parent in parents:
            raw = getattr(parent, "_eoh_sample_order", None)
            ids.append(int(raw) if raw is not None else hash(str(parent)))
        return ids

    def _build_prompt_record(self, update_id: int, *, prompt_offset: int = 0, record_index: int = 0) -> dict:
        op = self._operator_for_update(update_id, prompt_offset)
        parents = self._select_parents(op)
        if op == "e1":
            prompt = EoHPrompt.get_prompt_e1(self._task_description_str, parents, self._function_to_evolve)
        elif op == "e2":
            prompt = EoHPrompt.get_prompt_e2(self._task_description_str, parents, self._function_to_evolve)
        elif op == "m1":
            prompt = EoHPrompt.get_prompt_m1(self._task_description_str, parents[0], self._function_to_evolve)
        elif op == "m2":
            prompt = EoHPrompt.get_prompt_m2(self._task_description_str, parents[0], self._function_to_evolve)
        else:
            raise RuntimeError(f"unsupported EoH operator: {op}")
        parent_best = max((parent.score for parent in parents), key=self._utility)
        prompt_id = f"rl{update_id:03d}_{op}" if self._prompts_per_update == 1 and record_index == 0 else f"rl{update_id:03d}_{record_index:03d}_{op}"
        return EoHPrompt.build_prompt_record(
            prompt_id=prompt_id,
            prompt=prompt,
            operator_type=op,
            parent_best_score=parent_best,
            parent_codes=[str(parent) for parent in parents],
            parent_ids=self._parent_ids_for_samples(parents),
            population_best_score=self._best_population_score(),
            group_size=int(self._samples_per_prompt),
            reward_contract="bqr_v1",
            system_prompt=EoHPrompt.get_system_prompt(),
        )

    def _prompt_count_for_update(self, remaining_budget: int | None) -> int:
        if remaining_budget is None:
            return self._prompts_per_update
        return max(0, min(self._prompts_per_update, int(remaining_budget) // self._samples_per_prompt))

    def _build_prompt_records(self, update_id: int, *, remaining_budget: int | None = None) -> list[dict]:
        target = self._prompt_count_for_update(remaining_budget)
        if target <= 0:
            return []
        records: list[dict] = []
        used_prompts: set[str] = set()
        max_attempts = max(target * 4, target)
        attempt = 0
        while len(records) < target and attempt < max_attempts:
            record = self._build_prompt_record(update_id, prompt_offset=attempt, record_index=len(records))
            attempt += 1
            prompt_key = str(record.get("prompt") or "")
            if prompt_key in used_prompts:
                continue
            used_prompts.add(prompt_key)
            records.append(record)
        return records

    def _evaluate_programs_parallel(self, programs: list) -> list[tuple[Any, Any, Any]]:
        if self._num_evaluators <= 1 or len(programs) <= 1:
            return [self._evaluate_program_with_diag(program) for program in programs]

        results: list[tuple[Any, Any, Any]] = [(None, None, None)] * len(programs)
        future_to_idx = {self._evaluation_executor.submit(self._evaluate_program_with_diag, program): idx for idx, program in enumerate(programs)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = (None, 0.0, {"error": str(exc)})
        return results

    def _evaluate_program_with_diag(self, program):
        if hasattr(self._evaluator, "evaluate_program_with_profile"):
            started = time.time()
            packet = self._evaluator.evaluate_program_with_profile(program)
            if isinstance(packet, dict) and packet.get("score") is not None:
                diag = {"ok": True, "bucket": "ok", "error_type": None}
                diag.update(packet)
                return packet.get("score"), time.time() - started, diag
        if hasattr(self._evaluator, "evaluate_program_record_time_with_diag"):
            return self._evaluator.evaluate_program_record_time_with_diag(program)
        score, eval_time = self._evaluator.evaluate_program_record_time(program)
        return score, eval_time, None

    @staticmethod
    def _profile_from_diag(diag) -> list[float] | None:
        if not isinstance(diag, dict) or not isinstance(diag.get("performance_profile"), (list, tuple)):
            return None
        values = []
        for value in diag["performance_profile"]:
            try:
                parsed = float(value)
                if math.isfinite(parsed):
                    values.append(parsed)
            except Exception:
                pass
        return values or None

    @staticmethod
    def _clean_eval_diag(diag, *, score=None) -> dict:
        if isinstance(diag, dict):
            cleaned = {k: v for k, v in diag.items() if k not in {"performance_profile", "profile_score"}}
            if "ok" in cleaned or "bucket" in cleaned or "error_type" in cleaned:
                return {
                    "ok": bool(cleaned.get("ok", score is not None)),
                    "bucket": cleaned.get("bucket") or ("ok" if score is not None else "eval_or_timeout"),
                    "error_type": cleaned.get("error_type"),
                    **{k: v for k, v in cleaned.items() if k not in {"ok", "bucket", "error_type"}},
                }
        return {"ok": score is not None, "bucket": "ok" if score is not None else "eval_or_timeout", "error_type": None}

    def _register_candidate(self, item: dict, *, source: str = "grpo") -> bool:
        row, func, program_str = item["row"], item["func"], item["program_str"]
        meta, diag = item.get("meta") or {}, item.get("diag")
        score = item.get("score")
        finite_score = score is not None and math.isfinite(float(score))
        func.score = float(score) if finite_score else None
        func.evaluate_time = item.get("eval_time")
        func.algorithm = str(row.get("strategy") or "")
        func.sample_time = 0.0
        func.operator = str(meta.get("operator_type") or row.get("operator_type") or "")
        sample_order = int(item.get("sample_order") or row.get("sample_order") or self._tot_sample_nums)
        setattr(func, "_eoh_sample_order", sample_order)
        setattr(func, "_eoh_birth_sample_order", sample_order)
        setattr(func, "_eoh_birth_generation", int(self._population.generation))
        setattr(func, "_eoh_parent_id", meta.get("parent_ids") or row.get("parent_ids"))
        setattr(func, "_eoh_op", func.operator)
        setattr(func, "_eoh_source", "local")
        setattr(func, "_eval_diag", self._clean_eval_diag(diag, score=score))
        profile = self._profile_from_diag(diag)
        if profile:
            setattr(func, "_eoh_profile", profile)
        blocked_by_gate = (
            bool(row.get("exact_parent_copy"))
            or bool(row.get("random_algo"))
            or bool(row.get("score_metadata_leak"))
            or not bool(row.get("validity", True))
            or bool(row.get("failure_label"))
            or _check_score_metadata_leak(program_str)
            or (getattr(self._reward_fn, "detect_randomness", True) and _check_randomness(program_str))
        )
        if self._profiler is not None:
            self._profiler.register_function(func, program=program_str, source="local", record_best=finite_score and not blocked_by_gate)
        row["sample_order"] = sample_order
        if not finite_score:
            row["blocked_from_population"] = True
            return False
        if blocked_by_gate:
            row["blocked_from_population"] = True
            return False
        inserted, survived, _evicted = self._population.register_evolved_function(func)
        row["registered_to_population"] = bool(inserted)
        row["survived_main_population"] = bool(inserted and survived)
        row["blocked_from_population"] = not bool(inserted)
        if survived and self._profiler is not None:
            self._profiler.register_population(self._population)
        return bool(inserted)

    def _sample_initialize_population(self) -> None:
        init_started = time.time()
        prompt = EoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
        last_reported = -1
        try:
            while len(self._population) < self._pop_size and self._tot_sample_nums < self._initial_sample_nums_max:
                remaining = self._initial_sample_nums_max - self._tot_sample_nums
                batch_n = min(self._samples_per_prompt * self._num_samplers, remaining)
                records = self._sampler.get_strategy_and_function_records_batch(prompt, batch_n, strict_contract=True)
                eval_items = []
                programs = []
                for parsed in records:
                    self._tot_sample_nums += 1
                    func, program = parsed.get("func"), parsed.get("program")
                    if func is None or program is None:
                        continue
                    program_str = str(program)
                    if _check_score_metadata_leak(program_str) or (getattr(self._reward_fn, "detect_randomness", True) and _check_randomness(program_str)):
                        continue
                    eval_items.append((int(self._tot_sample_nums), parsed, func, program, program_str))
                    programs.append(program)

                for (sample_order, parsed, func, _program, program_str), (score, eval_time, diag) in zip(eval_items, self._evaluate_programs_parallel(programs)):
                    if score is None:
                        continue
                    func.score = float(score)
                    func.evaluate_time = eval_time
                    func.algorithm = str(parsed.get("strategy") or "")
                    func.sample_time = 0.0
                    func.operator = "i1"
                    setattr(func, "_eoh_sample_order", sample_order)
                    setattr(func, "_eoh_birth_sample_order", sample_order)
                    setattr(func, "_eoh_birth_generation", 0)
                    setattr(func, "_eoh_parent_id", None)
                    setattr(func, "_eoh_op", "i1")
                    setattr(func, "_eoh_source", "local")
                    setattr(func, "_eval_diag", self._clean_eval_diag(diag, score=score))
                    profile = self._profile_from_diag(diag)
                    if profile:
                        setattr(func, "_eoh_profile", profile)
                    if self._profiler is not None:
                        self._profiler.register_function(func, program=program_str, source="local")
                    self._population.add_initial_member(func)
                    if len(self._population) >= self._pop_size:
                        break
                if self._tot_sample_nums != last_reported:
                    print(
                        "[EoHRL:init] "
                        f"evaluated={self._tot_sample_nums}/{self._max_sample_nums or self._initial_sample_nums_max} "
                        f"pool={len(self._population)}/{self._pop_size} "
                        f"best={_fmt_score(self._best_population_score())}"
                    )
                    last_reported = self._tot_sample_nums
            if self._profiler is not None:
                self._profiler.register_population(self._population)
        finally:
            self._timing_totals["initialization_wall_elapsed"] += time.time() - init_started

    def _mark_events_after_registration(self, events: list[dict], candidates: list[dict]) -> None:
        by_index = {int((item.get("row") or {}).get("completion_index")): item.get("row") or {} for item in candidates if (item.get("row") or {}).get("completion_index") is not None}
        by_function = {str((item.get("row") or {}).get("function") or ""): item.get("row") or {} for item in candidates}
        active_codes = {str(func) for func in self._population.active_population}
        for event in events or []:
            row = by_index.get(int(event.get("completion_index") or 0)) or by_function.get(str(event.get("function") or ""))
            if row:
                event["sample_order"] = row.get("sample_order")
                event["registered_to_population"] = bool(row.get("registered_to_population"))
                event["blocked_from_population"] = bool(row.get("blocked_from_population"))
            event["survived_main_population"] = bool(event.get("registered_to_population") and str(event.get("function") or "") in active_codes)

    def _run_grpo_update(self) -> bool:
        update_started = time.time()
        if len(self._population) < self._pop_size:
            logger.info("[EoHRL] active population is not full; skipping GRPO")
            return False
        update_id = self._grpo_update_count + 1
        prompt_build_started = time.time()
        prompt_records = self._build_prompt_records(update_id, remaining_budget=None)
        prompt_build_elapsed = time.time() - prompt_build_started
        if not prompt_records:
            return False
        best_before = self._best_population_score()
        train_started = time.time()
        result = self._llm.model_manager.train_once(
            prompt_records=prompt_records,
            evaluator=self._evaluator,
            reward_fn=self._reward_fn,
            template_program=self._template_program,
            output_dir=self._default_log_dir(),
            update_id=update_id,
            reward_eval_workers=self._reward_eval_workers,
            enable_ast_gate=self._enable_ast_gate,
        )
        train_call_elapsed = time.time() - train_started
        summary = dict(result.get("reward_summary") or (result.get("metrics") or {}).get("reward_summary") or {})
        total = int(summary.get("total", 0) or result.get("total", 0) or 0)
        sample_order_base = self._tot_sample_nums
        self._tot_sample_nums += total
        registration_started = time.time()
        candidates = list(result.get("candidates") or [])
        for index, item in enumerate(candidates):
            row = item.get("row") or {}
            row["sample_order"] = sample_order_base + int(row.get("completion_index") or index + 1)
            item["sample_order"] = row["sample_order"]
        registered_count = sum(1 for item in candidates if self._register_candidate(item, source="local"))
        events = list(summary.get("reward_events") or result.get("recent_reward_events", []) or [])
        self._mark_events_after_registration(events, candidates)
        self._reward_events.extend(events)
        registration_elapsed = time.time() - registration_started
        token_usage = dict(summary.get("token_usage") or (result.get("metrics") or {}).get("token_usage") or {})
        self._accumulate_token_usage(token_usage)
        if not result.get("executed"):
            logger.warning("[EoHRL] GRPO update failed: %s", result.get("reason"))
            return False
        self._grpo_update_count = update_id
        best_after = self._best_population_score()
        metrics = dict(result.get("metrics") or {})
        timing = {
            **dict(metrics.get("timing") or {}),
            "grpo_update_wall_elapsed": time.time() - update_started,
            "prompt_build_wall_elapsed": prompt_build_elapsed,
            "train_call_wall_elapsed": train_call_elapsed,
            "candidate_registration_wall_elapsed": registration_elapsed,
        }
        self._accumulate_timing(timing)
        operator_types = [str(record.get("operator_type") or "") for record in prompt_records]
        operator_summary = operator_types[0] if len(set(operator_types)) == 1 else ",".join(operator_types)
        metrics.update(
            rl_update_id=update_id,
            operator_type=operator_summary,
            operator_types=operator_types,
            population_generation=self._population.generation,
            active_population_size=len(self._population),
            archive_size=self._population.archive_size,
            total_sample_nums=self._tot_sample_nums,
            best_before_rl=best_before,
            best_after_rl=best_after,
            registered_count=registered_count,
            token_usage=token_usage,
            token_usage_totals=dict(self._token_usage_totals),
            timing=timing,
            timing_totals=dict(self._timing_totals),
            operator_stats=operator_stats(events),
            reward_summary={**summary, "reward_events": events, "operator_stats": operator_stats(events), "token_usage": token_usage, "timing": dict(summary.get("timing") or {})},
        )
        if update_id % 100 == 0:
            saved = self._save_lora_artifact(f"lora_update_{update_id:03d}")
            if saved:
                metrics["saved_lora_path"] = saved
        self._best_curve.append({"rl_update_id": update_id, "population_generation": self._population.generation, "best": best_after})
        self._save_rl_update_metrics(update_id, metrics)
        self._save_checkpoint(tag=f"rl_{update_id:03d}")
        reward_stats = dict(summary.get("reward_stats") or {})
        print(
            "[EoHRL:grpo] "
            f"update={update_id} "
            f"gen={self._population.generation} "
            f"pop={self._population.archive_size} "
            f"active={len(self._population)}/{self._pop_size} "
            f"op={operator_summary} "
            f"rollouts={total} "
            f"valid={summary.get('valid_count', 0)} "
            f"parent_only={summary.get('parent_improve_only_count', 0)} "
            f"frontier={summary.get('frontier_improve_count', 0)} "
            f"non_improve={summary.get('valid_non_improving_count', 0)} "
            f"zero_std={float(summary.get('frac_reward_zero_std', 0.0)):.2f} "
            f"registered={registered_count} "
            f"tokens={token_usage.get('tokens_total', 0)} "
            f"in={token_usage.get('input_tokens_total', 0)} "
            f"out={token_usage.get('output_tokens_total', 0)} "
            f"time={timing.get('grpo_update_wall_elapsed', 0.0):.2f}s "
            f"reward_mean={_fmt_score(reward_stats.get('mean'))} "
            f"best={_fmt_score(best_before)}->{_fmt_score(best_after)}"
        )
        logger.info(
            "[EoHRL:grpo:metrics] update=%d/%s token_usage=%s token_totals=%s timing=%s timing_totals=%s",
            update_id,
            self._max_grpo_updates,
            token_usage,
            self._token_usage_totals,
            timing,
            self._timing_totals,
        )
        return True

    def _accumulate_token_usage(self, usage: dict | None) -> None:
        for key in self._token_usage_totals:
            try:
                self._token_usage_totals[key] += int((usage or {}).get(key, 0) or 0)
            except Exception:
                pass

    def _accumulate_timing(self, timing: dict | None) -> None:
        for key in self._timing_totals:
            if key == "run_wall_elapsed":
                continue
            try:
                self._timing_totals[key] += float((timing or {}).get(key, 0.0) or 0.0)
            except Exception:
                pass

    def _save_rl_update_metrics(self, update_id: int, metrics: dict) -> None:
        update_dir = os.path.join(self._default_log_dir(), "rl_training", f"rl_update_{int(update_id):03d}")
        os.makedirs(update_dir, exist_ok=True)
        _write_json_atomic(os.path.join(update_dir, "metrics.json"), metrics)

    def _default_log_dir(self) -> str:
        return getattr(self._profiler, "_log_dir", None) or "."

    def _checkpoint_path(self, tag: str = "latest") -> str:
        return os.path.join(self._checkpoint_dir, f"{tag}.json")

    def _save_checkpoint(self, *, tag: str = "latest") -> None:
        payload = {
            "population_generation": self._population.generation,
            "tot_sample_nums": self._tot_sample_nums,
            "rl_update_count": self._grpo_update_count,
            "population": [_serialize_function(func) for func in self._population.population],
            "best_curve": list(self._best_curve),
            "latest_saved_lora_path": self._latest_saved_lora_path,
            "final_saved_lora_path": self._final_saved_lora_path,
            "token_usage_totals": dict(self._token_usage_totals),
            "timing_totals": dict(self._timing_totals),
        }
        _write_json_atomic(self._checkpoint_path(tag), payload)
        if tag != "latest":
            _write_json_atomic(self._checkpoint_path("latest"), payload)

    def _restore_checkpoint(self) -> None:
        path = self._checkpoint_path("latest")
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as file:
            state = json.load(file)
        self._population = Population(
            pop_size=self._pop_size,
            generation=int(state.get("population_generation", 0) or 0),
            pop=_deserialize_functions(state.get("population", [])),
            minimize=self._minimize(),
        )
        self._tot_sample_nums = int(state.get("tot_sample_nums", 0) or 0)
        self._grpo_update_count = int(state.get("rl_update_count", 0) or 0)
        self._best_curve = list(state.get("best_curve") or [])
        self._latest_saved_lora_path = state.get("latest_saved_lora_path") or None
        self._final_saved_lora_path = state.get("final_saved_lora_path") or None
        self._token_usage_totals.update(dict(state.get("token_usage_totals") or {}))
        self._timing_totals.update(dict(state.get("timing_totals") or {}))

    def get_runtime_summary(self) -> dict:
        return {
            "rl_update_count": self._grpo_update_count,
            "population_generation": self._population.generation,
            "total_sample_nums": self._tot_sample_nums,
            "active_population_size": len(self._population),
            "archive_size": self._population.archive_size,
            "best_score": self._best_population_score(),
            "operator_cycle": list(self._operator_cycle),
            "samples_per_prompt": self._samples_per_prompt,
            "latest_saved_lora_path": self._latest_saved_lora_path,
            "final_saved_lora_path": self._final_saved_lora_path,
            "token_usage_totals": dict(self._token_usage_totals),
            "timing_totals": dict(self._timing_totals),
        }

    def _save_lora_artifact(self, name: str) -> str | None:
        manager = getattr(self._llm, "model_manager", None)
        if manager is None or not hasattr(manager, "export_current_adapter"):
            return None
        path = manager.export_current_adapter(os.path.join(self._checkpoint_dir, name))
        self._latest_saved_lora_path = path
        return path

    def _write_run_summary(self) -> None:
        path = os.path.join(self._default_log_dir(), "run_summary.json")
        _write_json_atomic(path, self.get_runtime_summary())

    def run(self):
        run_started = time.time()
        try:
            if len(self._population) < self._pop_size:
                self._sample_initialize_population()
            if len(self._population) < self._pop_size:
                print(
                    f"The search is terminated since EoH-RL only obtained {len(self._population)}/{self._pop_size} feasible algorithms during initialization."
                )
                self._finalize_run_timing(run_started)
                self._write_run_summary()
                return
            while self._continue_updates():
                try:
                    if not self._run_grpo_update():
                        break
                except KeyboardInterrupt:
                    break
                except Exception:
                    if self._debug_mode:
                        traceback.print_exc()
                        raise
                    logger.exception("[EoHRL] GRPO update failed; stopping to protect search state")
                    break
            self._finalize_run_timing(run_started)
            self._final_saved_lora_path = self._save_lora_artifact("lora_final")
            self._save_checkpoint(tag="latest_finish")
            self._write_run_summary()
            logger.info(
                "[EoHRL:summary] updates=%d/%s token_totals=%s timing_totals=%s best=%s",
                self._grpo_update_count,
                self._max_grpo_updates,
                self._token_usage_totals,
                self._timing_totals,
                self._best_population_score(),
            )
        finally:
            try:
                self._evaluation_executor.shutdown(cancel_futures=True)
            except Exception:
                pass
            try:
                self._llm.close()
            except Exception:
                pass

    def _finalize_run_timing(self, run_started: float) -> None:
        self._timing_totals["run_wall_elapsed"] = time.time() - run_started
