from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import random
import time
import traceback
from collections import deque
from typing import Optional

from .population import Population
from .prompt import EoHPrompt
from .rl.collapse import CollapseConfig, CollapseManager
from .rl.grpo_trainer import is_population_eligible_event, is_search_evidence_event, operator_stats
from .rl.operator_scheduler import AdaptiveOperatorConfig, AdaptiveOperatorScheduler
from .rl.reward import _check_randomness, _check_score_metadata_leak, build_reward_fn_from_task_rl
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
    "_eoh_lineage_tag": "lineage_tag",
}


def _checkpoint_dir(path: str | None, profiler=None) -> str:
    return path or os.path.join(getattr(profiler, "_log_dir", "") or ".", "checkpoints")


def _write_json_atomic(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


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
        try:
            payload["performance_profile"] = [float(value) for value in profile if value is not None]
        except TypeError:
            pass
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
            setattr(func, "_eoh_profile", [float(v) for v in row["performance_profile"] if v is not None])
        if isinstance(row.get("eval_diag"), dict):
            setattr(func, "_eval_diag", row["eval_diag"])
        funcs.append(func)
    return funcs


class EoH:
    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler=None,
        max_generations: Optional[int] = 10,
        max_sample_nums: Optional[int] = 100,
        pop_size: Optional[int] = 5,
        selection_num=2,
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
        self._llm = llm
        self._profiler = profiler
        self._debug_mode = debug_mode
        self._resume_mode = resume_mode
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        del multi_thread_or_process_eval

        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        del use_e2_operator, use_m1_operator, use_m2_operator
        evaluator_kwargs = dict(kwargs)
        self._task_name = evaluator_kwargs.pop("task_name", None)

        llm.debug_mode = debug_mode
        self._init_rl_component(evaluator_kwargs)
        self._pop_size = 10 if self._pop_size is None else int(self._pop_size)
        if self._pop_size != 10:
            raise ValueError("GRPO 模式要求 pop_size == 10")
        self._validate_adaptive_configs()

        self._population = Population(pop_size=self._pop_size, minimize=self._minimize())
        self._sampler = EoHSampler(llm, self._template_program_str, enable_ast_gate=self._enable_ast_gate)
        if not getattr(llm, "has_batch_draw", False):
            raise ValueError("EoH-RL 本地后训练入口要求 LLM 支持批量采样")
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **evaluator_kwargs)
        self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._num_evaluators)

        self._tot_sample_nums = 0
        configured_init_budget = 2 * int(self._pop_size)
        if self._max_sample_nums is not None:
            configured_init_budget = min(configured_init_budget, int(self._max_sample_nums))
        self._initial_sample_nums_max = max(0, configured_init_budget)
        self._bootstrap_done = False
        self._best_curve = []
        self._retained_core = []
        self._initial_candidates = []
        self._token_usage_totals = self._empty_token_usage()
        self._batch_sampler_logged = False
        if self._current_lora_path is None:
            self._current_lora_path = self._get_llm_lora_path("get_current_lora_path")
        if self._latest_lora_path is None:
            self._latest_lora_path = self._get_llm_lora_path("get_latest_lora_path") or self._current_lora_path
        if self._anchor_lora_path is None:
            self._anchor_lora_path = self._get_llm_lora_path("get_anchor_lora_path") or self._current_lora_path

        if self._profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

        if self._checkpoint_auto_resume:
            self._restore_checkpoint()

    def _init_rl_component(self, evaluator_kwargs: dict) -> None:
        """初始化 GRPO、奖励、checkpoint、自适应算子和坍缩控制组件。"""
        rl_config = dict(evaluator_kwargs.pop("rl_config", {}) or {})
        self._grpo_config = dict(evaluator_kwargs.pop("grpo_config", rl_config.get("grpo", {})) or {})
        self._lora_config = dict(evaluator_kwargs.pop("lora_config", rl_config.get("lora", {})) or {})
        reward_fn = evaluator_kwargs.pop("reward_fn", rl_config.get("reward_fn"))
        reward_config = dict(evaluator_kwargs.pop("reward_config", rl_config.get("reward_config", rl_config.get("task_rl", {}))) or {})
        self._grpo_enabled = bool(evaluator_kwargs.pop("enable_grpo", rl_config.get("enabled", False) or bool(self._grpo_config)))
        if not self._grpo_enabled:
            raise ValueError("EoH-RL 是在线 GRPO 专用入口；普通 EoH 请使用 llm4ad.method.eoh")
        self._reward_fn = reward_fn or (build_reward_fn_from_task_rl(reward_config, logger=logger) if self._grpo_enabled else None)

        self._prompts_per_update = max(1, int(self._grpo_config.get("prompts_per_update", 1) or 1))
        self._grpo_group_size = max(1, int(self._grpo_config.get("num_generations", 4) or 4))
        self._samples_per_prompt = max(1, int(evaluator_kwargs.pop("samples_per_prompt", rl_config.get("samples_per_prompt", self._grpo_group_size)) or self._grpo_group_size))
        self._reward_eval_workers = max(1, int(self._grpo_config.get("reward_eval_workers", self._num_evaluators) or 1))
        self._grpo_update_count = 0
        self._current_lora_path = self._grpo_config.get("current_lora_path")
        self._latest_lora_path = self._grpo_config.get("latest_lora_path", self._current_lora_path)
        self._anchor_lora_path = self._current_lora_path
        self._latest_training_timing = {}
        self._reward_events = deque(maxlen=max(1, int(self._grpo_config.get("recent_reward_event_window_size", self._grpo_config.get("event_ledger_max_recent", 256)) or 256)))

        self._checkpoint_auto_resume = bool(evaluator_kwargs.pop("checkpoint_auto_resume", rl_config.get("checkpoint_auto_resume", False)))
        checkpoint_dir = evaluator_kwargs.pop("checkpoint_dir", rl_config.get("checkpoint_dir"))
        adaptive_operator_dict = evaluator_kwargs.pop("adaptive_operator_config", rl_config.get("adaptive_operator"))
        collapse_dict = evaluator_kwargs.pop("collapse_config", rl_config.get("collapse"))
        self._enable_ast_gate = bool(evaluator_kwargs.pop("enable_ast_gate", rl_config.get("enable_ast_gate", False)))
        if adaptive_operator_dict is None:
            raise ValueError("EoH-RL requires adaptive_operator_config or rl_config.adaptive_operator")
        if collapse_dict is None:
            raise ValueError("EoH-RL requires collapse_config or rl_config.collapse")

        self._checkpoint_dir = _checkpoint_dir(checkpoint_dir, profiler=self._profiler)
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        self._adaptive_operator_config = AdaptiveOperatorConfig.from_dict(adaptive_operator_dict)
        self._operator_scheduler = AdaptiveOperatorScheduler(self._adaptive_operator_config)
        self._collapse_config = CollapseConfig.from_dict(collapse_dict)
        self._collapse_manager = CollapseManager(self._collapse_config)

    def _validate_adaptive_configs(self) -> None:
        """校验自适应算子与坍缩机制的运行约束。"""
        if int(self._grpo_config.get("trigger_every_n_generations", 1) or 1) != 1:
            raise ValueError("integrated online GRPO currently requires grpo.trigger_every_n_generations == 1")
        operators = [str(op).strip().lower() for op in self._grpo_config.get("operator_cycle", ["e1", "e2", "m1", "m2"])]
        if set(operators) != {"e1", "e2", "m1", "m2"}:
            raise ValueError("GRPO 训练阶段 operator_cycle 必须且只能包含 e1/e2/m1/m2")
        if int(self._selection_num or 0) < 2:
            raise ValueError("GRPO 训练阶段需要 selection_num >= 2 以支持 e1/e2 双父代算子")

    def _minimize(self) -> bool:
        return bool(getattr(self._reward_fn, "minimize", False))

    def _is_initialization_candidate(self, operator_type: str) -> bool:
        return str(operator_type or "").strip().lower() == "i1" and not self._bootstrap_done

    @staticmethod
    def _same_algorithm(left: Function, right: Function) -> bool:
        return str(left).strip() == str(right).strip()

    @staticmethod
    def _has_algorithm_description(func: Function) -> bool:
        return bool(str(getattr(func, "algorithm", "") or "").strip())

    def _unique_functions(self, funcs, *, require_score: bool = False) -> list[Function]:
        unique = []
        for func in funcs or []:
            if func is None:
                continue
            if require_score:
                try:
                    if getattr(func, "score", None) is None or not math.isfinite(float(func.score)):
                        continue
                except Exception:
                    continue
            if not any(self._same_algorithm(existing, func) for existing in unique):
                unique.append(func)
        return unique

    def _best_population_score(self):
        scores = [float(func.score) for func in self._population.population if getattr(func, "score", None) is not None and math.isfinite(float(func.score))]
        return (min(scores) if self._minimize() else max(scores)) if scores else None

    def _ordered_unique(self, funcs: list[Function] | None = None) -> list[Function]:
        funcs = self._population.population if funcs is None else funcs
        valid = []
        for func in funcs or []:
            try:
                if getattr(func, "score", None) is not None and math.isfinite(float(func.score)):
                    valid.append(func)
            except Exception:
                continue
        return self._unique_functions(sorted(valid, key=lambda f: float(f.score), reverse=not self._minimize()))

    def _utility(self, func: Function) -> float:
        score = float(func.score)
        return -score if self._minimize() else score

    @staticmethod
    def _median(values) -> float:
        items = sorted(float(value) for value in values if value is not None)
        if not items:
            return 0.0
        mid = len(items) // 2
        return float(items[mid]) if len(items) % 2 else 0.5 * (items[mid - 1] + items[mid])

    def _select_score_diverse(self, funcs: list[Function], *, count: int) -> list[Function]:
        ordered = self._ordered_unique(funcs)
        count = max(0, int(count))
        if count <= 0 or len(ordered) <= count:
            return ordered[:count]
        selected = ordered[: max(1, count // 2)]
        remaining = [func for func in ordered if not any(self._same_algorithm(func, kept) for kept in selected)]
        while remaining and len(selected) < count:
            candidate = max(
                remaining,
                key=lambda func: (
                    min((self._algorithm_diversity(func, kept) for kept in selected), default=0.0),
                    self._utility(func),
                ),
            )
            selected.append(candidate)
            remaining = [func for func in remaining if not self._same_algorithm(func, candidate)]
        return selected[:count]

    def _online_parent_candidates(self) -> list[Function]:
        return self._ordered_unique(list(self._population.population))[: max(1, int(self._pop_size or 1))]

    def _crossover_parent_candidates(self) -> list[Function]:
        return self._ordered_unique(list(self._population.population))

    def _rank_weights(self, ordered: list[Function]) -> list[float]:
        return [1.0 / rank for rank, _ in enumerate(self._unique_functions(ordered), start=1)]

    def _rank_sample_unique(self, ordered: list[Function], *, count: int) -> list[Function]:
        unique = self._unique_functions(ordered)
        count = max(0, int(count))
        if count <= 0 or len(unique) <= count:
            return unique[:count]
        selected, weights = [], self._rank_weights(unique)
        for _ in range(max(20, len(unique) * 4)):
            if len(selected) >= count:
                break
            candidate = random.choices(unique, weights=weights, k=1)[0]
            if not any(self._same_algorithm(existing, candidate) for existing in selected):
                selected.append(candidate)
        for candidate in unique:
            if len(selected) >= count:
                break
            if not any(self._same_algorithm(existing, candidate) for existing in selected):
                selected.append(candidate)
        return selected

    @staticmethod
    def _anchor_baseline_score(indivs: list[Function], cfg: dict, *, minimize: bool = False) -> Optional[float]:
        if str(cfg.get("anchor_baseline", "best")).strip().lower() != "best":
            raise ValueError("eoh_rl.grpo.anchor_baseline 当前主路径必须为 best")
        scores = [float(indiv.score) for indiv in indivs if getattr(indiv, "score", None) is not None and math.isfinite(float(indiv.score))]
        return (min(scores) if minimize else max(scores)) if scores else None

    @staticmethod
    def _parent_ids_for_samples(parents) -> Optional[list[int]]:
        ids = []
        for parent in parents or []:
            try:
                sample_id = int(getattr(parent, "_eoh_sample_order")) - 1
                if sample_id >= 0:
                    ids.append(sample_id)
            except Exception:
                continue
        return ids or None

    def _select_online_prompt_parents(self, *, count: int, slot: int, cfg: dict) -> list[Function]:
        del slot, cfg
        if self._collapse_manager.recovery_active and int(count) == 1:
            parent = self._select_recovery_mutation_parent()
            if parent is not None:
                return [parent]
        parents = self._rank_sample_unique(self._online_parent_candidates(), count=count)
        if not parents:
            raise RuntimeError("online GRPO requires a non-empty parent pool")
        return parents

    def _select_online_crossover_parents(self, *, count: int, slot: int, cfg: dict) -> list[Function]:
        del slot, cfg
        if self._collapse_manager.recovery_active:
            parents = self._select_recovery_crossover_parents(count=max(2, int(count)))
            if len(parents) >= min(max(2, int(count)), len(self._online_parent_candidates())):
                return parents[: int(count)]
        main_pool = self._online_parent_candidates()
        if not main_pool:
            raise RuntimeError("online GRPO crossover requires a non-empty parent pool")
        if int(count) <= 1:
            return self._rank_sample_unique(main_pool, count=count)
        p1 = random.choices(main_pool, weights=self._rank_weights(main_pool), k=1)[0]
        pool = self._crossover_parent_candidates()
        median_u = self._median(self._utility(func) for func in pool)
        quality_pool = [func for func in pool if self._utility(func) >= median_u and not self._same_algorithm(func, p1)]
        p2 = self._select_diverse_parent(p1, quality_pool or pool)
        if p2 is None:
            rest = [func for func in main_pool if not self._same_algorithm(func, p1)]
            if not rest:
                return []
            p2 = random.choices(rest, weights=self._rank_weights(rest), k=1)[0]
        parents = [p1, p2]
        parents += self._rank_sample_unique([func for func in pool if all(not self._same_algorithm(func, p) for p in parents)], count=int(count) - 2)
        return parents[: int(count)]

    def _recovery_anchor_parent(self) -> Function | None:
        core = self._ordered_unique(list(self._retained_core) + list(self._population.population))
        return core[0] if core else None

    def _select_recovery_mutation_parent(self) -> Function | None:
        anchor = self._recovery_anchor_parent()
        pool = self._online_parent_candidates()
        if anchor is not None:
            distant = self._select_diverse_parent(anchor, [func for func in pool if not self._same_algorithm(func, anchor)])
            if distant is not None:
                return distant
        return pool[0] if pool else None

    def _select_recovery_crossover_parents(self, *, count: int) -> list[Function]:
        anchor = self._recovery_anchor_parent()
        if anchor is None:
            return []
        pool = self._online_parent_candidates()
        partner = self._select_diverse_parent(anchor, [func for func in pool if not self._same_algorithm(func, anchor)])
        parents = [anchor] + ([partner] if partner is not None else [])
        parents += self._rank_sample_unique(
            [func for func in pool if all(not self._same_algorithm(func, parent) for parent in parents)],
            count=max(0, int(count) - len(parents)),
        )
        return parents[: int(count)]

    def _select_diverse_parent(self, parent: Function, candidates: list[Function]) -> Optional[Function]:
        scored = [
            (self._algorithm_diversity(parent, candidate), self._utility(candidate), candidate)
            for candidate in candidates
            if candidate is not None and not self._same_algorithm(parent, candidate)
        ]
        scored = [(dist, utility, candidate) for dist, utility, candidate in scored if dist > 0.0]
        if not scored:
            return None
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    def _structure_diversity(self, left_code: str, right_code: str) -> float:
        try:
            return float(self._reward_fn.symmetric_structure_diversity(left_code, right_code))
        except Exception:
            return 0.0

    def _algorithm_diversity(self, left: Function, right: Function) -> float:
        return max(
            self._structure_diversity(str(left), str(right)),
            self._behavior_diversity(left, right),
        )

    @staticmethod
    def _behavior_diversity(left: Function, right: Function) -> float:
        left_profile = getattr(left, "_eoh_profile", None)
        right_profile = getattr(right, "_eoh_profile", None)
        if not left_profile or not right_profile:
            return 0.0
        n = min(len(left_profile), len(right_profile))
        if n < 2:
            return 0.0
        xs = [float(v) for v in left_profile[:n]]
        ys = [float(v) for v in right_profile[:n]]
        mx, my = sum(xs) / n, sum(ys) / n
        vx, vy = [x - mx for x in xs], [y - my for y in ys]
        nx = math.sqrt(sum(v * v for v in vx))
        ny = math.sqrt(sum(v * v for v in vy))
        if nx <= 1.0e-12 or ny <= 1.0e-12:
            return 0.0
        corr = max(-1.0, min(1.0, sum(x * y for x, y in zip(vx, vy)) / (nx * ny)))
        return 0.5 * (1.0 - corr)

    def _expected_diverse_parent_diversity(self, candidates: list[Function]) -> float:
        pool = self._ordered_unique(candidates)
        if len(pool) < 2:
            return 0.0
        values = [self._algorithm_diversity(left, right) for idx, left in enumerate(pool) for right in pool[idx + 1:]]
        return max(0.0, min(1.0, sum(values) / len(values))) if values else 0.0

    def _population_diversity(self) -> float:
        return self._expected_diverse_parent_diversity(list(self._population.population))

    def _build_online_prompt_records(self, rl_update_id: int) -> list[dict]:
        if len(self._population) == 0:
            return []
        target_n = int(self._prompts_per_update)
        operators = list(self._grpo_config.get("operator_cycle", ["e1", "e2", "m1", "m2"]))
        population_best = self._best_population_score()
        context = self._adaptive_operator_context(cycle_index=rl_update_id - 1, population_best_score=population_best)
        if self._collapse_manager.recovery_active:
            sampled_ops = self._sample_recovery_ops(k=target_n, operator_cycle=operators, context=context)
        elif self._operator_scheduler.enabled:
            sampled_ops = self._operator_scheduler.select_ops(k=target_n, operator_cycle=operators, context=context)
        else:
            sampled_ops = [operators[(rl_update_id - 1 + idx) % len(operators)] for idx in range(target_n)]
        if self._collapse_manager.recovery_active and self._needs_recovery_i1(context):
            sampled_ops = (["i1"] + [op for op in sampled_ops if str(op).strip().lower() != "i1"])[:target_n]
        if target_n >= 2 and not self._collapse_manager.recovery_active:
            sampled_ops = self._ensure_mutation_prompt(sampled_ops=sampled_ops)

        records, used_prompt_texts = [], set()
        unique_parent_count = len(self._crossover_parent_candidates())
        required_crossover_parents = max(2, int(self._selection_num))

        def mutation_prompt(op_name: str, index: int):
            parent = self._select_online_prompt_parents(count=1, slot=index, cfg=self._grpo_config)[0]
            prompt_fn = EoHPrompt.get_prompt_m2 if op_name == "m2" else EoHPrompt.get_prompt_m1
            return prompt_fn(self._task_description_str, parent, self._function_to_evolve), parent.score, [str(parent)], self._parent_ids_for_samples([parent])

        for attempt in range(max(target_n * 4, len(sampled_ops), 1)):
            if len(records) >= target_n:
                break
            op = str(sampled_ops[attempt] if attempt < len(sampled_ops) else operators[(rl_update_id + attempt) % len(operators)]).strip().lower()
            parent_best, parent_codes, parent_ids = population_best, [], None
            if op in ("e1", "e2") and unique_parent_count < required_crossover_parents:
                op = "m2" if op == "e2" else "m1"
            try:
                if op == "i1":
                    prompt = EoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
                elif op in ("m1", "m2"):
                    prompt, parent_best, parent_codes, parent_ids = mutation_prompt(op, len(records))
                elif op in ("e1", "e2"):
                    parents = self._select_online_crossover_parents(count=self._selection_num, slot=len(records), cfg=self._grpo_config)
                    if len(parents) < required_crossover_parents:
                        prompt, parent_best, parent_codes, parent_ids = mutation_prompt("m2" if op == "e2" else "m1", len(records))
                        op = "m2" if op == "e2" else "m1"
                    else:
                        prompt_fn = EoHPrompt.get_prompt_e1 if op == "e1" else EoHPrompt.get_prompt_e2
                        prompt = prompt_fn(self._task_description_str, parents, self._function_to_evolve)
                        parent_best = self._anchor_baseline_score(parents, self._grpo_config, minimize=self._minimize())
                        parent_codes = [str(parent) for parent in parents]
                        parent_ids = self._parent_ids_for_samples(parents)
                else:
                    continue
            except RuntimeError:
                continue
            if self._collapse_manager.recovery_active:
                prompt = EoHPrompt.with_recovery_context(prompt)
            if prompt in used_prompt_texts:
                continue
            used_prompt_texts.add(prompt)
            records.append(
                EoHPrompt.build_prompt_record(
                    prompt_id=f"rl{rl_update_id:03d}_{len(records):03d}_{op}",
                    prompt=prompt,
                    operator_type=op,
                    parent_best_score=parent_best,
                    parent_codes=parent_codes,
                    parent_ids=parent_ids,
                    population_best_score=population_best,
                    group_size=int(self._grpo_group_size),
                    reward_contract=str(getattr(self._reward_fn, "reward_type", "bqr_v1")),
                    system_prompt=EoHPrompt.get_system_prompt(),
                )
            )
        return records

    @staticmethod
    def _ensure_mutation_prompt(*, sampled_ops: list[str]) -> list[str]:
        ops = [str(op).strip().lower() for op in sampled_ops or []]
        if any(op in {"m1", "m2"} for op in ops):
            return ops
        return (ops[:-1] + ["m2"]) if ops else ["m2"]

    def _adaptive_operator_context(self, *, cycle_index: int, population_best_score: Optional[float] = None) -> dict:
        target = max(1, int(self._pop_size or 1))
        diversity = self._population_diversity()
        collapse_state = self._collapse_manager.get_state()
        return {
            "cycle_index": int(cycle_index),
            "population_size": len(self._population),
            "target_population_size": target,
            "retained_core_size": max(1, len(self._retained_core)) if self._retained_core else 1,
            "population_best_score": population_best_score,
            "unique_parent_count": len(self._crossover_parent_candidates()),
            "recovery_active": bool(self._collapse_manager.recovery_active),
            "population_diversity": diversity,
            "recovery_diversity_target": float(collapse_state.get("recovery_diversity_target", 0.0) or 0.0),
            "operator_count": len({str(op).strip().lower() for op in self._grpo_config.get("operator_cycle", ["m1", "m2", "e1", "e2"]) if str(op).strip()}),
        }

    def _sample_recovery_ops(self, *, k: int, operator_cycle: list[str], context: dict) -> list[str]:
        return self._operator_scheduler.select_recovery_ops(k=k, operator_cycle=operator_cycle, context=context)

    @staticmethod
    def _needs_recovery_i1(context: dict) -> bool:
        diversity = float((context or {}).get("population_diversity", 0.0) or 0.0)
        target = float((context or {}).get("recovery_diversity_target", 0.0) or 0.0)
        period = max(2, int((context or {}).get("operator_count", 4) or 4))
        cycle_index = max(0, int((context or {}).get("cycle_index", 0) or 0))
        return bool(target > 0.0 and diversity < target and cycle_index % period == 0)

    def _is_invalid_program_for_population(self, program_str: str) -> bool:
        return bool(_check_score_metadata_leak(program_str) or (getattr(self._reward_fn, "detect_randomness", True) and _check_randomness(program_str)))

    def _candidate_block_reasons(self, meta: dict, program_str: str) -> list[str]:
        reasons = []
        if meta.get("failure_level") in ("no_output", "no_code", "no_function", "exec_error", "invalid_candidate"):
            reasons.append(str(meta.get("failure_level")))
        if not bool(meta.get("validity", True)):
            reasons.append("invalid_candidate")
        for key in ("random_algo", "exact_parent_copy", "score_metadata_leak"):
            if meta.get(key):
                reasons.append(key)
        if _check_score_metadata_leak(program_str):
            reasons.append("score_metadata_leak")
        return list(dict.fromkeys(reasons))

    def _reward_update_summary(self, reward_events: list[dict], *, cycle_index: int, population_best_score) -> dict:
        valid_events = [event for event in reward_events or [] if is_search_evidence_event(event)]
        registered_events = [event for event in reward_events or [] if is_population_eligible_event(event) and bool(event.get("registered_to_population"))]
        survived_events = [event for event in reward_events or [] if is_population_eligible_event(event) and bool(event.get("survived_main_population"))]
        return {
            "reward_events": list(reward_events or []),
            "operator_stats": operator_stats(reward_events or []),
            "frontier_improve_count": sum(bool(event.get("beats_frontier")) for event in valid_events),
            "parent_improve_count": sum(bool(event.get("beats_parent")) for event in valid_events),
            "registered_count": len(registered_events),
            "survived_count": len(survived_events),
            **self._adaptive_operator_context(cycle_index=cycle_index, population_best_score=population_best_score),
        }

    def _observe_collapse_decision(self, *, rl_update_id: int, best_after, reward_events: list[dict]) -> dict:
        if not self._collapse_manager.enabled:
            return {"should_collapse": False, "trigger": "disabled"}
        summary = self._reward_update_summary(reward_events, cycle_index=rl_update_id - 1, population_best_score=best_after)
        summary["population_diversity"] = self._population_diversity()
        return self._collapse_manager.observe_update(
            rl_update_id=rl_update_id,
            best_after=best_after,
            population_stats={"size": len(self._population)},
            summary=summary,
            population_size=len(self._population),
            minimize=self._minimize(),
            target_population_size=max(1, int(self._pop_size or 1)),
            recovery_active=bool(self._collapse_manager.recovery_active),
        )

    def _select_collapse_retained_core(self, ordered: list[Function]) -> list[Function]:
        if len(ordered) <= 1:
            return ordered[:1]
        best = ordered[0]
        median_u = self._median(self._utility(func) for func in ordered)
        candidates = [func for func in ordered if not self._same_algorithm(func, best) and self._utility(func) >= median_u]
        diverse = max(candidates or ordered[1:], key=lambda func: (self._algorithm_diversity(best, func), self._utility(func)), default=None)
        return [best] + ([diverse] if diverse is not None else [])

    def _execute_population_collapse(self, *, rl_update_id: int, decision: dict) -> dict | None:
        ordered = self._ordered_unique(list(self._population.population))
        if not ordered:
            return None
        retained = self._select_collapse_retained_core(ordered)
        before_size = len(self._population)
        self._retained_core = list(retained)
        self._population.replace_population(retained)
        event = {
            "rl_update_id": int(rl_update_id),
            "trigger": decision.get("trigger", "adaptive_collapse"),
            "population_size_before": before_size,
            "population_size_after": len(self._population),
            "retained_core_size": len(retained),
            "retained_core_scores": [getattr(func, "score", None) for func in retained],
            "target_population_size": int(self._pop_size or 0),
            "population_diversity": float(decision.get("population_diversity", 0.0) or 0.0),
            "diversity_reference": float(decision.get("diversity_reference", 0.0) or 0.0),
            "recovery_diversity_target": float(decision.get("recovery_diversity_target", 0.0) or 0.0),
            "operator_count": int(decision.get("operator_count", 4) or 4),
        }
        logger.info("[EoHRL] adaptive collapse: update=%d active %d->%d", rl_update_id, before_size, len(self._population))
        return self._collapse_manager.record_collapse(event)

    def _finish_recovery_if_full(self, *, rl_update_id: int, reward_events: list[dict]) -> dict | None:
        if not self._collapse_manager.recovery_active or len(self._population) < int(self._pop_size or 0):
            return None
        summary = self._reward_update_summary(reward_events, cycle_index=rl_update_id - 1, population_best_score=self._best_population_score())
        diversity = self._population_diversity()
        if not self._collapse_manager.recovery_ready_to_finish(
            rl_update_id=rl_update_id,
            population_diversity=diversity,
            frontier_improve_count=int(summary.get("frontier_improve_count", 0) or 0),
        ):
            return None
        self._retained_core = []
        return self._collapse_manager.finish_recovery(
            {
                "rl_update_id": int(rl_update_id),
                "population_size": len(self._population),
                "population_diversity": float(diversity),
                "frontier_improve_count": int(summary.get("frontier_improve_count", 0) or 0),
                "parent_improve_count": int(summary.get("parent_improve_count", 0) or 0),
                "reason": "population_full_and_diversity_restored",
            }
        )

    def _recovery_state_summary(self) -> dict:
        target = max(1, int(self._pop_size or 1))
        collapse_state = self._collapse_manager.get_state()
        return {
            "active": bool(self._collapse_manager.recovery_active),
            "active_size": len(self._population),
            "target_size": target,
            "diversity_target": float(collapse_state.get("recovery_diversity_target", 0.0) or 0.0),
        }

    def _refresh_reward_event_population_survival(self, reward_events: list[dict]) -> None:
        population_codes = {str(func).strip() for func in self._population.population}
        for event in reward_events or []:
            text = str(event.get("function") or "").strip()
            event["survived_main_population"] = bool(text and event.get("registered_to_population") and is_population_eligible_event(event) and text in population_codes)

    def _llm_policy_owner(self):
        return self._llm.model_manager

    def _get_llm_lora_path(self, method_name: str) -> str | None:
        return getattr(self._llm_policy_owner(), method_name)()

    def _run_grpo_update(self) -> bool:
        if not self._grpo_enabled or self._reward_fn is None:
            return False
        if not self._collapse_manager.recovery_active and (not self._bootstrap_done or len(self._population) < int(self._pop_size or 0)):
            logger.info("[EoHRL] 种群尚未完成初始化，跳过 GRPO update")
            return False
        target_prompts = int(self._prompts_per_update)
        required_budget = int(self._grpo_group_size) * target_prompts
        remaining_budget = None if self._max_sample_nums is None else int(self._max_sample_nums - self._tot_sample_nums)
        if remaining_budget is not None and remaining_budget < required_budget:
            logger.info("[EoHRL] remaining completion budget=%d < one GRPO update=%d", remaining_budget, required_budget)
            return False
        update_id = self._grpo_update_count + 1
        recovery_active_at_update = bool(self._collapse_manager.recovery_active)
        self._pending_grpo_sample_order = int(self._tot_sample_nums)
        records = self._build_online_prompt_records(update_id)
        if not records:
            return False
        best_before = self._best_population_score()
        pop_stats_before = self._get_population_score_stats()
        started_at = time.time()
        result = self._llm_policy_owner().train_once(
            prompt_records=records,
            evaluator=self._evaluator,
            reward_fn=self._reward_fn,
            template_program=self._template_program,
            output_dir=self._grpo_config.get("output_dir") or self._default_log_dir(),
            update_id=update_id,
            reward_eval_workers=self._reward_eval_workers,
            register_candidate=self._register_candidate,
            enable_ast_gate=bool(getattr(self, "_enable_ast_gate", False)),
            recovery_active=recovery_active_at_update,
        )
        if not result.get("executed"):
            total = int(result.get("total", 0) or 0)
            if result.get("samples_consumed"):
                failed_summary = dict((result.get("metrics") or {}).get("reward_summary") or {})
                self._tot_sample_nums += total
                self._accumulate_token_usage(failed_summary.get("token_usage") or failed_summary)
                for event in result.get("recent_reward_events", []) or []:
                    self._reward_events.append(event)
            logger.warning("[EoHRL] GRPO backend 未完成执行: reason=%s total=%s", result.get("reason"), total)
            return False

        self._grpo_update_count = update_id
        metrics = result.get("metrics") or {}
        self._latest_lora_path = metrics.get("latest_lora_path") or self._latest_lora_path
        self._current_lora_path = metrics.get("current_lora_path") or self._latest_lora_path
        reward_summary = dict(result.get("reward_summary") or metrics.get("reward_summary") or {})
        reward_events = list(reward_summary.get("reward_events") or result.get("recent_reward_events", []) or [])
        for event in reward_events:
            self._reward_events.append(event)
        self._population.increment_generation()
        self._tot_sample_nums += int(reward_summary.get("total", 0) or 0)
        self._accumulate_token_usage(reward_summary.get("token_usage") or reward_summary)
        best_after = self._best_population_score()
        self._refresh_reward_event_population_survival(reward_events)
        if self._operator_scheduler.enabled:
            summary = self._reward_update_summary(reward_events, cycle_index=update_id - 1, population_best_score=best_after)
            if self._collapse_manager.recovery_active:
                self._operator_scheduler.update_recovery_from_summary(summary)
            else:
                self._operator_scheduler.update_from_summary(summary)
        collapse_decision = self._observe_collapse_decision(rl_update_id=update_id, best_after=best_after, reward_events=reward_events)
        collapse_event = self._execute_population_collapse(rl_update_id=update_id, decision=collapse_decision) if collapse_decision.get("should_collapse") else None
        recovery_finish_event = self._finish_recovery_if_full(rl_update_id=update_id, reward_events=reward_events)
        post_best = self._best_population_score()
        metrics.update(
            {
                "rl_update_id": update_id,
                "best_before_rl": best_before,
                "best_after_rl": post_best,
                "effective_best_score": post_best,
                "population_score_stats_before_rl": pop_stats_before,
                "population_score_stats_after_rl": self._get_population_score_stats(),
                "generation": self._population.generation,
                "operator_stats": operator_stats(reward_events),
                "collapse_decision": collapse_decision,
                "collapse_event": collapse_event,
                "recovery_finish_event": recovery_finish_event,
                "recovery": self._recovery_state_summary(),
                "adaptive_operator": self._operator_scheduler.get_state(),
                "collapse_manager": self._collapse_manager.get_state(),
                "reward_summary": {**reward_summary, "reward_events": reward_events, "operator_stats": operator_stats(reward_events)},
            }
        )
        metrics.update(self._adaptive_operator_context(cycle_index=update_id - 1, population_best_score=post_best))
        if self._operator_scheduler.enabled:
            operator_cycle = list(self._grpo_config.get("operator_cycle", []))
            if self._collapse_manager.recovery_active:
                probs = self._operator_scheduler.compute_recovery_probabilities(operator_cycle=operator_cycle, context=metrics)
                metrics["recovery_operator_probabilities"] = probs
            else:
                probs = self._operator_scheduler.compute_probabilities(operator_cycle=operator_cycle, context=metrics)
                metrics["adaptive_operator_probabilities"] = probs
            if probs:
                metrics["operator_probability_source"] = "recovery_operator" if self._collapse_manager.recovery_active else "adaptive_operator"
                metrics["adaptive_skew"] = max(float(value) for value in probs.values())
                metrics["adaptive_entropy"] = -sum(float(value) * math.log(max(float(value), 1.0e-12)) for value in probs.values())
        timing = dict(metrics.get("timing") or {})
        timing.update({"total_update_elapsed": time.time() - started_at, "rl_update_id": update_id})
        metrics["timing"] = timing
        self._latest_training_timing = timing
        if bool(metrics.get("accepted_lora", False)):
            self._reload_llm_after_grpo(self._current_lora_path)
        self._best_curve.append({"rl_update_id": update_id, "generation": self._population.generation, "best_at_trigger": best_before})
        logger.info("[Timing][EoHRLGRPO] %s", json.dumps({"update": update_id, "prompts": len(records), "completions": int(reward_summary.get("total", 0) or 0), "timing": timing, "accepted_lora": bool(metrics.get("accepted_lora", False)), "adaptive_operator": bool(self._operator_scheduler.enabled), "collapse": bool(self._collapse_manager.enabled)}, ensure_ascii=False, default=str))
        self._save_rl_update_metrics(rl_update_id=update_id, metrics=metrics)
        self._save_checkpoint(tag=f"rl_{update_id:03d}")
        return True

    def _reload_llm_after_grpo(self, lora_path: str | None, *, force_reload: bool = False) -> None:
        if not lora_path:
            return
        owner = self._llm_policy_owner()
        if force_reload:
            owner.set_current_lora_path(lora_path)
        else:
            owner.prepare_for_inference(force_reload=False)
        logger.info("[EoHRL] 已激活训练后最新 LoRA: %s", lora_path)

    def _save_rl_update_metrics(self, *, rl_update_id: int, metrics: dict) -> None:
        try:
            update_dir = os.path.join(self._default_log_dir(), "rl_training", f"rl_update_{int(rl_update_id):03d}")
            os.makedirs(update_dir, exist_ok=True)
            payload = dict(metrics or {})
            reward_summary = dict(payload.pop("reward_summary", {}) or {})
            payload.update(reward_summary)
            payload.update({"rl_update_id": int(rl_update_id), "best_before": payload.get("best_before_rl"), "best_after": payload.get("effective_best_score", payload.get("best_after_rl")), "population_size": len(self._population), "current_lora_path": self._current_lora_path, "latest_lora_path": self._latest_lora_path, "operator_scheduler_state": self._operator_scheduler.get_state(), "collapse_decision": dict(payload.get("collapse_decision") or {}), "collapse_event": payload.get("collapse_event"), "recovery_finish_event": payload.get("recovery_finish_event")})
            with open(os.path.join(update_dir, "metrics.json"), "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        except Exception:
            if self._debug_mode:
                logger.exception("save rl update metrics failed")

    def _sample_evaluate_register_batch(
        self,
        prompt: str,
        *,
        operator_type: str,
        parent_ids=None,
        n: Optional[int] = None,
        strict_contract: bool = True,
    ) -> bool:
        remaining_budget = None if self._max_sample_nums is None else int(self._max_sample_nums - self._tot_sample_nums)
        if remaining_budget is not None and remaining_budget <= 0:
            return False
        n = self._samples_per_prompt if n is None else max(1, int(n))
        if remaining_budget is not None:
            n = min(n, remaining_budget)
        if n <= 0:
            return False
        batch_started_at = time.time()
        records = self._sampler.get_strategy_and_function_records_batch(prompt, n, strict_contract=strict_contract)
        batch_elapsed = time.time() - batch_started_at
        if not records:
            return False
        self._record_generation_token_usage(prompt=prompt, responses=[record.get("raw_response", "") for record in records])

        sample_time_per_item = batch_elapsed / max(1, len(records))
        if not self._batch_sampler_logged:
            logger.info("[EoHRL] 批量采样成功: 返回 %d 条结果, 耗时 %.1fs", n, batch_elapsed)
            self._batch_sampler_logged = True
        logger.info(
            "[EoHRL] batch 采样: generation=%s, batch_elapsed=%.1fs, n=%d",
            self._population.generation,
            batch_elapsed,
            len(records),
        )

        eval_items, stats = [], {"missing_idea": 0, "missing_func": 0, "program_failed": 0, "registered": 0}
        for parsed in records:
            self._tot_sample_nums += 1
            sample_order = int(self._tot_sample_nums)
            strategy = parsed.get("strategy")
            func = parsed.get("func")
            if strategy is None:
                stats["missing_idea"] += 1
            if func is None:
                stats["missing_func"] += 1
                continue
            setattr(func, "_eoh_sample_order", sample_order)
            program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
            if program is None:
                stats["program_failed"] += 1
                continue
            future = self._evaluation_executor.submit(self._evaluate_program_with_diag, program)
            eval_items.append((future, strategy or "", func, str(program)))

        registered = False
        for future, strategy, func, program_str in eval_items:
            score, eval_time, diag = future.result()
            inserted = self._register_candidate(
                func=func,
                strategy=strategy,
                program_str=program_str,
                score=score,
                eval_time=eval_time,
                diag=diag,
                sample_time=sample_time_per_item,
                operator_type=operator_type,
                parent_ids=parent_ids,
            )
            registered = registered or inserted
            stats["registered"] += int(bool(inserted))
        logger.info(
            "[EoHRL] batch 解析/评估统计: returned=%d, submitted=%d, registered=%d, missing_idea=%d, missing_func=%d, program_failed=%d",
            len(records), len(eval_items), stats["registered"], stats["missing_idea"], stats["missing_func"], stats["program_failed"],
        )
        return registered

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
        return score, eval_time, {"failure_level": "exec_error", "detail": "evaluation returned None"} if score is None else None

    @staticmethod
    def _performance_profile_from_diag(diag) -> list[float] | None:
        if not isinstance(diag, dict):
            return None
        profile = diag.get("performance_profile")
        if not isinstance(profile, (list, tuple)):
            return None
        values = []
        for value in profile:
            try:
                parsed = float(value)
                if math.isfinite(parsed):
                    values.append(parsed)
            except Exception:
                continue
        return values or None

    def _register_candidate(
        self,
        *,
        func: Function,
        program_str: str,
        score,
        eval_time,
        diag,
        strategy: str = "",
        sample_time: float = 0.0,
        operator_type: str = "",
        parent_ids=None,
        meta: dict | None = None,
    ) -> bool:
        grpo_path = meta is not None
        meta = meta or {}
        if grpo_path and getattr(func, "_eoh_sample_order", None) is None:
            self._pending_grpo_sample_order = int(getattr(self, "_pending_grpo_sample_order", self._tot_sample_nums) or self._tot_sample_nums) + 1
            setattr(func, "_eoh_sample_order", self._pending_grpo_sample_order)
        op = str(meta.get("operator_type") or operator_type or "grpo").strip().lower()
        parent_ids = meta.get("parent_ids", parent_ids)
        source = "integrated_grpo" if grpo_path else getattr(self._sampler.llm, "last_source", "evolution")
        func.score, func.evaluate_time, func.algorithm, func.sample_time, func.operator = score, eval_time, str(meta.get("strategy") or strategy or ""), sample_time, op
        for attr, value in (("_eoh_op", op), ("_eoh_parent_id", parent_ids), ("_eoh_source", source)):
            setattr(func, attr, value)
        if isinstance(diag, dict):
            setattr(func, "_eval_diag", diag)
        profile = self._performance_profile_from_diag(diag)
        if profile:
            setattr(func, "_eoh_profile", profile)
        if getattr(func, "_eoh_sample_order", None) is None:
            setattr(func, "_eoh_sample_order", int(self._tot_sample_nums))
        setattr(func, "_eoh_birth_sample_order", int(getattr(func, "_eoh_sample_order", self._tot_sample_nums) or self._tot_sample_nums))
        setattr(func, "_eoh_birth_generation", int(getattr(self._population, "generation", 0) or 0))

        if self._profiler is not None:
            self._profiler.register_function(func, program=program_str, source=source)

        bootstrap_phase = (not grpo_path) and self._is_initialization_candidate(op)
        reasons = self._candidate_block_reasons(meta, program_str) if grpo_path else []
        if not self._has_algorithm_description(func):
            if grpo_path:
                meta["blocked_from_population"] = True
                meta["population_gate_reasons"] = list(dict.fromkeys(list(reasons) + ["missing_algorithm_description"]))
            logger.debug("[EoHRL] 候选已评估但未入 population: missing algorithm description, operator=%s", op)
            return False
        recovery_i1 = bool(grpo_path and op == "i1" and self._collapse_manager.recovery_active)
        if not bootstrap_phase and op == "i1" and not recovery_i1:
            if grpo_path:
                reasons.append("i1_after_bootstrap")
            else:
                return False
        if grpo_path and (reasons or score is None or not math.isfinite(float(score))):
            meta["blocked_from_population"] = True
            meta["population_gate_reasons"] = list(dict.fromkeys(reasons or ["non_finite_score"]))
            return False
        if (not grpo_path) and self._is_invalid_program_for_population(program_str):
            logger.debug(
                "[EoHRL] 候选已评估但未入 population: invalid program metadata/randomness, operator=%s",
                op,
            )
            return False

        if bootstrap_phase:
            self._initial_candidates = self._ordered_unique(list(self._initial_candidates) + [func])
            return True

        inserted, just_filled, _evicted = self._population.register_evolved_function(func)

        if (inserted or bootstrap_phase) and not self._bootstrap_done and just_filled:
            self._bootstrap_done = True
        if inserted:
            if grpo_path:
                meta["blocked_from_population"] = False
                meta["population_gate_reasons"] = []
        else:
            if grpo_path:
                meta["blocked_from_population"] = True
                meta["population_gate_reasons"] = list(dict.fromkeys(list(meta.get("population_gate_reasons") or []) + ["not_selected_for_population"]))
        return inserted

    def _save_checkpoint(self, *, tag: str | None = None) -> None:
        try:
            state = {
                "generation": int(self._population.generation),
                "tot_sample_nums": int(self._tot_sample_nums),
                "rl_update_count": int(self._grpo_update_count),
                "best_curve": list(getattr(self, "_best_curve", [])),
                "population": [_serialize_function(func) for func in self._population.population],
                "adaptive_operator_state": self._operator_scheduler.get_state(),
                "collapse_state": self._collapse_manager.get_state(),
                "bootstrap_done": bool(self._bootstrap_done),
                "token_usage_totals": dict(self._token_usage_totals),
                "current_lora_path": self._current_lora_path,
                "latest_lora_path": self._latest_lora_path or self._get_llm_lora_path("get_latest_lora_path"),
                "deployed_lora_path": self._get_llm_lora_path("get_deployed_lora_path"),
                "anchor_lora_path": self._get_llm_lora_path("get_anchor_lora_path"),
                "best_lora_path": self._get_llm_lora_path("get_best_lora_path"),
                "best_score": self._best_population_score(),
            }
            latest = os.path.join(self._checkpoint_dir, "latest.json")
            path = os.path.join(self._checkpoint_dir, f"ckpt_{tag}.json") if tag else latest
            for target in {path, latest}:
                _write_json_atomic(target, state)
            logger.info("checkpoint 已保存: %s", path)
        except Exception:
            if self._debug_mode:
                logger.exception("save checkpoint failed")

    def _restore_checkpoint(self) -> None:
        latest = os.path.join(self._checkpoint_dir, "latest.json")
        if not os.path.exists(latest):
            logger.info("未找到 checkpoint, 将从头开始")
            return
        try:
            with open(latest, "r", encoding="utf-8") as file:
                state = json.load(file)
            model_manager = self._llm_policy_owner()
            restored_population = [func for func in _deserialize_functions(state.get("population", [])) if self._has_algorithm_description(func)]
            population = Population(
                pop_size=self._pop_size,
                generation=int(state.get("generation", 0) or 0),
                pop=restored_population,
                minimize=self._minimize(),
            )
            lora_paths = {
                key: state.get(key)
                for key in ("best_lora_path", "latest_lora_path", "deployed_lora_path", "anchor_lora_path", "current_lora_path")
                if state.get(key) is not None
            }
            self._apply_resume_lora_paths(model_manager, lora_paths)
            self._population = population
            self._operator_scheduler.load_state(state.get("adaptive_operator_state"))
            self._collapse_manager.load_state(state.get("collapse_state"))
            self._retained_core = list(self._population.population[: min(len(self._population.population), 2)]) if self._collapse_manager.recovery_active else []
            self._tot_sample_nums = int(state.get("tot_sample_nums", 0) or 0)
            self._grpo_update_count = int(state.get("rl_update_count", 0) or 0)
            self._best_curve = list(state.get("best_curve", []))
            self._token_usage_totals = self._normalize_token_usage(state.get("token_usage_totals"))
            target_pop_size = int(self._pop_size or 0)
            self._bootstrap_done = bool(state.get("bootstrap_done", False) or (target_pop_size > 0 and len(self._population) >= target_pop_size))
            self._resume_mode = True
            self._current_lora_path = self._get_llm_lora_path("get_current_lora_path") or self._current_lora_path
            self._latest_lora_path = self._get_llm_lora_path("get_latest_lora_path") or self._current_lora_path
            self._anchor_lora_path = self._get_llm_lora_path("get_anchor_lora_path") or self._anchor_lora_path
            logger.info("[EoHRL] 从 checkpoint 恢复 LoRA: current=%s latest=%s", self._current_lora_path, self._latest_lora_path)
            self._reload_llm_after_grpo(self._current_lora_path, force_reload=True)
        except Exception:
            logger.exception("restore checkpoint failed")
            raise

    def _apply_resume_lora_paths(self, model_manager, lora_paths: dict) -> None:
        if not lora_paths:
            return
        order = ("best", "latest", "deployed", "anchor", "current")
        resolved = {name: lora_paths.get(f"{name}_lora_path") for name in order}
        resolved = {name: path for name, path in resolved.items() if path is not None}
        validator = getattr(model_manager, "_assert_lora", None)
        for path in resolved.values():
            if callable(validator):
                validator(path)
            elif not os.path.isdir(os.path.expanduser(str(path))):
                raise ValueError(f"invalid lora path: {path}")
        previous = {name: getattr(model_manager, f"get_{name}_lora_path")() for name in order}
        try:
            for name in order:
                if name in resolved:
                    getattr(model_manager, f"set_{name}_lora_path")(resolved[name])
        except Exception:
            logger.exception("[EoHRL] checkpoint LoRA 路径恢复失败，回退到恢复前状态")
            for name, path in previous.items():
                try:
                    getattr(model_manager, f"set_{name}_lora_path")(path)
                except Exception:
                    logger.warning("[EoHRL] 回退 %s LoRA 路径失败: %s", name, path, exc_info=True)
            raise

    def _default_log_dir(self) -> str:
        return (getattr(self._profiler, "_log_dir", None) if self._profiler is not None else None) or "./rl_training"

    def _finalize_initial_population(self) -> None:
        selected = self._select_score_diverse(list(self._initial_candidates), count=int(self._pop_size or 0))
        self._population.replace_population(selected)
        self._bootstrap_done = len(self._population) >= int(self._pop_size or 0)
        logger.info(
            "[EoHRL] 初始化选优完成: candidates=%d, population=%d",
            len(self._initial_candidates),
            len(self._population),
        )

    def _iteratively_init_population(self):
        no_progress_rounds = 0
        while self._tot_sample_nums < int(self._initial_sample_nums_max):
            remaining_budget = None if self._max_sample_nums is None else int(self._max_sample_nums - self._tot_sample_nums)
            if remaining_budget is not None and remaining_budget <= 0:
                break
            before_evaluated = self._tot_sample_nums
            before_pool = len(self._initial_candidates)
            try:
                prompt = EoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
                batch_n = self._samples_per_prompt
                if remaining_budget is not None:
                    batch_n = min(batch_n, remaining_budget)
                self._sample_evaluate_register_batch(prompt, operator_type="i1", n=batch_n, strict_contract=False)
                logger.info("[EoHRL] 初始化进度: evaluated=%d/%d, candidates=%d, target_pop=%d", self._tot_sample_nums, self._initial_sample_nums_max, len(self._initial_candidates), self._pop_size)
                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    print(f"Note: During initialization, EoH gets {len(self._initial_candidates)} candidate algorithms after {self._initial_sample_nums_max} trails.")
                    break
                after_pool = len(self._initial_candidates)
                if self._tot_sample_nums == before_evaluated and after_pool == before_pool:
                    no_progress_rounds += 1
                    if no_progress_rounds >= 10:
                        logger.warning("[EoHRL] 初始化连续 %d 轮无有效评估，提前结束初始化以避免空转；evaluated=%d, candidates=%d/%d", no_progress_rounds, self._tot_sample_nums, after_pool, self._pop_size)
                        break
                else:
                    no_progress_rounds = 0
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    raise
                logger.warning("[EoHRL] 初始化采样/评估异常，继续下一轮", exc_info=True)
                continue
        self._finalize_initial_population()

    def _iteratively_use_online_rl(self):
        while (self._max_generations is None or int(self._grpo_update_count) < int(self._max_generations)) and (
            self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums
        ):
            try:
                previous_generation = self._population.generation
                executed = self._run_grpo_update()
                if not executed:
                    logger.info("[EoHRL] 在线 RL update 未执行，结束 RL 主循环")
                    break
                if self._profiler is not None and self._population.generation > previous_generation:
                    self._profiler.register_population(self._population)
            except KeyboardInterrupt:
                break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    raise
                logger.exception("[EoHRL] 在线 RL update 失败，已停止主循环以避免重复破坏状态")
                break

    def run(self):
        try:
            if not self._bootstrap_done and len(self._population) < int(self._pop_size or 0):
                self._iteratively_init_population()
                if len(self._population) < int(self._pop_size or 0):
                    print(
                        f"The search is terminated since EoH unable to obtain {int(self._pop_size or 0)} feasible algorithms during initialization. "
                        f"Initialization generated {self._initial_sample_nums_max} samples. "
                        f"Please also check your evaluation implementation and LLM implementation."
                    )
                    return
            self._iteratively_use_online_rl()
            self._save_checkpoint(tag="latest_finish")
        finally:
            try:
                self._evaluation_executor.shutdown(cancel_futures=True)
            except Exception:
                pass
            try:
                self._write_run_summary()
            except Exception:
                pass
            if self._profiler is not None:
                self._profiler.finish()
            self._sampler.llm.close()

    def _write_run_summary(self) -> None:
        model_manager = self._llm_policy_owner()
        payload = {
            "samples_per_prompt": int(self._samples_per_prompt),
            "max_generations": self._max_generations,
            "max_sample_nums": self._max_sample_nums,
            "pop_size": int(self._pop_size),
            "selection_num": int(self._selection_num),
            "current_lora_path": model_manager.get_current_lora_path(),
            "best_lora_path": model_manager.get_best_lora_path(),
            "best_curve": list(self._best_curve or []),
            "sample_accounting": self._sample_accounting(),
            "token_usage": dict(self._token_usage_totals),
            "rl_update_count": int(self._grpo_update_count),
        }
        path = os.path.join(self._default_log_dir(), "run_summary.json")
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)

    def _sample_accounting(self) -> dict:
        profiler = self._profiler
        return {
            "grpo_completion_samples": int(self._tot_sample_nums),
            "profiler_function_samples": int(getattr(profiler, "_num_samples", 0)) if profiler else 0,
            "profiler_success_samples": int(getattr(profiler, "_evaluate_success_program_num", 0)) if profiler else 0,
            "profiler_failed_samples": int(getattr(profiler, "_evaluate_failed_program_num", 0)) if profiler else 0,
            "reward_event_count": len(getattr(self, "_reward_events", []) or []),
        }

    def _get_population_score_stats(self) -> dict:
        scores = sorted([float(func.score) for func in self._population.population if getattr(func, "score", None) is not None], reverse=not self._minimize())
        if not scores:
            return {"best": None, "median": None, "worst": None, "mean": None, "size": 0}
        mid = len(scores) // 2
        median = (scores[mid - 1] + scores[mid]) / 2.0 if len(scores) % 2 == 0 else scores[mid]
        return {"best": scores[0], "median": median, "worst": scores[-1], "mean": sum(scores) / len(scores), "size": len(scores)}

    def get_runtime_summary(self) -> dict:
        return {
            "tot_sample_nums": self._tot_sample_nums,
            "sample_accounting": self._sample_accounting(),
            "generation": self._population.generation,
            "rl_update_count": self._grpo_update_count,
            "population_size": len(self._population),
            "best_population_score": self._best_population_score(),
            "current_lora_path": self._current_lora_path,
            "latest_lora_path": self._latest_lora_path,
            "training_timings": dict(getattr(self, "_latest_training_timing", {}) or {}),
            "token_usage": dict(self._token_usage_totals),
        }

    @staticmethod
    def _empty_token_usage() -> dict:
        return {"input_tokens_total": 0, "unique_input_tokens_total": 0, "output_tokens_total": 0, "tokens_total": 0}

    @classmethod
    def _normalize_token_usage(cls, value) -> dict:
        totals = cls._empty_token_usage()
        if isinstance(value, dict):
            for key in totals:
                try:
                    totals[key] = max(0, int(value.get(key, 0) or 0))
                except Exception:
                    totals[key] = 0
        return totals

    def _accumulate_token_usage(self, usage) -> None:
        current = self._normalize_token_usage(self._token_usage_totals)
        update = self._normalize_token_usage(usage)
        for key in current:
            current[key] += update[key]
        self._token_usage_totals = current

    def _record_generation_token_usage(self, *, prompt: str, responses: list[str]) -> None:
        owner = self._llm_policy_owner()
        prompt_counter = getattr(owner, "_count_prompt_tokens", None)
        prompt_tokens = 0
        if callable(prompt_counter):
            prompt_tokens = int(prompt_counter([{"role": "user", "content": str(prompt or "").strip()}]) or 0)
        output_tokens = sum(self._count_text_tokens(response) for response in responses or [])
        input_total = prompt_tokens * len(responses or [])
        self._accumulate_token_usage({
            "input_tokens_total": input_total,
            "unique_input_tokens_total": prompt_tokens,
            "output_tokens_total": output_tokens,
            "tokens_total": input_total + output_tokens,
        })

    def _count_text_tokens(self, text: str) -> int:
        tokenizer = getattr(self._llm_policy_owner(), "_tokenizer", None)
        if tokenizer is None:
            return 0
        try:
            return len(tokenizer.encode(str(text or ""), add_special_tokens=False))
        except Exception:
            return 0

__all__ = ["EoH"]
