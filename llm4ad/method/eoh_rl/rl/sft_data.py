from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger(__name__)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


_SFT_FORBIDDEN_CALL_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "breakpoint",
        "input",
    }
)
_STRICT_FORBIDDEN_CALL_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "breakpoint",
    }
)
_DISALLOWED_ATTR_PREFIXES = frozenset({"__"})
_DISALLOWED_INSIDE_GENERATED_FUNCTION = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFunctionDef,
    ast.Try,
    ast.Raise,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)

_TASK_TEMPLATE_MODULE_MAP = {
    "tsp_construct": "llm4ad.task.optimization.tsp_construct.template",
    "online_bin_packing": "llm4ad.task.optimization.online_bin_packing.template",
    "cvrp_construct": "llm4ad.task.optimization.cvrp_construct.template",
    "jssp_construct": "llm4ad.task.optimization.jssp_construct.template",
    "knapsack_construct": "llm4ad.task.optimization.knapsack_construct.template",
    "vrptw_construct": "llm4ad.task.optimization.vrptw_construct.template",
    "ovrp_construct": "llm4ad.task.optimization.ovrp_construct.template",
    "bp_1d_construct": "llm4ad.task.optimization.bp_1d_construct.template",
    "set_cover_construct": "llm4ad.task.optimization.set_cover_construct.template",
    "qap_construct": "llm4ad.task.optimization.qap_construct.template",
    "cflp_construct": "llm4ad.task.optimization.cflp_construct.template",
}

_TEMPLATE_SEARCH_ROOTS = [
    ("optimization", None),
    ("optimization", "co_bench"),
    ("science_discovery", None),
    ("machine_learning", None),
]


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str = ""


def parse_function_signature_from_template(template_program: str) -> tuple[str, list[str]]:
    tree = ast.parse(template_program)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            arg_names = [arg.arg for arg in getattr(node.args, "posonlyargs", [])]
            arg_names.extend(arg.arg for arg in getattr(node.args, "args", []))
            arg_names.extend(arg.arg for arg in getattr(node.args, "kwonlyargs", []))
            return node.name, arg_names
    raise ValueError("template_program has no function definition")


def validate_sft_data_code(code: str, *, template_program: str | None = None) -> GateResult:
    if not isinstance(code, str) or not code.strip():
        return GateResult(False, "empty_code")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return GateResult(False, "syntax_error")
    except Exception:
        return GateResult(False, "ast_parse_error")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return GateResult(False, "no_functiondef")
    function_node = functions[0]
    if template_program is not None:
        try:
            expected_name, expected_args = parse_function_signature_from_template(template_program)
        except Exception:
            return GateResult(False, "template_parse_failed")
        matched = None
        for node in functions:
            if node.name == expected_name:
                matched = node
                break
        if matched is None:
            return GateResult(False, "no_matching_functiondef")
        function_node = matched
        got_args = [arg.arg for arg in getattr(function_node.args, "posonlyargs", [])]
        got_args.extend(arg.arg for arg in getattr(function_node.args, "args", []))
        got_args.extend(arg.arg for arg in getattr(function_node.args, "kwonlyargs", []))
        if got_args != list(expected_args):
            return GateResult(False, "signature_args_mismatch")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = None
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            if call_name in _SFT_FORBIDDEN_CALL_NAMES:
                return GateResult(False, f"call_forbidden:{call_name}")
    return GateResult(True, "")


def validate_generated_code(
    code: str,
    *,
    template_program: str | None = None,
    allowed_import_modules: Sequence[str] | None = None,
) -> GateResult:
    """在线 RL 生成代码 AST gate：只接受目标函数和模板允许的顶层 import。"""
    if not isinstance(code, str) or not code.strip():
        return GateResult(False, "empty_code")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return GateResult(False, "syntax_error")
    except Exception:
        return GateResult(False, "ast_parse_error")

    function_node = _target_functiondef(tree, template_program)
    if function_node is None:
        return GateResult(False, "no_matching_functiondef" if template_program else "no_functiondef")

    if template_program is not None:
        sig_error = _check_signature_match(function_node, template_program)
        if sig_error is not None:
            return sig_error

    allowed_template_imports = (
        _allowed_toplevel_imports_from_template(template_program)
        if template_program is not None
        else frozenset()
    )
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), (ast.Constant, ast.Str)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)) and allowed_template_imports:
            if _normalize_import_module(node) in allowed_template_imports:
                continue
            return GateResult(False, "toplevel_import_not_in_template")
        return GateResult(False, "toplevel_stmt_forbidden")

    for node in ast.walk(function_node):
        if isinstance(node, _DISALLOWED_INSIDE_GENERATED_FUNCTION):
            return GateResult(False, f"node_forbidden:{type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if any(str(node.attr).startswith(prefix) for prefix in _DISALLOWED_ATTR_PREFIXES):
                return GateResult(False, "dunder_attr_forbidden")
        if isinstance(node, ast.Call):
            call_name = _get_call_name(node)
            if call_name in _STRICT_FORBIDDEN_CALL_NAMES:
                return GateResult(False, f"call_forbidden:{call_name}")
        if isinstance(node, ast.Name) and node.id in {"__builtins__"}:
            return GateResult(False, "builtins_ref_forbidden")
    if allowed_import_modules:
        return GateResult(False, "import_whitelist_not_supported")
    return GateResult(True, "")


def _target_functiondef(tree: ast.Module, template_program: str | None) -> ast.FunctionDef | None:
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None
    if template_program:
        try:
            expected_name, _ = parse_function_signature_from_template(template_program)
        except Exception:
            return functions[0]
        for function_node in functions:
            if function_node.name == expected_name:
                return function_node
        return None
    return functions[0]


def _check_signature_match(function_node: ast.FunctionDef, template_program: str) -> GateResult | None:
    try:
        expected_name, expected_args = parse_function_signature_from_template(template_program)
    except Exception:
        return GateResult(False, "template_parse_failed")
    if function_node.name != expected_name:
        return GateResult(False, "signature_name_mismatch")
    got_args = [arg.arg for arg in getattr(function_node.args, "posonlyargs", [])]
    got_args.extend(arg.arg for arg in getattr(function_node.args, "args", []))
    got_args.extend(arg.arg for arg in getattr(function_node.args, "kwonlyargs", []))
    if got_args != list(expected_args):
        return GateResult(False, "signature_args_mismatch")
    return None


def _normalize_import_module(node: ast.Import | ast.ImportFrom) -> frozenset[str]:
    names: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            base = str(alias.name or "").split(".", 1)[0]
            if base:
                names.add(base)
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            base = str(node.module).split(".", 1)[0]
            if base:
                names.add(base)
        for alias in node.names:
            if alias.name == "*":
                continue
            base = str(alias.name or "").split(".", 1)[0]
            if base:
                names.add(base)
    return frozenset(names)


def _allowed_toplevel_imports_from_template(template_program: str) -> frozenset[frozenset[str]]:
    if template_program is None:
        return frozenset()
    try:
        tree = ast.parse(template_program)
    except Exception:
        return frozenset()
    signatures: set[frozenset[str]] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            signatures.add(_normalize_import_module(node))
    return frozenset(signatures)


def _get_call_name(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def normalize_sft_source_dirs(raw_value) -> list[str]:
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        raise ValueError("eoh_rl.sft.source_dirs 必须为字符串或字符串列表")
    source_dirs: list[str] = []
    for value in values:
        raw_path = str(value).strip()
        if not raw_path:
            continue
        path = os.path.abspath(os.path.expanduser(raw_path))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"SFT 数据源目录不存在: {path}")
        source_dirs.append(path)
    if not source_dirs:
        raise ValueError("eoh_rl.sft.source_dirs 不能为空")
    return source_dirs


def build_sft_dataset_from_source_dirs(*, cleaner, source_dirs: list[str], output_path: str, logger_override=None) -> str:
    active_logger = logger_override or logger
    raw_paths: list[str] = []
    for source_dir in source_dirs:
        for root, _dirs, files in os.walk(source_dir):
            if "raw.jsonl" in files:
                raw_paths.append(os.path.join(root, "raw.jsonl"))
    raw_paths = sorted(set(raw_paths))
    if not raw_paths:
        raise FileNotFoundError(f"配置的 SFT 数据源目录下未找到 raw.jsonl: {', '.join(source_dirs)}")

    all_raw_records: list[dict] = []
    task_records: dict[str, list[dict]] = defaultdict(list)
    for raw_path in raw_paths:
        records = cleaner.load_jsonl(raw_path)
        active_logger.info("SFT 数据源: %s -> 原始 %d 条", raw_path, len(records))
        all_raw_records.extend(records)
        for record in records:
            task_records[str(record.get("task") or "")].append(record)

    merged_raw_path = os.path.join(os.path.dirname(output_path), "sft_train_merged_raw.jsonl")
    cleaner.save_jsonl(all_raw_records, merged_raw_path)
    active_logger.info("SFT 数据源原始合并完成: 共 %d 条 -> %s", len(all_raw_records), merged_raw_path)

    cleaned_by_task: dict[str, list[dict]] = {}
    for task_name, raw_records in task_records.items():
        active_logger.info("SFT 任务聚合清洗: %s -> 原始 %d 条", task_name or "<empty>", len(raw_records))
        cleaned_by_task[task_name] = cleaner.clean_records(raw_records)

    if not cleaned_by_task:
        cleaner.save_jsonl([], output_path)
        merged_count = 0
    else:
        merged_count = cleaner.merge_task_records(task_records=cleaned_by_task, output_path=output_path, logger_override=active_logger)
    active_logger.info("SFT 数据统一清洗完成: 共 %d 条 -> %s", merged_count, output_path)
    return output_path


class SFTDataCleaner:
    def __init__(
        self,
        *,
        score_filter_min: float | None = None,
        score_filter_max: float | None = None,
        top_ratio: float = 0.3,
        dedup_strategy: str = "ast_struct",
        dedup_field: str = "response",
        balance_tasks: bool = True,
        max_per_task: int | None = None,
        filter_invalid_code: bool = True,
        enable_ast_gate: bool = True,
        ast_gate_use_task_template: bool = True,
        stratified_top_k: bool = True,
        novelty_ratio: float = 0.2,
    ):
        self._score_filter_min = score_filter_min
        self._score_filter_max = score_filter_max
        self._top_ratio = float(top_ratio)
        self._dedup_strategy = str(dedup_strategy)
        self._dedup_field = str(dedup_field)
        self._balance_tasks = bool(balance_tasks)
        self._max_per_task = None if max_per_task is None else int(max_per_task)
        self._filter_invalid_code_enabled = bool(filter_invalid_code)
        self._enable_ast_gate = bool(enable_ast_gate)
        self._ast_gate_use_task_template = bool(ast_gate_use_task_template)
        self._stratified_top_k = bool(stratified_top_k)
        self._novelty_ratio = float(novelty_ratio)
        self._template_cache: dict[str, str | None] = {}

    def load_jsonl(self, path: str) -> list[dict]:
        records: list[dict] = []
        if not os.path.exists(path):
            return records
        with open(path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("跳过无法解析的行: %s:%d", path, line_num)
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    def save_jsonl(self, records: list[dict], path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        logger.info("已保存 %d 条数据到 %s", len(records), path)

    def clean_records(self, records: list[dict]) -> list[dict]:
        if not records:
            return []
        items = self._filter_by_score(records)
        items = self._filter_invalid_code(items)
        items = self._filter_by_ast_gate(items)
        items = self._dedup(items)
        items = self._select_top_k(items)
        return self._format_for_sft(items)

    def merge_task_records(self, *, task_records: dict[str, list[dict]], output_path: str, logger_override=None) -> int:
        active_logger = logger_override or logger
        merged: list[dict] = []
        if self._balance_tasks:
            non_empty_counts = sorted(len(records) for records in task_records.values() if records)
            if self._max_per_task is not None:
                cap = self._max_per_task
            else:
                cap = non_empty_counts[len(non_empty_counts) // 2] if non_empty_counts else 0
            if cap <= 0 and non_empty_counts:
                cap = 1
            for task_name, records in task_records.items():
                kept = records[: min(len(records), cap)]
                merged.extend(kept)
                active_logger.info("  任务 %s: 保留 %d / %d 条", task_name, len(kept), len(records))
        else:
            for records in task_records.values():
                merged.extend(records)
        active_logger.info("合并结果: 共 %d 条, 来自 %d 个任务", len(merged), len(task_records))
        self.save_jsonl(merged, output_path)
        return len(merged)

    def _filter_by_score(self, records: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for record in records:
            raw_score = record.get("score")
            if raw_score is None:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if self._score_filter_min is not None and score < self._score_filter_min:
                continue
            if self._score_filter_max is not None and score > self._score_filter_max:
                continue
            filtered.append(record)
        removed = len(records) - len(filtered)
        if removed > 0:
            logger.info("  质量过滤: 移除 %d 条, 保留 %d 条", removed, len(filtered))
        return filtered

    def _filter_invalid_code(self, records: list[dict]) -> list[dict]:
        if not self._filter_invalid_code_enabled:
            return records
        filtered: list[dict] = []
        for record in records:
            code = str(record.get(self._dedup_field) or "")
            if not code.strip():
                continue
            try:
                ast.parse(code)
                filtered.append(record)
                continue
            except SyntaxError:
                pass
            try:
                indented = "\n".join("    " + line for line in code.strip().splitlines())
                ast.parse("def _f():\n" + indented)
                filtered.append(record)
            except SyntaxError:
                continue
        removed = len(records) - len(filtered)
        if removed > 0:
            logger.info("  无效代码过滤: 移除 %d 条(无法 ast 解析), 保留 %d 条", removed, len(filtered))
        return filtered

    def _filter_by_ast_gate(self, records: list[dict]) -> list[dict]:
        if not self._enable_ast_gate:
            return records
        filtered: list[dict] = []
        reject_counts: dict[str, int] = {}
        for record in records:
            code = str(record.get(self._dedup_field) or "")
            if not code.strip():
                reject_counts["empty_code"] = reject_counts.get("empty_code", 0) + 1
                continue
            template_program = None
            if self._ast_gate_use_task_template:
                template_program = self._get_template_program_for_task(str(record.get("task") or ""))
            result = validate_sft_data_code(code, template_program=template_program)
            if (not result.ok) and template_program is not None:
                wrapped = self._wrap_body_as_function(code, template_program)
                if wrapped is not None:
                    result = validate_sft_data_code(wrapped, template_program=template_program)
            if result.ok:
                filtered.append(record)
            else:
                reject_counts[result.reason] = reject_counts.get(result.reason, 0) + 1
        removed = len(records) - len(filtered)
        if removed > 0:
            top = sorted(reject_counts.items(), key=lambda item: item[1], reverse=True)[:5]
            logger.info(
                "  AST门禁过滤: 移除 %d 条, 保留 %d 条; top原因: %s",
                removed,
                len(filtered),
                ", ".join(f"{key}={value}" for key, value in top),
            )
        return filtered

    def _dedup(self, records: list[dict]) -> list[dict]:
        groups: dict[str, dict] = {}
        for record in records:
            code = str(record.get(self._dedup_field) or "")
            dedup_key = self._compute_dedup_key(code)
            best = groups.get(dedup_key)
            score = self._score_value(record)
            if best is None or score > self._score_value(best):
                groups[dedup_key] = record
        deduped = list(groups.values())
        removed = len(records) - len(deduped)
        if removed > 0:
            logger.info("  代码去重(%s): 移除 %d 条重复, 保留 %d 条", self._dedup_strategy, removed, len(deduped))
        else:
            logger.info("  代码去重(%s): 无重复, 保留全部 %d 条", self._dedup_strategy, len(deduped))
        return deduped

    def _select_top_k(self, records: list[dict]) -> list[dict]:
        if not records:
            return []
        if not self._stratified_top_k:
            ordered = sorted(records, key=self._score_value, reverse=True)
            keep = max(1, int(len(ordered) * self._top_ratio))
            selected = ordered[:keep]
            logger.info("  Top-%.0f%%: 从 %d 条中选取 %d 条", self._top_ratio * 100.0, len(ordered), len(selected))
            return selected
        return self._select_top_k_stratified(records)

    def _select_top_k_stratified(self, records: list[dict]) -> list[dict]:
        ordered = sorted(records, key=self._score_value, reverse=True)
        keep = max(1, int(len(ordered) * self._top_ratio))
        novelty_keep = int(round(keep * max(0.0, min(1.0, self._novelty_ratio))))
        quality_keep = max(0, keep - novelty_keep)
        buckets: dict[str, list[dict]] = defaultdict(list)
        for record in ordered:
            code = str(record.get(self._dedup_field) or "")
            buckets[self._compute_ast_struct_key(code)].append(record)
        for items in buckets.values():
            items.sort(key=self._score_value, reverse=True)
        bucket_keys = sorted(buckets, key=lambda key: self._score_value(buckets[key][0]), reverse=True)

        selected: list[dict] = []
        used: set[int] = set()

        def take(bucket_key: str, depth: int) -> dict | None:
            bucket = buckets.get(bucket_key) or []
            if depth >= len(bucket):
                return None
            record = bucket[depth]
            record_id = id(record)
            if record_id in used:
                return None
            used.add(record_id)
            return record

        depth = 0
        while len(selected) < quality_keep:
            progressed = False
            for bucket_key in bucket_keys:
                if len(selected) >= quality_keep:
                    break
                record = take(bucket_key, depth)
                if record is not None:
                    selected.append(record)
                    progressed = True
            if not progressed:
                break
            depth += 1

        remaining: list[tuple[int, float, dict]] = []
        for bucket_key, bucket in buckets.items():
            size = len(bucket)
            for record in bucket:
                if id(record) in used:
                    continue
                remaining.append((size, -self._score_value(record), record))
        remaining.sort(key=lambda item: (item[0], item[1]))
        for _size, _neg_score, record in remaining:
            if len(selected) >= keep or len(selected) >= quality_keep + novelty_keep:
                break
            used.add(id(record))
            selected.append(record)

        logger.info(
            "  StratifiedTopK: N=%d, k=%d (quality=%d, novelty=%d), buckets=%d, AST_unique≈%d",
            len(ordered),
            keep,
            quality_keep,
            novelty_keep,
            len(buckets),
            len(bucket_keys),
        )
        return selected

    def _format_for_sft(self, records: list[dict]) -> list[dict]:
        formatted: list[dict] = []
        for record in records:
            prompt = str(record.get("prompt") or "")
            strategy = str(record.get("strategy") or "")
            response = str(record.get("response") or "")
            output = f"{strategy}\n{response}" if strategy and response else (response or strategy)
            if not prompt or not output:
                continue
            formatted.append(
                {
                    "instruction": prompt,
                    "output": output,
                    "task": record.get("task", ""),
                    "scale": record.get("scale", ""),
                    "score": record.get("score"),
                }
            )
        return formatted

    def _get_template_program_for_task(self, task_name: str) -> str | None:
        task_name = str(task_name or "").strip()
        if not task_name:
            return None
        if task_name in self._template_cache:
            return self._template_cache[task_name]
        module_candidates = []
        explicit = _TASK_TEMPLATE_MODULE_MAP.get(task_name)
        if explicit:
            module_candidates.append(explicit)
        module_candidates.extend(
            [
                f"llm4ad.task.optimization.{task_name}.template",
                f"llm4ad.task.science_discovery.{task_name}.template",
                f"llm4ad.task.machine_learning.{task_name}.template",
            ]
        )
        template_program = None
        for module_name in module_candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            candidate = getattr(module, "template_program", None)
            if isinstance(candidate, str) and candidate.strip():
                template_program = candidate
                break
        if template_program is None:
            for file_path in self._candidate_template_paths(task_name):
                if not os.path.isfile(file_path):
                    continue
                try:
                    namespace: dict[str, object] = {}
                    with open(file_path, "r", encoding="utf-8") as file:
                        exec(compile(file.read(), file_path, "exec"), namespace, namespace)
                except Exception:
                    continue
                candidate = namespace.get("template_program")
                if isinstance(candidate, str) and candidate.strip():
                    template_program = candidate
                    break
        self._template_cache[task_name] = template_program
        return template_program

    @staticmethod
    def _candidate_template_paths(task_name: str) -> list[str]:
        paths = []
        for family, subdir in _TEMPLATE_SEARCH_ROOTS:
            parts = [_PROJECT_ROOT, "llm4ad", "task", family]
            if subdir is not None:
                parts.append(subdir)
            parts.extend([task_name, "template.py"])
            paths.append(os.path.join(*parts))
        return paths

    @staticmethod
    def _wrap_body_as_function(body_code: str, template_program: str) -> str | None:
        try:
            fn_name, arg_names = parse_function_signature_from_template(template_program)
        except Exception:
            return None
        header = f"def {fn_name}({', '.join(arg_names)}):"
        body = "\n".join("    " + line for line in str(body_code or "").splitlines())
        return header + "\n" + (body if body.strip() else "    pass\n")

    def _compute_ast_struct_key(self, code: str) -> str:
        return self._compute_dedup_key(code, strategy_override="ast_struct")

    def _compute_dedup_key(self, code: str, strategy_override: str | None = None) -> str:
        strategy = strategy_override or self._dedup_strategy
        if strategy == "md5":
            return hashlib.md5(code.encode("utf-8")).hexdigest()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            normalized = " ".join(code.split())
            return hashlib.md5(normalized.encode("utf-8")).hexdigest()
        tree = self._strip_docstrings(tree)
        tree = self._clear_locations(tree)
        if strategy == "ast_norm":
            tree = self._normalize_variable_names(tree)
        dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return hashlib.md5(dumped.encode("utf-8")).hexdigest()

    @staticmethod
    def _strip_docstrings(node: ast.AST) -> ast.AST:
        for child in ast.walk(node):
            if not isinstance(child, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            body = getattr(child, "body", [])
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                child.body = body[1:]
        return node

    @staticmethod
    def _clear_locations(node: ast.AST) -> ast.AST:
        for child in ast.walk(node):
            for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
                if hasattr(child, attr):
                    setattr(child, attr, 0)
        return node

    @classmethod
    def _normalize_variable_names(cls, node: ast.AST) -> ast.AST:
        protected = set(dir(builtins)) | {
            "np",
            "numpy",
            "math",
            "range",
            "len",
            "enumerate",
            "zip",
            "sorted",
            "min",
            "max",
            "sum",
            "abs",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "str",
            "bool",
            "None",
            "True",
            "False",
            "print",
        }
        rename_map: dict[str, str] = {}
        counter = [0]

        def map_name(name: str) -> str:
            if name in protected:
                return name
            if name not in rename_map:
                rename_map[name] = f"_v{counter[0]}"
                counter[0] += 1
            return rename_map[name]

        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in child.args.args + child.args.posonlyargs + child.args.kwonlyargs:
                    arg.arg = map_name(arg.arg)
                if child.args.vararg:
                    child.args.vararg.arg = map_name(child.args.vararg.arg)
                if child.args.kwarg:
                    child.args.kwarg.arg = map_name(child.args.kwarg.arg)
            if isinstance(child, ast.Name):
                child.id = map_name(child.id)
            if isinstance(child, ast.arg):
                child.arg = map_name(child.arg)
        return node

    @staticmethod
    def _score_value(record: dict) -> float:
        score = record.get("score")
        try:
            return float(score)
        except (TypeError, ValueError):
            return float("-inf")
