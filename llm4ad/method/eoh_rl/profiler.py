from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any


def _fmt(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "None"


class EoHProfiler:
    """EoH-RL 最小日志器：只保留恢复、分析日志和样本审计需要的信息。"""

    def __init__(self, log_dir: str | None = None, *, log_style: str = "complex", create_random_path: bool = False, **_):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_dir = os.path.join(log_dir or ".", timestamp) if create_random_path else log_dir
        self._log_style = log_style
        self._num_samples = self._evaluate_success_program_num = self._evaluate_failed_program_num = 0
        self._cur_best_program_score, self._samples_json_dir, self._population_dir = float("-inf"), "", ""
        self._cur_gen = -1
        if self._log_dir:
            self._samples_json_dir = os.path.join(self._log_dir, "samples")
            self._population_dir = os.path.join(self._log_dir, "population")
            for path in (self._samples_json_dir, self._population_dir):
                os.makedirs(path, exist_ok=True)

    @staticmethod
    def _safe_run_id(value: str) -> str:
        run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
        if not run_id:
            raise ValueError("run_id 不能为空")
        return run_id

    @classmethod
    def create_run_context(cls, base_dir: str, *, run_id: str | None = None, run_log_dir: str | None = None, allow_existing: bool = False) -> dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        explicit_dir = str(run_log_dir or os.environ.get("EOH_RL_RUN_LOG_DIR", "")).strip()
        explicit_id = str(run_id or os.environ.get("EOH_RL_RUN_ID", "")).strip()
        if explicit_dir:
            run_log_dir, source = os.path.abspath(os.path.expanduser(explicit_dir)), "explicit_log_dir"
            run_id = os.path.basename(os.path.normpath(run_log_dir)) or timestamp
        else:
            run_id = cls._safe_run_id(explicit_id) if explicit_id else timestamp
            run_log_dir, source = os.path.join(os.path.abspath(base_dir), run_id), "explicit_run_id" if explicit_id else "timestamp"
        os.makedirs(run_log_dir, exist_ok=bool(allow_existing and (explicit_dir or explicit_id)))
        return {"timestamp": timestamp, "run_id": run_id, "run_log_dir": run_log_dir, "run_dir_source": source}

    @staticmethod
    def write_run_manifest(run_log_dir: str, payload: dict[str, Any], *, external_path: str | None = None) -> dict[str, Any]:
        manifest = {**dict(payload or {}), "run_log_dir": run_log_dir, "pid": os.getpid(), "updated_at": datetime.now().isoformat()}
        external = external_path or os.environ.get("EOH_RL_RUN_MANIFEST_PATH", "").strip()
        for path in [os.path.join(run_log_dir, "run_manifest.json")] + ([os.path.abspath(os.path.expanduser(external))] if external else []):
            EoHProfiler._write_json(path, manifest)
        return manifest

    def record_parameters(self, llm, evaluation, method) -> None:
        if not self._log_dir:
            return
        max_generations = getattr(method, "_max_generations", None)
        if max_generations is None:
            max_generations = getattr(method, "_max_grpo_updates", None)
        self._append_run_log({"llm": llm.__class__.__name__, "problem": evaluation.__class__.__name__, "method": "EoHRL", "max_generations": max_generations, "max_sample_nums": getattr(method, "_max_sample_nums", None), "pop_size": getattr(method, "_pop_size", None), "samples_per_prompt": getattr(method, "_samples_per_prompt", None), "grpo": getattr(method, "_grpo_config", None)})

    def register_function(
        self,
        function,
        program: str = "",
        *,
        resume_mode: bool = False,
        source: str | None = None,
        record_best: bool = True,
    ) -> None:
        self._num_samples += 1
        score = getattr(function, "score", None)
        self._evaluate_success_program_num += int(score is not None)
        self._evaluate_failed_program_num += int(score is None)
        if record_best and score is not None and float(score) > self._cur_best_program_score:
            self._cur_best_program_score = float(score)
            if not resume_mode:
                self._write_sample(function, program, source=source, record_type="best")
        if not resume_mode:
            self._write_sample(function, program, source=source, record_type="history")
            if self._log_style == "complex":
                sample_source = self._display_source(function, source)
                print(
                    "[EoHRL:candidate] "
                    f"sample={self._num_samples} "
                    f"source={sample_source} "
                    f"op={getattr(function, 'operator', None)} "
                    f"score={_fmt(getattr(function, 'score', None))} "
                    f"best={_fmt(self._cur_best_program_score)} "
                    f"eval={_fmt(getattr(function, 'evaluate_time', None), 2)}s"
                )

    def register_population(self, pop) -> None:
        generation = int(getattr(pop, "generation", 0) or 0)
        if not self._log_dir or generation == self._cur_gen:
            return
        self._cur_gen = generation
        self._write_json(os.path.join(self._population_dir, f"pop_{generation}.json"), [self._serialize_function(func) for func in getattr(pop, "population", []) or []])

    def finish(self) -> None:
        return None

    def _write_sample(self, function, program: str, *, source: str | None, record_type: str) -> None:
        if not self._log_dir:
            return
        sample_order = int(getattr(function, "_eoh_sample_order", None) or self._num_samples)
        payload = self._serialize_sample(function, sample_order=sample_order, source=source)
        lower = ((sample_order - 1) // 200) * 200 + 1
        filename = "samples_best.json" if record_type == "best" else f"samples_{lower}~{lower + 199}.json"
        path = os.path.join(self._samples_json_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as file:
                rows = json.load(file)
        except Exception:
            rows = []
        rows.append(payload)
        self._write_json(path, rows)

    @staticmethod
    def _serialize_sample(func, *, sample_order: int, source: str | None) -> dict:
        return {
            "sample_order": sample_order,
            "algorithm": getattr(func, "algorithm", "") or "",
            "function": str(func),
            "score": getattr(func, "score", None),
            "program": "",
            "parent_id": getattr(func, "_eoh_parent_id", None),
            "op": getattr(func, "_eoh_op", None) or getattr(func, "operator", None),
            "source": EoHProfiler._display_source(func, source),
            "eval_diag": EoHProfiler._clean_eval_diag(getattr(func, "_eval_diag", None)),
        }

    @staticmethod
    def _serialize_function(func) -> dict:
        payload = {key: value for key, value in {"algorithm": getattr(func, "algorithm", ""), "function": str(func), "score": getattr(func, "score", None), "evaluate_time": getattr(func, "evaluate_time", None), "sample_time": getattr(func, "sample_time", None), "operator": getattr(func, "operator", None)}.items()}
        for attr, key in (("_eoh_sample_order", "sample_order"), ("_eoh_parent_id", "parent_id"), ("_eoh_op", "op"), ("_eoh_source", "source"), ("_eoh_birth_sample_order", "birth_sample_order"), ("_eoh_birth_generation", "birth_generation"), ("_eoh_lineage_tag", "lineage_tag")):
            value = getattr(func, attr, None)
            if value is not None:
                payload[key] = value
        diag = getattr(func, "_eval_diag", None)
        if isinstance(diag, dict):
            payload["eval_diag"] = EoHProfiler._clean_eval_diag(diag)
        return payload

    @staticmethod
    def _display_source(func, source: str | None) -> str:
        raw = getattr(func, "_eoh_source", None) or source or "local"
        return "local" if str(raw).strip().lower() in {"initialization", "grpo", "resident", "resident_grpo"} else str(raw)

    @staticmethod
    def _clean_eval_diag(diag) -> dict | None:
        if not isinstance(diag, dict):
            return None
        blocked = {"performance_profile", "profile_score"}
        cleaned = {k: v for k, v in diag.items() if k not in blocked}
        if "ok" not in cleaned and "bucket" not in cleaned and "error_type" not in cleaned:
            return None
        return {
            "ok": bool(cleaned.get("ok", False)),
            "bucket": cleaned.get("bucket"),
            "error_type": cleaned.get("error_type"),
            **{k: v for k, v in cleaned.items() if k not in {"ok", "bucket", "error_type"}},
        }

    def _append_run_log(self, payload: dict[str, Any]) -> None:
        path = os.path.join(self._log_dir, "run_log.txt")
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, default=str, indent=2) + "\n")

    @staticmethod
    def _write_json(path: str, payload) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)


__all__ = ["EoHProfiler"]
