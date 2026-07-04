from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK_REGISTRY: dict[str, dict[str, str]] = {
    "tsp": {
        "config_name": "eoh_online_tsp.yaml",
        "display_name": "TSP",
        "task_key": "tsp",
        "task_name": "tsp_construct",
        "evaluation_module": "llm4ad.task.optimization.tsp_construct",
        "evaluation_class": "TSPEvaluation",
        "logger_name": "run_eoh_online_tsp",
    },
    "cvrp": {
        "config_name": "eoh_online_cvrp.yaml",
        "display_name": "CVRP",
        "task_key": "cvrp",
        "task_name": "cvrp_construct",
        "evaluation_module": "llm4ad.task.optimization.cvrp_construct",
        "evaluation_class": "CVRPEvaluation",
        "logger_name": "run_eoh_online_cvrp",
    },
    "jssp": {
        "config_name": "eoh_online_jssp.yaml",
        "display_name": "JSSP",
        "task_key": "jssp",
        "task_name": "jssp_construct",
        "evaluation_module": "llm4ad.task.optimization.jssp_construct",
        "evaluation_class": "JSSPEvaluation",
        "logger_name": "run_eoh_online_jssp",
    },
    "obp": {
        "config_name": "eoh_online_obp.yaml",
        "display_name": "OBP",
        "task_key": "obp",
        "task_name": "online_bin_packing",
        "evaluation_module": "llm4ad.task.optimization.online_bin_packing",
        "evaluation_class": "OBPEvaluation",
        "logger_name": "run_eoh_online_obp",
    },
}

METHOD_REGISTRY: dict[str, dict[str, str]] = {
    "eoh": {
        "display_name": "EoH",
        "method_module": "llm4ad.method.eoh",
        "method_class": "EoH",
        "profiler_module": "llm4ad.method.eoh.profiler",
        "profiler_class": "EoHProfiler",
        "legacy_params_key": "evolution",
    },
    "funsearch": {
        "display_name": "FunSearch",
        "method_module": "llm4ad.method.funsearch",
        "method_class": "FunSearch",
        "profiler_module": "llm4ad.method.funsearch.profiler",
        "profiler_class": "FunSearchProfiler",
        "legacy_params_key": "funsearch",
    },
    "mcts_ahd": {
        "display_name": "MCTS_AHD",
        "method_module": "llm4ad.method.mcts_ahd",
        "method_class": "MCTS_AHD",
        "profiler_module": "llm4ad.method.mcts_ahd.profiler",
        "profiler_class": "MAProfiler",
        "legacy_params_key": "mcts_ahd",
    },
    "reevo": {
        "display_name": "ReEvo",
        "method_module": "llm4ad.method.reevo",
        "method_class": "ReEvo",
        "profiler_module": "llm4ad.method.reevo.profiler",
        "profiler_class": "ReEvoProfiler",
        "legacy_params_key": "reevo",
    },
}

METHOD_PARAM_OWNERS: dict[str, set[str]] = {
    "max_generations": {"eoh"},
    "use_e2_operator": {"eoh"},
    "use_m1_operator": {"eoh"},
    "use_m2_operator": {"eoh"},
    "init_size": {"mcts_ahd"},
    "alpha": {"mcts_ahd"},
    "lambda_0": {"mcts_ahd"},
    "mutation_rate": {"reevo"},
    "samples_per_prompt": {"funsearch"},
    "pop_size": {"eoh", "mcts_ahd", "reevo"},
    "selection_num": {"eoh", "mcts_ahd"},
}


def run_from_config(*, config_path: str, task_family: str, dry_run: bool = False) -> None:
    spec = _task_spec(task_family)
    raw_root = _apply_runtime_overrides(_load_yaml_root(config_path))
    cfg = _load_online_run_config(raw_root, config_path)
    _validate_pure_online_config(cfg)

    model_name, model_config_path, model_cfg = _load_selected_model_config(cfg)
    task_name, scale_name, task_params, scale_method_overrides = _resolve_task_config(cfg, spec)
    method_name, method_spec, method_params = _resolve_method_config(cfg)
    method_params = _apply_scale_method_overrides(method_name, method_params, scale_method_overrides)
    _validate_method_params(method_name, method_params)
    run_context = _create_run_context(dict(_require(cfg, "logging")), model_cfg, spec, method_name)

    resolved_preview = {
        "config_path": os.path.abspath(config_path),
        "task_family": task_family,
        "method_name": method_name,
        "method_params": method_params,
        "model_config_path": model_config_path,
        "online_model": _public_model_snapshot(model_cfg),
        "task_name": task_name,
        "scale_name": scale_name,
        "task_params": task_params,
        "run_context": run_context,
        "api_key_configured": _api_key_configured(model_cfg),
    }
    if dry_run:
        print(json.dumps(resolved_preview, indent=2, ensure_ascii=False, default=str))
        return

    _resolve_api_key(model_cfg)
    os.makedirs(run_context["run_log_dir"], exist_ok=False)
    logger_name = f"run_{method_name}_online_{task_family}"
    _configure_logging(run_context["run_log_dir"], logger_name)
    logger = logging.getLogger(logger_name)

    resolved_cfg = _sanitize_secrets(dict(raw_root))
    resolved_cfg["_selected_online_model"] = {
        "name": model_name,
        "config_path": model_config_path,
        "config": _sanitize_secrets(model_cfg),
    }
    _write_yaml(os.path.join(run_context["run_log_dir"], "config_resolved.yaml"), resolved_cfg)

    config_snapshot = _build_config_snapshot(
        config_path=config_path,
        model_config_path=model_config_path,
        model_cfg=model_cfg,
        method_name=method_name,
        method_params=method_params,
        task_name=task_name,
        scale_name=scale_name,
        task_params=task_params,
        run_context=run_context,
    )
    _write_json(os.path.join(run_context["run_log_dir"], "config.json"), config_snapshot)
    _write_run_manifest(run_context["run_log_dir"], {**config_snapshot, "status": "starting"})

    logger.info("============================================================")
    logger.info("在线流程: %s + %s", method_spec["display_name"], spec["display_name"])
    logger.info("配置来源: %s", os.path.abspath(config_path))
    logger.info("在线模型: provider=%s model=%s backend=%s",
                model_cfg.get("provider"), model_cfg.get("model"), model_cfg.get("backend"))
    logger.info("方法参数: %s", method_params)
    logger.info("规模: %s, 参数: %s", scale_name, task_params)
    logger.info("日志目录: %s", run_context["run_log_dir"])
    logger.info("============================================================")

    _ensure_writable_mplconfig()
    llm = _build_online_llm(model_cfg)
    task = _build_evaluation(spec, task_params)
    profiler = _build_profiler(method_spec, run_context["run_log_dir"], cfg)
    method = _build_method(
        method_spec=method_spec,
        llm=llm,
        profiler=profiler,
        evaluation=task,
        method_params=method_params,
    )
    _write_effective_runtime(run_context["run_log_dir"], method, model_cfg, method_name, method_params)

    status = "completed"
    try:
        method.run()
        status = _classify_run_status(method, profiler, method_params)
    except Exception:
        status = "failed"
        logger.exception("在线 %s 运行失败", method_spec["display_name"])
        raise
    finally:
        run_summary = _build_run_summary(
            run_log_dir=run_context["run_log_dir"],
            method=method,
            profiler=profiler,
            model_cfg=model_cfg,
            method_name=method_name,
            status=status,
        )
        _write_json(os.path.join(run_context["run_log_dir"], "run_summary.json"), run_summary)
        _write_run_manifest(
            run_context["run_log_dir"],
            {**config_snapshot, "status": status, "runtime_summary": run_summary},
        )
        logger.info("本次运行概要: %s", run_summary)


def _load_yaml_root(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件根节点必须是 dict: {path}")
    return data


def _load_online_run_config(root: dict[str, Any], path: str) -> dict[str, Any]:
    if "online_run" in root and isinstance(root["online_run"], dict):
        return dict(root["online_run"])
    if "eoh_online" in root and isinstance(root["eoh_online"], dict):
        # 兼容早期纯 EoH 在线配置；新配置统一使用 online_run.method。
        cfg = dict(root["eoh_online"])
        cfg.setdefault("method", {"name": "eoh", "params": dict(cfg.get("evolution") or {})})
        return cfg
    raise ValueError(f"配置文件缺少 online_run 根字段: {path}")


def _apply_runtime_overrides(root: dict[str, Any]) -> dict[str, Any]:
    text = os.environ.get("EOH_ONLINE_RUNTIME_OVERRIDES_JSON", "").strip()
    if not text:
        return root
    try:
        overrides = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("EOH_ONLINE_RUNTIME_OVERRIDES_JSON 必须是合法 JSON") from exc
    if not isinstance(overrides, dict):
        raise ValueError("EOH_ONLINE_RUNTIME_OVERRIDES_JSON 根节点必须是 dict")
    overrides = _align_runtime_override_root(root, overrides)
    return _merge_runtime_overrides(dict(root), overrides)


def _align_runtime_override_root(root: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    aligned = dict(overrides)
    if "online_run" in root and "online_run" not in aligned and "eoh_online" in aligned:
        aligned["online_run"] = aligned.pop("eoh_online")
    elif "eoh_online" in root and "eoh_online" not in aligned and "online_run" in aligned:
        aligned["eoh_online"] = aligned.pop("online_run")
    return aligned


def _merge_runtime_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = _deep_merge_dict(base, overrides)
    for root_key in ("online_run", "eoh_online"):
        base_section = base.get(root_key)
        override_section = overrides.get(root_key)
        result_section = result.get(root_key)
        if not all(isinstance(item, dict) for item in (base_section, override_section, result_section)):
            continue
        base_method = _section_method_name(base_section)
        override_method = _section_method_name(override_section, default=base_method)
        if override_method is None or override_method == base_method:
            continue
        method_cfg = result_section.get("method")
        if not isinstance(method_cfg, dict):
            method_cfg = {"name": override_method}
        replacement_params = _replacement_method_params(override_section, override_method)
        method_cfg["name"] = override_method
        method_cfg["params"] = replacement_params
        result_section["method"] = method_cfg
    return result


def _section_method_name(section: dict[str, Any], default: str | None = "eoh") -> str | None:
    method_cfg = section.get("method")
    if isinstance(method_cfg, str):
        return _normalize_method_name(method_cfg)
    if isinstance(method_cfg, dict):
        return _normalize_method_name(method_cfg.get("name") or default or "eoh")
    if "evolution" in section:
        return "eoh"
    return default


def _replacement_method_params(section: dict[str, Any], method_name: str) -> dict[str, Any]:
    method_cfg = section.get("method")
    if isinstance(method_cfg, dict) and "params" in method_cfg:
        return dict(method_cfg.get("params") or {})
    legacy_key = METHOD_REGISTRY[method_name]["legacy_params_key"]
    return dict(section.get(legacy_key) or {})


def _deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _validate_pure_online_config(cfg: dict[str, Any]) -> None:
    llm_mode = str(cfg.get("llm_mode", "online")).lower()
    if llm_mode != "online":
        raise ValueError("online_run.llm_mode 必须为 online")
    forbidden = [
        "rl",
        "grpo",
        "lora",
        "sft",
        "adaptive_operator",
        "collapse",
        "vllm",
        "local_model_path",
        "inference_backend",
    ]
    present = [key for key in forbidden if key in cfg]
    if present:
        raise ValueError("纯在线方法配置不能包含 RL/本地训练字段: " + ", ".join(present))


def _load_selected_model_config(cfg: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    online_model = _require(cfg, "online_model")
    if isinstance(online_model, str):
        model_name = online_model
        model_overrides: dict[str, Any] = {}
        model_config_path = None
    elif isinstance(online_model, dict):
        model_name = str(_require(online_model, "name"))
        model_overrides = dict(online_model.get("overrides") or {})
        model_config_path = online_model.get("config_path")
    else:
        raise ValueError("online_run.online_model 必须是字符串或 dict")

    if model_config_path:
        path = _resolve_path(model_config_path)
    else:
        config_dir = cfg.get("online_model_config_dir") or "configs/run_eoh/online_models"
        path = _resolve_path(os.path.join(str(config_dir), f"{model_name}.yaml"))
    model_root = _load_yaml_root(path)
    model_cfg = model_root.get("online_model", model_root)
    if not isinstance(model_cfg, dict):
        raise ValueError(f"在线模型配置根节点必须是 dict: {path}")
    merged = _deep_merge_dict(dict(model_cfg), model_overrides)
    merged.setdefault("name", model_name)
    merged.setdefault("provider", model_name)
    merged.setdefault("backend", "openai_compatible")
    return model_name, os.path.abspath(path), merged


def _resolve_method_config(cfg: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    method_cfg = cfg.get("method") or {"name": "eoh", "params": dict(cfg.get("evolution") or {})}
    if isinstance(method_cfg, str):
        method_name = _normalize_method_name(method_cfg)
        legacy_key = METHOD_REGISTRY[method_name]["legacy_params_key"]
        params = dict(cfg.get(legacy_key) or {})
    elif isinstance(method_cfg, dict):
        method_name = str(method_cfg.get("name") or "eoh")
        params = dict(method_cfg.get("params") or {})
        if not params:
            normalized = _normalize_method_name(method_name)
            legacy_key = METHOD_REGISTRY[normalized]["legacy_params_key"]
            params = dict(cfg.get(legacy_key) or {})
    else:
        raise ValueError("online_run.method 必须是字符串或 dict")
    method_name = _normalize_method_name(method_name)
    method_spec = METHOD_REGISTRY[method_name]
    if method_name == "eoh" and "max_generations" not in params and "evolution" in cfg:
        params = dict(cfg["evolution"])
    return method_name, method_spec, params


def _apply_scale_method_overrides(
        method_name: str,
        method_params: dict[str, Any],
        scale_method_overrides: dict[str, Any],
) -> dict[str, Any]:
    params = dict(method_params)
    if method_name in {"eoh", "mcts_ahd", "reevo"} and "pop_size" in scale_method_overrides:
        params["pop_size"] = scale_method_overrides["pop_size"]
    return params


def _normalize_method_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "mcts": "mcts_ahd",
        "mctsahd": "mcts_ahd",
        "mcts_ahd": "mcts_ahd",
        "reevo": "reevo",
        "re_evo": "reevo",
        "fun_search": "funsearch",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in METHOD_REGISTRY:
        raise ValueError(f"不支持的在线方法: {name}，可选: {', '.join(sorted(METHOD_REGISTRY))}")
    return normalized


def _validate_method_params(method_name: str, params: dict[str, Any]) -> None:
    if not isinstance(params, dict):
        raise ValueError(f"{method_name}.params 必须是 dict")
    common_required = ["max_sample_nums", "num_samplers", "num_evaluators"]
    missing = [key for key in common_required if key not in params]
    if missing:
        raise ValueError(f"{method_name}.params 缺少字段: {', '.join(missing)}")
    if method_name in {"eoh", "mcts_ahd", "reevo"} and "pop_size" not in params:
        raise ValueError(f"{method_name}.params 缺少字段: pop_size")
    if method_name == "eoh" and "max_generations" not in params:
        raise ValueError("eoh.params 缺少字段: max_generations")
    stale_params = sorted(
        key for key in params
        if key in METHOD_PARAM_OWNERS and method_name not in METHOD_PARAM_OWNERS[key]
    )
    if stale_params:
        raise ValueError(
            f"{method_name}.params 包含其他方法的残留字段: {', '.join(stale_params)}，"
            "请为当前 method.name 重写完整 params"
        )


def _resolve_task_config(cfg: dict[str, Any], spec: dict[str, str]) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    task_cfg = dict(_require(cfg, spec["task_key"]))
    task_name = str(task_cfg.get("task_name", spec["task_name"]))
    if task_name != spec["task_name"]:
        raise ValueError(f"{spec['display_name']} 配置中的 task_name 应为 {spec['task_name']}，当前为: {task_name}")
    scale_name = str(_require(task_cfg, "default_scale"))
    scales = dict(_require(task_cfg, "scales"))
    scale_cfg = dict(_require(scales, scale_name))
    scale_method_overrides: dict[str, Any] = {}
    if "pop_size" in scale_cfg:
        scale_method_overrides["pop_size"] = int(scale_cfg["pop_size"])
    task_params = {key: value for key, value in scale_cfg.items() if key != "pop_size"}
    return task_name, scale_name, task_params, scale_method_overrides


def _create_run_context(
        logging_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        spec: dict[str, str],
        method_name: str,
) -> dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    explicit_dir = str(os.environ.get("EOH_ONLINE_RUN_LOG_DIR") or logging_cfg.get("run_log_dir") or "").strip()
    explicit_id = str(os.environ.get("EOH_ONLINE_RUN_ID") or logging_cfg.get("run_id") or "").strip()
    allow_existing = bool(logging_cfg.get("allow_existing_run_dir", False))

    if explicit_dir:
        run_log_dir = os.path.abspath(os.path.expanduser(explicit_dir))
        run_id = os.path.basename(os.path.normpath(run_log_dir)) or timestamp
        source = "explicit_log_dir"
    else:
        log_base_dir = _resolve_path(logging_cfg.get("log_base_dir") or "logs")
        base_dir = os.path.join(log_base_dir, spec["display_name"], f"{method_name}_online")
        if explicit_id:
            run_id = _safe_run_id(explicit_id)
            source = "explicit_run_id"
        else:
            run_name = (
                logging_cfg.get("run_name")
                or f"online_{method_name}_{model_cfg.get('provider', 'model')}_np1_{spec['display_name']}_run1"
            )
            run_id = f"{_safe_run_id(run_name)}_{timestamp}"
            source = "timestamp"
        run_log_dir = os.path.join(base_dir, run_id)

    if os.path.exists(run_log_dir) and not allow_existing:
        raise FileExistsError(f"日志目录已存在，如需复用请设置 logging.allow_existing_run_dir=true: {run_log_dir}")
    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "run_log_dir": run_log_dir,
        "run_dir_source": source,
    }


def _build_evaluation(spec: dict[str, str], task_params: dict[str, Any]):
    module = importlib.import_module(spec["evaluation_module"])
    evaluation_cls = getattr(module, spec["evaluation_class"])
    return evaluation_cls(**task_params)


def _build_profiler(method_spec: dict[str, str], run_log_dir: str, cfg: dict[str, Any]):
    profiler_module = importlib.import_module(method_spec["profiler_module"])
    profiler_cls = getattr(profiler_module, method_spec["profiler_class"])
    profiler_cfg = dict(cfg.get("profiler") or {})
    profiler_cfg.setdefault("log_style", "complex")
    return profiler_cls(log_dir=run_log_dir, create_random_path=False, **profiler_cfg)


def _build_method(*, method_spec: dict[str, str], llm, profiler, evaluation, method_params: dict[str, Any]):
    method_module = importlib.import_module(method_spec["method_module"])
    method_cls = getattr(method_module, method_spec["method_class"])
    return method_cls(llm=llm, profiler=profiler, evaluation=evaluation, **method_params)


def _build_online_llm(model_cfg: dict[str, Any]):
    backend = str(model_cfg.get("backend", "openai_compatible")).lower()
    api_key = _resolve_api_key(model_cfg)
    generation_kwargs = dict(model_cfg.get("generation") or {})
    timeout = float(model_cfg.get("timeout", 120))
    if backend in {"openai_compatible", "openai"}:
        from llm4ad.tools.llm.llm_api_openai import OpenAIAPI

        llm = OpenAIAPI(
            base_url=str(_require(model_cfg, "base_url")).rstrip("/"),
            api_key=api_key,
            model=str(_require(model_cfg, "model")),
            timeout=timeout,
            generation_kwargs=generation_kwargs,
            client_kwargs=dict(model_cfg.get("client") or {}),
        )
    elif backend in {"https", "https_legacy"}:
        from llm4ad.tools.llm.llm_api_https import HttpsApi

        llm = HttpsApi(
            host=str(_require(model_cfg, "host")),
            key=api_key,
            model=str(_require(model_cfg, "model")),
            timeout=timeout,
            **generation_kwargs,
        )
    else:
        raise ValueError(f"不支持的在线模型 backend: {backend}")
    # 日志中只保留可公开元信息，不保存 API key。
    llm._provider = model_cfg.get("provider")
    llm._api_key_env = model_cfg.get("api_key_env")
    return llm


def _resolve_api_key(model_cfg: dict[str, Any]) -> str:
    api_key = str(model_cfg.get("api_key") or "").strip()
    env_name = str(model_cfg.get("api_key_env") or "").strip()
    if not api_key and env_name:
        api_key = str(os.environ.get(env_name, "")).strip()
    if not api_key:
        hint = f"环境变量 {env_name}" if env_name else "online model 配置中的 api_key"
        raise ValueError(f"在线模型 API key 为空，请设置 {hint}")
    return api_key


def _ensure_writable_mplconfig() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    path = os.path.join("/tmp", "llm4ad_mplconfig")
    os.makedirs(path, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = path


def _api_key_configured(model_cfg: dict[str, Any]) -> bool:
    env_name = str(model_cfg.get("api_key_env") or "").strip()
    return bool(str(model_cfg.get("api_key") or "").strip() or (env_name and os.environ.get(env_name)))


def _build_config_snapshot(
        *,
        config_path: str,
        model_config_path: str,
        model_cfg: dict[str, Any],
        method_name: str,
        method_params: dict[str, Any],
        task_name: str,
        scale_name: str,
        task_params: dict[str, Any],
        run_context: dict[str, str],
) -> dict[str, Any]:
    return {
        "script": os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        "entry_script": os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        "config_path": os.path.abspath(config_path),
        "llm_mode": "online",
        "online_model_provider": model_cfg.get("provider"),
        "online_model_name": model_cfg.get("model"),
        "online_model_backend": model_cfg.get("backend"),
        "online_model_config_path": model_config_path,
        "online_model_base_url": model_cfg.get("base_url"),
        "method_name": method_name,
        "method_params": method_params,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "task_for_evolve": task_name,
        "task_name": task_name,
        "scale_for_evolve": scale_name,
        "scale_name": scale_name,
        "scale_params": task_params,
        "task_params": task_params,
        "timestamp": run_context["timestamp"],
        "run_id": run_context["run_id"],
        "run_log_dir": run_context["run_log_dir"],
        "run_dir_source": run_context["run_dir_source"],
        "detail_log_path": os.path.join(run_context["run_log_dir"], "run_log.txt"),
    }


def _write_effective_runtime(run_log_dir: str, method, model_cfg: dict[str, Any], method_name: str, method_params: dict[str, Any]) -> None:
    payload = {
        "llm_mode": "online",
        "method_name": method_name,
        "method_params": method_params,
        "llm_backend": model_cfg.get("backend"),
        "online_model_provider": model_cfg.get("provider"),
        "online_model_name": model_cfg.get("model"),
        "num_samplers_effective": int(getattr(method, "_num_samplers", 0)),
        "num_evaluators_effective": int(getattr(method, "_num_evaluators", 0)),
        "samples_per_prompt": int(getattr(method, "_samples_per_prompt", 1)),
        "max_generations": _safe_int(getattr(method, "_max_generations", None)),
        "max_sample_nums": _safe_int(getattr(method, "_max_sample_nums", None)),
        "pop_size": _safe_int(getattr(method, "_pop_size", None)),
        "selection_num": _safe_int(getattr(method, "_selection_num", None)),
        "multi_thread_or_process_eval": getattr(method, "_multi_thread_or_process_eval", None),
        "adaptive_operator_enabled": False,
        "collapse_enabled": False,
        "rl_enabled": False,
    }
    _write_json(os.path.join(run_log_dir, "effective_runtime.json"), payload)


def _build_run_summary(
        *,
        run_log_dir: str,
        method,
        profiler,
        model_cfg: dict[str, Any],
        method_name: str,
        status: str,
) -> dict[str, Any]:
    return {
        "method_name": method_name,
        "samples_per_prompt": int(getattr(method, "_samples_per_prompt", 1)),
        "max_generations": _safe_int(getattr(method, "_max_generations", None)),
        "max_sample_nums": _safe_int(getattr(method, "_max_sample_nums", None)),
        "pop_size": _safe_int(getattr(method, "_pop_size", None)),
        "selection_num": _safe_int(getattr(method, "_selection_num", None)),
        "llm_mode": "online",
        "online_model_provider": model_cfg.get("provider"),
        "online_model_name": model_cfg.get("model"),
        "status": status,
        "total_samples": int(getattr(profiler, "_num_samples", 0)),
        "success_samples": int(getattr(profiler, "_evaluate_success_program_num", 0)),
        "failed_samples": int(getattr(profiler, "_evaluate_failed_program_num", 0)),
        "best_score": _jsonable_score(getattr(profiler, "_cur_best_program_score", None)),
        "best_sample_order": getattr(profiler, "_cur_best_program_sample_order", None),
        "total_sample_time": float(getattr(profiler, "_tot_sample_time", 0.0)),
        "total_evaluate_time": float(getattr(profiler, "_tot_evaluate_time", 0.0)),
        "population_generation": int(getattr(getattr(method, "_population", None), "generation", 0)),
        "population_size": int(len(getattr(method, "_population", []))),
        "program_database_samples": _program_database_size(method),
        "best_curve": _collect_best_curve(run_log_dir),
    }


def _classify_run_status(method, profiler, method_params: dict[str, Any]) -> str:
    profiler_samples = int(getattr(profiler, "_num_samples", 0))
    budget_samples = _safe_int(getattr(method, "_tot_sample_nums", None))
    observed_samples = budget_samples if budget_samples is not None else profiler_samples
    population = getattr(method, "_population", None)
    generation = int(getattr(population, "generation", 0)) if population is not None else 0
    population_size = len(population) if population is not None else 0
    selection_num = _safe_int(getattr(method, "_selection_num", None))
    if population is not None and generation == 0 and (
            population_size == 0 or (selection_num is not None and population_size < selection_num)):
        return "failed_initialization"
    if observed_samples == 0 and profiler_samples == 0:
        return "failed_no_samples"
    target_samples = _safe_int(method_params.get("max_sample_nums"))
    if target_samples is not None and observed_samples >= target_samples:
        return "completed"
    return "early_terminated"


def _program_database_size(method) -> int | None:
    database = getattr(method, "_database", None)
    islands = getattr(database, "islands", None)
    if islands is None:
        return None
    total = 0
    for island in islands:
        clusters = getattr(island, "clusters", {})
        for cluster in clusters.values():
            total += len(getattr(cluster, "programs", []))
    return total


def _collect_best_curve(run_log_dir: str) -> list[float | None]:
    samples_dir = os.path.join(run_log_dir, "samples")
    if not os.path.isdir(samples_dir):
        return []
    files = [
        name for name in os.listdir(samples_dir)
        if name.startswith("samples_") and name != "samples_best.json" and name.endswith(".json")
    ]
    files.sort(key=_sample_file_start)
    curve: list[float | None] = []
    best = float("-inf")
    for name in files:
        path = os.path.join(samples_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as file:
                rows = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            score = row.get("score") if isinstance(row, dict) else None
            if isinstance(score, (int, float)):
                best = max(best, float(score))
            curve.append(None if best == float("-inf") else best)
    return curve


def _sample_file_start(name: str) -> int:
    match = re.search(r"samples_(\d+)~", name)
    return int(match.group(1)) if match else 0


def _write_run_manifest(run_log_dir: str, payload: dict[str, Any]) -> None:
    manifest = dict(payload)
    manifest.setdefault("run_log_dir", run_log_dir)
    manifest["pid"] = os.getpid()
    manifest["updated_at"] = datetime.now().isoformat()
    _write_json(os.path.join(run_log_dir, "run_manifest.json"), manifest)


def _configure_logging(run_log_dir: str, logger_name: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(run_log_dir, "run.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.getLogger(logger_name).setLevel(logging.INFO)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def _write_yaml(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


def _public_model_snapshot(model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": model_cfg.get("name"),
        "provider": model_cfg.get("provider"),
        "backend": model_cfg.get("backend"),
        "model": model_cfg.get("model"),
        "base_url": model_cfg.get("base_url"),
        "host": model_cfg.get("host"),
        "api_key_env": model_cfg.get("api_key_env"),
        "timeout": model_cfg.get("timeout"),
        "generation": model_cfg.get("generation"),
    }


def _sanitize_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"api_key", "key", "authorization", "access_token", "secret", "secret_key"}:
                clean[key] = "***REDACTED***" if item else item
            else:
                clean[key] = _sanitize_secrets(item)
        return clean
    if isinstance(value, list):
        return [_sanitize_secrets(item) for item in value]
    return value


def _jsonable_score(value: Any) -> Any:
    if isinstance(value, (int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_score(item) for item in value]
    return str(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_path(path: str | os.PathLike[str]) -> str:
    text = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(text):
        return text
    return os.path.abspath(PROJECT_ROOT / text)


def _safe_run_id(value: str) -> str:
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    run_id = run_id.strip("._-")
    if not run_id:
        raise ValueError("run_id 不能为空或仅包含非法路径字符")
    return run_id


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"配置缺少字段: {key}")
    return mapping[key]


def _task_spec(task_family: str) -> dict[str, str]:
    if task_family not in TASK_REGISTRY:
        raise ValueError(f"不支持的 task_family: {task_family}")
    return TASK_REGISTRY[task_family]


def _default_config_path(task_family: str) -> str:
    return str(PROJECT_ROOT / "configs" / "run_eoh" / _task_spec(task_family)["config_name"])


def main(task_family: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a pure online LLM4AD method.")
    if task_family is None:
        parser.add_argument(
            "--task-family",
            choices=sorted(TASK_REGISTRY),
            default=os.environ.get("EOH_ONLINE_TASK_FAMILY", "tsp"),
            help="Task family to run.",
        )
    parser.add_argument(
        "--config",
        default=os.environ.get("EOH_ONLINE_CONFIG_PATH"),
        help="Path to online_run YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("EOH_ONLINE_DRY_RUN", "").strip().lower() in {"1", "true", "yes"},
        help="Validate and print resolved config without calling the online API.",
    )
    args = parser.parse_args()
    selected_task = task_family or args.task_family
    config_path = args.config or _default_config_path(selected_task)
    run_from_config(config_path=config_path, task_family=selected_task, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
