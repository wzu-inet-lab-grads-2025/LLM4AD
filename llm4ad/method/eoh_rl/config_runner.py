from __future__ import annotations

import json
import logging
import os
import sys
import warnings

import yaml
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", f"llm4ad_matplotlib_{os.getuid()}"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

from .args import apply_runtime_defaults
from .eoh_rl import EoH
from .profiler import EoHProfiler
from .rl.grpo_trainer import ResidentGRPOLLm, ResidentGRPOPolicy, build_reward_fn_from_task_rl
from .rl.sft_train import normalize_lora_config, run_sft_pipeline
from ...task.optimization.cvrp_construct import CVRPEvaluation
from ...task.optimization.jssp_construct import JSSPEvaluation
from ...task.optimization.mixed_scale_evaluation import create_mixed_scale_evaluation
from ...task.optimization.online_bin_packing import OBPEvaluation
from ...task.optimization.tsp_construct import TSPEvaluation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TASK_REGISTRY = {
    "tsp": {"logger_name": "run_eoh_local_rl_tsp", "log_dir_name": "TSP", "evaluation_cls": TSPEvaluation, "task_key": "tsp"},
    "obp": {"logger_name": "run_eoh_local_rl_obp", "log_dir_name": "OBP", "evaluation_cls": OBPEvaluation, "task_key": "obp"},
    "cvrp": {"logger_name": "run_eoh_local_rl_cvrp", "log_dir_name": "CVRP", "evaluation_cls": CVRPEvaluation, "task_key": "cvrp"},
    "jssp": {"logger_name": "run_eoh_local_rl_jssp", "log_dir_name": "JSSP", "evaluation_cls": JSSPEvaluation, "task_key": "jssp"},
}


def run_from_config(*, config_path: str, task_family: str | None = None) -> None:
    """从 YAML 启动本地 SFT adapter + resident Unsloth EoH-RL。"""
    raw_cfg_root = _apply_runtime_overrides(_load_yaml_root(config_path))
    raw_cfg = dict(_require(raw_cfg_root, "eoh_rl"))
    _validate_config_shape(raw_cfg)
    cfg = apply_runtime_defaults(raw_cfg)
    raw_cfg_root = {**raw_cfg_root, "eoh_rl": cfg}
    task_family = _select_task_family(task_family, cfg)
    if task_family not in TASK_REGISTRY:
        raise ValueError(f"不支持的 task_family: {task_family}")
    spec = TASK_REGISTRY[task_family]

    task_cfg = dict(_require(cfg, spec["task_key"]))
    evolution_cfg = dict(_require(cfg, "evolution"))
    grpo_cfg = dict(_require(cfg, "grpo"))
    rl_cfg = dict(_require(cfg, "rl"))
    lora_cfg = normalize_lora_config(dict(_require(cfg, "lora")))
    reward_cfg = dict(_require(cfg, "task_rl_common"))
    logging_cfg = dict(_require(cfg, "logging"))
    sft_cfg = dict(_require(cfg, "sft"))
    grpo_cfg.setdefault("reward_eval_workers", int(evolution_cfg.get("num_evaluators", 1) or 1))
    grpo_cfg["prompts_per_update"] = max(
        int(grpo_cfg.get("prompts_per_update", 1) or 1),
        int(evolution_cfg.get("num_samplers", 1) or 1),
    )
    if int(rl_cfg.get("samples_per_prompt", grpo_cfg.get("num_generations", 1))) != int(grpo_cfg.get("num_generations", 1)):
        raise ValueError("rl.samples_per_prompt 必须等于 grpo.num_generations")

    training_gpus = list(_require(cfg, "training_gpus"))
    if not list(_require(cfg, "inference_gpus")) or not training_gpus:
        raise ValueError("inference_gpus / training_gpus 不能为空")
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices_from_cfg(cfg)
    if cfg.get("pytorch_alloc_conf"):
        os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(cfg["pytorch_alloc_conf"]).strip()
    run_context = _create_run_context(logging_cfg, spec, allow_existing=bool(rl_cfg.get("checkpoint_auto_resume", True)))
    run_log_dir = run_context["run_log_dir"]
    checkpoint_dir = os.path.join(run_log_dir, "checkpoints")
    for subdir in ("online_grpo", "updates", "checkpoints"):
        os.makedirs(os.path.join(run_log_dir, subdir), exist_ok=True)
    _configure_logging(run_log_dir, spec["logger_name"])
    run_logger = logging.getLogger(spec["logger_name"])

    task, task_name, scale_name, task_params = _build_task(task_cfg, evolution_cfg, spec)
    _seed_runtime(int(grpo_cfg["seed"]))
    model_path = str(_require(cfg, "local_model_path")).strip()
    manifest = {
        "script": os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        "config_path": config_path,
        "model_path": model_path,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "inference_quantization": cfg.get("inference_quantization"),
        "task_name": task_name,
        "scale_name": scale_name,
        "task_params": task_params,
        "timestamp": run_context["timestamp"],
        "run_id": run_context["run_id"],
        "run_log_dir": run_log_dir,
        "run_dir_source": run_context["run_dir_source"],
        "detail_log_path": os.path.join(run_log_dir, "run_log.txt"),
        "phase1_sft_lora_path": sft_cfg.get("load_lora_path"),
        "cold_start_from_base": str(sft_cfg.get("mode") or "cold_start") == "cold_start",
        "sft_mode": str(sft_cfg.get("mode") or "cold_start"),
        "grpo_enabled": bool(rl_cfg["enabled"]),
        "reward_mode": reward_cfg["reward_mode"],
        "reward_config": reward_cfg,
        "seed": int(grpo_cfg["seed"]),
        "initial_population_path": rl_cfg.get("initial_population_path"),
        "online_update_count": int(evolution_cfg["max_generations"]),
        "prompts_per_update": int(grpo_cfg["prompts_per_update"]),
        "responses_per_prompt": int(grpo_cfg["num_generations"]),
        "planned_online_completion_count": (
            int(evolution_cfg["max_generations"])
            * int(grpo_cfg["prompts_per_update"])
            * int(grpo_cfg["num_generations"])
        ),
        "grpo_config": grpo_cfg,
    }
    _write_yaml(os.path.join(run_log_dir, "config_resolved.yaml"), raw_cfg_root)
    run_logger.info("=" * 60)
    run_logger.info("本地流程: 可选 SFT + EoH-RL")
    run_logger.info("配置来源: %s", config_path)
    run_logger.info("规模: %s, 参数: %s", scale_name, task_params)
    run_logger.info("日志目录: %s", run_log_dir)
    run_logger.info("=" * 60)

    sft_runtime = run_sft_pipeline(sft_config=sft_cfg, lora_config=lora_cfg, model_path=model_path, training_gpus=training_gpus, logger_override=run_logger)
    logging.getLogger("llm4ad.tools.rl.task_registry").info(
        "加载任务: %s/yaml_task_default -> %s.%s, 参数: %s",
        task_name,
        task.__class__.__module__,
        task.__class__.__name__,
        task_params,
    )
    manifest.update(
        {
            "phase1_sft_lora_path": sft_runtime.get("initial_lora_path"),
            "cold_start_from_base": not bool(sft_runtime.get("initial_lora_path")),
            "sft_mode": str(sft_runtime.get("mode") or manifest.get("sft_mode") or "cold_start"),
        }
    )
    _write_json(os.path.join(run_log_dir, "config.json"), manifest)
    EoHProfiler.write_run_manifest(run_log_dir, manifest)

    reward_fn = build_reward_fn_from_task_rl(reward_cfg, run_logger)
    run_logger.info("开始 Phase 2 EoH-RL，GRPO=%s", "开启" if rl_cfg["enabled"] else "关闭")
    policy = ResidentGRPOPolicy(
        model_name_or_path=model_path,
        grpo_config=grpo_cfg,
        adapter_config=lora_cfg,
        training_gpus=training_gpus,
        initial_adapter_path=sft_runtime.get("initial_lora_path"),
        quantization=cfg.get("inference_quantization"),
    )
    llm = ResidentGRPOLLm(policy=policy)

    method = EoH(
        llm=llm,
        profiler=EoHProfiler(log_dir=run_log_dir, log_style="complex", create_random_path=False),
        evaluation=task,
        max_sample_nums=_require(evolution_cfg, "max_sample_nums"),
        max_generations=_require(evolution_cfg, "max_generations"),
        pop_size=int(_require(evolution_cfg, "pop_size")),
        num_samplers=int(_require(evolution_cfg, "num_samplers")),
        num_evaluators=int(_require(evolution_cfg, "num_evaluators")),
        debug_mode=False,
        enable_grpo=bool(_require(rl_cfg, "enabled")),
        samples_per_prompt=int(_require(rl_cfg, "samples_per_prompt")),
        grpo_config=grpo_cfg,
        reward_fn=reward_fn,
        checkpoint_dir=checkpoint_dir,
        checkpoint_auto_resume=bool(rl_cfg.get("checkpoint_auto_resume", True)),
        initial_population_path=rl_cfg.get("initial_population_path"),
        enable_ast_gate=bool(rl_cfg.get("enable_ast_gate", False)),
        save_final_lora=bool(rl_cfg.get("save_final_lora", False)),
        compress_history=bool(rl_cfg.get("compress_history", True)),
    )
    status, runtime_summary = "completed", {}
    try:
        method.run()
        runtime_summary = method.get_runtime_summary()
        run_logger.info("本次运行概要: %s", runtime_summary)
    except Exception:
        status = "failed"
        runtime_summary = method.get_runtime_summary()
        run_logger.exception("EoH-RL 运行失败")
        raise
    finally:
        final_manifest = dict(manifest)
        final_manifest.update({"status": status, "runtime_summary": runtime_summary})
        EoHProfiler.write_run_manifest(run_log_dir, final_manifest)


def _select_task_family(task_family: str | None, cfg: dict) -> str:
    value = task_family or cfg.get("task_family") or cfg.get("problem")
    if not value:
        raise ValueError("task_family 不能为空；可由入口脚本传入，或在 YAML 写 task_family/problem")
    return str(value).strip().lower()


def _validate_config_shape(cfg: dict) -> None:
    runtime_sections = {"grpo", "evolution", "rl", "vllm", "lora", "task_rl_common"}
    present = sorted(runtime_sections.intersection(cfg))
    if present:
        raise ValueError("训练参数不再写入 YAML；请删除这些 section，必要时只通过 eoh_rl.args 覆盖标量: " + ", ".join(present))


def _build_task(task_cfg: dict, evolution_cfg: dict, spec: dict):
    mixed_scales = dict(task_cfg.get("mixed_scales") or {})
    if bool(mixed_scales.get("enabled", False)):
        task_name = str(_require(task_cfg, "task_name"))
        return create_mixed_scale_evaluation(task_name, mixed_scales), task_name, "mixed", {"mixed_scales": dict(_require(mixed_scales, "scales"))}
    scale_name = str(_require(task_cfg, "default_scale"))
    scale_cfg = dict(_require(dict(_require(task_cfg, "scales")), scale_name))
    if "pop_size" in scale_cfg:
        evolution_cfg["pop_size"] = int(scale_cfg["pop_size"])
    task_name = str(_require(task_cfg, "task_name"))
    task_params = {key: value for key, value in scale_cfg.items() if key != "pop_size"}
    return spec["evaluation_cls"](**task_params), task_name, scale_name, task_params


def _load_yaml_root(path: str) -> dict:
    data = OmegaConf.load(path)
    OmegaConf.resolve(data)
    if isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=True)
    if not isinstance(data, dict) or "eoh_rl" not in data or not isinstance(data["eoh_rl"], dict):
        raise ValueError(f"配置文件缺少 eoh_rl 根字段: {path}")
    return data


def _apply_runtime_overrides(root: dict) -> dict:
    text = os.environ.get("EOH_RL_RUNTIME_OVERRIDES_JSON", "").strip()
    if not text:
        return root
    overrides = json.loads(text)
    if not isinstance(overrides, dict):
        raise ValueError("EOH_RL_RUNTIME_OVERRIDES_JSON 根节点必须是 dict")
    resolved = OmegaConf.create(_deep_merge_dict(dict(root), overrides))
    OmegaConf.resolve(resolved)
    return OmegaConf.to_container(resolved, resolve=True) if isinstance(resolved, DictConfig) else dict(resolved)


def _deep_merge_dict(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        result[key] = _deep_merge_dict(dict(result[key]), value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _require(mapping: dict, key: str):
    if key not in mapping:
        raise ValueError(f"配置缺少字段: {key}")
    return mapping[key]


def _cuda_visible_devices_from_cfg(cfg: dict) -> str:
    ids = sorted({int(value) for value in list(_require(cfg, "inference_gpus")) + list(_require(cfg, "training_gpus"))})
    return ",".join(str(value) for value in ids)


def _seed_runtime(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_run_context(logging_cfg: dict, spec: dict, *, allow_existing: bool) -> dict:
    return EoHProfiler.create_run_context(
        os.path.join(str(logging_cfg.get("log_base_dir") or os.path.join(PROJECT_ROOT, "logs")).strip(), spec["log_dir_name"], "eoh_local_rl"),
        run_id=os.environ.get("EOH_RL_RUN_ID", "").strip() or None,
        run_log_dir=os.environ.get("EOH_RL_RUN_LOG_DIR", "").strip() or None,
        allow_existing=allow_existing,
    )


def _configure_logging(run_log_dir: str, logger_name: str) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", handlers=[logging.FileHandler(os.path.join(run_log_dir, "run.log"), encoding="utf-8"), logging.StreamHandler(sys.stdout)], force=True)
    logging.getLogger(logger_name).setLevel(logging.INFO)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
    warnings.filterwarnings("ignore", message=r".*TRL currently supports vLLM versions.*")
    warnings.filterwarnings("ignore", message=r".*different tokenizers for different LoRAs.*")
    for name in ("vllm", "trl", "transformers", "unsloth"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def _write_yaml(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


__all__ = ["run_from_config"]
