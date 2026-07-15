from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class EoHRLArgs:
    enable_grpo: bool = True
    max_steps: int = 500
    max_samples: int = 2000
    n_prompts: int = 1
    n_generations: int = 4
    population_size: int = 10
    num_samplers: int = 1
    num_evaluators: int = 4

    model_max_length: int = 4096
    max_prompt_length: int = 4000
    max_completion_length: int = 1000
    lora_rank: int = 32
    lr: float = 5.0e-5
    beta: float = 0.04
    epsilon: float = 0.15
    epsilon_high: float = 0.28
    seed: int = 42
    initial_population_path: str | None = None
    reward_mode: str = "vc_pair"
    pair_se_multiplier: float = 1.0
    pair_margin: float = 1.0e-4
    pair_positive_reward: float = 1.0
    pair_neutral_reward: float = 0.0
    pair_negative_reward: float = -0.5
    save_final_lora: bool = False
    compress_history: bool = True

    temperature: float = 1.0
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.8
    vllm_group_port: int = 51215
    quiet_training_output: bool = False

    invalid_reward: float = -1.0
    detect_randomness: bool = True

    @classmethod
    def from_mapping(cls, data: dict | None) -> "EoHRLArgs":
        raw = dict(data or {})
        names = {field.name for field in fields(cls)}
        return cls(**{key: raw[key] for key in names if key in raw})


def apply_runtime_defaults(cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    args = EoHRLArgs.from_mapping(cfg.get("args"))
    server_port = _first_int(cfg.get("inference_ports"), 22003)

    cfg["lora"] = _merge(
        {
            "r": args.lora_rank,
            "alpha": 2 * args.lora_rank,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        cfg.get("lora"),
    )
    cfg["sft"] = _merge(_sft_defaults(args, cfg.get("sft")), cfg.get("sft"))
    cfg["evolution"] = _merge(
        {
            "max_generations": args.max_steps,
            "max_sample_nums": args.max_samples,
            "pop_size": args.population_size,
            "num_samplers": args.num_samplers,
            "num_evaluators": args.num_evaluators,
        },
        cfg.get("evolution"),
    )
    cfg["grpo"] = _merge(_grpo_defaults(args, server_port), cfg.get("grpo"))
    group_size = int(cfg["grpo"]["num_generations"])
    cfg["grpo"]["generation_batch_size"] = group_size
    cfg["grpo"]["per_device_train_batch_size"] = group_size
    cfg["rl"] = _merge(
        {
            "enabled": args.enable_grpo,
            "enable_ast_gate": False,
            "samples_per_prompt": args.n_generations,
            "checkpoint_auto_resume": True,
            "initial_population_path": args.initial_population_path,
            "save_final_lora": args.save_final_lora,
            "compress_history": args.compress_history,
        },
        cfg.get("rl"),
    )
    cfg["task_rl_common"] = _merge(
        {
            "minimize": False,
            "detect_randomness": args.detect_randomness,
            "reward_parse_fail": -1.00,
            "reward_exec_fail": -0.80,
            "reward_none_return": -0.60,
            "reward_random": -0.70,
            "reward_leak": -0.70,
            "epsilon": 1.0e-4,
            "q3_scale": 0.50,
            "reward_mode": args.reward_mode,
            "pair_se_multiplier": args.pair_se_multiplier,
            "pair_margin": args.pair_margin,
            "pair_positive_reward": args.pair_positive_reward,
            "pair_neutral_reward": args.pair_neutral_reward,
            "pair_negative_reward": args.pair_negative_reward,
        },
        cfg.get("task_rl_common"),
    )
    cfg.setdefault("inference_quantization", "bitsandbytes")
    return cfg


def _sft_defaults(args: EoHRLArgs, user_sft: dict | None) -> dict:
    user_sft = dict(user_sft or {})
    mode = user_sft.get("mode") or ("load_existing" if user_sft.get("load_lora_path") else "cold_start")
    return {
        "mode": mode,
        "learning_rate": 2.0e-4,
        "num_epochs": 2,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_seq_length": args.model_max_length,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "quantization": "4bit",
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "seed": args.seed,
        "load_lora_path": None,
        "data_cleaner": {
            "top_ratio": 0.2,
            "balance_tasks": True,
            "score_filter_min": None,
            "score_filter_max": None,
            "dedup_strategy": "ast_struct",
            "filter_invalid_code": True,
            "max_per_task": None,
        },
    }


def _grpo_defaults(args: EoHRLArgs, server_port: int) -> dict:
    return {
        "prompts_per_update": args.n_prompts,
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_model_impl": "vllm",
        "vllm_enable_sleep_mode": False,
        "vllm_server_host": "127.0.0.1",
        "vllm_server_port": server_port,
        "vllm_server_timeout": 240.0,
        "vllm_group_port": args.vllm_group_port,
        "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
        "vllm_tensor_parallel_size": 1,
        "vllm_max_model_length": args.model_max_length,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.n_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "epsilon_high": args.epsilon_high,
        "learning_rate": args.lr,
        "lr_scheduler_type": "constant",
        "optim": "adamw_8bit",
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "scale_rewards": "group",
        "loss_type": "dapo",
        "num_iterations": 1,
        "importance_sampling_level": "token",
        "mask_truncated_completions": True,
        "vllm_importance_sampling_correction": True,
        "num_train_epochs": 1,
        "generation_batch_size": args.n_generations,
        "per_device_train_batch_size": args.n_generations,
        "gradient_accumulation_steps": 1,
        "max_seq_length": args.model_max_length,
        "warmup_ratio": 0.0,
        "weight_decay": 0.1,
        "max_grad_norm": 0.1,
        "bf16": True,
        "fp16": False,
        "quantization": "4bit",
        "gradient_checkpointing": True,
        "seed": args.seed,
        "logging_steps": 1,
        "save_strategy": "no",
        "remove_unused_columns": False,
        "report_to": "none",
        "disable_tqdm": False,
        "logging_strategy": "steps",
        "log_level": "warning",
        "log_level_replica": "warning",
        "quiet_training_output": args.quiet_training_output,
    }


def _merge(defaults: dict, overrides: dict | None) -> dict:
    result = deepcopy(defaults)
    for key, value in dict(overrides or {}).items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _first_int(values, default: int) -> int:
    return int(values[0]) if isinstance(values, (list, tuple)) and values else int(default)


__all__ = ["EoHRLArgs", "apply_runtime_defaults"]
