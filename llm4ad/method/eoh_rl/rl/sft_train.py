from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from .sft_data import SFTDataCleaner, build_sft_dataset_from_source_dirs, normalize_sft_source_dirs

logger = logging.getLogger(__name__)


def _torch_module():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("SFT 训练需要安装 torch") from exc
    return torch


def normalize_optional_path(value) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return os.path.abspath(os.path.expanduser(raw))


def normalize_lora_config(lora_config: dict | None) -> dict:
    cfg = dict(lora_config or {})
    rank = int(cfg.get("r", cfg.get("lora_rank", 32)))
    alpha = int(cfg.get("alpha", cfg.get("lora_alpha", 2 * rank)))
    dropout = float(cfg.get("dropout", cfg.get("lora_dropout", 0.0)))
    target_modules = list(
        cfg.get("target_modules")
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    return {
        **cfg,
        "r": rank,
        "alpha": alpha,
        "dropout": dropout,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": target_modules,
    }


def _load_adapter_config(lora_path: str) -> dict | None:
    cfg_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_target_modules(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def assert_lora_compatible(*, lora_path: str, expected_lora_config: dict) -> None:
    path = normalize_optional_path(lora_path)
    if path is None or not os.path.isdir(path):
        raise ValueError(f"invalid lora path: {lora_path}")
    cfg = _load_adapter_config(path)
    if cfg is None:
        raise ValueError(f"LoRA checkpoint missing adapter_config.json: {path}")
    expected_modules = _normalize_target_modules(expected_lora_config.get("target_modules"))
    actual_modules = _normalize_target_modules(cfg.get("target_modules"))
    if expected_modules is not None and actual_modules is not None and set(expected_modules) != set(actual_modules):
        raise ValueError(
            "LoRA target_modules mismatch.\n"
            f"  lora_path={path}\n"
            f"  expected={sorted(set(expected_modules))}\n"
            f"  got={sorted(set(actual_modules))}"
        )
    for key in ("r", "lora_alpha", "lora_dropout"):
        if key in cfg and key in expected_lora_config and cfg.get(key) != expected_lora_config.get(key):
            raise ValueError(
                f"LoRA {key} mismatch: expected={expected_lora_config.get(key)} got={cfg.get(key)} (path={path})"
            )


def aggressive_cuda_release_after_sft() -> None:
    torch = _torch_module()
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        for _ in range(5):
            gc.collect()
            for device_idx in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(device_idx):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            torch.cuda.empty_cache()
        torch.cuda.set_device(0)
    except Exception as exc:
        logger.debug("aggressive_cuda_release_after_sft 跳过: %s", exc)


@dataclass(frozen=True)
class SFTModePlan:
    mode: str
    output_dir: str | None
    train_jsonl_path: str | None
    lora_output_path: str | None
    load_lora_path: str | None
    should_train: bool
    should_load_lora: bool


def resolve_sft_mode(sft_config: dict) -> SFTModePlan:
    """
    解析 SFT 启动模式，并兼容旧字段 skip_sft/cold_start_from_base。

    参数:
        sft_config: YAML 中 eoh_rl.sft 配置字典。
    """
    cfg = dict(sft_config or {})
    if not cfg:
        raise ValueError("eoh_rl.sft 不能为空；必须显式配置 SFT/初始 LoRA 语义")
    output_dir = normalize_optional_path(cfg.get("output_dir"))

    mode = str(cfg.get("mode") or "").strip().lower()
    if not mode:
        skip_sft = bool(cfg.get("skip_sft"))
        cold_start = bool(cfg.get("cold_start_from_base"))
        if cold_start and not skip_sft:
            raise ValueError("eoh_rl.sft.cold_start_from_base=true 时必须同时设置 skip_sft=true")
        if not mode and not skip_sft:
            mode = "train"
        elif not mode and cold_start:
            mode = "cold_start"
        elif not mode:
            mode = "load_existing"

    if mode not in {"train", "load_existing", "cold_start"}:
        raise ValueError(f"不支持的 eoh_rl.sft.mode: {mode}")

    load_lora_path = normalize_optional_path(cfg.get("load_lora_path"))
    if mode == "train":
        if output_dir is None:
            raise ValueError("eoh_rl.sft.output_dir 不能为空")
        train_jsonl_path = os.path.join(output_dir, "sft_train.jsonl")
        default_lora_path = os.path.join(output_dir, "lora_sft")
        return SFTModePlan(
            mode=mode,
            output_dir=output_dir,
            train_jsonl_path=train_jsonl_path,
            lora_output_path=default_lora_path,
            load_lora_path=default_lora_path,
            should_train=True,
            should_load_lora=True,
        )
    if mode == "load_existing":
        if output_dir is None and load_lora_path is None:
            raise ValueError("eoh_rl.sft.load_existing 模式要求至少提供 output_dir 或 load_lora_path")
        base_dir = output_dir or (os.path.dirname(load_lora_path) if load_lora_path else None)
        train_jsonl_path = os.path.join(base_dir, "sft_train.jsonl") if base_dir else None
        default_lora_path = os.path.join(base_dir, "lora_sft") if base_dir else None
        path = load_lora_path or default_lora_path
        return SFTModePlan(
            mode=mode,
            output_dir=base_dir,
            train_jsonl_path=train_jsonl_path,
            lora_output_path=default_lora_path,
            load_lora_path=path,
            should_train=False,
            should_load_lora=True,
        )
    return SFTModePlan(
        mode=mode,
        output_dir=output_dir,
        train_jsonl_path=os.path.join(output_dir, "sft_train.jsonl") if output_dir else None,
        lora_output_path=os.path.join(output_dir, "lora_sft") if output_dir else None,
        load_lora_path=None,
        should_train=False,
        should_load_lora=False,
    )


def run_sft_pipeline(
    *,
    sft_config: dict,
    lora_config: dict,
    model_path: str,
    training_gpus: list[int] | None,
    logger_override=None,
) -> dict:
    """
    执行 SFT 准备流程，返回后续 resident LoRA 初始化信息。

    参数:
        sft_config: YAML 中 eoh_rl.sft 配置字典。
        lora_config: LoRA adapter 配置。
        model_path: 基模型路径。
        training_gpus: SFT 使用的物理 GPU 编号列表。
        logger_override: 可选 logger，用于写入 run 级日志。
    """
    active_logger = logger_override or logger
    plan = resolve_sft_mode(sft_config)
    expected_lora_config = normalize_lora_config(lora_config)
    if plan.should_train:
        if plan.output_dir is None or plan.train_jsonl_path is None:
            raise ValueError("SFT train 模式缺少 output_dir")
        os.makedirs(plan.output_dir, exist_ok=True)
    payload = {
        "mode": plan.mode,
        "output_dir": plan.output_dir,
        "train_jsonl_path": plan.train_jsonl_path,
        "lora_output_path": plan.lora_output_path,
        "initial_lora_path": plan.load_lora_path,
    }
    if not plan.should_train:
        if plan.should_load_lora:
            if not plan.load_lora_path:
                raise FileNotFoundError(f"SFT LoRA 不存在: {plan.load_lora_path}")
            assert_lora_compatible(lora_path=plan.load_lora_path, expected_lora_config=expected_lora_config)
            active_logger.info("跳过 Phase 1，使用已有 SFT LoRA: %s", plan.load_lora_path)
        else:
            active_logger.info("跳过 Phase 1，且不加载 SFT LoRA：直接从基座开始 EoH-RL")
        return payload

    source_dirs = normalize_sft_source_dirs(sft_config.get("source_dirs"))
    cleaner_cfg = dict(sft_config.get("data_cleaner") or {})
    cleaner = SFTDataCleaner(
        score_filter_min=cleaner_cfg.get("score_filter_min"),
        score_filter_max=cleaner_cfg.get("score_filter_max"),
        top_ratio=float(cleaner_cfg.get("top_ratio", 0.3)),
        dedup_strategy=str(cleaner_cfg.get("dedup_strategy", "ast_struct")),
        balance_tasks=bool(cleaner_cfg.get("balance_tasks", True)),
        max_per_task=cleaner_cfg.get("max_per_task"),
        filter_invalid_code=bool(cleaner_cfg.get("filter_invalid_code", True)),
        enable_ast_gate=bool(cleaner_cfg.get("enable_ast_gate", True)),
        ast_gate_use_task_template=bool(cleaner_cfg.get("ast_gate_use_task_template", True)),
        stratified_top_k=bool(cleaner_cfg.get("stratified_top_k", True)),
        novelty_ratio=float(cleaner_cfg.get("novelty_ratio", 0.2)),
    )
    train_jsonl_path = build_sft_dataset_from_source_dirs(
        cleaner=cleaner,
        source_dirs=source_dirs,
        output_path=plan.train_jsonl_path,
        logger_override=active_logger,
    )
    trainer = SFTTrainer(
        model_path=model_path,
        output_dir=plan.output_dir,
        data_path=train_jsonl_path,
        lora_config=expected_lora_config,
        sft_config=sft_config,
        training_gpus=training_gpus,
    )
    payload["initial_lora_path"] = trainer.train()
    return payload


class SFTTrainer:
    def __init__(
        self,
        *,
        model_path: str,
        output_dir: str,
        data_path: str,
        lora_config: dict,
        sft_config: dict,
        training_gpus: list[int] | None = None,
    ):
        """
        参数:
            model_path: 基模型路径。
            output_dir: SFT 输出目录。
            data_path: SFT JSONL 数据路径。
            lora_config: LoRA adapter 配置。
            sft_config: SFT 训练超参数。
            training_gpus: SFT 使用的物理 GPU 编号列表。
        """
        self.model_path = str(model_path)
        self.output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
        self.data_path = os.path.abspath(os.path.expanduser(str(data_path)))
        self.lora_config = normalize_lora_config(lora_config)
        self.sft_config = dict(sft_config or {})
        self.training_gpus = [int(item) for item in (training_gpus or [])] or None
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, data_path: str | None = None) -> str:
        torch = _torch_module()
        if self._use_sft_subprocess():
            return self._train_in_subprocess(data_path=data_path)

        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments

        save_path = os.path.join(self.output_dir, "lora_sft")
        records = self._load_data(data_path)
        if not records:
            raise ValueError("SFT 数据为空，请先检查 source_dirs 与清洗结果")

        saved_accelerate_device = os.environ.get("ACCELERATE_TORCH_DEVICE")
        try:
            visible_idx = self._single_visible_training_index()
            if visible_idx is not None:
                os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{visible_idx}"
                if torch.cuda.is_available():
                    torch.cuda.set_device(visible_idx)
                logger.info("[SFT] 设置 ACCELERATE_TORCH_DEVICE=%s", os.environ["ACCELERATE_TORCH_DEVICE"])

            logger.info("=" * 60)
            logger.info("SFT 训练开始")
            logger.info("=" * 60)

            quantization = str(self.sft_config.get("quantization", "4bit")).strip().lower()
            bf16 = bool(self.sft_config.get("bf16", True))
            bnb_config = None
            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
            }
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
            device_map = self._device_map_for_current_process()
            if device_map is not None:
                load_kwargs["device_map"] = device_map

            model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
            if quantization in {"4bit", "8bit"}:
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=bool(self.sft_config.get("gradient_checkpointing", True)),
                )
            elif bool(self.sft_config.get("gradient_checkpointing", True)):
                model.gradient_checkpointing_enable()

            peft_config = LoraConfig(
                r=int(self.lora_config["r"]),
                lora_alpha=int(self.lora_config["lora_alpha"]),
                lora_dropout=float(self.lora_config["lora_dropout"]),
                target_modules=list(self.lora_config["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

            dataset = self._build_dataset(records, tokenizer, dataset_cls=Dataset)
            effective_batch = int(self.sft_config.get("per_device_train_batch_size", 4)) * int(
                self.sft_config.get("gradient_accumulation_steps", 4)
            )
            total_steps = max(1, len(dataset) // max(1, effective_batch)) * int(self.sft_config.get("num_epochs", 2))
            logger.info(
                "[SFT] 训练规模: %d 样本, effective_batch=%d, total_steps≈%d, epochs=%d",
                len(dataset),
                effective_batch,
                total_steps,
                int(self.sft_config.get("num_epochs", 2)),
            )

            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=int(self.sft_config.get("num_epochs", 2)),
                per_device_train_batch_size=int(self.sft_config.get("per_device_train_batch_size", 4)),
                gradient_accumulation_steps=int(self.sft_config.get("gradient_accumulation_steps", 4)),
                learning_rate=float(self.sft_config.get("learning_rate", 2e-4)),
                warmup_ratio=float(self.sft_config.get("warmup_ratio", 0.1)),
                weight_decay=float(self.sft_config.get("weight_decay", 0.01)),
                max_grad_norm=float(self.sft_config.get("max_grad_norm", 1.0)),
                lr_scheduler_type=str(self.sft_config.get("lr_scheduler_type", "cosine")),
                bf16=bf16,
                logging_steps=int(self.sft_config.get("logging_steps", 10)),
                save_strategy="epoch",
                save_total_limit=2,
                seed=int(self.sft_config.get("seed", 42)),
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=True,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest", label_pad_token_id=-100),
            )
            trainer.train()

            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            with open(os.path.join(save_path, "sft_config.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "model_path": self.model_path,
                        "lora_r": int(self.lora_config["r"]),
                        "lora_alpha": int(self.lora_config["lora_alpha"]),
                        "lora_dropout": float(self.lora_config["lora_dropout"]),
                        "lora_target_modules": list(self.lora_config["target_modules"]),
                        "learning_rate": float(self.sft_config.get("learning_rate", 2e-4)),
                        "num_epochs": int(self.sft_config.get("num_epochs", 2)),
                        "per_device_batch_size": int(self.sft_config.get("per_device_train_batch_size", 4)),
                        "gradient_accumulation_steps": int(self.sft_config.get("gradient_accumulation_steps", 4)),
                        "max_prompt_length": int(self.sft_config.get("max_prompt_length", 4000)),
                        "max_completion_length": int(self.sft_config.get("max_completion_length", 1000)),
                        "max_seq_length": int(self.sft_config.get("max_seq_length", 4096)),
                        "warmup_ratio": float(self.sft_config.get("warmup_ratio", 0.1)),
                        "weight_decay": float(self.sft_config.get("weight_decay", 0.01)),
                        "max_grad_norm": float(self.sft_config.get("max_grad_norm", 1.0)),
                        "quantization": quantization,
                        "lr_scheduler_type": str(self.sft_config.get("lr_scheduler_type", "cosine")),
                        "data_records": len(records),
                        "dataset_size": len(dataset),
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            logger.info("SFT LoRA adapter 已保存: %s", save_path)

            del trainer
            del model
            del tokenizer
            del dataset
            self._cleanup_gpu_memory()
            aggressive_cuda_release_after_sft()
            logger.info("=" * 60)
            logger.info("SFT 训练完成, adapter 路径: %s", save_path)
            logger.info("=" * 60)
        finally:
            if saved_accelerate_device is None:
                os.environ.pop("ACCELERATE_TORCH_DEVICE", None)
            else:
                os.environ["ACCELERATE_TORCH_DEVICE"] = saved_accelerate_device
        return save_path

    def _load_data(self, data_path: str | None = None) -> list[dict]:
        path = os.path.abspath(os.path.expanduser(str(data_path or self.data_path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SFT 数据文件不存在: {path}")
        records: list[dict] = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                if record.get("instruction") and record.get("output"):
                    records.append(record)
        logger.info("加载 SFT 数据: %d 条, 来自 %s", len(records), path)
        return records

    def _build_dataset(self, records: list[dict], tokenizer, *, dataset_cls):
        has_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
        if has_chat_template:
            logger.info("[SFT] 使用 chat_template 构造 prompt（与推理/GRPO 一致）")
        else:
            logger.info("[SFT] 模型无 chat_template，回退为 instruction + newline 拼接")

        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        skipped = 0
        skip_reasons: dict[str, int] = {}
        for record in records:
            instruction = str(record.get("instruction") or "")
            output = str(record.get("output") or "")
            if not instruction.strip() or not output.strip():
                skipped += 1
                continue
            try:
                tokenized = self._tokenize_single(tokenizer, instruction, output, use_chat_template=has_chat_template)
            except Exception as exc:
                skipped += 1
                reason = f"exception:{type(exc).__name__}"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            if tokenized is None:
                skipped += 1
                skip_reasons["too_short_or_empty_response"] = skip_reasons.get("too_short_or_empty_response", 0) + 1
                continue
            input_ids_list.append(tokenized[0])
            labels_list.append(tokenized[1])
        if skipped > 0:
            top_reasons = ", ".join(
                f"{key}={value}" for key, value in sorted(skip_reasons.items(), key=lambda item: item[1], reverse=True)[:5]
            )
            logger.info("[SFT] tokenize 跳过 %d 条%s", skipped, f"; top原因: {top_reasons}" if top_reasons else "")
        if not input_ids_list:
            raise ValueError("SFT tokenize 后数据为空；请检查 chat_template/tokenizer、max_seq_length 或 output 字段")
        dataset = dataset_cls.from_dict({"input_ids": input_ids_list, "labels": labels_list})
        avg_len = sum(len(item) for item in input_ids_list) / max(len(input_ids_list), 1)
        avg_resp = sum(sum(1 for token in labels if token != -100) for labels in labels_list) / max(len(labels_list), 1)
        logger.info("构建 Dataset: %d 条, 平均 token 长度: %.0f, 平均 response token: %.0f", len(dataset), avg_len, avg_resp)
        return dataset

    def _tokenize_single(self, tokenizer, instruction: str, output: str, *, use_chat_template: bool):
        min_response_tokens = 20
        if use_chat_template:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_ids = tokenizer.encode(instruction + "\n", add_special_tokens=True)
        eos_token = tokenizer.eos_token or ""
        response_ids = tokenizer.encode(output + eos_token, add_special_tokens=False)
        if not response_ids or len(response_ids) < min_response_tokens:
            return None

        max_completion_length = int(self.sft_config.get("max_completion_length", 1000))
        max_seq_length = int(self.sft_config.get("max_seq_length", 4096))
        max_prompt_length = int(self.sft_config.get("max_prompt_length", 4000))
        if max_completion_length and len(response_ids) > max_completion_length:
            response_ids = response_ids[:max_completion_length]
        if len(response_ids) > max_seq_length:
            response_ids = response_ids[:max_seq_length]
        prompt_budget = max(0, max_seq_length - len(response_ids))
        if max_prompt_length:
            prompt_budget = min(prompt_budget, max_prompt_length)
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        full_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        return full_ids, labels

    def _use_sft_subprocess(self) -> bool:
        if os.environ.get("LLM4AD_SFT_SUBPROCESS") == "1":
            return False
        if not self.training_gpus:
            return False
        if len(self.training_gpus) > 1:
            return True
        try:
            visible_idx = self._single_visible_training_index()
        except ValueError:
            return True
        return visible_idx is not None and visible_idx != 0

    def _train_in_subprocess(self, data_path: str | None = None) -> str:
        if not self.training_gpus:
            raise RuntimeError("_train_in_subprocess 需要非空 training_gpus")
        payload = {
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "data_path": os.path.abspath(os.path.expanduser(str(data_path or self.data_path))),
            "lora_config": self.lora_config,
            "sft_config": self.sft_config,
            "training_gpus": [int(item) for item in self.training_gpus],
        }
        fd, config_path = tempfile.mkstemp(suffix=".json", prefix="llm4ad_sft_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in self.training_gpus)
            env["LLM4AD_SFT_SUBPROCESS"] = "1"
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            python_path = env.get("PYTHONPATH", "").strip()
            env["PYTHONPATH"] = repo_root + (os.pathsep + python_path if python_path else "")
            logger.info("[SFT] 使用子进程训练（CUDA_VISIBLE_DEVICES=%s）", env["CUDA_VISIBLE_DEVICES"])
            subprocess.check_call([sys.executable, "-m", "llm4ad.method.eoh_rl.rl.sft_train", "--worker-config", config_path], env=env)
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass
        save_path = os.path.join(self.output_dir, "lora_sft")
        if not os.path.isdir(save_path):
            raise RuntimeError(f"子进程 SFT 结束但未找到输出目录: {save_path}")
        return save_path

    @staticmethod
    def _cleanup_gpu_memory() -> None:
        torch = _torch_module()
        gc.collect()
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for device_idx in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(device_idx)
            logger.info("GPU %d: %.1fGB 空闲 / %.1fGB 总计", device_idx, free / 1e9, total / 1e9)

    def _single_visible_training_index(self) -> int | None:
        if not self.training_gpus or len(self.training_gpus) != 1:
            return None
        return _visible_cuda_index_for_physical(int(self.training_gpus[0]))

    def _device_map_for_current_process(self):
        if not self.training_gpus:
            return "auto"
        if len(self.training_gpus) == 1:
            return {"": _visible_cuda_index_for_physical(int(self.training_gpus[0]))}
        return "auto"


def _visible_cuda_index_for_physical(physical_gpu_id: int) -> int:
    physical_gpu_id = int(physical_gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return physical_gpu_id
    values = [item.strip() for item in visible.split(",") if item.strip()]
    try:
        mapping = [int(item) for item in values]
    except ValueError:
        return physical_gpu_id
    if physical_gpu_id not in mapping:
        raise ValueError(
            f"物理 GPU {physical_gpu_id} 不在当前 CUDA_VISIBLE_DEVICES={visible!r} 中；"
            "请保证 inference_gpus / training_gpus 配置合理。"
        )
    return mapping.index(physical_gpu_id)


def _run_worker_from_config(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    trainer = SFTTrainer(
        model_path=str(payload["model_path"]),
        output_dir=str(payload["output_dir"]),
        data_path=str(payload["data_path"]),
        lora_config=dict(payload.get("lora_config") or {}),
        sft_config=dict(payload.get("sft_config") or {}),
        training_gpus=list(payload.get("training_gpus") or []),
    )
    trainer.train()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "--worker-config":
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", force=True)
        return _run_worker_from_config(args[1])
    raise SystemExit("仅支持 --worker-config <path> 子进程模式")


if __name__ == "__main__":
    raise SystemExit(main())
