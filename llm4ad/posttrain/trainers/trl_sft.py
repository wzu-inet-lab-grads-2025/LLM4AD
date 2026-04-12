from __future__ import annotations

import json
from pathlib import Path

from llm4ad.posttrain.schemas import ModelArtifactRecord

from .artifacts import build_artifact_version_id, prepare_training_output_dir
from .base import TrainerBackendBase


class TrlSftTrainer(TrainerBackendBase):
    def train(
        self, dataset_path, *, model_spec, train_config, output_dir: str | None = None
    ):
        version_id = build_artifact_version_id(model_spec["round_id"])
        train_root = output_dir or train_config.output_root
        artifact_dir = prepare_training_output_dir(train_root, version_id)

        if train_config.execution_mode == "subprocess":
            return self._train_via_subprocess(
                dataset_path,
                model_spec=model_spec,
                train_config=train_config,
                artifact_dir=artifact_dir,
                version_id=version_id,
            )
        return self._train_in_process(
            dataset_path,
            model_spec=model_spec,
            train_config=train_config,
            artifact_dir=artifact_dir,
            version_id=version_id,
        )

    def _train_in_process(
        self,
        dataset_path,
        *,
        model_spec,
        train_config,
        artifact_dir: Path,
        version_id: str,
    ):
        try:
            from datasets import load_dataset
            from peft import LoraConfig
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
            )
            from trl import SFTTrainer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Posttrain SFT requires requirements-posttrain.txt dependencies."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(train_config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(train_config.base_model)
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        peft_config = None
        if train_config.output_adapter_only:
            peft_config = LoraConfig(
                r=train_config.lora_r,
                lora_alpha=train_config.lora_alpha,
                lora_dropout=train_config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

        training_args = TrainingArguments(
            output_dir=str(artifact_dir),
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            learning_rate=train_config.learning_rate,
            num_train_epochs=train_config.num_train_epochs,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            max_seq_length=train_config.max_length,
        )
        trainer.train()
        trainer.save_model(str(artifact_dir))

        metadata = {
            "trainer": "trl_sft",
            "dataset_path": str(dataset_path),
            "execution_mode": "python",
        }
        (artifact_dir / "train_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return ModelArtifactRecord(
            version_id=version_id,
            round_id=model_spec["round_id"],
            artifact_type="adapter"
            if train_config.output_adapter_only
            else "checkpoint",
            path=str(artifact_dir),
            base_model=train_config.base_model,
            metadata=metadata,
        )

    def _train_via_subprocess(
        self,
        dataset_path,
        *,
        model_spec,
        train_config,
        artifact_dir: Path,
        version_id: str,
    ):
        metadata = {
            "trainer": "trl_sft",
            "dataset_path": str(dataset_path),
            "execution_mode": "subprocess",
        }
        (artifact_dir / "train_request.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        raise NotImplementedError(
            "Subprocess-based SFT training is reserved but not implemented yet."
        )
