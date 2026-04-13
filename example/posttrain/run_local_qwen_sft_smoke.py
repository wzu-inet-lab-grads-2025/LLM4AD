from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from llm4ad.posttrain.config import TrainerConfig
from llm4ad.posttrain.trainers.trl_sft import TrlSftTrainer


def main():
    dataset_path = Path("artifacts/posttrain/datasets/round_0001_outcome.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Expected dataset not found. Run example/posttrain/run_local_qwen_toy_orchestrator.py first."
        )

    trainer = TrlSftTrainer()
    artifact = trainer.train(
        str(dataset_path),
        model_spec={"round_id": 1},
        train_config=TrainerConfig(
            backend="trl_sft",
            base_model="models/Qwen2.5-Coder-1.5B-Instruct",
            output_adapter_only=True,
            execution_mode="python",
            output_root="artifacts/posttrain/training_sft_smoke",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1.0,
            learning_rate=1e-5,
            max_length=256,
            max_prompt_length=128,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            gradient_checkpointing=True,
            use_cpu=False,
            lora_target_modules="all-linear",
        ),
    )
    print(json.dumps(asdict(artifact), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
