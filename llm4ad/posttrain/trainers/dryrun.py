from __future__ import annotations

import json

from llm4ad.posttrain.schemas import ModelArtifactRecord

from .artifacts import build_artifact_version_id, prepare_training_output_dir
from .base import TrainerBackendBase


class DryRunTrainer(TrainerBackendBase):
    def train(
        self, dataset_path, *, model_spec, train_config, output_dir: str | None = None
    ):
        version_id = build_artifact_version_id(model_spec["round_id"])
        train_root = output_dir or train_config.output_root
        artifact_dir = prepare_training_output_dir(train_root, version_id)

        payload = {
            "trainer": "dryrun",
            "dataset_path": str(dataset_path),
            "base_model": train_config.base_model,
            "note": "Dry-run artifact for end-to-end orchestrator validation.",
        }
        (artifact_dir / "dryrun_artifact.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        return ModelArtifactRecord(
            version_id=version_id,
            round_id=model_spec["round_id"],
            artifact_type="dryrun",
            path=str(artifact_dir),
            base_model=train_config.base_model,
            metadata=payload,
        )
