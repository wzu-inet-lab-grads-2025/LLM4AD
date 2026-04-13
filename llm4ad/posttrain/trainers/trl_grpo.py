from __future__ import annotations

from .base import TrainerBackendBase


class TrlGrpoTrainer(TrainerBackendBase):
    def train(
        self, dataset_path, *, model_spec, train_config, output_dir: str | None = None
    ):
        raise NotImplementedError(
            "GRPO training is reserved for the next implementation stage."
        )
