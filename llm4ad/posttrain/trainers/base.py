from __future__ import annotations


class TrainerBackendBase:
    def train(self, dataset_path, *, model_spec, train_config, output_dir: str):
        raise NotImplementedError
