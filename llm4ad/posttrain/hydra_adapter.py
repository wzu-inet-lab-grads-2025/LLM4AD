from __future__ import annotations

from typing import Any

from .config_loader import build_posttrain_config


def load_posttrain_config_from_hydra(config_like: Any):
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Hydra/OmegaConf is not installed. Use dataclass+YAML config loading or install hydra-core."
        ) from exc

    raw_data = OmegaConf.to_container(config_like, resolve=True)
    if not isinstance(raw_data, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    return build_posttrain_config(raw_data)
