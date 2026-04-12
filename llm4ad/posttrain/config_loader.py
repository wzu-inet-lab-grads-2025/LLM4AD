from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

from .config import (
    BuilderConfig,
    CollectorConfig,
    EventStoreConfig,
    GateConfig,
    PostTrainConfig,
    RegistryConfig,
    ReplayBufferConfig,
    RewardConfig,
    RoundConfig,
    ServeConfig,
    SynthesizeConfig,
    TrainerConfig,
    WorkflowConfig,
)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in baseline env
    yaml = None


_CONFIG_TYPES = {
    "workflow": WorkflowConfig,
    "round": RoundConfig,
    "event_store": EventStoreConfig,
    "collector": CollectorConfig,
    "replay_buffer": ReplayBufferConfig,
    "builder": BuilderConfig,
    "reward": RewardConfig,
    "trainer": TrainerConfig,
    "gate": GateConfig,
    "registry": RegistryConfig,
    "serve": ServeConfig,
    "synthesize": SynthesizeConfig,
}


def _resolve_optional_type(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is None:
        return tp
    args = [arg for arg in get_args(tp) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return tp


def _coerce_value(field_type: Any, value: Any) -> Any:
    field_type = _resolve_optional_type(field_type)
    if value is None:
        return None
    if is_dataclass(field_type):
        return _instantiate_dataclass(field_type, value)
    origin = get_origin(field_type)
    if origin is tuple:
        item_types = get_args(field_type)
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            return tuple(value)
    return value


def _instantiate_dataclass(cls, data: dict[str, Any]):
    kwargs = {}
    for field_info in fields(cls):
        if field_info.name not in data:
            continue
        kwargs[field_info.name] = _coerce_value(field_info.type, data[field_info.name])
    return cls(**kwargs)


def _normalize_paths(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    normalized = dict(data)
    for section in ("registry", "event_store"):
        section_data = normalized.get(section)
        if not isinstance(section_data, dict):
            continue
        for key in ("artifact_root", "root_subdir"):
            if (
                key in section_data
                and section_data[key]
                and not Path(section_data[key]).is_absolute()
            ):
                section_data[key] = str((base_dir / section_data[key]).resolve())
    return normalized


def load_posttrain_config(path: str | Path) -> PostTrainConfig:
    config_path = Path(path).expanduser().resolve()
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        raw_data = json.loads(config_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML posttrain configs.")
        raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported posttrain config suffix: {suffix}")

    raw_data = _normalize_paths(raw_data, config_path.parent)
    kwargs = {}
    for name, cls in _CONFIG_TYPES.items():
        section_data = raw_data.get(name)
        if section_data is None:
            continue
        kwargs[name] = _instantiate_dataclass(cls, section_data)
    if "workflow" not in kwargs:
        raise ValueError("Posttrain config must include a workflow section.")
    return PostTrainConfig(**kwargs)
