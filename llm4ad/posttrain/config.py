from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class RoundConfig:
    max_samples_per_round: int | None = None
    max_generations_per_round: int | None = None
    stop_granularity: Literal["sample", "operator", "generation"] = "sample"
    rebuild_method_between_rounds: bool = True
    allow_same_process_resume: bool = True


@dataclass(slots=True)
class EventStoreConfig:
    enabled: bool = True
    root_subdir: str = "posttrain/events"
    split_by_event_type: bool = True


@dataclass(slots=True)
class CollectorConfig:
    include_history_logs: bool = True
    include_event_store: bool = True
    output_as: Literal["records"] = "records"


@dataclass(slots=True)
class ReplayBufferConfig:
    enabled: bool = True
    recent_capacity: int = 5000
    elite_capacity: int = 1000
    diversity_capacity: int = 2000


@dataclass(slots=True)
class BuilderConfig:
    include_init_phase: bool = False
    dedup: bool = True
    synthetic_ratio: float = 0.1
    synthetic_mode: Literal["mixed", "staged", "disabled"] = "mixed"


@dataclass(slots=True)
class RewardConfig:
    provider: str = "local_eval"
    shapers: list[str] = field(default_factory=lambda: ["normalize"])
    allow_external_provider: bool = True


@dataclass(slots=True)
class TrainerConfig:
    backend: Literal["trl_sft", "trl_dpo", "trl_grpo"] = "trl_sft"
    base_model: str = ""
    output_adapter_only: bool = True


@dataclass(slots=True)
class GateConfig:
    use_fixed_validation: bool = True
    use_smoke_test: bool = True
    smoke_compare_lines: tuple[str, ...] = ("candidate", "active", "base")
    allow_auto_split_validation: bool = True
    validation_split_seed: int = 42


@dataclass(slots=True)
class RegistryConfig:
    artifact_root: str = "artifacts/posttrain"
    version_naming: Literal["round_timestamp"] = "round_timestamp"
    keep_previous: bool = True


@dataclass(slots=True)
class ServeConfig:
    backend: Literal["vllm"] = "vllm"
    openai_compatible: bool = True
    switch_only_at_round_end: bool = True


@dataclass(slots=True)
class SynthesizeConfig:
    enabled: bool = False
    allow_teacher_local_or_remote: bool = True
    record_provenance: bool = True


@dataclass(slots=True)
class WorkflowConfig:
    workflow_name: str
    task_name: str
    method_name: str
    experiment_name: str | None = None


@dataclass(slots=True)
class PostTrainConfig:
    workflow: WorkflowConfig
    round: RoundConfig = field(default_factory=RoundConfig)
    event_store: EventStoreConfig = field(default_factory=EventStoreConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    builder: BuilderConfig = field(default_factory=BuilderConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    synthesize: SynthesizeConfig = field(default_factory=SynthesizeConfig)
