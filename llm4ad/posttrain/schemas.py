from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RunContext:
    task_name: str
    workflow_name: str
    experiment_name: str
    method_name: str
    run_id: str
    round_id: int
    phase: str
    sample_id: str | None = None
    operator: str | None = None
    parents: list[int] | None = None
    generation: int | None = None
    cluster_id: int | None = None
    thread_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMRequestEvent:
    run_id: str
    round_id: int
    sample_id: str
    phase: str
    prompt: str | None
    messages: list[dict[str, Any]] | None
    image64s: list[str] | None
    model_id: str | None
    sampling_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponseEvent:
    run_id: str
    round_id: int
    sample_id: str
    raw_response: str | None
    latency_seconds: float | None


@dataclass(slots=True)
class ParseEvent:
    run_id: str
    round_id: int
    sample_id: str
    parse_ok: bool
    program_ok: bool
    error_type: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class EvalTraceRecord:
    run_id: str
    round_id: int
    sample_id: str
    score: Any
    eval_time: float | None
    compile_ok: bool | None
    runtime_ok: bool | None
    timeout: bool
    error_type: str | None
    error_message: str | None
    score_breakdown: dict[str, Any] | None = None


@dataclass(slots=True)
class SampleRecord:
    run_id: str
    round_id: int
    sample_id: str
    method_name: str
    phase: str
    operator: str | None
    parents: list[int] | None
    generation: int | None
    prompt: str | None
    messages: list[dict[str, Any]] | None
    response: str | None
    function: str | None
    program: str | None
    algorithm: str | None
    score: Any
    sample_time: float | None
    eval_time: float | None
    observation: Any = None
    image64: Any = None
    provenance: dict[str, Any] | None = None


@dataclass(slots=True)
class DatasetManifest:
    round_id: int
    dataset_type: str
    source_runs: list[str]
    source_rounds: list[int]
    record_count: int
    output_path: str | None
    config_digest: str


@dataclass(slots=True)
class ModelArtifactRecord:
    version_id: str
    round_id: int
    artifact_type: str
    path: str
    base_model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromotionRecord:
    round_id: int
    candidate_version: str
    previous_active_version: str | None
    promoted: bool
    reason: str
    static_gate_passed: bool
    smoke_gate_passed: bool
