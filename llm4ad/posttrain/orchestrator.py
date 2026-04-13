from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .augment import DedupAugmenter
from .builders import OutcomeBuilder, PreferenceBuilder, VerifierBuilder
from .collector import UnifiedCollector
from .formatters import TrlChatFormatter, TrlPreferenceFormatter, VerifierJsonlFormatter
from .gates import PromotionGate
from .registry import ModelRegistry
from .replay_buffer import ReplayBuffer
from .schemas import DatasetManifest
from .trainers import DryRunTrainer, TrlDpoTrainer, TrlGrpoTrainer, TrlSftTrainer


class PostTrainOrchestrator:
    def __init__(
        self,
        config,
        runtime,
        event_store,
        collector=None,
        replay_buffer=None,
        registry=None,
        gate=None,
        serve_manager=None,
    ):
        self.config = config
        self.runtime = runtime
        self.event_store = event_store
        self.collector = collector or UnifiedCollector(config.collector)
        self.replay_buffer = replay_buffer or ReplayBuffer(config.replay_buffer)
        workflow_key = "/".join(
            [
                config.workflow.task_name,
                config.workflow.workflow_name,
                runtime.experiment_name,
            ]
        )
        self.registry = registry or ModelRegistry(
            config.registry.artifact_root, workflow_key
        )
        self.gate = gate or PromotionGate(config=config.gate)
        self.serve_manager = serve_manager or self._build_default_serve_manager()

    def run_round(
        self,
        workflow,
        *,
        resume_path: str | None = None,
        validation_spec=None,
        smoke_spec=None,
    ):
        search_report = workflow.run_search_round(resume_path=resume_path)
        validation_spec = validation_spec or workflow.build_validation_spec()
        smoke_spec = smoke_spec or workflow.build_smoke_spec()
        records = self.collect_round_data(workflow, search_report)
        records = self.run_augment(records)
        records = self.run_synthesize(records)
        self.replay_buffer.add_records(records)
        dataset_manifests = self.build_datasets(records, search_report=search_report)
        candidate = self.train_candidate(dataset_manifests)
        gate_report = self.evaluate_candidate(
            candidate,
            validation_spec=validation_spec,
            smoke_spec=smoke_spec,
            search_report=search_report,
        )
        promotion = self.maybe_promote(candidate, gate_report)
        report = {
            "run_id": self.runtime.run_id,
            "round_id": self.runtime.round_id,
            "search_report": search_report,
            "dataset_manifests": {
                name: asdict(manifest) for name, manifest in dataset_manifests.items()
            },
            "candidate": None if candidate is None else asdict(candidate),
            "gate_report": gate_report,
            "promotion": None if promotion is None else asdict(promotion),
        }
        report["round_manifest_path"] = self.write_round_manifest(
            report, log_dir=search_report.get("log_dir")
        )
        return report

    def collect_round_data(self, workflow, search_report) -> list:
        return self.collector.collect_round_records(
            log_dir=search_report.get("log_dir"),
            event_store=self.event_store,
            run_id=self.runtime.run_id,
            round_id=self.runtime.round_id,
            method_name=workflow.config.workflow.method_name,
        )

    def run_augment(self, records):
        return DedupAugmenter().augment(records, context=self.runtime.snapshot())

    def run_synthesize(self, records):
        return records

    def build_datasets(self, records, *, search_report) -> dict[str, DatasetManifest]:
        datasets_dir = self._get_datasets_dir(search_report.get("log_dir"))
        backend = self.config.trainer.backend
        manifests = {}

        if backend in {"dryrun", "trl_sft"}:
            examples = OutcomeBuilder().build(
                records, config=self.config.builder, context=self.runtime.snapshot()
            )
            output_path = TrlChatFormatter().format(
                examples,
                output_dir=datasets_dir,
                dataset_name=f"round_{self.runtime.round_id:04d}_outcome",
                context=self.runtime.snapshot(),
            )
            manifests["outcome"] = DatasetManifest(
                round_id=self.runtime.round_id,
                dataset_type="outcome",
                source_runs=[self.runtime.run_id],
                source_rounds=[self.runtime.round_id],
                record_count=len(examples),
                output_path=output_path,
                config_digest="outcome_v1",
            )
        elif backend == "trl_dpo":
            examples = PreferenceBuilder().build(
                records, config=self.config.builder, context=self.runtime.snapshot()
            )
            output_path = TrlPreferenceFormatter().format(
                examples,
                output_dir=datasets_dir,
                dataset_name=f"round_{self.runtime.round_id:04d}_preference",
                context=self.runtime.snapshot(),
            )
            manifests["preference"] = DatasetManifest(
                round_id=self.runtime.round_id,
                dataset_type="preference",
                source_runs=[self.runtime.run_id],
                source_rounds=[self.runtime.round_id],
                record_count=len(examples),
                output_path=output_path,
                config_digest="preference_v1",
            )
        elif backend == "trl_grpo":
            examples = VerifierBuilder().build(
                records, config=self.config.builder, context=self.runtime.snapshot()
            )
            output_path = VerifierJsonlFormatter().format(
                examples,
                output_dir=datasets_dir,
                dataset_name=f"round_{self.runtime.round_id:04d}_verifier",
                context=self.runtime.snapshot(),
            )
            manifests["verifier"] = DatasetManifest(
                round_id=self.runtime.round_id,
                dataset_type="verifier",
                source_runs=[self.runtime.run_id],
                source_rounds=[self.runtime.round_id],
                record_count=len(examples),
                output_path=output_path,
                config_digest="verifier_v1",
            )
        else:
            raise ValueError(f"Unsupported trainer backend: {backend}")

        return manifests

    def train_candidate(self, dataset_manifests):
        if not dataset_manifests:
            return None
        backend = self.config.trainer.backend
        model_spec = {"round_id": self.runtime.round_id}
        if backend == "dryrun":
            manifest = dataset_manifests["outcome"]
            trainer = DryRunTrainer()
        elif backend == "trl_sft":
            manifest = dataset_manifests["outcome"]
            trainer = TrlSftTrainer()
        elif backend == "trl_dpo":
            manifest = dataset_manifests["preference"]
            trainer = TrlDpoTrainer()
        elif backend == "trl_grpo":
            manifest = dataset_manifests["verifier"]
            trainer = TrlGrpoTrainer()
        else:
            raise ValueError(f"Unsupported trainer backend: {backend}")

        if manifest.record_count == 0 or not manifest.output_path:
            return None

        candidate = trainer.train(
            manifest.output_path,
            model_spec=model_spec,
            train_config=self.config.trainer,
        )
        self.registry.register_candidate(candidate)
        return candidate

    def evaluate_candidate(
        self, candidate, *, validation_spec=None, smoke_spec=None, search_report=None
    ):
        if candidate is None:
            return {
                "passed": False,
                "static": {
                    "passed": False,
                    "details": {"reason": "No candidate was trained."},
                },
                "smoke": {
                    "passed": False,
                    "details": {"reason": "No candidate was trained."},
                },
            }
        active = self.registry.get_active()
        base = {
            "version_id": "base",
            "path": self.config.trainer.base_model,
        }
        gate_context = {
            "run_context": self.runtime.snapshot(),
            "smoke_resume_path": None
            if search_report is None
            else search_report.get("log_dir"),
        }
        return self.gate.evaluate(
            asdict(candidate) if is_dataclass(candidate) else candidate,
            active,
            base,
            validation_spec=validation_spec,
            smoke_spec=smoke_spec,
            context=gate_context,
        )

    def maybe_promote(self, candidate, gate_report):
        if candidate is None or not gate_report.get("passed"):
            return None
        promotion = self.registry.promote_candidate(self.runtime.round_id)
        if self.serve_manager is not None:
            self.serve_manager.switch(promotion.candidate_version)
        return promotion

    def _build_default_serve_manager(self):
        serve_cfg = self.config.serve
        if serve_cfg.backend != "vllm":
            return None
        if not serve_cfg.gpus or not serve_cfg.ports:
            return None
        from .serve.model_router import RegistryModelRouter
        from .serve.vllm_manager import VLLMServeManager

        return VLLMServeManager(
            RegistryModelRouter(self.registry),
            tokenizer_path=serve_cfg.tokenizer_path,
            gpus=list(serve_cfg.gpus),
            ports=list(serve_cfg.ports),
        )

    def write_round_manifest(self, report: dict, *, log_dir: str | None):
        rounds_dir = self._get_rounds_dir(log_dir)
        round_dir = rounds_dir / f"round_{self.runtime.round_id:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = round_dir / "round_manifest.json"
        manifest_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return str(manifest_path)

    def _get_datasets_dir(self, log_dir: str | None) -> Path:
        if log_dir is None:
            base_dir = Path(self.config.registry.artifact_root) / "datasets"
        else:
            base_dir = Path(log_dir) / "posttrain" / "datasets"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def _get_rounds_dir(self, log_dir: str | None) -> Path:
        if log_dir is None:
            base_dir = Path(self.config.registry.artifact_root) / "rounds"
        else:
            base_dir = Path(log_dir) / "posttrain" / "rounds"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
