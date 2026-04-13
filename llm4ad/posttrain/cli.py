from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .config_loader import load_posttrain_config


def _load_symbol(import_path: str):
    module_name, symbol_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _load_json_arg(value: str | None):
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def cmd_validate_config(args):
    config = load_posttrain_config(args.config)
    print(
        json.dumps(
            {
                "workflow": config.workflow.workflow_name,
                "task": config.workflow.task_name,
                "method": config.workflow.method_name,
                "trainer_backend": config.trainer.backend,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


def cmd_show_registry(args):
    from .registry import ModelRegistry

    registry = ModelRegistry(args.artifact_root, args.workflow_key)
    print(
        json.dumps(
            {
                "active": registry.get_active(),
                "previous": registry.get_previous(),
                "candidate": registry.get_candidate(),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


def cmd_rollback(args):
    from .registry import ModelRegistry

    registry = ModelRegistry(args.artifact_root, args.workflow_key)
    result = registry.rollback()
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=True))


def cmd_run_workflow(args):
    from .event_store import EventStore
    from .orchestrator import PostTrainOrchestrator
    from .runtime import PostTrainRuntime

    config = load_posttrain_config(args.config)
    llm_factory = _load_symbol(args.llm_factory)
    evaluation_factory = _load_symbol(args.evaluation_factory)
    workflow_fn = _load_symbol(args.workflow_fn)
    profiler_factory = (
        None if args.profiler_factory is None else _load_symbol(args.profiler_factory)
    )

    llm = llm_factory(config=config)
    evaluation = evaluation_factory(config=config)
    profiler = None if profiler_factory is None else profiler_factory(config=config)

    runtime = PostTrainRuntime(
        config,
        workflow_name=config.workflow.workflow_name,
        task_name=config.workflow.task_name,
        method_name=config.workflow.method_name,
    )

    event_root = (
        Path(args.event_root)
        if args.event_root
        else Path("logs") / "posttrain_cli" / runtime.run_id / "events"
    )
    event_store = EventStore(event_root)

    method_kwargs = _load_json_arg(args.method_kwargs) or {}
    workflow = workflow_fn(
        config=config,
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        method_kwargs=method_kwargs,
        runtime=runtime,
        event_store=event_store,
        resume_path=args.resume_path,
    )

    if args.use_orchestrator:
        orchestrator = PostTrainOrchestrator(config, runtime, event_store)
        result = orchestrator.run_round(
            workflow,
            resume_path=args.resume_path,
        )
    else:
        result = workflow.run_search_round(resume_path=args.resume_path)

    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m llm4ad.posttrain.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("--config", required=True)
    validate_parser.set_defaults(func=cmd_validate_config)

    show_registry_parser = subparsers.add_parser("show-registry")
    show_registry_parser.add_argument("--artifact-root", required=True)
    show_registry_parser.add_argument("--workflow-key", required=True)
    show_registry_parser.set_defaults(func=cmd_show_registry)

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--artifact-root", required=True)
    rollback_parser.add_argument("--workflow-key", required=True)
    rollback_parser.set_defaults(func=cmd_rollback)

    run_parser = subparsers.add_parser("run-workflow")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--workflow-fn", required=True)
    run_parser.add_argument("--llm-factory", required=True)
    run_parser.add_argument("--evaluation-factory", required=True)
    run_parser.add_argument("--profiler-factory")
    run_parser.add_argument("--method-kwargs")
    run_parser.add_argument("--resume-path")
    run_parser.add_argument("--event-root")
    run_parser.add_argument("--use-orchestrator", action="store_true")
    run_parser.set_defaults(func=cmd_run_workflow)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
