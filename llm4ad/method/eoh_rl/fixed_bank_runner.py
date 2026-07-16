from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from ...base import SecureEvaluator, TextFunctionProgramConverter
from .args import apply_runtime_defaults
from .config_runner import TASK_REGISTRY, _apply_runtime_overrides, _build_task, _load_yaml_root, _select_task_family
from .controlled_eval import (
    build_query_bank,
    evaluate_fixed_bank,
    load_checkpoint_parents,
    load_query_bank,
    reevaluate_parents,
    same_valid_curves,
    same_valid_summaries,
    save_query_bank,
)
from .rl.grpo_trainer import ResidentGRPOPolicy, build_reward_fn_from_task_rl


def _runtime(config_path: str, task_family: str | None):
    root = _apply_runtime_overrides(_load_yaml_root(config_path))
    cfg = apply_runtime_defaults(dict(root["eoh_rl"]))
    family = _select_task_family(task_family, cfg)
    spec = TASK_REGISTRY[family]
    evolution = dict(cfg["evolution"])
    task, task_name, scale, params = _build_task(dict(cfg[spec["task_key"]]), evolution, spec)
    return cfg, family, task, task_name, scale, params


def build(args) -> None:
    cfg, family, task, task_name, scale, params = _runtime(args.config, args.task_family)
    minimize = bool(cfg["task_rl_common"]["minimize"])
    evaluator = SecureEvaluator(task)
    template_program = TextFunctionProgramConverter.text_to_program(task.template_program)
    parents = load_checkpoint_parents(args.checkpoint)
    parents.sort(key=lambda item: -float(item.score) if minimize else float(item.score), reverse=True)
    count = min(len(parents), max(2, math.ceil((args.size + 2) / 4)))
    if count < len(parents):
        parents = [parents[round(index * (len(parents) - 1) / (count - 1))] for index in range(count)]
    parents = reevaluate_parents(
        parents,
        evaluator=evaluator,
        template_program=template_program,
    )
    records = build_query_bank(
        parents,
        task_description=task.task_description,
        template_function=TextFunctionProgramConverter.text_to_function(task.template_program),
        size=args.size,
        group_size=args.group_size,
        minimize=minimize,
        seed=args.seed,
    )
    payload = save_query_bank(args.output, records, {"task_family": family, "task_name": task_name, "scale": scale, "task_params": params, "source_checkpoint": os.path.abspath(args.checkpoint)})
    print(json.dumps({"bank_id": payload["bank_id"], "query_count": len(records), "output": os.path.abspath(args.output)}, ensure_ascii=False))


def evaluate(args) -> None:
    cfg, family, task, task_name, scale, params = _runtime(args.config, args.task_family)
    bank = load_query_bank(args.bank)
    records = bank["records"]
    generations = int(records[0]["group_size"])
    if any(int(record["group_size"]) != generations for record in records):
        raise ValueError("fixed bank contains mixed group sizes")
    grpo = dict(cfg["grpo"])
    grpo.update(seed=args.seed, num_generations=generations, generation_batch_size=generations, per_device_train_batch_size=generations)
    gpus = list(cfg["training_gpus"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gpu)) for gpu in sorted(set(gpus + list(cfg["inference_gpus"]))))
    adapter = None if args.base_model else args.adapter or cfg["sft"].get("load_lora_path")
    policy = ResidentGRPOPolicy(
        model_name_or_path=cfg["local_model_path"],
        grpo_config=grpo,
        adapter_config=cfg["lora"],
        training_gpus=gpus,
        initial_adapter_path=adapter,
        quantization=cfg.get("inference_quantization"),
    )
    try:
        result = evaluate_fixed_bank(
            policy=policy,
            records=records,
            evaluator=SecureEvaluator(task),
            template_program=TextFunctionProgramConverter.text_to_program(task.template_program),
            output_dir=args.output,
            reward=build_reward_fn_from_task_rl(cfg["task_rl_common"]),
        )
        _write_json(Path(args.output, "fixed_bank_manifest.json"), {
            "bank_id": bank["bank_id"], "bank_path": os.path.abspath(args.bank), "adapter_path": None if adapter is None else os.path.abspath(adapter),
            "base_model": bool(args.base_model), "seed": args.seed, "task_family": family, "task_name": task_name, "scale": scale, "task_params": params,
            "summary": result["summary"],
        })
    finally:
        policy.close()


def compare(args) -> None:
    event_sets, bank_id = {}, None
    for item in args.result:
        label, path = item.split("=", 1)
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        if bank_id is not None and payload.get("bank_id") != bank_id:
            raise ValueError("fixed-bank results use different query banks")
        bank_id = payload.get("bank_id")
        event_sets[label] = payload["events"]
    payload = {"bank_id": bank_id, "same_valid": same_valid_summaries(event_sets, minimize=args.minimize), "same_valid_curves": same_valid_curves(event_sets, minimize=args.minimize)}
    _write_json(Path(args.output), payload)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--config", required=True)
    build_parser.add_argument("--checkpoint", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--task-family")
    build_parser.add_argument("--size", type=int, default=100)
    build_parser.add_argument("--group-size", type=int, default=4)
    build_parser.add_argument("--seed", type=int, default=42)
    build_parser.set_defaults(run=build)

    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--bank", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--adapter")
    eval_parser.add_argument("--base-model", action="store_true")
    eval_parser.add_argument("--task-family")
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.set_defaults(run=evaluate)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--result", action="append", required=True, metavar="LABEL=PATH")
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--minimize", action="store_true")
    compare_parser.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
