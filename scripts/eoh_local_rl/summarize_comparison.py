#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from pathlib import Path


def load(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def variant(run_id, enabled, reward_mode):
    if "_b0_fixed_" in run_id:
        return "b0_fixed"
    if "_vc_pair_lr0_" in run_id:
        return "vc_pair_lr0"
    return reward_mode if enabled else "b0_fixed"


def update_stats(run_dir):
    eligible = parent = frontier = paired_positive = zero_std = 0
    optimizer_steps = 0
    update_count = 0
    for path in sorted((run_dir / "updates").glob("update_*/metrics.json")):
        metrics = load(path)
        summary = metrics.get("reward_summary") or {}
        funnel = summary.get("funnel") or {}
        eligible += int(funnel.get("population_eligible_count", 0) or 0)
        parent += int(summary.get("parent_improve_count", 0) or 0)
        frontier += int(summary.get("frontier_improve_count", 0) or 0)
        paired_positive += int((summary.get("paired_outcome_counts") or {}).get("positive", 0) or 0)
        zero_std += float(summary.get("frac_reward_zero_std", 0) or 0)
        optimizer_steps += int(metrics.get("optimizer_steps_this_update", 0) or 0)
        update_count += 1
    return {
        "parent_improve_given_eligible": parent / eligible if eligible else 0.0,
        "frontier_improve_given_eligible": frontier / eligible if eligible else 0.0,
        "paired_positive_given_eligible": paired_positive / eligible if eligible else 0.0,
        "zero_std_update_rate": zero_std / update_count if update_count else 0.0,
        "optimizer_steps": optimizer_steps,
    }


def read_run(run_dir):
    config_path = run_dir / "config.json"
    summary_path = run_dir / "run_summary.json"
    manifest_path = run_dir / "run_manifest.json"
    checkpoint = run_dir / "checkpoints" / "latest_finish.json"
    if not all(path.is_file() for path in (config_path, summary_path, manifest_path, checkpoint)):
        return None
    config, summary, manifest = load(config_path), load(summary_path), load(manifest_path)
    if int(config.get("online_update_count", 0) or 0) == 0:
        return None
    if manifest.get("status") != "completed" or int(summary.get("online_update_count", 0) or 0) != int(config["online_update_count"]):
        return None
    enabled = bool(config["grpo_enabled"])
    funnel = summary.get("online_funnel") or {}
    curve = load(checkpoint).get("best_curve", [])
    best_values = [float(row["best"]) for row in curve if row.get("best") is not None]
    return {
        "run_id": config["run_id"],
        "variant": variant(config["run_id"], enabled, str(config.get("reward_mode") or "vc_pair")),
        "seed": int(config["seed"]),
        "grpo_enabled": enabled,
        "learning_rate": float((config.get("grpo_config") or {}).get("learning_rate", 0) or 0),
        "initial_population_path": summary.get("initial_population_path"),
        "updates": int(summary.get("online_update_count", 0) or 0),
        "completions": int(funnel.get("completion_count", 0) or 0),
        "eligible_rate": float(funnel.get("population_eligible_rate", 0) or 0),
        "final_best": summary.get("best_score"),
        "mean_best_over_updates": statistics.fmean(best_values) if best_values else None,
        "wall_seconds": float((summary.get("timing_totals") or {}).get("run_wall_elapsed", 0) or 0),
        **update_stats(run_dir),
    }


def paired_summary(rows):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["variant"]] = row
    pairs = []
    for seed, variants in sorted(by_seed.items()):
        if {"b0_fixed", "vc_pair"}.issubset(variants):
            b0, b1 = variants["b0_fixed"], variants["vc_pair"]
            pairs.append({
                "seed": seed,
                "b0_final_best": b0["final_best"],
                "vc_pair_final_best": b1["final_best"],
                "vc_pair_minus_b0": float(b1["final_best"]) - float(b0["final_best"]),
                "same_initial_population": b0["initial_population_path"] == b1["initial_population_path"],
            })
    deltas = [row["vc_pair_minus_b0"] for row in pairs]
    return {
        "pair_count": len(pairs),
        "mean_vc_pair_minus_b0": statistics.fmean(deltas) if deltas else None,
        "median_vc_pair_minus_b0": statistics.median(deltas) if deltas else None,
        "vc_pair_win_count": sum(delta > 0 for delta in deltas),
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    rows = [row for path in sorted(args.experiment_dir.iterdir()) if path.is_dir() if (row := read_run(path))]
    if not rows:
        raise SystemExit(f"No completed runs found in {args.experiment_dir}")
    fields = list(rows[0])
    with (args.experiment_dir / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    paired = paired_summary(rows)
    with (args.experiment_dir / "paired_summary.json").open("w", encoding="utf-8") as file:
        json.dump(paired, file, indent=2, ensure_ascii=False)
    print(json.dumps(paired, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
