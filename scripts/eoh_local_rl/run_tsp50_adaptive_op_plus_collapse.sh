#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_GPU="${TARGET_GPU:-0}"
export TARGET_PORT="${TARGET_PORT:-22001}"
export TARGET_GROUP_PORT="${TARGET_GROUP_PORT:-51213}"
export TSP_SCALE="${TSP_SCALE:-small}"
export TSP_LABEL="${TSP_LABEL:-50}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-adaptive_op_plus_collapse_np1_TSP50}"
export SFT_LOAD_LORA_PATH="${SFT_LOAD_LORA_PATH:-/home/yuanyilun/projects/LLM4AD/data/SFT/tsp/small/lora_sft}"

exec "${SCRIPT_DIR}/_run_tsp_adaptive_op_plus_collapse.sh" "$@"
