#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_GPU="${TARGET_GPU:-1}"
export TARGET_PORT="${TARGET_PORT:-22002}"
export TARGET_GROUP_PORT="${TARGET_GROUP_PORT:-51214}"
export TSP_SCALE="${TSP_SCALE:-medium-100}"
export TSP_LABEL="${TSP_LABEL:-100}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-adaptive_op_plus_collapse_np1_TSP100}"
export SFT_LOAD_LORA_PATH="${SFT_LOAD_LORA_PATH:-/home/yuanyilun/projects/LLM4AD/data/SFT/tsp/medium-100/lora_sft}"

exec "${SCRIPT_DIR}/_run_tsp_adaptive_op_plus_collapse.sh" "$@"
