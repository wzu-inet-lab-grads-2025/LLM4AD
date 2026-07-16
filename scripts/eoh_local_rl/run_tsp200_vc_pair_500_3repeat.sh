#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LORA_PATH="/home/yuanyilun/projects/LLM4AD/data/SFT/tsp/medium-200/lora_sft"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
[ -d "${LORA_PATH}" ] || { echo "[ERROR] Missing SFT LoRA: ${LORA_PATH}" >&2; exit 1; }

for repeat in 1 2 3; do
  EXPERIMENT_ID=tsp200_vc_pair_500_3repeat RUN_BATCH_ID="${RUN_STAMP}" RUN_STAMP="${RUN_STAMP}" \
  RUN_ID_PREFIX="tsp200_vc_pair_persistent_TSP200_seed42_repeat${repeat}" \
  VARIANT=vc_pair_persistent ENABLE_GRPO=1 SEED=42 INITIAL_POPULATION_PATH= \
  TARGET_GPU=2 TARGET_PORT=22003 TARGET_GROUP_PORT=51215 \
  SFT_MODE=load_existing SFT_LOAD_LORA_PATH="${LORA_PATH}" \
  MAX_STEPS=500 N_GENERATIONS=4 POPULATION_SIZE=10 REWARD_MODE=vc_pair \
  TRAINER_LIFECYCLE=persistent SAVE_FINAL_LORA=1 COMPRESS_HISTORY=1 \
  "${SCRIPT_DIR}/run_tsp200_eoh_rl.sh"
done
