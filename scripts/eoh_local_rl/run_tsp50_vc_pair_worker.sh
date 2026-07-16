#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
VARIANTS="${VARIANTS:?VARIANTS is required}"
SEEDS="${SEEDS:-42,43,44}"
EXPERIMENT_ID="${EXPERIMENT_ID:-tsp50_vc_pair_pilot}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
INITIAL_BATCH="${INITIAL_BATCH:-${PROJECT_ROOT}/logs/TSP/eoh_local_rl/tsp50_grpo_500_6seed/20260712_run01}"
TARGET_PORT=$((22001 + GPU_ID))
TARGET_GROUP_PORT=$((51213 + GPU_ID))

run_variant() {
  local variant="$1" seed="$2" enabled reward_mode lr lifecycle save_lora
  case "${variant}" in
    b0_fixed) enabled=0; reward_mode=vc_pair; lr=""; lifecycle=persistent ;;
    vc_pair_persistent) enabled=1; reward_mode=vc_pair; lr=""; lifecycle=persistent ;;
    vc_pair_reset) enabled=1; reward_mode=vc_pair; lr=""; lifecycle=reset ;;
    vc_pair_lr0) enabled=1; reward_mode=vc_pair; lr=0.0; lifecycle=persistent ;;
    aggregate_reset) enabled=1; reward_mode=aggregate; lr=""; lifecycle=reset ;;
    validity_reset) enabled=1; reward_mode=validity_only; lr=""; lifecycle=reset ;;
    shuffled_reset) enabled=1; reward_mode=performance_shuffled; lr=""; lifecycle=reset ;;
    *) echo "[ERROR] Unknown variant: ${variant}" >&2; exit 1 ;;
  esac
  save_lora=1
  [ "${enabled}" = 1 ] && [ "${lr}" != 0.0 ] || save_lora=0
  local initial="${INITIAL_BATCH}/tsp50_grpo_500_6seed_init_TSP50_seed${seed}_20260712_run01/checkpoints/latest_finish.json"
  [ -f "${initial}" ] || { echo "[ERROR] Missing initial population: ${initial}" >&2; exit 1; }
  EXPERIMENT_ID="${EXPERIMENT_ID}" RUN_BATCH_ID="${RUN_STAMP}" RUN_STAMP="${RUN_STAMP}" \
  VARIANT="${variant}" ENABLE_GRPO="${enabled}" SEED="${seed}" INITIAL_POPULATION_PATH="${initial}" \
  TARGET_GPU="${GPU_ID}" TARGET_PORT="${TARGET_PORT}" TARGET_GROUP_PORT="${TARGET_GROUP_PORT}" \
  MAX_STEPS="${MAX_STEPS:-100}" N_GENERATIONS=4 POPULATION_SIZE=10 LR="${lr}" REWARD_MODE="${reward_mode}" \
  TRAINER_LIFECYCLE="${lifecycle}" \
  SAVE_FINAL_LORA="${save_lora}" COMPRESS_HISTORY=1 \
  "${SCRIPT_DIR}/run_tsp50_eoh_rl.sh"
}

for seed in ${SEEDS//,/ }; do
  for variant in ${VARIANTS//,/ }; do
    run_variant "${variant}" "${seed}"
  done
done
