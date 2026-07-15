#!/bin/bash

set -euo pipefail

GPU_ID=0 VARIANTS=b0_fixed,vc_pair_lr0 SEEDS="${SEEDS:-42,43,44,45,46,47}" \
EXPERIMENT_ID="${EXPERIMENT_ID:-tsp50_vc_pair_500_6seed}" \
RUN_STAMP="${RUN_STAMP:-20260715_vc_pair_500_6seed_run01}" MAX_STEPS="${MAX_STEPS:-500}" \
"$(dirname "$0")/run_tsp50_vc_pair_worker.sh"
