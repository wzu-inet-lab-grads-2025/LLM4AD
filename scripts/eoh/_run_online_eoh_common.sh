#!/bin/bash

set -euo pipefail

TASK_FAMILY="${1:?missing task family}"
TASK_DISPLAY="${2:?missing task display name}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-/home/yuanyilun/miniconda3/envs/LLM4AD}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
RUNS_PER_VARIANT="${RUNS_PER_VARIANT:-1}"
DRY_RUN="${DRY_RUN:-0}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/run_eoh/eoh_online_${TASK_FAMILY}.yaml}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PROJECT_ROOT}/logs}"
RUN_ROOT_TS="$(date +%Y%m%d_%H%M%S)"
ENTRY_SCRIPT="${PROJECT_ROOT}/example/run_eoh/${TASK_FAMILY}/run_eoh.py"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/llm4ad_mplconfig}"
mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

if [ ! -x "${PYTHON}" ]; then
  echo "[ERROR] Python not found or not executable: ${PYTHON}" >&2
  exit 1
fi
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [ ! -f "${ENTRY_SCRIPT}" ]; then
  echo "[ERROR] Entry script not found: ${ENTRY_SCRIPT}" >&2
  exit 1
fi

BASE_RUNTIME_OVERRIDES="${EOH_ONLINE_RUNTIME_OVERRIDES_JSON:-}"
if [ -z "${BASE_RUNTIME_OVERRIDES}" ]; then
  BASE_RUNTIME_OVERRIDES="{}"
fi
EOH_ONLINE_RUNTIME_OVERRIDES_JSON="$(
  ONLINE_MODEL_OVERRIDE="${ONLINE_MODEL:-}" \
  LOG_BASE_DIR="${LOG_BASE_DIR}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  "${PYTHON}" -c 'import json,os,sys
import yaml
def merge(base, overrides):
    result = dict(base)
    for key, value in (overrides or {}).items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result
try:
    base = json.loads(sys.argv[1]) if sys.argv[1].strip() else {}
except json.JSONDecodeError as exc:
    raise SystemExit(f"EOH_ONLINE_RUNTIME_OVERRIDES_JSON is not valid JSON: {exc}") from exc
with open(os.environ["CONFIG_PATH"], "r", encoding="utf-8") as file:
    root = yaml.safe_load(file) or {}
root_key = "online_run" if isinstance(root.get("online_run"), dict) else "eoh_online"
if root_key == "online_run" and "online_run" not in base and "eoh_online" in base:
    base["online_run"] = base.pop("eoh_online")
elif root_key == "eoh_online" and "eoh_online" not in base and "online_run" in base:
    base["eoh_online"] = base.pop("online_run")
forced = {root_key: {"logging": {"log_base_dir": os.environ["LOG_BASE_DIR"]}}}
model = os.environ.get("ONLINE_MODEL_OVERRIDE", "").strip()
if model:
    forced[root_key]["online_model"] = {"name": model}
print(json.dumps(merge(base, forced), ensure_ascii=False, separators=(",", ":")))' "${BASE_RUNTIME_OVERRIDES}"
)"
export EOH_ONLINE_RUNTIME_OVERRIDES_JSON

if ! CFG_OUTPUT="$(
  TASK_FAMILY="${TASK_FAMILY}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  "${PYTHON}" -c 'import json,os,sys,yaml
sys.path.insert(0, os.environ["PROJECT_ROOT"])
from llm4ad.method.eoh.config_runner import (
    _apply_runtime_overrides,
    _apply_scale_method_overrides,
    _load_online_run_config,
    _load_selected_model_config,
    _load_yaml_root,
    _resolve_method_config,
    _resolve_task_config,
    _task_spec,
    _validate_method_params,
    _validate_pure_online_config,
)
config_path = sys.argv[1]
task_family = os.environ["TASK_FAMILY"]
spec = _task_spec(task_family)
root = _apply_runtime_overrides(_load_yaml_root(config_path))
data = _load_online_run_config(root, config_path)
_validate_pure_online_config(data)
model_name, _, _ = _load_selected_model_config(data)
task_name, scale, scale_cfg, scale_method_overrides = _resolve_task_config(data, spec)
method_name, _, params = _resolve_method_config(data)
params = _apply_scale_method_overrides(method_name, params, scale_method_overrides)
_validate_method_params(method_name, params)
print(model_name)
print(method_name)
print(task_name)
print(scale)
print(json.dumps(scale_cfg, ensure_ascii=False, sort_keys=True))
print(json.dumps(params, ensure_ascii=False, sort_keys=True))' "${CONFIG_PATH}"
)"; then
  echo "[ERROR] Failed to resolve online run config: ${CONFIG_PATH}" >&2
  exit 1
fi
readarray -t CFG_VALUES <<< "${CFG_OUTPUT}"
if [ "${#CFG_VALUES[@]}" -lt 6 ]; then
  echo "[ERROR] Failed to resolve online run config: ${CONFIG_PATH}" >&2
  exit 1
fi

ONLINE_MODEL_EFFECTIVE="${CFG_VALUES[0]}"
METHOD_NAME="${CFG_VALUES[1]}"
TASK_NAME="${CFG_VALUES[2]}"
SCALE_NAME="${CFG_VALUES[3]}"
TASK_PARAMS_JSON="${CFG_VALUES[4]}"
METHOD_PARAMS_JSON="${CFG_VALUES[5]}"
RUN_ROOT="${LOG_BASE_DIR}/${TASK_DISPLAY}/${METHOD_NAME}_online"

echo "Online ${METHOD_NAME} ${TASK_DISPLAY} runner. RUNS_PER_VARIANT=${RUNS_PER_VARIANT} DRY_RUN=${DRY_RUN}"
echo "Run root: ${RUN_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Model: ${ONLINE_MODEL_EFFECTIVE}"
echo "Task: ${TASK_NAME} scale=${SCALE_NAME} params=${TASK_PARAMS_JSON}"
echo "Method params: ${METHOD_PARAMS_JSON}"
echo

for i in $(seq 1 "${RUNS_PER_VARIANT}"); do
  RUN_ID="online_${METHOD_NAME}_${ONLINE_MODEL_EFFECTIVE}_np1_${TASK_DISPLAY}_run${i}_${RUN_ROOT_TS}"
  RUN_LOG_DIR="${RUN_ROOT}/${RUN_ID}"
  echo "============================================================"
  echo "Online ${TASK_DISPLAY}: run ${i}/${RUNS_PER_VARIANT}"
  echo "run_id=${RUN_ID}"
  echo "run_log_dir=${RUN_LOG_DIR}"

  if [ "${DRY_RUN}" = "1" ]; then
    EOH_ONLINE_RUN_ID="${RUN_ID}" \
    EOH_ONLINE_RUN_LOG_DIR="${RUN_LOG_DIR}" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" "${ENTRY_SCRIPT}" --config "${CONFIG_PATH}" --dry-run
    continue
  fi

  EOH_ONLINE_RUN_ID="${RUN_ID}" \
  EOH_ONLINE_RUN_LOG_DIR="${RUN_LOG_DIR}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON}" "${ENTRY_SCRIPT}" --config "${CONFIG_PATH}"
done
