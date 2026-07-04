#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-/home/yuanyilun/miniconda3/envs/LLM4AD}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
RUNS_PER_VARIANT="${RUNS_PER_VARIANT:-3}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_GPU="${TARGET_GPU:-1}"
TARGET_PORT="${TARGET_PORT:-22002}"
TARGET_GROUP_PORT="${TARGET_GROUP_PORT:-51214}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-/home/yuanyilun/models/deepseek-coder-7b-instruct-v1.5}"

SOURCE_CONFIG="${SOURCE_CONFIG:-${PROJECT_ROOT}/configs/run_eoh_local_rl/eoh_local_rl_cvrp.yaml}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-${PROJECT_ROOT}/logs/CVRP/eoh_local_rl}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PROJECT_ROOT}/logs}"
RUN_ROOT_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT_BASE}"
CONFIG_PATH="${SOURCE_CONFIG}"

if [ ! -x "${PYTHON}" ]; then
  echo "[ERROR] Python not found or not executable: ${PYTHON}" >&2
  exit 1
fi
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [ ! -f "${PROJECT_ROOT}/example/run_eoh/cvrp/run_eoh_local_rl.py" ]; then
  echo "[ERROR] Project root is invalid: ${PROJECT_ROOT}" >&2
  exit 1
fi
if [ ! -d "${LOCAL_MODEL_PATH}" ]; then
  echo "[ERROR] Local model path not found: ${LOCAL_MODEL_PATH}" >&2
  exit 1
fi

BASE_RUNTIME_OVERRIDES="${EOH_RL_RUNTIME_OVERRIDES_JSON:-}"
if [ -z "${BASE_RUNTIME_OVERRIDES}" ]; then
  BASE_RUNTIME_OVERRIDES="{}"
fi
EOH_RL_RUNTIME_OVERRIDES_JSON="$(
  TARGET_GPU="${TARGET_GPU}" \
  TARGET_PORT="${TARGET_PORT}" \
  TARGET_GROUP_PORT="${TARGET_GROUP_PORT}" \
  LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH}" \
  LOG_BASE_DIR="${LOG_BASE_DIR}" \
  "${PYTHON}" -c 'import json,os,sys
def merge(base, overrides):
    result=dict(base)
    for key,value in (overrides or {}).items():
        result[key]=merge(result.get(key, {}), value) if isinstance(value,dict) and isinstance(result.get(key),dict) else value
    return result
try:
    base=json.loads(sys.argv[1]) if sys.argv[1].strip() else {}
except json.JSONDecodeError as exc:
    raise SystemExit(f"EOH_RL_RUNTIME_OVERRIDES_JSON is not valid JSON: {exc}") from exc
gpu=int(os.environ["TARGET_GPU"])
port=int(os.environ["TARGET_PORT"])
group_port=int(os.environ["TARGET_GROUP_PORT"])
forced={
    "eoh_rl": {
        "local_model_path": os.environ["LOCAL_MODEL_PATH"],
        "inference_gpus": [gpu],
        "training_gpus": [gpu],
        "inference_ports": [port],
        "grpo": {
            "vllm_server_port": port,
            "vllm_group_port": group_port,
        },
        "logging": {
            "log_base_dir": os.environ["LOG_BASE_DIR"],
        },
    }
}
print(json.dumps(merge(base, forced), ensure_ascii=False, separators=(",", ":")))' "${BASE_RUNTIME_OVERRIDES}"
)"
export EOH_RL_RUNTIME_OVERRIDES_JSON

readarray -t CFG_VALUES < <("${PYTHON}" -c 'import json,os,sys,yaml; root=yaml.safe_load(open(sys.argv[1],"r",encoding="utf-8"));
def merge(base, overrides):
    result=dict(base)
    for key,value in (overrides or {}).items():
        result[key]=merge(result.get(key, {}), value) if isinstance(value,dict) and isinstance(result.get(key),dict) else value
    return result
text=os.environ.get("EOH_RL_RUNTIME_OVERRIDES_JSON","").strip(); root=merge(root,json.loads(text)) if text else root; data=root["eoh_rl"]; cvrp=data["cvrp"]; scale=cvrp["default_scale"]; scale_cfg=cvrp["scales"][scale]; evo=data["evolution"]; grpo=data["grpo"]; sft=data.get("sft",{}); mode=str(sft.get("mode","cold_start")); lora="<none/cold_start>" if mode=="cold_start" else (sft.get("load_lora_path") or os.path.join(str(sft.get("output_dir","")),"lora_sft")); gate=grpo.get("lora_gate",{}).get("enabled", str(grpo.get("deploy_policy","gated")).lower()=="gated"); print(data["inference_gpus"][0]); print(data["inference_ports"][0]); print(grpo["vllm_group_port"]); print(evo["max_sample_nums"]); print(evo["max_generations"]); print(grpo["num_generations"]); print(evo["pop_size"]); print(cvrp["task_name"]); print(scale); print(scale_cfg["problem_size"]); print(scale_cfg["n_instance"]); print(mode); print(lora); print(str(data.get("adaptive_operator",{}).get("enabled",False)).lower()); print(str(data.get("collapse",{}).get("enabled",False)).lower()); print(str(gate).lower())' "${CONFIG_PATH}")

TARGET_GPU="${CFG_VALUES[0]}"
TARGET_PORT="${CFG_VALUES[1]}"
TARGET_GROUP_PORT="${CFG_VALUES[2]}"
MAX_SAMPLE_NUMS="${CFG_VALUES[3]}"
EOH_MAX_GENERATIONS="${CFG_VALUES[4]}"
NUM_GENERATIONS="${CFG_VALUES[5]}"
POP_SIZE="${CFG_VALUES[6]}"
TASK_NAME="${CFG_VALUES[7]}"
SCALE_NAME="${CFG_VALUES[8]}"
PROBLEM_SIZE="${CFG_VALUES[9]}"
N_INSTANCE="${CFG_VALUES[10]}"
SFT_MODE="${CFG_VALUES[11]}"
SFT_LORA_PATH="${CFG_VALUES[12]}"
ADAPTIVE_OPERATOR_ENABLED="${CFG_VALUES[13]}"
COLLAPSE_ENABLED="${CFG_VALUES[14]}"
LORA_GATE_ENABLED="${CFG_VALUES[15]}"

for i in $(seq 1 "${RUNS_PER_VARIANT}"); do
  RUN_ID="adaptive_op_plus_collapse_np1_CVRP_run${i}_${RUN_ROOT_TS}"
  RUN_LOG_DIR="${RUN_ROOT}/${RUN_ID}"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "============================================================"
    echo "EoH-RL dry run ${i}/${RUNS_PER_VARIANT}: ${RUN_ID}"
    echo "GPU=${TARGET_GPU} port=${TARGET_PORT} group_port=${TARGET_GROUP_PORT} max_sample_nums=${MAX_SAMPLE_NUMS} max_generations=${EOH_MAX_GENERATIONS} num_generations=${NUM_GENERATIONS} pop_size=${POP_SIZE}"
    echo "mechanisms adaptive_operator=${ADAPTIVE_OPERATOR_ENABLED} collapse=${COLLAPSE_ENABLED} lora_gate=${LORA_GATE_ENABLED}"
    echo "task=${TASK_NAME} scale=${SCALE_NAME} problem_size=${PROBLEM_SIZE} n_instance=${N_INSTANCE} sft_mode=${SFT_MODE} sft_lora=${SFT_LORA_PATH}"
    echo "config=${CONFIG_PATH}"
    continue
  fi
  EOH_RL_CONFIG_PATH="${CONFIG_PATH}" EOH_RL_RUN_ID="${RUN_ID}" EOH_RL_RUN_LOG_DIR="${RUN_LOG_DIR}" PYTHONUNBUFFERED=1 "${PYTHON}" "${PROJECT_ROOT}/example/run_eoh/cvrp/run_eoh_local_rl.py"
done
