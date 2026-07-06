#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-/home/yuanyilun/miniconda3/envs/LLM4AD}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"

RUNS_PER_VARIANT="${RUNS_PER_VARIANT:-3}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_GPU="${TARGET_GPU:-0}"
TARGET_PORT="${TARGET_PORT:-22001}"
TARGET_GROUP_PORT="${TARGET_GROUP_PORT:-$((TARGET_PORT + 29212))}"
TSP_SCALE="${TSP_SCALE:-small}"
TSP_LABEL="${TSP_LABEL:-${TSP_SCALE}}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-eoh_rl_grpo_np1_TSP${TSP_LABEL}}"

LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-/home/yuanyilun/models/deepseek-coder-7b-instruct-v1.5}"
SFT_MODE="${SFT_MODE:-load_existing}"
SFT_LOAD_LORA_PATH="${SFT_LOAD_LORA_PATH:-/home/yuanyilun/projects/LLM4AD/data/SFT/tsp/${TSP_SCALE}/lora_sft}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${PROJECT_ROOT}/data/SFT/tsp/${TSP_SCALE}}"

SOURCE_CONFIG="${SOURCE_CONFIG:-${PROJECT_ROOT}/configs/run_eoh_local_rl/eoh_local_rl_tsp.yaml}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-${PROJECT_ROOT}/logs/TSP/eoh_local_rl}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PROJECT_ROOT}/logs}"
RUN_ROOT_TS="$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="${SOURCE_CONFIG}"

if [ ! -x "${PYTHON}" ]; then
  echo "[ERROR] Python not found or not executable: ${PYTHON}" >&2
  exit 1
fi
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [ ! -f "${PROJECT_ROOT}/example/run_eoh/tsp/run_eoh_local_rl.py" ]; then
  echo "[ERROR] Project root is invalid: ${PROJECT_ROOT}" >&2
  exit 1
fi
if [ ! -d "${LOCAL_MODEL_PATH}" ]; then
  echo "[ERROR] Local model path not found: ${LOCAL_MODEL_PATH}" >&2
  exit 1
fi
if [ "${SFT_MODE}" != "cold_start" ] && [ ! -d "${SFT_LOAD_LORA_PATH}" ]; then
  echo "[ERROR] SFT LoRA path not found: ${SFT_LOAD_LORA_PATH}" >&2
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
  TSP_SCALE="${TSP_SCALE}" \
  SFT_MODE="${SFT_MODE}" \
  SFT_LOAD_LORA_PATH="${SFT_LOAD_LORA_PATH}" \
  SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR}" \
  MAX_STEPS="${MAX_STEPS:-}" \
  MAX_SAMPLES="${MAX_SAMPLES:-}" \
  N_GENERATIONS="${N_GENERATIONS:-}" \
  POPULATION_SIZE="${POPULATION_SIZE:-}" \
  NUM_EVALUATORS="${NUM_EVALUATORS:-}" \
  LR="${LR:-}" \
  BETA="${BETA:-}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}" \
  "${PYTHON}" -c 'import json, os, sys

FORBIDDEN = {"grpo", "evolution", "rl", "vllm", "lora", "task_rl_common"}

def merge(base, overrides):
    result = dict(base)
    for key, value in (overrides or {}).items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result

try:
    base = json.loads(sys.argv[1]) if sys.argv[1].strip() else {}
except json.JSONDecodeError as exc:
    raise SystemExit(f"EOH_RL_RUNTIME_OVERRIDES_JSON is not valid JSON: {exc}") from exc

eoh_base = base.get("eoh_rl")
if isinstance(eoh_base, dict):
    for key in FORBIDDEN:
        eoh_base.pop(key, None)

args = {"vllm_group_port": int(os.environ["TARGET_GROUP_PORT"])}
env_to_arg = {
    "MAX_STEPS": "max_steps",
    "MAX_SAMPLES": "max_samples",
    "N_GENERATIONS": "n_generations",
    "POPULATION_SIZE": "population_size",
    "NUM_EVALUATORS": "num_evaluators",
    "LR": "lr",
    "BETA": "beta",
    "GPU_MEMORY_UTILIZATION": "gpu_memory_utilization",
}
for env_key, arg_key in env_to_arg.items():
    value = os.environ.get(env_key, "").strip()
    if not value:
        continue
    args[arg_key] = float(value) if arg_key in {"lr", "beta", "gpu_memory_utilization"} else int(value)

sft = {
    "mode": os.environ["SFT_MODE"],
    "output_dir": os.environ["SFT_OUTPUT_DIR"],
    "load_lora_path": None if os.environ["SFT_MODE"] == "cold_start" else os.environ["SFT_LOAD_LORA_PATH"],
}
forced = {
    "eoh_rl": {
        "local_model_path": os.environ["LOCAL_MODEL_PATH"],
        "inference_gpus": [int(os.environ["TARGET_GPU"])],
        "training_gpus": [int(os.environ["TARGET_GPU"])],
        "inference_ports": [int(os.environ["TARGET_PORT"])],
        "args": args,
        "sft": sft,
        "logging": {"log_base_dir": os.environ["LOG_BASE_DIR"]},
        "tsp": {"default_scale": os.environ["TSP_SCALE"]},
    }
}
print(json.dumps(merge(base, forced), ensure_ascii=False, separators=(",", ":")))' "${BASE_RUNTIME_OVERRIDES}"
)"
export EOH_RL_RUNTIME_OVERRIDES_JSON

cd "${PROJECT_ROOT}"

readarray -t CFG_VALUES < <("${PYTHON}" -c 'import importlib.util, json, os, sys, yaml

args_path = os.path.join(sys.argv[2], "llm4ad", "method", "eoh_rl", "args.py")
spec = importlib.util.spec_from_file_location("eohrl_args", args_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
apply_runtime_defaults = module.apply_runtime_defaults

def merge(base, overrides):
    result = dict(base)
    for key, value in (overrides or {}).items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result

with open(sys.argv[1], "r", encoding="utf-8") as file:
    root = yaml.safe_load(file)
text = os.environ.get("EOH_RL_RUNTIME_OVERRIDES_JSON", "").strip()
root = merge(root, json.loads(text)) if text else root
data = apply_runtime_defaults(root["eoh_rl"])
tsp = data["tsp"]
scale = tsp["default_scale"]
scale_cfg = tsp["scales"][scale]
evo = data["evolution"]
grpo = data["grpo"]
sft = data.get("sft", {})
mode = str(sft.get("mode", "cold_start"))
lora = "<none/cold_start>" if mode == "cold_start" else (sft.get("load_lora_path") or "")
print(data["inference_gpus"][0])
print(data["inference_ports"][0])
print(grpo["vllm_group_port"])
print(evo["max_sample_nums"])
print(evo["max_generations"])
print(grpo["num_generations"])
print(evo["pop_size"])
print(tsp["task_name"])
print(scale)
print(scale_cfg["problem_size"])
print(scale_cfg["n_instance"])
print(mode)
print(lora)' "${CONFIG_PATH}" "${PROJECT_ROOT}")

TARGET_GPU="${CFG_VALUES[0]}"
TARGET_PORT="${CFG_VALUES[1]}"
TARGET_GROUP_PORT="${CFG_VALUES[2]}"
MAX_SAMPLE_NUMS="${CFG_VALUES[3]}"
EOH_MAX_GENERATIONS="${CFG_VALUES[4]}"
NUM_GENERATIONS="${CFG_VALUES[5]}"
POP_SIZE="${CFG_VALUES[6]}"
TSP_TASK="${CFG_VALUES[7]}"
TSP_SCALE="${CFG_VALUES[8]}"
TSP_PROBLEM_SIZE="${CFG_VALUES[9]}"
TSP_INSTANCE_COUNT="${CFG_VALUES[10]}"
SFT_MODE="${CFG_VALUES[11]}"
SFT_LORA_PATH="${CFG_VALUES[12]}"

for i in $(seq 1 "${RUNS_PER_VARIANT}"); do
  RUN_ID="${RUN_ID_PREFIX}_run${i}_${RUN_ROOT_TS}"
  RUN_LOG_DIR="${RUN_ROOT_BASE}/${RUN_ID}"

  echo "============================================================"
  echo "EoH-RL TSP run ${i}/${RUNS_PER_VARIANT}: ${RUN_ID}"
  echo "GPU=${TARGET_GPU} port=${TARGET_PORT} group_port=${TARGET_GROUP_PORT} init_sample_budget=${MAX_SAMPLE_NUMS} max_grpo_updates=${EOH_MAX_GENERATIONS} num_generations=${NUM_GENERATIONS} pop_size=${POP_SIZE}"
  echo "task=${TSP_TASK} scale=${TSP_SCALE} problem_size=${TSP_PROBLEM_SIZE} n_instance=${TSP_INSTANCE_COUNT} sft_mode=${SFT_MODE} sft_lora=${SFT_LORA_PATH}"
  echo "config=${CONFIG_PATH}"

  if [ "${DRY_RUN}" = "1" ]; then
    continue
  fi

  EOH_RL_CONFIG_PATH="${CONFIG_PATH}" \
  EOH_RL_RUN_ID="${RUN_ID}" \
  EOH_RL_RUN_LOG_DIR="${RUN_LOG_DIR}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON}" "${PROJECT_ROOT}/example/run_eoh/tsp/run_eoh_local_rl.py"
done
