import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm4ad.method.eoh_rl import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("EOH_RL_CONFIG_PATH", "").strip())
    args = parser.parse_args()
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/run_eoh_local_rl/eoh_local_rl_jssp.yaml")
    run_from_config(config_path=config_path, task_family="jssp")


if __name__ == "__main__":
    main()
