import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm4ad.method.eoh_rl import run_from_config


def main() -> None:
    config_path = os.environ.get("EOH_RL_CONFIG_PATH", "").strip()
    if not config_path:
        raise ValueError("必须通过 EOH_RL_CONFIG_PATH 指定 YAML 配置文件")
    run_from_config(config_path=config_path, task_family="cvrp")


if __name__ == "__main__":
    main()
