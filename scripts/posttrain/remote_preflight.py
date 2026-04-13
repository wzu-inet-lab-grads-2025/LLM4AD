from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


def _module_installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _safe_torch_info():
    if not _module_installed("torch"):
        return {
            "installed": False,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        }

    import torch

    devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": devices,
    }


def _disk_info(path: Path):
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "free_gb": round(usage.free / (1024**3), 2),
        "total_gb": round(usage.total / (1024**3), 2),
    }


def build_report(model_path: Path):
    workspace = Path.cwd()
    report = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "venv": os.environ.get("VIRTUAL_ENV"),
        },
        "workspace": {
            "cwd": str(workspace),
            "disk": _disk_info(workspace),
        },
        "modules": {
            name: _module_installed(name)
            for name in [
                "torch",
                "transformers",
                "datasets",
                "peft",
                "accelerate",
                "trl",
                "matplotlib",
            ]
        },
        "torch": _safe_torch_info(),
        "model": {
            "path": str(model_path),
            "exists": model_path.exists(),
            "has_config": (model_path / "config.json").exists(),
            "has_weights": any(model_path.glob("*.safetensors"))
            or any(model_path.glob("*.bin")),
        },
        "artifacts_dir_writable": None,
    }

    artifacts_dir = workspace / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    probe = artifacts_dir / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report["artifacts_dir_writable"] = True
    except OSError:
        report["artifacts_dir_writable"] = False

    required_modules_ok = all(
        report["modules"][name] for name in ["torch", "transformers"]
    )
    model_ok = (
        report["model"]["exists"]
        and report["model"]["has_config"]
        and report["model"]["has_weights"]
    )
    report["ready_for_local_inference"] = (
        required_modules_ok and model_ok and report["torch"]["cuda_available"]
    )
    report["ready_for_real_sft"] = (
        report["ready_for_local_inference"]
        and all(
            report["modules"][name]
            for name in ["datasets", "peft", "accelerate", "trl"]
        )
        and report["artifacts_dir_writable"]
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen2.5-Coder-1.5B-Instruct")
    args = parser.parse_args()

    report = build_report(Path(args.model_path))
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if not report["ready_for_local_inference"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
