from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_artifact_version_id(round_id: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"round_{round_id:04d}_{stamp}"


def prepare_training_output_dir(output_root: str, version_id: str) -> Path:
    output_dir = Path(output_root) / version_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
