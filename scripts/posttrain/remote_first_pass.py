from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, workdir: Path):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=workdir, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-dryrun", action="store_true")
    parser.add_argument("--skip-sft-smoke", action="store_true")
    parser.add_argument("--skip-real-loop", action="store_true")
    parser.add_argument("--enable-smoke", action="store_true")
    parser.add_argument("--smoke-lines", nargs="+", default=["candidate", "base"])
    parser.add_argument("--llm-max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    py = sys.executable

    if not args.skip_preflight:
        _run(
            [
                py,
                "scripts/posttrain/remote_preflight.py",
                "--model-path",
                args.model_path,
            ],
            workdir=root,
        )

    if not args.skip_inference:
        _run(
            [
                py,
                "example/posttrain/run_local_qwen_inference.py",
                "--model-path",
                args.model_path,
                "--max-new-tokens",
                "64",
                "--temperature",
                "0.0",
            ],
            workdir=root,
        )

    if not args.skip_dryrun:
        _run(
            [
                py,
                "example/posttrain/run_local_qwen_toy_orchestrator.py",
                "--trainer-backend",
                "dryrun",
                "--disable-smoke",
                "--llm-max-new-tokens",
                str(args.llm_max_new_tokens),
            ],
            workdir=root,
        )

    if not args.skip_sft_smoke:
        _run(
            [py, "example/posttrain/run_local_qwen_sft_smoke.py"],
            workdir=root,
        )

    if not args.skip_real_loop:
        cmd = [
            py,
            "example/posttrain/run_local_qwen_toy_orchestrator.py",
            "--trainer-backend",
            "trl_sft",
            "--llm-max-new-tokens",
            str(args.llm_max_new_tokens),
        ]
        if args.enable_smoke:
            cmd.extend(["--smoke-max-samples", "1", "--smoke-lines", *args.smoke_lines])
        else:
            cmd.append("--disable-smoke")
        _run(cmd, workdir=root)


if __name__ == "__main__":
    main()
