from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from llm4ad.tools.llm.local_transformers import LocalTransformersLLM


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="models/Qwen2.5-Coder-1.5B-Instruct",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser


def main():
    args = build_parser().parse_args()

    llm = LocalTransformersLLM(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    prompt = (
        "Write a Python function named `priority(item, bins)` that returns a NumPy array "
        "of scores for online bin packing. Only output the function body."
    )
    try:
        response = llm.draw_sample(prompt)
        print("=== Model Path ===")
        print(args.model_path)
        print("\n=== Response ===")
        print(response)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
