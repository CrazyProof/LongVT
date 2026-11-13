#!/usr/bin/env python3
"""Normalize FSDP-merged Qwen2.5-VL safetensors keys.

Transformers expects vision parameters under ``visual.*`` while the FSDP save
path used by this project emitted them under ``model.visual.*``. The text
parameters already use the standard ``model.*`` prefix. This script validates
the merged checkpoint against the base model's key map, rewrites only the
vision prefix, and saves a corrected safetensors file for an atomic swap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def base_weight_map(base_model: Path) -> dict[str, str]:
    index_path = base_model / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing base model index: {index_path}")
    return json.loads(index_path.read_text())["weight_map"]


def normalize_keys(merged_model: Path, expected_keys: set[str]) -> dict:
    source = merged_model / "model.safetensors"
    if not source.exists():
        raise FileNotFoundError(f"missing merged weights: {source}")

    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        for original_key in handle.keys():
            key = (
                original_key.removeprefix("model.")
                if original_key.startswith("model.visual.")
                else original_key
            )
            if key in tensors:
                raise ValueError(f"duplicate key after normalization: {key}")
            tensors[key] = handle.get_tensor(original_key)

    actual_keys = set(tensors)
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    if missing or unexpected:
        raise ValueError(
            "key mismatch after normalization: "
            f"missing={sorted(missing)[:20]} unexpected={sorted(unexpected)[:20]}"
        )
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_keys = set(base_weight_map(args.base_model))
    tensors = normalize_keys(args.merged_model, expected_keys)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"format": "pt"})
    print(f"Normalized {len(tensors)} tensors -> {args.output}")


if __name__ == "__main__":
    main()
