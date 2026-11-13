#!/usr/bin/env python3
"""Rescore stored LongVT generations without rerunning model inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from custom_rewards.lmms_lab_recipe import compute_score  # noqa: E402


def prediction_text(row: dict) -> str:
    value = row.get("filtered_resps") or row.get("resps") or ""
    if isinstance(value, list):
        value = value[-1] if value else ""
    return str(value)


def score_row(row: dict) -> dict:
    score = compute_score(
        data_source="longvt",
        solution_str=prediction_text(row),
        ground_truth=str(row["target"]),
        extra_info={"question": row.get("input", "")},
    )
    return {
        "doc_id": str(row["doc_id"]),
        "acc_score": float(score["acc_score"]),
        "format_score": float(score["format_reward_score"]),
        "predict_str": score["predict_str"],
        "ground_truth": score["ground_truth"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output = args.output or args.samples.with_name("rejudged.jsonl")
    summary_path = output.with_suffix(".summary.json")
    rows = [json.loads(line) for line in args.samples.read_text().splitlines()]
    rows.sort(key=lambda row: int(row["doc_id"]))
    if args.limit:
        rows = rows[: args.limit]

    completed = {}
    if output.exists():
        for line in output.read_text().splitlines():
            item = json.loads(line)
            completed[item["doc_id"]] = item
    pending = [row for row in rows if str(row["doc_id"]) not in completed]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as handle, ThreadPoolExecutor(args.workers) as executor:
        futures = {executor.submit(score_row, row): row for row in pending}
        for future in as_completed(futures):
            item = future.result()
            completed[item["doc_id"]] = item
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{len(completed)}/{len(rows)}] "
                f"doc={item['doc_id']} acc={item['acc_score']:.0f}"
            )

    if len(completed) != len(rows):
        raise RuntimeError(f"missing scores: {len(rows) - len(completed)}")

    ordered = [completed[str(row["doc_id"])] for row in rows]
    summary = {
        "samples": str(args.samples),
        "count": len(ordered),
        "acc_score": sum(item["acc_score"] for item in ordered) / len(ordered),
        "format_score": sum(item["format_score"] for item in ordered) / len(ordered),
        "judge_model": os.getenv("OPENAI_MODEL_NAME"),
        "judge_base_url": os.getenv("OPENAI_BASE_URL"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
