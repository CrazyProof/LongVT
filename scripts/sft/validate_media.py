#!/usr/bin/env python3
"""Verify that every media URL in LongVT SFT annotations exists locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def collect_refs(messages: object) -> set[str]:
    refs: set[str] = set()
    if not isinstance(messages, list):
        return refs
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            for key in ("image_url", "video_url"):
                value = item.get(key)
                url = value.get("url") if isinstance(value, dict) else value
                if url:
                    refs.add(str(url).removeprefix("file://"))
    return refs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/data/caojingqi/LongVT/data/annotations"),
    )
    parser.add_argument(
        "--media",
        type=Path,
        default=Path("/data/caojingqi/LongVT/media"),
    )
    args = parser.parse_args()

    parquet_files = sorted(args.annotations.glob("longvt_sft_*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No SFT annotations found in {args.annotations}")

    total_refs = 0
    total_missing = 0
    for parquet in parquet_files:
        refs: set[str] = set()
        table = pq.read_table(parquet, columns=["messages"])
        for row in table.to_pylist():
            refs.update(collect_refs(row["messages"]))

        missing = sorted(ref for ref in refs if not (args.media / ref).is_file())
        total_refs += len(refs)
        total_missing += len(missing)
        print(f"{parquet.name}: refs={len(refs)} missing={len(missing)}")
        for ref in missing[:20]:
            print(f"  MISSING {ref}")

    print(f"TOTAL_REFS={total_refs} TOTAL_MISSING={total_missing}")
    if total_missing:
        print("MEDIA_COVERAGE_FAILED")
        return 1
    print("MEDIA_COVERAGE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
