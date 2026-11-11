#!/usr/bin/env python3
"""Select a small, media-complete LongVT SFT smoke dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
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


def content_flags(messages: object) -> tuple[bool, bool]:
    has_image = False
    has_video = False
    if not isinstance(messages, list):
        return has_image, has_video
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            image_value = item.get("image_url")
            image_url = (
                image_value.get("url")
                if isinstance(image_value, dict)
                else image_value
            )
            video_value = item.get("video_url")
            video_url = (
                video_value.get("url")
                if isinstance(video_value, dict)
                else video_value
            )
            has_image |= bool(image_url)
            has_video |= bool(video_url)
    return has_image, has_video


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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/caojingqi/LongVT/datasets/longvt-sft-smoke/train.parquet"),
    )
    parser.add_argument("--scan-rows", type=int, default=128)
    args = parser.parse_args()

    parquet_files = sorted(args.annotations.glob("longvt_sft_*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No SFT annotations found in {args.annotations}")

    selected_rows: list[dict] = []
    selected_manifest: list[dict] = []
    schema: pa.Schema | None = None

    for parquet in parquet_files:
        parquet_file = pq.ParquetFile(parquet)
        best: dict | None = None
        best_meta: dict | None = None

        scanned = 0
        for batch in parquet_file.iter_batches(batch_size=16, columns=["id", "messages"]):
            for row in batch.to_pylist():
                refs = collect_refs(row["messages"])
                if not refs or any(not (args.media / ref).is_file() for ref in refs):
                    continue
                total_bytes = sum((args.media / ref).stat().st_size for ref in refs)
                has_image, has_video = content_flags(row["messages"])
                serialized_messages = json.dumps(row["messages"])
                has_tool_call = "<tool_call>" in serialized_messages
                has_tool_response = "<tool_response>" in serialized_messages
                score = (
                    int(has_video) * 4
                    + int(has_image) * 2
                    + int(has_tool_call) * 3
                    + int(has_tool_response) * 2
                )
                candidate_meta = {
                    "dataset": parquet.name,
                    "id": row["id"],
                    "media": sorted(refs),
                    "media_bytes": total_bytes,
                    "has_image": has_image,
                    "has_video": has_video,
                    "has_tool_call": has_tool_call,
                    "has_tool_response": has_tool_response,
                }
                if (
                    best_meta is None
                    or score > best_meta["_score"]
                    or (
                        score == best_meta["_score"]
                        and total_bytes < best_meta["media_bytes"]
                    )
                ):
                    best = row
                    best_meta = {**candidate_meta, "_score": score}
                scanned += 1
                if scanned >= args.scan_rows:
                    break
            if scanned >= args.scan_rows:
                break
        if best is None:
            raise SystemExit(f"No media-complete row found in {parquet}")
        selected_rows.append(best)
        best_meta.pop("_score")
        selected_manifest.append(best_meta)
        schema = schema or batch.schema
        print(
            f"{parquet.name}: id={best['id']} "
            f"media={len(best_meta['media'])} bytes={best_meta['media_bytes']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected_rows, schema=schema), args.output)
    manifest_path = args.output.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps({"rows": selected_manifest}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"SMOKE_ROWS={len(selected_rows)}")
    print(f"SMOKE_PARQUET={args.output}")
    print(f"SMOKE_MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
