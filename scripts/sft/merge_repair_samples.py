#!/usr/bin/env python3
"""Merge targeted LongVT repair runs into a complete evaluation sample file."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

FAILURE_MARKERS = (
    "REQUEST_FAILED",
    "Tool execution failed",
    "Video file not found",
    "Error code:",
)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or "doc_id" not in row:
            raise ValueError(f"{path}:{line_number}: missing doc_id")
        rows.append(row)
    return rows


def response_text(row: dict) -> str:
    value = row.get("filtered_resps") or row.get("resps") or ""
    if isinstance(value, list):
        value = value[-1] if value else ""
    return str(value)


def validate_replacement(row: dict, path: Path) -> None:
    doc_id = str(row["doc_id"])
    response = response_text(row)
    failures = [marker for marker in FAILURE_MARKERS if marker in response]
    if failures:
        raise ValueError(
            f"{path}: doc_id={doc_id} still contains failure marker(s): "
            + ", ".join(failures)
        )
    if "<answer>" not in response or "</answer>" not in response:
        raise ValueError(f"{path}: doc_id={doc_id} has no complete answer tag")
    if not str(row.get("target", "")).strip():
        raise ValueError(f"{path}: doc_id={doc_id} has an empty target")


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--replacement",
        type=Path,
        nargs="+",
        required=True,
        help="Repair-run JSONL files containing one or more rows",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    input_paths = {args.base.resolve(), *(path.resolve() for path in args.replacement)}
    if args.output.resolve() in input_paths:
        raise SystemExit("--output must differ from every input file")

    base_rows = read_jsonl(args.base)
    if args.expected_count is not None and len(base_rows) != args.expected_count:
        raise ValueError(
            f"base has {len(base_rows)} rows, expected {args.expected_count}"
        )

    merged = {str(row["doc_id"]): row for row in base_rows}
    if len(merged) != len(base_rows):
        raise ValueError("base file contains duplicate doc_id values")
    order = [str(row["doc_id"]) for row in base_rows]

    replaced: set[str] = set()
    for path in args.replacement:
        replacements = read_jsonl(path)
        if not replacements:
            raise ValueError(f"{path}: no rows found")
        for row in replacements:
            validate_replacement(row, path)
            doc_id = str(row["doc_id"])
            if doc_id not in merged:
                raise ValueError(f"{path}: doc_id={doc_id} is absent from base")
            if doc_id in replaced:
                raise ValueError(f"doc_id={doc_id} appears in multiple replacements")
            merged[doc_id] = row
            replaced.add(doc_id)

    output_rows = [merged[doc_id] for doc_id in order]
    if len(output_rows) != len(base_rows):
        raise RuntimeError("merged row count changed unexpectedly")
    if len({str(row["doc_id"]) for row in output_rows}) != len(output_rows):
        raise RuntimeError("merged file contains duplicate doc_id values")

    write_jsonl_atomic(args.output, output_rows)
    print(f"Base rows:          {len(base_rows)}")
    print(f"Replaced rows:      {len(replaced)}")
    print(f"Replacement doc_id: {', '.join(sorted(replaced, key=int))}")
    print(f"Output rows:        {len(output_rows)}")
    print(f"Output:             {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
