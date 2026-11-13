#!/usr/bin/env python3
"""Write a compact report from LongVT samples and rejudged scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.samples.read_text().splitlines()]
    summary = json.loads(args.summary.read_text())
    token_counts = [
        token
        for row in rows
        for token in (row.get("token_counts") or [{}])
    ]
    tool_calls = sum(
        "<tool_call>" in str(row.get("filtered_resps") or row.get("resps") or "")
        for row in rows
    )
    tool_errors = sum(
        any(
            marker in str(row.get("filtered_resps") or row.get("resps") or "")
            for marker in ("Tool execution failed", "Video file not found", "Error code:")
        )
        for row in rows
    )
    answer_tags = sum(
        "<answer>" in str(row.get("filtered_resps") or row.get("resps") or "")
        and "</answer>" in str(row.get("filtered_resps") or row.get("resps") or "")
        for row in rows
    )

    report = f"""# LongVT SFT Full Evaluation

## Configuration

```text
Model: /data/caojingqi/LongVT/runs/longvt_7b_sft_8gpu_20261112_120910/merged-hf
Global video frames: {args.frames}
MCP crop frame cap: 64
Max pixels: 50,176
Max context: 12,288
Judge: {summary.get("judge_model")}
Judge endpoint: {summary.get("judge_base_url")}
```

This is the conservative 64-frame local evaluation profile, not the paper's
512-frame / 8 x 80 GB configuration.

## Result

```text
Examples:          {summary["count"]}
Accuracy:          {summary["acc_score"]:.4f}
Format score:      {summary["format_score"]:.4f}
Tool calls:        {tool_calls} / {len(rows)}
Tool errors:       {tool_errors} / {len(rows)}
Complete answers:  {answer_tags} / {len(rows)}
Input tokens:      {sum(int(item.get("input_tokens", 0)) for item in token_counts)}
Output tokens:     {sum(int(item.get("output_tokens", 0)) for item in token_counts)}
```

## Artifacts

```text
Samples: {args.samples}
Summary: {args.summary}
```
"""
    args.output.write_text(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
