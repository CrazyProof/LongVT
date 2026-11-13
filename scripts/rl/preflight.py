#!/usr/bin/env python3
"""Check LongVT RL/RFT data and runtime readiness without using a GPU."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
ANNOTATIONS = Path(os.getenv("LONGVT_ANNOTATION_DIR", REPO_ROOT / "data/annotations"))
MEDIA = Path(os.getenv("LONGVT_VIDEO_DIR", REPO_ROOT / "media"))
MODEL = Path(
    os.getenv(
        "LONGVT_RL_MODEL",
        REPO_ROOT / "runs/longvt_7b_sft_8gpu_20261112_120910/merged-hf",
    )
)
TOOL_CONFIG = REPO_ROOT / "examples/video_tools/config/local/mcp_tool_config.yaml"


def rl_video_names(path: Path) -> list[str]:
    rows = pq.read_table(path, columns=["videos"]).to_pylist()
    return [
        os.path.basename(video["video"].removeprefix("file://"))
        for row in rows
        for video in (row["videos"] or [])
    ]


def rft_video_names(path: Path) -> list[str]:
    rows = pq.read_table(path, columns=["messages"]).to_pylist()
    names = []
    for row in rows:
        for message in row["messages"]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                video_url = part.get("video_url") if isinstance(part, dict) else None
                if isinstance(video_url, dict) and video_url.get("url"):
                    names.append(os.path.basename(video_url["url"]))
    return names


def media_check(names: list[str]) -> dict:
    missing = sorted({name for name in names if not (MEDIA / name).exists()})
    return {"references": len(names), "unique": len(set(names)), "missing": missing}


def main() -> int:
    rl_train = ANNOTATIONS / "longvt_rl_selfqa_1k6.parquet"
    rl_val = ANNOTATIONS / "longvt_rl_val_114.parquet"
    rft = ANNOTATIONS / "longvt_rft_selftrace_15k3.parquet"

    data = {
        "rl_train_rows": pq.ParquetFile(rl_train).metadata.num_rows,
        "rl_val_rows": pq.ParquetFile(rl_val).metadata.num_rows,
        "rft_rows": pq.ParquetFile(rft).metadata.num_rows,
        "rl_train_media": media_check(rl_video_names(rl_train)),
        "rl_val_media": media_check(rl_video_names(rl_val)),
        "rft_media": media_check(rft_video_names(rft)),
    }

    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("verl", "vllm", "sglang", "ray", "hydra", "flash_attn")
    }
    tool_config = yaml.safe_load(TOOL_CONFIG.read_text())
    server_config = Path(tool_config["tools"][0]["mcp"]["mcp_servers_config_path"])
    mcp_config = json.loads(server_config.read_text())
    server = next(iter(mcp_config["mcpServers"].values()))
    runtime = {
        "modules": modules,
        "model_exists": MODEL.exists(),
        "tool_config_exists": server_config.exists(),
        "mcp_command_exists": Path(server["command"]).exists(),
        "mcp_server_exists": Path(server["args"][0]).exists(),
    }

    media_ready = all(
        not data[key]["missing"] for key in ("rl_train_media", "rl_val_media", "rft_media")
    )
    data_ready = media_ready and all(
        data[key] > 0 for key in ("rl_train_rows", "rl_val_rows", "rft_rows")
    )
    runtime_ready = runtime["model_exists"] and all(
        runtime[key] for key in ("tool_config_exists", "mcp_command_exists", "mcp_server_exists")
    ) and all(modules[name] for name in ("verl", "vllm", "ray", "hydra"))

    result = {
        "annotations": str(ANNOTATIONS),
        "media_root": str(MEDIA),
        "model": str(MODEL),
        "data": data,
        "runtime": runtime,
        "data_ready": data_ready,
        "runtime_ready": runtime_ready,
    }
    print(json.dumps(result, indent=2))
    print("RL_DATA_OK" if data_ready else "RL_DATA_INCOMPLETE")
    print("RL_RUNTIME_OK" if runtime_ready else "RL_RUNTIME_INCOMPLETE")
    return 0 if data_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
