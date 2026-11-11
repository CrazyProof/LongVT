#!/usr/bin/env python3
"""Validate a LongVT SFT config, media coverage, and smoke preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import yaml

from lmms_engine.datasets import DatasetConfig
from lmms_engine.datasets.iterable.vision_iterable_dataset import (
    VisionSFTIterableDataset,
)
from lmms_engine.datasets.processor import (
    ProcessorConfig,
    Qwen2_5_VLDataProcessor,
)
from lmms_engine.launch.cli import filter_training_args
from lmms_engine.models import ModelConfig
from lmms_engine.train import TrainingArguments


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


def validate_media(annotations: Path, media: Path) -> None:
    files = sorted(annotations.glob("longvt_sft_*.parquet"))
    if not files:
        raise RuntimeError(f"No SFT annotations found in {annotations}")
    total_refs = 0
    total_missing = 0
    for parquet in files:
        refs: set[str] = set()
        for row in pq.read_table(parquet, columns=["messages"]).to_pylist():
            refs.update(collect_refs(row["messages"]))
        missing = [ref for ref in refs if not (media / ref).is_file()]
        total_refs += len(refs)
        total_missing += len(missing)
        print(f"MEDIA {parquet.name}: refs={len(refs)} missing={len(missing)}")
    print(f"TOTAL_REFS={total_refs} TOTAL_MISSING={total_missing}")
    if total_missing:
        raise RuntimeError("Media coverage failed")


def validate_config(path: Path) -> tuple[dict, DatasetConfig, ModelConfig, TrainingArguments]:
    data = yaml.safe_load(path.read_text())
    dataset_config = DatasetConfig(**data["dataset_config"])
    model_config = ModelConfig(**data["model_config"])
    trainer_args = TrainingArguments(**filter_training_args(data["trainer_args"]))
    print(
        f"CONFIG {path.name}: datasets={len(dataset_config.datasets)} "
        f"attention={model_config.attn_implementation} steps={trainer_args.max_steps}"
    )
    return data, dataset_config, model_config, trainer_args


def validate_model(model_path: Path) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"Missing model index: {index_path}")
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (model_path / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing model shards: {missing}")
    print(f"MODEL {model_path}: shards={len(shards)}")


def validate_smoke_preprocessing(
    data: dict,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    media: Path,
) -> None:
    datasets = dataset_config.datasets
    if len(datasets) != 1:
        print("SMOKE_PREPROCESS_SKIPPED: config is not a single-dataset smoke config")
        return
    smoke_path = Path(datasets[0]["path"])
    if "smoke" not in smoke_path.name and "smoke" not in smoke_path.parent.name:
        print("SMOKE_PREPROCESS_SKIPPED: dataset path is not smoke data")
        return

    processor_config = ProcessorConfig(**data["dataset_config"]["processor_config"])
    processor = Qwen2_5_VLDataProcessor(processor_config)
    processor.build()
    dataset = object.__new__(VisionSFTIterableDataset)
    dataset.processor = processor
    dataset.config = SimpleNamespace(
        fps=dataset_config.fps,
        object_storage=dataset_config.object_storage,
        video_backend=dataset_config.video_backend,
        video_sampling_strategy=dataset_config.video_sampling_strategy,
        video_max_pixels=dataset_config.video_max_pixels,
        video_max_frames=dataset_config.video_max_frames,
        video_min_pixels=dataset_config.video_min_pixels,
        frame_num=dataset_config.frame_num,
    )
    rows = pq.read_table(smoke_path).to_pylist()
    for index, row in enumerate(rows, start=1):
        inputs = dataset.load_from_json(row, str(media))
        pixel_shape = (
            tuple(inputs["pixel_values"].shape)
            if inputs.get("pixel_values") is not None
            else None
        )
        print(
            f"SMOKE {index}/{len(rows)} id={row['id']} "
            f"tokens={tuple(inputs['input_ids'].shape)} pixels={pixel_shape}"
        )
    print(f"SMOKE_PREPROCESS_OK rows={len(rows)} model={model_config.load_from_pretrained_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/data/caojingqi/LongVT/examples/sft/local/longvt_7b_sft_smoke.yaml"
        ),
    )
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
        "--skip-media",
        action="store_true",
        help="Skip full annotation/media coverage scan.",
    )
    args = parser.parse_args()

    data, dataset_config, model_config, _ = validate_config(args.config)
    validate_model(Path(model_config.load_from_pretrained_path))
    if not args.skip_media:
        validate_media(args.annotations, args.media)
    validate_smoke_preprocessing(data, dataset_config, model_config, args.media)
    print("SFT_PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
