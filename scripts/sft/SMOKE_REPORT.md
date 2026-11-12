# LongVT SFT Smoke Report

Date: 2026-11-12

## Environment

```text
Python        3.11.17
PyTorch       2.6.0+cu124
torchvision   0.21.0+cu124
Transformers  5.5.0
DeepSpeed     0.18.9
FlashAttention 2.7.4.post1
```

Hardware: four RTX 4090 24 GB GPUs in FSDP2 mode.

## Input coverage

The smoke Parquet contains one sample from each of the eight SFT datasets:

```text
GeminiCoT
LLaVA-CoT
LongVideoReason
LongVideoReflection
OpenVLThinker
TVG
Video-R1
WeMath
```

The samples cover images, videos, tool calls, and tool responses. All selected
media decoded successfully, and all 480,060 unique full-dataset media
references were present.

## Training result

```text
step 0: loss=3.16015625, grad_norm=456.0, tokens/s=722
step 1: loss=2.013671875, grad_norm=65.0, tokens/s=205
training duration: 97.9 seconds
```

The run completed two optimizer steps without OOM. The model-only checkpoint
contains four FSDP2 model shards and no optimizer state, approximately 16 GB.

The checkpoint was merged successfully into a standard Hugging Face model with
729 tensors, matching the Qwen2.5-VL weight key count.

## Output

```text
/data/caojingqi/LongVT/runs/longvt_7b_sft_smoke_20261112_111331
```

Expected files include `config.yaml`, `checkpoint-2`, `merged-hf`, and
`train.log`.

## Compatibility fixes exercised

- Transformers 5.5 Qwen2.5-VL `video_metadata` is removed before collation.
- `save_only_model: true` omits optimizer, scheduler, RNG, and dataloader state.
- The launcher renders a per-run config so output directories do not collide.
