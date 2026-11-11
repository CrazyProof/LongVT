# Local LongVT SFT

## Preconditions

- Training environment: `/data/caojingqi/lmms-engine/.venv`
- Base model: `/data/caojingqi/LongVT/models/Qwen2.5-VL-7B-Instruct`
- Annotations: `/data/caojingqi/LongVT/data/annotations`
- Media root: `/data/caojingqi/LongVT/media`

Verified runtime:

```text
Python        3.11.17
PyTorch       2.6.0+cu124
torchvision   0.21.0+cu124
Triton        3.2.0
Transformers  5.5.0
DeepSpeed     0.18.9
FlashAttention 2.7.4.post1
```

Do not run `uv sync` in `lmms-engine`; the current `uv.lock` does not describe
the verified CUDA 12.4 environment.

## Validate media

```bash
/data/caojingqi/lmms-engine/.venv/bin/python \
  /data/caojingqi/LongVT/scripts/sft/validate_media.py
```

The final line must be `MEDIA_COVERAGE_OK`.

## Prepare and run smoke

```bash
/data/caojingqi/lmms-engine/.venv/bin/python \
  /data/caojingqi/LongVT/scripts/sft/prepare_smoke.py

CUDA_VISIBLE_DEVICES=1,5 \
  /data/caojingqi/LongVT/scripts/sft/run_sft.sh \
  /data/caojingqi/LongVT/examples/sft/local/longvt_7b_sft_smoke.yaml \
  2
```

The smoke run performs two optimizer steps, writes a model-only checkpoint,
and stores logs under `/data/caojingqi/LongVT/runs`.

## Run full SFT

Start with all eight GPUs idle:

```bash
/data/caojingqi/LongVT/scripts/sft/run_sft.sh \
  /data/caojingqi/LongVT/examples/sft/local/longvt_7b_sft_8gpu.yaml \
  8
```

The eight-GPU configuration uses conservative 16K packing and 128 video
frames. It is not a reproduction of the 51.2K/512-frame configuration.
