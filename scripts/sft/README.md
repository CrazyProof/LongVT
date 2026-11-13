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

Alternatively, run the complete config, model, media, and smoke-processing
preflight:

```bash
/data/caojingqi/lmms-engine/.venv/bin/python \
  /data/caojingqi/LongVT/scripts/sft/preflight.py
```

The final line must be `SFT_PREFLIGHT_OK`.

The verified two-step result is documented in [SMOKE_REPORT.md](SMOKE_REPORT.md).

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

Alternatively, wait until two GPUs each have at least 20 GiB free and launch
the smoke automatically:

```bash
/data/caojingqi/LongVT/scripts/sft/wait_for_gpus_and_run_smoke.sh
```

## Run full SFT

Start with all eight GPUs idle:

```bash
/data/caojingqi/LongVT/scripts/sft/run_sft.sh \
  /data/caojingqi/LongVT/examples/sft/local/longvt_7b_sft_8gpu.yaml \
  8
```

The eight-GPU configuration uses conservative 8K packing and 64 video
frames. It is not a reproduction of the 51.2K/512-frame configuration.

Before a long run, exercise the full datasets and packing path with the
ten-step stability configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  /data/caojingqi/LongVT/scripts/sft/run_sft.sh \
  /data/caojingqi/LongVT/examples/sft/local/longvt_7b_sft_stability.yaml \
  6
```

The verified result is documented in
[STABILITY_REPORT.md](STABILITY_REPORT.md).

The completed 3000-step run is documented in
[TRAINING_REPORT.md](TRAINING_REPORT.md).

## Local evaluation runner

Use the repository-local runner for repeatable Base / SFT / RFT comparisons:

```bash
GPU=5 PROCS=4 LIMIT=5 MAX_FRAMES=64 INFERENCE_ONLY=1 \
  scripts/sft/run_local_eval.sh /path/to/model
```

Omit `LIMIT` for the full 652-example benchmark and set
`INFERENCE_ONLY=0` to require the configured LLM judge. The runner starts a
local vLLM server, waits for its health endpoint, runs the LongVT tool task,
and stops the server on exit.
