# LongVT SFT Stability Report

Date: 2026-11-12

## Configuration

```text
GPUs: 6 x RTX 4090 24 GB
Datasets: all 9 official SFT entries over 8 Parquet files
Packing: first_fit, length 8192
Video: 64 frames, 50,176 max pixels
FSDP2 reshard_after_forward: true
Dataloader workers: 0
Steps: 10
```

## Result

The ten-step run completed without OOM, resume-state errors, or data-path
failures. Loss moved from `2.1602` at step 0 to `1.0358` at step 9.
Global packed sequence length was approximately 7,000-8,000 tokens per step.

The model-only checkpoint contains six FSDP2 shards totaling approximately
16 GB and no optimizer state.

## Output

```text
/data/caojingqi/LongVT/runs/longvt_7b_sft_stability_20261112_115852
```

This validates the conservative 8K/64 frame configuration for the full SFT
data stream. It is not equivalent to the paper's 51.2K packing and 512-frame
configuration.
