# LongVT SFT Training Report

Date: 2026-11-12 to 2026-11-13

## Configuration

```text
GPUs: 6 x RTX 4090 24 GB
Base model: Qwen/Qwen2.5-VL-7B-Instruct
Datasets: 9 official SFT entries over 8 Parquet files
Packing: first_fit, length 8192
Video: 64 frames, 50,176 max pixels
FSDP2 reshard_after_forward: true
Dataloader workers: 0
Trainable parameters: 8.292166656 B
```

This is the verified conservative single-node profile. It is not the paper's
51.2K packing / 512-frame / 8 x 80 GB configuration.

## Result

```text
Completed steps: 3000 / 3000
Training duration: 22h 27m 47s
Total tokens: approximately 129M
Step 0 loss: 2.1602
Final step loss: 0.8594
Final learning rate: 0
Final grad norm: 2.421875
```

The run completed without OOM or data-path failures. The final FSDP2 checkpoint
was merged into a standard Hugging Face model and verified to contain 729
tensors. A short image-generation inference completed successfully on one
RTX 4090.

## Final model

```text
/data/caojingqi/LongVT/runs/longvt_7b_sft_8gpu_20261112_120910/merged-hf
```

The run also retains `train.log`, `config.yaml`, `config.input.yaml`, and
`compute_summary.json`.
