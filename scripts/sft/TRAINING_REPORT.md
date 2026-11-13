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
was merged into a standard Hugging Face model.

The first merge used `model.visual.*` keys for the 390 vision tensors, whereas
Hugging Face and vLLM expect `visual.*`. The text weights were already correct.
`fix_merged_hf_keys.py` normalized the vision prefix without changing tensor
values. The corrected checkpoint has exactly the same 729 keys as the base
model, and vLLM 0.8.3 loaded all weights successfully on one RTX 4090.

Transformers 5.5 metadata (`text_config`, tokenizer and processor layouts) is
not readable by the pinned vLLM/Transformers 4.51 evaluation server. The final
model therefore uses the base model's compatible metadata layout. The original
Transformers 5 metadata files are retained with a `.bak` suffix.

## Final model

```text
/data/caojingqi/LongVT/runs/longvt_7b_sft_8gpu_20261112_120910/merged-hf
```

The run also retains `train.log`, `config.yaml`, `config.input.yaml`, and
`compute_summary.json`.
