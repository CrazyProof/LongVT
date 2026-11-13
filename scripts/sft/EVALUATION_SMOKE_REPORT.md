# LongVT SFT Evaluation Smoke Report

Date: 2026-11-13

## Scope

This smoke test validates the post-training inference path. It is not a
benchmark score because only five examples were used and the external judge
credentials had expired.

```text
Model: /data/caojingqi/LongVT/runs/longvt_7b_sft_8gpu_20261112_120910/merged-hf
GPU: 1 x RTX 4090 24 GB
vLLM: 0.8.3
Task: longvt tool-calling evaluation
Frames: 64
Max pixels: 50,176
Examples: 5
```

## Result

```text
Inference:       5 / 5 completed
API calls:       8
Input tokens:    37,777
Output tokens:   1,841
Tool use:        3 / 5 samples
Format score:    0.64
Accuracy:        invalid (external judge returned HTTP 401)
```

The model produced both `<think>` and `<answer>` content in all five samples.
Three samples issued a real `crop_video` tool call and processed the returned
image frames. This verifies model loading, chat formatting, tool parsing, MCP
execution, result capture, and postprocessing.

## Weight compatibility finding

The initial merged checkpoint contained 390 `model.visual.*` tensors. The
standard Qwen2.5-VL and vLLM layouts require those tensors as `visual.*`.
Text weights already had the correct `model.*` prefix.

`fix_merged_hf_keys.py` rewrites only the vision prefix. After normalization,
all 729 tensor names exactly match the base model, with no missing or unexpected
keys. vLLM then loaded 15.6271 GiB of model weights successfully.

Transformers 5.5 also saved nested configuration, tokenizer, and processor
metadata that Transformers 4.51 / vLLM 0.8.3 cannot read. The corrected model
uses the base model's compatible metadata files; tensor values are unchanged.

## Required follow-up

Renew or replace the judge API credential in `/data/caojingqi/judge.env`.
Run a judge preflight before a full evaluation so authentication failures
cannot be silently converted to zero accuracy.
