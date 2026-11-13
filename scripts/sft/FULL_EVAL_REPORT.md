# LongVT SFT Full Evaluation

## Configuration

```text
Model: runs/longvt_7b_sft_8gpu_20261112_120910/merged-hf
Global video frames: 64
MCP crop frame cap: 64
Max pixels: 50,176
Max context: 12,288
Judge: DeepSeek-V4.1-Flash
```

This is the conservative 64-frame local evaluation profile, not the paper's
512-frame / 8 x 80 GB configuration.

## Result

```text
Examples:          652
Accuracy:          0.0261 (17 / 652)
Format score:      0.6061
Tool calls:        430 / 652
Tool errors:       1 / 652
Complete answers:  652 / 652
Input tokens:      5,130,093
Output tokens:     256,431
```

All 652 document IDs are unique and every response has a complete
`<answer>...</answer>` block. The remaining tool error is a model-generated
filename case mismatch (`nTpNJK3_YJg.mp4` instead of
`nTpNJk3_YJg.mp4`), not a missing benchmark file.

## Repairs

Four examples from the first full run were rerun and merged by `doc_id`:

```text
368  corrupt 6lRTHAIJDIU.mp4; evaluated with a decoded repair proxy
369  same corrupt source video as 368
478  incomplete answer on the first run
500  incomplete answer on the first run
```

The original damaged video is retained as
`6lRTHAIJDIU.mp4.problem-backup`. Scores were recomputed over the complete
canonical sample file rather than averaged across runs.

## Artifacts

```text
Canonical samples: runs/longvt_7b_sft_8gpu_20261112_120910/eval-full-64f-v3/longvt-rft/canonical_samples.jsonl
Final scores:      runs/longvt_7b_sft_8gpu_20261112_120910/eval-full-64f-v3/longvt-rft/canonical_rejudged.jsonl
Summary:           runs/longvt_7b_sft_8gpu_20261112_120910/eval-full-64f-v3/longvt-rft/canonical_rejudged.summary.json
```
