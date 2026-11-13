# LongVT RL And RFT

The official training order is:

```text
cold-start SFT -> agentic GRPO RL -> self-trace RFT
```

## Verified local data

```text
RL train: 1,668 samples
RL val:     114 samples
RFT:     15,354 samples
```

All referenced videos are present under `/data/caojingqi/LongVT/media`.

## Local tool configuration

Use:

```text
examples/video_tools/config/local/mcp_tool_config.yaml
```

The MCP server resolves relative video basenames using `LONGVT_VIDEO_DIR`.
Set it before starting RL:

```bash
export LONGVT_VIDEO_DIR=/data/caojingqi/LongVT/media
export LONGVT_MCP_MAX_FRAMES=64
```

## Preflight

```bash
/data/caojingqi/miniconda3/envs/longvt/bin/python \
  scripts/rl/preflight.py
```

The full official GRPO script targets 8 nodes x 8 GPUs. A local single-node
smoke should be prepared only after the full SFT evaluation and comparison
baselines are complete.

The conservative smoke launcher is:

```bash
LLM_AS_A_JUDGE_BASE=http://<judge-host>:<port>/v1 \
  bash examples/video_tools/local/longvt_7b_rl_smoke.sh
```

It defaults to two optimizer steps, two rollouts per prompt, an 8K prompt cap,
a 2K response cap, and at most two assistant/tool turns. It uses vLLM rollout
by default because the current environment has vLLM but not SGLang. Set
`ROLLOUT_NAME=sglang` when the official SGLang environment is installed.

## Dedicated environment

Do not modify the `longvt` conda environment while it is serving evaluation.
Create the official RL environment separately:

```bash
bash scripts/rl/create_env.sh --dry-run
bash scripts/rl/create_env.sh
```

The script refuses to overwrite an existing environment and uses the official
Torch 2.6 / SGLang 0.4.6.post5 / FlashAttention 2.7.4 stack with Megatron
disabled for the first single-node smoke.
