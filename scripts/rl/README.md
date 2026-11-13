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
