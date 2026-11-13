#!/bin/bash
set -euo pipefail

usage() {
  echo "usage: MODEL_DIR=... $0 [model_dir]" >&2
  echo "Optional: GPU PORT OUT PROCS LIMIT MAX_FRAMES MCP_MAX_FRAMES INFERENCE_ONLY" >&2
  exit 2
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
fi

ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
MODEL_DIR=${1:-${MODEL_DIR:-}}
[ -n "$MODEL_DIR" ] || usage

GPU=${GPU:-5}
PORT=${PORT:-8000}
PROCS=${PROCS:-4}
MAX_FRAMES=${MAX_FRAMES:-64}
MCP_MAX_FRAMES=${MCP_MAX_FRAMES:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
MAX_IMAGES=${MAX_IMAGES:-128}
MAX_SEQS=${MAX_SEQS:-4}
MASTER_PORT=${MASTER_PORT:-12345}
LIMIT=${LIMIT:-}
INFERENCE_ONLY=${INFERENCE_ONLY:-0}
OFFLINE=${OFFLINE:-1}

HF_HOME=${HF_HOME:-/data/caojingqi/huggingface}
JUDGE_ENV=${JUDGE_ENV:-/data/caojingqi/judge.env}
LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-/data/caojingqi/lmms-eval}
VLLM_PYTHON=${VLLM_PYTHON:-/data/caojingqi/miniconda3/envs/longvt/bin/python}
OUT=${OUT:-$ROOT/runs/eval-$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUT"

export HF_HOME
export LMMS_EVAL_HOME="$HF_HOME"
export LONGVT_VIDEO_DIR="$ROOT/media"
export LONGVT_MCP_MAX_FRAMES="$MCP_MAX_FRAMES"
export FORCE_QWENVL_VIDEO_READER=av
export LD_LIBRARY_PATH="/data/caojingqi/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export DECORD_EOF_RETRY_MAX=409600
export LMMS_EVAL_USE_CACHE=False

if [ "$OFFLINE" = "1" ]; then
  export HF_HUB_OFFLINE=1
fi

if [ "$INFERENCE_ONLY" = "1" ]; then
  export USE_LLM_JUDGE=False
else
  set -a
  source "$JUDGE_ENV"
  set +a
  export USE_LLM_JUDGE=True
  export https_proxy=${https_proxy:-http://127.0.0.1:7890}
  export http_proxy=${http_proxy:-http://127.0.0.1:7890}
  export no_proxy=${no_proxy:+$no_proxy,}localhost,127.0.0.1,10.3.208.0/24
  export NO_PROXY=$no_proxy
  "$LMMS_EVAL_DIR/.venv/bin/python" "$ROOT/scripts/sft/check_judge.py"
fi

export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1"

CUDA_VISIBLE_DEVICES="$GPU" \
HF_HOME="$HF_HOME" \
  "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name longvt-eval \
  --chat-template "$ROOT/examples/eval/tool_call_qwen2_5_vl.jinja" \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --port "$PORT" \
  --gpu-memory-utilization 0.95 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --limit-mm-per-prompt "image=$MAX_IMAGES,video=0" \
  --enforce-eager \
  --disable-log-requests \
  >"$OUT/vllm.log" 2>&1 &
vllm_pid=$!
trap 'kill "$vllm_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    echo "vLLM exited before becoming ready; see $OUT/vllm.log" >&2
    tail -80 "$OUT/vllm.log" >&2
    exit 1
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

limit_args=()
if [ -n "$LIMIT" ]; then
  limit_args=(--limit "$LIMIT")
fi

"$LMMS_EVAL_DIR/.venv/bin/torchrun" \
  --nproc_per_node="$PROCS" \
  --master_port="$MASTER_PORT" \
  -m lmms_eval \
  --model async_openai \
  --model_args "model_version=longvt-eval,mcp_server_path=$ROOT/examples/video_tools/mcp_server.py,fps=1,max_frames=$MAX_FRAMES,max_pixels=50176,base_url=$OPENAI_API_BASE,api_key=EMPTY,num_cpus=1,timeout=12000,is_qwen3_vl=False" \
  --tasks longvt \
  --batch_size 1 \
  --output_path "$OUT" \
  --log_samples \
  --include_path "$ROOT/lmms_eval_tasks" \
  "${limit_args[@]}"
