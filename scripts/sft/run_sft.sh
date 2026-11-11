#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/caojingqi
ENGINE="$ROOT/lmms-engine"
PYTHON="$ENGINE/.venv/bin/python"
TORCHRUN="$ENGINE/.venv/bin/torchrun"
CONFIG="${1:?usage: run_sft.sh CONFIG [NPROC] [OUTPUT_ROOT]}"
NPROC="${2:-8}"
OUTPUT_ROOT="${3:-$ROOT/LongVT/runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="$(basename "${CONFIG%.*}")_${STAMP}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

test -x "$PYTHON" || { echo "Missing Python: $PYTHON" >&2; exit 1; }
test -x "$TORCHRUN" || { echo "Missing torchrun: $TORCHRUN" >&2; exit 1; }
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=decord
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONPATH="$ENGINE/src${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

echo "CONFIG=$CONFIG"
echo "NPROC=$NPROC"
echo "OUTPUT_DIR=$OUTPUT_DIR"

cd "$ENGINE"
"$TORCHRUN" \
  --nproc_per_node="$NPROC" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-18000}" \
  -m lmms_engine.launch.cli \
  config_yaml="$CONFIG" \
  trainer_args.output_dir="$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
