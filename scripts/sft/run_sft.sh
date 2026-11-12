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
RENDERED_CONFIG="$OUTPUT_DIR/config.input.yaml"

test -x "$PYTHON" || { echo "Missing Python: $PYTHON" >&2; exit 1; }
test -x "$TORCHRUN" || { echo "Missing torchrun: $TORCHRUN" >&2; exit 1; }
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=decord
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONPATH="$ENGINE/src${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

"$PYTHON" - "$CONFIG" "$RENDERED_CONFIG" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

import yaml

source, target, output_dir = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text())
config.setdefault("trainer_args", {})["output_dir"] = output_dir
Path(target).write_text(yaml.safe_dump(config, sort_keys=False))
PY

echo "CONFIG=$CONFIG"
echo "RENDERED_CONFIG=$RENDERED_CONFIG"
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
  config_yaml="$RENDERED_CONFIG" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
