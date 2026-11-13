#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
CONDA_BASE=${CONDA_BASE:-/data/caojingqi/miniconda3}
ENV_NAME=${ENV_NAME:-longvt-rl}
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
DRY_RUN=${DRY_RUN:-0}
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

if [ -e "$ENV_PREFIX" ]; then
  echo "Environment already exists: $ENV_PREFIX" >&2
  echo "Choose another ENV_NAME or remove it explicitly after review." >&2
  exit 1
fi

commands=(
  "$CONDA_BASE/bin/conda create -n $ENV_NAME python=3.10 -y"
  "PATH=$ENV_PREFIX/bin:\$PATH USE_MEGATRON=0 USE_SGLANG=1 bash $ROOT/scripts/install_vllm_sglang_mcore.sh"
  "PATH=$ENV_PREFIX/bin:\$PATH pip install --no-deps -e $ROOT"
  "$ENV_PREFIX/bin/python $ROOT/scripts/rl/preflight.py"
)

if [ "$DRY_RUN" = "1" ]; then
  printf '%s\n' "${commands[@]}"
  exit 0
fi

build_dir=$(mktemp -d /data/caojingqi/longvt-rl-build.XXXXXX)
trap 'rm -rf "$build_dir"' EXIT

"$CONDA_BASE/bin/conda" create -n "$ENV_NAME" python=3.10 -y

cd "$build_dir"
PATH="$ENV_PREFIX/bin:$PATH" \
USE_MEGATRON=0 \
USE_SGLANG=1 \
  bash "$ROOT/scripts/install_vllm_sglang_mcore.sh"

PATH="$ENV_PREFIX/bin:$PATH" pip install --no-deps -e "$ROOT"

cd "$ROOT"
LONGVT_VIDEO_DIR=${LONGVT_VIDEO_DIR:-$ROOT/media} \
  "$ENV_PREFIX/bin/python" "$ROOT/scripts/rl/preflight.py"
