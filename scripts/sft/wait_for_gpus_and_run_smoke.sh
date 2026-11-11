#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/caojingqi/LongVT
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
REQUIRED_GPUS="${REQUIRED_GPUS:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"

while true; do
  mapfile -t available < <(
    nvidia-smi \
      --query-gpu=index,memory.free \
      --format=csv,noheader,nounits |
      awk -F', ' -v min="$MIN_FREE_MIB" '$2 >= min { print $1 }'
  )

  printf '%s available GPUs with >= %s MiB free:' \
    "$(date -Iseconds)" "$MIN_FREE_MIB"
  printf ' %s' "${available[@]:-}"
  printf '\n'

  if ((${#available[@]} >= REQUIRED_GPUS)); then
    selected="$(IFS=,; echo "${available[*]:0:REQUIRED_GPUS}")"
    echo "Starting smoke on GPUs: $selected"
    CUDA_VISIBLE_DEVICES="$selected" \
      "$ROOT/scripts/sft/run_sft.sh" \
      "$ROOT/examples/sft/local/longvt_7b_sft_smoke.yaml" \
      "$REQUIRED_GPUS"
    exit $?
  fi

  sleep "$POLL_SECONDS"
done
