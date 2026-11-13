#!/bin/bash
set -euo pipefail

RUN_DIR=${1:?usage: wait_and_rejudge.sh RUN_DIR [expected_count]}
EXPECTED=${2:-652}
JUDGE_ENV=${JUDGE_ENV:-/data/caojingqi/judge.env}
LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-/data/caojingqi/lmms-eval}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORKERS=${WORKERS:-8}
POLL_SECONDS=${POLL_SECONDS:-30}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-28800}

set -a
source "$JUDGE_ENV"
set +a
export https_proxy=${https_proxy:-http://127.0.0.1:7890}
export http_proxy=${http_proxy:-http://127.0.0.1:7890}

"$LMMS_EVAL_DIR/.venv/bin/python" "$REPO_ROOT/scripts/sft/check_judge.py"

elapsed=0
sample_file=""
while [ "$elapsed" -lt "$MAX_WAIT_SECONDS" ]; do
  shopt -s nullglob
  files=("$RUN_DIR"/*/*_samples_longvt.jsonl)
  shopt -u nullglob
  if [ "${#files[@]}" -eq 1 ]; then
    lines=$(wc -l < "${files[0]}")
    if [ "$lines" -ge "$EXPECTED" ]; then
      sample_file=${files[0]}
      break
    fi
    echo "Waiting: ${files[0]} has $lines/$EXPECTED samples"
  else
    echo "Waiting for the completed samples file under $RUN_DIR"
  fi
  sleep "$POLL_SECONDS"
  elapsed=$((elapsed + POLL_SECONDS))
done

if [ -z "$sample_file" ]; then
  echo "Timed out after ${MAX_WAIT_SECONDS}s waiting for $EXPECTED samples." >&2
  exit 1
fi

rejudged=${sample_file%_samples_longvt.jsonl}_rejudged.jsonl
"$LMMS_EVAL_DIR/.venv/bin/python" "$REPO_ROOT/scripts/sft/rejudge_samples.py" \
  --samples "$sample_file" \
  --output "$rejudged" \
  --workers "$WORKERS"

summary=${rejudged%.jsonl}.summary.json
report=$(dirname "$sample_file")/FULL_EVAL_REPORT.md
"$LMMS_EVAL_DIR/.venv/bin/python" "$REPO_ROOT/scripts/sft/write_eval_report.py" \
  --samples "$sample_file" \
  --summary "$summary" \
  --output "$report" \
  --frames 64
