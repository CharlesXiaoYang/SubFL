#!/usr/bin/env bash
# Runs the commands of one experiments/<study>.txt file, one after another.
# Usage: bash run.sh experiments/<study>.txt [GPU [NUM_GPUS]]
#   runs every NUM_GPUS-th line, starting at line GPU+1, on GPU (defaults: GPU 0, NUM_GPUS 1).
#   DRY_RUN=1 only prints the commands. Logs go to logs/<study>/; completed runs are skipped.
set -u
gpu=${2:-0}
n=${3:-1}
if [ ! -f "${1:-}" ] || ! [[ $gpu =~ ^[0-9]+$ && $n =~ ^[0-9]+$ ]] || [ "$gpu" -ge "$n" ]; then
  echo "usage: bash run.sh experiments/<study>.txt [GPU [NUM_GPUS]]  (GPU < NUM_GPUS)" >&2
  exit 1
fi
list=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
study=$(basename "$1" .txt)
cd "$(dirname "$0")"
tag_re='--run_tag ([^ ]+)'
out_re='--output_root ([^ ]+)'
i=0
while IFS= read -r cmd; do
  [ -n "$cmd" ] || continue
  i=$((i + 1))
  [ $(((i - 1) % n)) -eq "$gpu" ] || continue
  [[ $cmd =~ $tag_re ]] && name=${BASH_REMATCH[1]}
  [[ $cmd =~ $out_re ]] && out=${BASH_REMATCH[1]}
  if grep -qsE '"status": *"completed"' "$out"/*/run_status.json; then
    echo "skip $name (completed)"
    continue
  fi
  log=logs/$study/$name.log
  if [ -n "${DRY_RUN:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES=$gpu $cmd > $log"
    continue
  fi
  mkdir -p "logs/$study"
  echo "$(date '+%F %T') GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$log" 2>&1 < /dev/null || echo "failed: $name (see $log)"
done < "$list"
