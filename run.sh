#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

(
  while true; do
    date -u '+LOG_HEARTBEAT %Y-%m-%dT%H:%M:%SZ'
    sleep 20
  done
) &
heartbeat_pid=$!
trap 'kill "$heartbeat_pid" 2>/dev/null || true' EXIT

python -m pip install --no-cache-dir -r requirements.txt
echo "RUN_CONFIG_BEGIN"
cat config.json
echo "RUN_CONFIG_END"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
pids=()
for local_rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="$local_rank" LOCAL_RANK="$local_rank" LOCAL_WORLD_SIZE=8 \
    python -u experiment.py &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
