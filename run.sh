#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python -m pip install --no-cache-dir -r requirements.txt
echo "RUN_CONFIG_BEGIN"
cat config.json
echo "RUN_CONFIG_END"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
torchrun --standalone --nproc_per_node=8 experiment.py

