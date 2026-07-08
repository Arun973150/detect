#!/usr/bin/env bash
# Semantic VLM branch: DPO label generation (72B-AWQ via vLLM, ~4-6h) then
# DPO fine-tune of the 7B (~4-6h). A100-80GB.
set -euo pipefail
cd "$(dirname "$0")/.."

# vLLM is installed here (not in setup) because it pins its own torch;
# keep it isolated to this phase.
pip show vllm >/dev/null 2>&1 || pip install -e ".[label]"

# LABEL_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ is the budget option (~2-3x
# faster/cheaper than 72B, slightly weaker rationales).
python -m piksign.train.label_dpo --model "${LABEL_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"

python -m piksign.train.train_dpo

echo "VLM branch complete -> ${PIKSIGN_CKPT:-checkpoints}/vlm_dpo"
