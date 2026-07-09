#!/usr/bin/env bash
# Semantic VLM branch: DPO label generation (72B-AWQ via vLLM, ~4-6h) then
# DPO fine-tune of the 7B (~4-6h). A100-80GB.
set -euo pipefail
cd "$(dirname "$0")/.."

# vLLM is installed here (not in setup) because it pins its own torch.
# PINNED: latest vLLM pulls a torch built for newer CUDA than RunPod's
# drivers (12.8) support; 0.10.x matches the torch-2.8/cu128 era.
pip show vllm >/dev/null 2>&1 || pip install "vllm==0.10.2"

# LABEL_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ is the budget option (~2-3x
# faster/cheaper than 72B, slightly weaker rationales).
python -m piksign.train.label_dpo --model "${LABEL_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"

python -m piksign.train.train_dpo

echo "VLM branch complete -> ${PIKSIGN_CKPT:-checkpoints}/vlm_dpo"
