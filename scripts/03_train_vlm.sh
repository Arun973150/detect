#!/usr/bin/env bash
# Semantic VLM branch: DPO label generation (72B-AWQ via vLLM, ~4-6h) then
# DPO fine-tune of the 7B (~4-6h). A100-80GB.
set -euo pipefail
cd "$(dirname "$0")/.."

# RunPod templates enable hf_transfer, a fast downloader that hangs silently
# on unstable connections; xet has similar issues on network volumes.
# Force the plain, retrying HTTP downloader for the big model pulls.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

# vLLM is installed here (not in setup) because it pins its own torch.
# PINNED as a matched pair: latest vLLM pulls a torch built for newer CUDA
# than RunPod's drivers (12.8) support, and vllm 0.10.x needs the
# transformers-4.55 API (all_special_tokens_extended was removed later).
pip show vllm >/dev/null 2>&1 || pip install "vllm==0.10.2" "transformers==4.55.2"
python -c "import transformers as t; assert t.__version__.startswith('4.55'), t.__version__" \
    || pip install "transformers==4.55.2"

# LABEL_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ is the budget option (~2-3x
# faster/cheaper than 72B, slightly weaker rationales).
python -m piksign.train.label_dpo --model "${LABEL_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"

python -m piksign.train.train_dpo

echo "VLM branch complete -> ${PIKSIGN_CKPT:-checkpoints}/vlm_dpo"
