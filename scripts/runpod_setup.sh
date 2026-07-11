
#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod (PyTorch template, A100 SXM 80GB,
# 50GB container disk + 250GB network volume mounted at /workspace).
set -euo pipefail

# ---- everything persistent lives on the network volume -----------------
export PIKSIGN_DATA=/workspace/data
export PIKSIGN_CKPT=/workspace/checkpoints
export HF_HOME=/workspace/hf
mkdir -p "$PIKSIGN_DATA" "$PIKSIGN_CKPT" "$HF_HOME"

# persist env for future shells
cat >> ~/.bashrc <<'EOF'
export PIKSIGN_DATA=/workspace/data
export PIKSIGN_CKPT=/workspace/checkpoints
export HF_HOME=/workspace/hf
EOF

# ---- install ------------------------------------------------------------
cd "$(dirname "$0")/.."
pip install -U pip
pip install -e ".[dpo,gen]"
# vLLM (DPO labeling) pulls its own torch pin; install last and only when needed:
#   pip install -e ".[label]"

echo
echo "Setup complete. Suggested order:"
echo "  bash scripts/00_download_all.sh    # old full pipeline: COCO + ShareGPT"
echo "  bash scripts/01_recon.sh           # old COCO VAE recon"
echo "  bash scripts/02_train_experts.sh   # old expert training"
echo "  bash scripts/10_prepare_modern_expert_data.sh  # new A1/A4 data: modern reals + GPT-Image-Edit"
echo "  bash scripts/11_train_modern_experts.sh        # new A1/A4 retrain"
echo "  bash scripts/03_train_vlm.sh       # DPO semantic branch"
echo "  bash scripts/04_eval.sh            # eval"
