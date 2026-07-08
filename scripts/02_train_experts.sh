#!/usr/bin/env bash
# Train the four pixel-artifact experts. GPU, ~1.5-2h each on A100.
set -euo pipefail
cd "$(dirname "$0")/.."
D="${PIKSIGN_DATA:-data}/processed/pairs"

# Generalization experts (A1/A2): paper's conservative alpha=1.0 keeps the DINOv2
# features mostly frozen, which generalizes best to unseen generators.
python -m piksign.train.train_expert --name a1_sd21 --lora-alpha 1.0 \
    --real-dir "$D/sd21_recon/real" --fake-dir "$D/sd21_recon/fake"

python -m piksign.train.train_expert --name a2_flux --lora-alpha 1.0 \
    --real-dir "$D/flux_recon/real" --fake-dir "$D/flux_recon/fake"

# Target-specific experts (A3/A4): stronger alpha=8.0 lets the adapter lock onto
# Nano Banana / GPT-4o fingerprints. Tune down if val bacc >> in-the-wild eval.
python -m piksign.train.train_expert --name a3_nanobanana --lora-alpha 8.0 \
    --real-dir "$D/nanobanana/real" --fake-dir "$D/nanobanana/fake" \
    --val-real "$D/nanobanana_val/real" --val-fake "$D/nanobanana_val/fake"

python -m piksign.train.train_expert --name a4_gpt4o --lora-alpha 8.0 \
    --real-dir "$D/gpt4o/real" --fake-dir "$D/gpt4o/fake" \
    --val-real "$D/gpt4o_val/real" --val-fake "$D/gpt4o_val/fake"

echo "expert training complete."
