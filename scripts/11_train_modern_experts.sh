#!/usr/bin/env bash
# Retrain the A1 SD-family and A4 GPT-edit experts on modern real negatives.
set -euo pipefail
cd "$(dirname "$0")/.."

D="${PIKSIGN_DATA:-data}/processed/pairs"

python -m piksign.train.train_expert --name a1_sd21 --lora-alpha 1.0 \
    --real-dir "$D/sd21_modern_recon/real" --fake-dir "$D/sd21_modern_recon/fake" \
    --val-real "$D/sd21_modern_recon_val/real" --val-fake "$D/sd21_modern_recon_val/fake"

# a2 is the Gemini/Nano Banana workhorse; "gentle" launder (0.3) since the FLUX
# fingerprint is fragile and dies under heavy augmentation (verified earlier).
python -m piksign.train.train_expert --name a2_flux_gentle --lora-alpha 8.0 --launder-prob 0.3 \
    --real-dir "$D/flux_modern_recon/real" --fake-dir "$D/flux_modern_recon/fake" \
    --val-real "$D/flux_modern_recon_val/real" --val-fake "$D/flux_modern_recon_val/fake"

python -m piksign.train.train_expert --name a4_gpt4o --lora-alpha 8.0 \
    --real-dir "$D/gpt4o_modern/real" --fake-dir "$D/gpt4o_modern/fake" \
    --val-real "$D/gpt4o_modern_val/real" --val-fake "$D/gpt4o_modern_val/fake"

echo "modern A1/A4 expert training complete."