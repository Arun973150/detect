#!/usr/bin/env bash
# Train the four pixel-artifact experts. GPU, ~1.5-2h each on A100.
set -euo pipefail
cd "$(dirname "$0")/.."
D="${PIKSIGN_DATA:-data}/processed/pairs"

python -m piksign.train.train_expert --name a1_sd21 \
    --real-dir "$D/sd21_recon/real" --fake-dir "$D/sd21_recon/fake"

python -m piksign.train.train_expert --name a2_flux \
    --real-dir "$D/flux_recon/real" --fake-dir "$D/flux_recon/fake"

python -m piksign.train.train_expert --name a3_nanobanana \
    --real-dir "$D/nanobanana/real" --fake-dir "$D/nanobanana/fake" \
    --val-real "$D/nanobanana_val/real" --val-fake "$D/nanobanana_val/fake"

python -m piksign.train.train_expert --name a4_gpt4o \
    --real-dir "$D/gpt4o/real" --fake-dir "$D/gpt4o/fake" \
    --val-real "$D/gpt4o_val/real" --val-fake "$D/gpt4o_val/fake"

echo "expert training complete."
