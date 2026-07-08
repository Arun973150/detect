#!/usr/bin/env bash
# VAE reconstruction pair sets for the generalization experts. GPU, ~2h total.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${PIKSIGN_DATA:-data}/raw/coco/train2017_12000"

python -m piksign.recon.vae_reconstruct --vae sd21 --input "$SRC"
python -m piksign.recon.vae_reconstruct --vae flux --input "$SRC"

python -m piksign.audit --real "${PIKSIGN_DATA:-data}/processed/pairs/sd21_recon/real" \
                        --fake "${PIKSIGN_DATA:-data}/processed/pairs/sd21_recon/fake"
python -m piksign.audit --real "${PIKSIGN_DATA:-data}/processed/pairs/flux_recon/real" \
                        --fake "${PIKSIGN_DATA:-data}/processed/pairs/flux_recon/fake"

echo "reconstruction phase complete."
