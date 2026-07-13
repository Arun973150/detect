#!/usr/bin/env bash
# Build the modern A1/A4 training data without COCO or ShareGPT-4o as primary data.
#
# Expected real-photo source:
#   Put FODB/DPED/OpenImages/YFCC-style real images under:
#     ${PIKSIGN_DATA:-data}/raw/modern_reals
#   or set MODERN_REALS_DIR=/path/to/real/photo/root
#
# Outputs:
#   processed/reals/modern_{train,val}
#   processed/pairs/sd21_modern_recon{,_val}
#   processed/pairs/gpt4o_{ultraedit,hqedit_edit}
#   processed/pairs/gpt4o_modern{,_val}
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${PIKSIGN_DATA:-data}"
PAIRS="$DATA/processed/pairs"
REAL_SRC="${MODERN_REALS_DIR:-$DATA/raw/modern_reals}"
REAL_TRAIN="$DATA/processed/reals/modern_train"
REAL_VAL="$DATA/processed/reals/modern_val"

REAL_N="${MODERN_REALS_N:-40000}"
SD_RECON_N="${SD_RECON_N:-0}"
GPT_ULTRAEDIT_N="${GPT_ULTRAEDIT_N:-16000}"
GPT_HQEDIT_N="${GPT_HQEDIT_N:-16000}"

if [ ! -d "$REAL_SRC" ]; then
    echo "missing real-photo source: $REAL_SRC"
    echo "put FODB/DPED/OpenImages/YFCC-style real images there or set MODERN_REALS_DIR"
    exit 1
fi

python -m piksign.download.image_pool \
    --input "$REAL_SRC" \
    --out "$REAL_TRAIN" \
    --val-out "$REAL_VAL" \
    --n "$REAL_N"

python -m piksign.recon.vae_reconstruct --vae sd21 \
    --input "$REAL_TRAIN" \
    --out-name sd21_modern_recon \
    --n "$SD_RECON_N"

python -m piksign.recon.vae_reconstruct --vae sd21 \
    --input "$REAL_VAL" \
    --out-name sd21_modern_recon_val \
    --n "$SD_RECON_N"

# FLUX-VAE recon on the same modern reals: a2 was the best pixel catcher on
# Gemini/Nano Banana (its autoencoder is in the FLUX family), so retraining it
# on modern reals matters most for the actual target.
python -m piksign.recon.vae_reconstruct --vae flux \
    --input "$REAL_TRAIN" \
    --out-name flux_modern_recon \
    --n "$SD_RECON_N"

python -m piksign.recon.vae_reconstruct --vae flux \
    --input "$REAL_VAL" \
    --out-name flux_modern_recon_val \
    --n "$SD_RECON_N"

python -m piksign.download.gptimageedit \
    --part-prefix gpt-edit/ultraedit.tar.gz.part \
    --out-name gpt4o_ultraedit \
    --n "$GPT_ULTRAEDIT_N"

python -m piksign.download.gptimageedit \
    --part-prefix gpt-edit/hqedit.tar.gz.part \
    --include-task edit \
    --out-name gpt4o_hqedit_edit \
    --n "$GPT_HQEDIT_N"

python -m piksign.download.image_pool \
    --input "$PAIRS/gpt4o_ultraedit/fake" \
    --input "$PAIRS/gpt4o_hqedit_edit/fake" \
    --out "$PAIRS/gpt4o_modern/fake" \
    --val-every 0

python -m piksign.download.image_pool \
    --input "$PAIRS/gpt4o_ultraedit_val/fake" \
    --input "$PAIRS/gpt4o_hqedit_edit_val/fake" \
    --out "$PAIRS/gpt4o_modern_val/fake" \
    --val-every 0

python -m piksign.download.image_pool \
    --input "$REAL_TRAIN" \
    --out "$PAIRS/gpt4o_modern/real" \
    --val-every 0

python -m piksign.download.image_pool \
    --input "$REAL_VAL" \
    --out "$PAIRS/gpt4o_modern_val/real" \
    --val-every 0

python -m piksign.audit --real "$PAIRS/gpt4o_modern/real" \
                        --fake "$PAIRS/gpt4o_modern/fake"
python -m piksign.audit --real "$PAIRS/sd21_modern_recon/real" \
                        --fake "$PAIRS/sd21_modern_recon/fake"
python -m piksign.audit --real "$PAIRS/flux_modern_recon/real" \
                        --fake "$PAIRS/flux_modern_recon/fake"

echo "modern expert data ready."