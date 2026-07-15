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

# Modern reals: by default pull them directly from OpenFake (current 2025-2026
# photos: Pexels/DOCCI/Reddit in-the-wild). Set OPENFAKE_REALS=0 to instead use
# your own $REAL_SRC folder (VISION/FODB/your phone photos). Either way, drop
# your own phone photos into $REAL_SRC too - image_pool merges all of it.
mkdir -p "$REAL_SRC"
if [ "${OPENFAKE_REALS:-1}" = "1" ]; then
    # Non-fatal: the OpenFake streams can segfault mid-iteration on a bad row
    # (native pyarrow/PIL abort, uncatchable in Python). Never let that kill the
    # whole pipeline - keep whatever reals already landed and push on to recon.
    # Both pulls are resumable (they skip files already on disk).
    python -m piksign.download.openfake --config reddit --split test --label real \
        --out "$REAL_SRC/openfake_reddit" --n "${OPENFAKE_REDDIT_N:-15000}" \
        || echo "[warn] openfake reddit stream ended early - keeping what downloaded"
    if [ "${OPENFAKE_CORE_N:-25000}" != "0" ]; then
        python -m piksign.download.openfake --config core --split train --label real \
            --min-date "${MODERN_MIN_DATE:-2024}" \
            --out "$REAL_SRC/openfake_core" --n "${OPENFAKE_CORE_N:-25000}" \
            || echo "[warn] openfake core stream ended early - keeping what downloaded"
    fi
fi

if [ -z "$(ls -A "$REAL_SRC" 2>/dev/null)" ]; then
    echo "no real photos under $REAL_SRC"
    echo "enable OPENFAKE_REALS=1 (default) or add your own images / set MODERN_REALS_DIR"
    exit 1
fi

# --max-side 640: OpenFake reals arrive at ~9MP while the GPT fakes are ~0.26MP.
# Unresized, resolution itself becomes the label (a4 "learns" big=real) and the
# 1024px recon crops land in smooth 9MP regions where the VAE reconstructs
# perfectly, so a1/a2 flatline at 50%. Downscaling restores parity and puts the
# whole busy frame inside every crop.
python -m piksign.download.image_pool \
    --input "$REAL_SRC" \
    --out "$REAL_TRAIN" \
    --val-out "$REAL_VAL" \
    --max-side "${REAL_MAX_SIDE:-640}" \
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