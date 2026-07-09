#!/usr/bin/env bash
# Acquire all training data. No GPU needed - run on a cheap CPU pod attached
# to the same network volume to save A100 hours.
set -euo pipefail
cd "$(dirname "$0")/.."

# user's piksign_data (manifests + official sft.jsonl)
python -m piksign.download.drive

# COCO reals: 12k train (VAE recon sources + semantic reals) and full val (eval reals)
# --delete-zip frees the 19GB archive immediately (important on small pod disks)
python -m piksign.download.coco --split train2017 --n 12000 --delete-zip
python -m piksign.download.coco --split val2017 --n 0 --delete-zip

# Nano Banana pairs: Apple CDN edits + OpenImages S3 sources, piksign split
python -m piksign.download.pico_banana --workers 16

# GPT-4o edit pairs from ShareGPT-4o-Image
python -m piksign.download.sharegpt4o --n 16000

# semantic corpus: Echo-4o surreal fakes + COCO reals
python -m piksign.download.echo4o --n 5000 --reals-dir "${PIKSIGN_DATA:-data}/raw/coco/train2017_12000"

# audit every pair set - MUST pass before training
python -m piksign.audit --real "${PIKSIGN_DATA:-data}/processed/pairs/nanobanana/real" \
                        --fake "${PIKSIGN_DATA:-data}/processed/pairs/nanobanana/fake"
python -m piksign.audit --real "${PIKSIGN_DATA:-data}/processed/pairs/gpt4o/real" \
                        --fake "${PIKSIGN_DATA:-data}/processed/pairs/gpt4o/fake"

echo "download phase complete."
