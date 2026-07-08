"""Dataset acquisition tools. Each module is a standalone CLI:

    python -m piksign.download.drive         # user's piksign_data from Google Drive
    python -m piksign.download.pico_banana   # Nano Banana pairs (Apple CDN + OpenImages S3)
    python -m piksign.download.sharegpt4o    # GPT-4o edit pairs (HuggingFace)
    python -m piksign.download.coco          # COCO reals (VAE recon + semantic + eval)
    python -m piksign.download.echo4o        # Echo-4o surreal fakes (HuggingFace)
    python -m piksign.download.gen_eval_api  # fresh eval fakes via Gemini/OpenAI APIs
"""

import shutil
from pathlib import Path

from PIL import Image


def normalize_save(img: Image.Image, out: Path, quality: int = 95) -> None:
    """Single normalization funnel for ALL training images, both classes.

    Everything becomes JPEG q95 4:4:4. This kills the format shortcut
    (e.g. Nano Banana outputs are PNG while OpenImages sources are JPEG);
    q95 is mild relative to the training augmentation so the generator
    fingerprint survives.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.jpg")
    img.convert("RGB").save(tmp, format="JPEG", quality=quality, subsampling=0)
    shutil.move(tmp, out)
