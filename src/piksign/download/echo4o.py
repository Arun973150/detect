"""Semantic-branch corpus: Echo-4o surreal GPT-4o images (fake) + photo reals.

The semantic VLM must ONLY see semantic differences, so this corpus is:
  fake: semantically implausible GPT-4o generations (Echo-4o surreal subset)
  real: ordinary photographs (COCO subset by default)
Low-level cues are destroyed later by online pixel-scrambling during DPO
labeling/training, not here.

Outputs:
  data/processed/semantic/fake/<idx>.jpg
  data/processed/semantic/real/<idx>.jpg

    python -m piksign.download.echo4o --n 5000 --reals-dir data/raw/coco/train2017_12000
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, list_images, processed_dir

Image.MAX_IMAGE_PIXELS = None

SURREAL_HINTS = re.compile(r"surreal|fantasy|imagin", re.I)


def load_echo4o(repo: str, config: str | None, split: str):
    from datasets import get_dataset_config_names, load_dataset

    cfg = config
    if cfg is None:
        try:
            configs = get_dataset_config_names(repo)
        except Exception:  # noqa: BLE001
            configs = []
        for c in configs:
            if SURREAL_HINTS.search(c):
                cfg = c
                break
    print(f"loading {repo} config={cfg} split={split} ...")
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    return ds


def find_image_column(ds) -> str:
    from datasets import Image as HFImage
    cols = [n for n, f in ds.features.items() if isinstance(f, HFImage)]
    if not cols:
        raise SystemExit(f"no image column found; features: {ds.features}")
    return cols[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="Yejy53/Echo-4o-Image",
                    help="HF dataset with GPT-4o surreal generations")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--reals-dir", type=Path, required=True,
                    help="directory of real photos to sample the real class from (e.g. COCO subset)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # fakes: Echo-4o surreal generations
    fake_out = ensure(processed_dir("semantic", "fake"))
    ds = load_echo4o(args.repo_id, args.config, args.split)
    col = find_image_column(ds)
    print(f"image column: {col}  rows: {len(ds)}")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    written = 0
    for idx in tqdm(indices, desc="echo4o fakes"):
        if written >= args.n:
            break
        dst = fake_out / f"{idx:07d}.jpg"
        if dst.exists():
            written += 1
            continue
        try:
            img = ds[idx][col]
            if img is None:
                continue
            normalize_save(img.convert("RGB"), dst)
            written += 1
        except Exception as e:  # noqa: BLE001
            print(f"[skip {idx}] {e}")
    print(f"fakes: {written} -> {fake_out}")

    # reals: sampled photos, same normalization funnel
    real_out = ensure(processed_dir("semantic", "real"))
    reals = list_images(args.reals_dir)
    if len(reals) < args.n:
        print(f"WARNING: only {len(reals)} reals available (< {args.n})")
    picked = rng.sample(reals, min(args.n, len(reals)))
    for i, p in enumerate(tqdm(picked, desc="semantic reals")):
        dst = real_out / f"{i:07d}.jpg"
        if dst.exists():
            continue
        try:
            with Image.open(p) as im:
                normalize_save(im.convert("RGB"), dst)
        except Exception as e:  # noqa: BLE001
            print(f"[skip {p}] {e}")
    print(f"reals: {len(picked)} -> {real_out}")


if __name__ == "__main__":
    main()
