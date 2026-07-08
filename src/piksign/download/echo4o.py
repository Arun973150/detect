"""Semantic-branch corpus: Echo-4o surreal GPT-4o images (fake) + photo reals.

The semantic VLM must ONLY see semantic differences, so this corpus is:
  fake: semantically implausible GPT-4o generations (Echo-4o surreal subset)
  real: ordinary photographs (COCO subset by default)
Low-level cues are destroyed later by online pixel-scrambling during DPO
labeling/training, not here.

STREAMING by default: Echo-4o is ~180k images; we keep only --n of them and
never download the full dataset to disk.

Outputs:
  data/processed/semantic/fake/<idx>.jpg
  data/processed/semantic/real/<idx>.jpg

    python -m piksign.download.echo4o --n 5000 --reals-dir data/raw/coco/train2017_12000
"""
from __future__ import annotations

import argparse
import random
import re
from itertools import chain
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, list_images, processed_dir

Image.MAX_IMAGE_PIXELS = None

SURREAL_HINTS = re.compile(r"surreal|fantasy|imagin", re.I)


def pick_config(repo: str, wanted: str | None):
    from datasets import get_dataset_config_names
    if wanted:
        return wanted
    try:
        configs = get_dataset_config_names(repo)
    except Exception:  # noqa: BLE001
        return None
    for c in configs or []:
        if SURREAL_HINTS.search(c):
            return c
    return configs[0] if configs else None


def detect_image_column(example: dict) -> str:
    cols = [k for k, v in example.items() if isinstance(v, Image.Image)]
    if not cols:
        raise SystemExit(f"no image column found; keys: {list(example)}")
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
    ap.add_argument("--no-streaming", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    streaming = not args.no_streaming
    cfg = pick_config(args.repo_id, args.config)
    print(f"loading {args.repo_id} config={cfg} split={args.split} streaming={streaming} ...")
    ds = (load_dataset(args.repo_id, cfg, split=args.split, streaming=streaming)
          if cfg else load_dataset(args.repo_id, split=args.split, streaming=streaming))
    ds = ds.shuffle(seed=args.seed, buffer_size=1000) if streaming else ds.shuffle(seed=args.seed)

    it = iter(ds)
    first = next(it)
    col = detect_image_column(first)
    print(f"image column: {col}")

    fake_out = ensure(processed_dir("semantic", "fake"))
    written = 0
    bar = tqdm(total=args.n, desc="echo4o fakes")
    for row in chain([first], it):
        if written >= args.n:
            break
        dst = fake_out / f"{written:07d}.jpg"
        if dst.exists():
            written += 1
            bar.update(1)
            continue
        try:
            img = row[col]
            if img is None:
                continue
            normalize_save(img.convert("RGB"), dst)
            written += 1
            bar.update(1)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {e}")
    bar.close()
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
