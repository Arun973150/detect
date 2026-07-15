"""Normalize one or more image folders into a reusable train/val image pool.

This is used for the modern real class: FODB/DPED/OpenImages/YFCC-style
photos are collected outside this repo, then pushed through the same JPEG
normalization funnel as generated images before expert training.

    python -m piksign.download.image_pool \
        --input /workspace/reals/fodb \
        --input /workspace/reals/dped \
        --out data/processed/reals/modern_train \
        --val-out data/processed/reals/modern_val \
        --n 40000
"""
from __future__ import annotations

import argparse
import random
import zlib
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, list_images

Image.MAX_IMAGE_PIXELS = None


def dest_name(path: Path, index: int) -> str:
    key = str(path.resolve()).replace("\\", "/")
    digest = zlib.crc32(key.encode()) & 0xFFFFFFFF
    return f"{index:02d}_{digest:08x}_{path.stem}.jpg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, action="append", required=True,
                    help="image directory; repeat for multiple sources")
    ap.add_argument("--out", type=Path, required=True, help="train/output directory")
    ap.add_argument("--val-out", type=Path, default=None, help="optional validation directory")
    ap.add_argument("--n", type=int, default=0, help="cap images after shuffling (0 = all)")
    ap.add_argument("--max-side", type=int, default=0,
                    help="downscale so max(w,h) <= this before normalizing (0 = keep native). "
                         "Use for full-res reals paired against low-res fakes: without it, "
                         "resolution itself becomes the label and VAE recon crops land in "
                         "smooth regions with no learnable fingerprint.")
    ap.add_argument("--val-every", type=int, default=20,
                    help="send every kth stable-hash image to --val-out; 0 disables")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths: list[tuple[int, Path]] = []
    for i, root in enumerate(args.input):
        imgs = list_images(root)
        print(f"{len(imgs)} images under {root}")
        paths.extend((i, p) for p in imgs)
    if not paths:
        raise SystemExit("no images found")

    rng = random.Random(args.seed)
    rng.shuffle(paths)
    if args.n and args.n < len(paths):
        paths = paths[:args.n]

    ensure(args.out)
    if args.val_out:
        ensure(args.val_out)

    train_n = 0
    val_n = 0
    for i, p in tqdm(paths, desc="image pool"):
        name = dest_name(p, i)
        use_val = (
            args.val_out is not None and args.val_every > 0 and
            (zlib.crc32(str(p.resolve()).encode()) % args.val_every == 0)
        )
        dst = (args.val_out if use_val else args.out) / name
        if dst.exists():
            if use_val:
                val_n += 1
            else:
                train_n += 1
            continue
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if args.max_side and max(im.size) > args.max_side:
                    im.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
                normalize_save(im, dst)
            if use_val:
                val_n += 1
            else:
                train_n += 1
        except Exception as e:  # noqa: BLE001
            print(f"[skip {p}] {e}")

    print(f"done: {train_n} train -> {args.out}")
    if args.val_out:
        print(f"      {val_n} val   -> {args.val_out}")


if __name__ == "__main__":
    main()