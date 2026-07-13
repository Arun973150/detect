"""Stream OpenFake (ComplexDataLab/OpenFake) - modern (2025-2026) reals & fakes.

Why: COCO/VISION/FODB reals are old (2014-2020 cameras), so the experts
false-positive on modern computational-photography phone output. OpenFake's
reals are current (LAION/Pexels/DOCCI/ImageNet/Reddit), and its 'reddit' config
is in-the-wild real-user photos - the closest public proxy for a modern phone
domain. Its fakes come from current commercial generators (gpt-image,
gemini/nano-banana, imagen, flux, midjourney, ...), so one dataset serves both
the negative class (fix false positives) and the positive class (real targets).

Schema: image (embedded), label (real|fake), model (generator or real source),
prompt, type, release_date. Configs: 'core' (train/val/test), 'reddit' (test).
License note: proprietary-generator subsets are research/non-commercial.

Examples:
  # modern in-the-wild reals for the negative class
  python -m piksign.download.openfake --config reddit --split test --label real \
      --out data/raw/modern_reals/openfake_reddit --n 20000

  # modern reals from core (Pexels/DOCCI/...), 2024 onward
  python -m piksign.download.openfake --config core --split train --label real \
      --min-date 2024 --out data/raw/modern_reals/openfake_core --n 20000

  # Gemini / Nano Banana fakes (your primary target)
  python -m piksign.download.openfake --config core --split train --label fake \
      --model-contains gemini,nano,imagen --out data/raw/openfake_fakes/gemini --n 8000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure

Image.MAX_IMAGE_PIXELS = None

REPO = "ComplexDataLab/OpenFake"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=REPO)
    ap.add_argument("--config", default="core", help="'core' or 'reddit'")
    ap.add_argument("--split", default="train", help="train/validation/test (reddit: test)")
    ap.add_argument("--label", choices=["real", "fake"], default=None,
                    help="keep only this label (default: both)")
    ap.add_argument("--model-contains", default=None,
                    help="comma list; keep rows whose 'model' contains any (substring, case-insensitive)")
    ap.add_argument("--min-date", default=None,
                    help="keep rows with release_date >= this (lexical YYYY or YYYY-MM-DD)")
    ap.add_argument("--out", type=Path, required=True, help="output image directory")
    ap.add_argument("--n", type=int, default=20000, help="max images to keep")
    ap.add_argument("--no-streaming", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    wanted_models = [m.strip().lower() for m in args.model_contains.split(",")] if args.model_contains else None
    out = ensure(args.out)
    have = len(list(out.glob("*.jpg")))
    print(f"{have} images already present at {out}, target {args.n}")
    if have >= args.n:
        print("nothing to do")
        return

    streaming = not args.no_streaming
    print(f"loading {args.repo_id} config={args.config} split={args.split} streaming={streaming} ...")
    ds = load_dataset(args.repo_id, args.config, split=args.split, streaming=streaming)
    ds = ds.shuffle(seed=args.seed, buffer_size=2000) if streaming else ds.shuffle(seed=args.seed)

    def keep(row) -> bool:
        if args.label and str(row.get("label", "")).lower() != args.label:
            return False
        if wanted_models:
            m = str(row.get("model", "")).lower()
            if not any(w in m for w in wanted_models):
                return False
        if args.min_date:
            rd = str(row.get("release_date", "") or "")
            if rd and rd < args.min_date:
                return False
        return True

    written = have
    seen = 0
    bar = tqdm(total=args.n, initial=have, desc=f"openfake[{args.label or 'all'}]")
    for row in ds:
        if written >= args.n:
            break
        seen += 1
        if not keep(row):
            continue
        img = row.get("image")
        if img is None:
            continue
        dst = out / f"{written:07d}.jpg"
        if dst.exists():
            written += 1
            bar.update(1)
            continue
        try:
            normalize_save(img.convert("RGB"), dst)
            written += 1
            bar.update(1)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {e}")
    bar.close()
    print(f"done: {written - have} new ({written} total) from {seen} rows scanned -> {out}")
    if wanted_models:
        print(f"      (filtered model contains any of {wanted_models})")


if __name__ == "__main__":
    main()
