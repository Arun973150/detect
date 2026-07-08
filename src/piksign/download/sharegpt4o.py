"""GPT-4o aligned edit pairs from ShareGPT-4o-Image (HuggingFace) for expert A4.

The dataset (FreedomIntelligence/ShareGPT-4o-Image) contains text-to-image and
text-and-image-to-image samples produced by GPT-4o. We want the EDITING subset:
(input image, GPT-4o output image) gives aligned real/fake pairs, the same
structure as Pico-Banana.

STREAMING by default: we never download the whole dataset to disk, only the
--n pairs we keep (important on small pod disks). Column names are discovered
defensively from the first record.

Outputs:
  data/processed/pairs/gpt4o/{real,fake}/<idx>.jpg          (train)
  data/processed/pairs/gpt4o_val/{real,fake}/<idx>.jpg      (5% val)

    python -m piksign.download.sharegpt4o --n 16000
"""
from __future__ import annotations

import argparse
import re
from itertools import chain

from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, processed_dir

Image.MAX_IMAGE_PIXELS = None

SOURCE_HINTS = re.compile(r"input|source|orig|before|condition", re.I)
OUTPUT_HINTS = re.compile(r"output|edit|result|after|target|gpt", re.I)


def pick_config(repo: str, wanted: str | None):
    from datasets import get_dataset_config_names
    try:
        configs = get_dataset_config_names(repo)
    except Exception:  # noqa: BLE001
        return None
    if not configs:
        return None
    if wanted:
        for c in configs:
            if wanted.lower() in c.lower():
                return c
    for c in configs:
        if OUTPUT_HINTS.search(c) or "image_to_image" in c.lower() or "editing" in c.lower():
            return c
    return configs[0]


def detect_pair_columns(example: dict) -> tuple[str, str]:
    img_cols = [k for k, v in example.items() if isinstance(v, Image.Image)]
    if len(img_cols) < 2:
        raise SystemExit(
            "ERROR: could not find (input image, output image) columns - this looks like "
            "the text-to-image subset. Re-run with --config pointing at the editing subset. "
            f"Image columns found: {img_cols}. Keys: {list(example)}"
        )
    src = next((c for c in img_cols if SOURCE_HINTS.search(c)), None)
    out = next((c for c in img_cols if OUTPUT_HINTS.search(c) and c != src), None)
    if src is None or out is None:
        src, out = img_cols[0], img_cols[1]
        print(f"WARNING: assigning by position: source={src} output={out} - verify with audit!")
    else:
        print(f"source column: {src}   output column: {out}")
    return src, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="FreedomIntelligence/ShareGPT-4o-Image")
    ap.add_argument("--config", default=None, help="dataset config name (auto-detected otherwise)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=16000, help="max pairs")
    ap.add_argument("--val-every", type=int, default=20, help="every k-th pair goes to val (5%%)")
    ap.add_argument("--no-streaming", action="store_true",
                    help="download the full dataset instead of streaming (needs big disk)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    cfg = pick_config(args.repo_id, args.config)
    streaming = not args.no_streaming
    print(f"loading {args.repo_id} config={cfg} split={args.split} streaming={streaming} ...")
    ds = (load_dataset(args.repo_id, cfg, split=args.split, streaming=streaming)
          if cfg else load_dataset(args.repo_id, split=args.split, streaming=streaming))
    # seeded shuffle works for both modes (buffer-based when streaming)
    ds = ds.shuffle(seed=args.seed, buffer_size=1000) if streaming else ds.shuffle(seed=args.seed)

    it = iter(ds)
    first = next(it)
    src_col, out_col = detect_pair_columns(first)

    roots = {
        False: processed_dir("pairs", "gpt4o"),
        True: processed_dir("pairs", "gpt4o_val"),
    }
    for r in roots.values():
        ensure(r / "real")
        ensure(r / "fake")

    written = 0
    bar = tqdm(total=args.n, desc="gpt4o pairs")
    for row in chain([first], it):
        if written >= args.n:
            break
        is_val = (written % args.val_every) == 0
        root = roots[is_val]
        real_dst = root / "real" / f"{written:07d}.jpg"
        fake_dst = root / "fake" / f"{written:07d}.jpg"
        if real_dst.exists() and fake_dst.exists():
            written += 1
            bar.update(1)
            continue
        try:
            real, fake = row[src_col], row[out_col]
            if real is None or fake is None:
                continue
            real, fake = real.convert("RGB"), fake.convert("RGB")
            if real.size != fake.size:
                real = real.resize(fake.size, Image.LANCZOS)
            normalize_save(fake, fake_dst)
            normalize_save(real, real_dst)
            written += 1
            bar.update(1)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {e}")
    bar.close()
    print(f"done: {written} pairs (~1/{args.val_every} in val)")
    print("CAVEAT: audit whether the 'real' inputs are genuine photos; if the audit or "
          "eval looks off, fall back to COCO reals for this expert (see README).")


if __name__ == "__main__":
    main()
