"""GPT-4o aligned edit pairs from ShareGPT-4o-Image (HuggingFace) for expert A4.

The dataset (FreedomIntelligence/ShareGPT-4o-Image) contains text-to-image and
text-and-image-to-image samples produced by GPT-4o. We want the EDITING subset:
(input image, GPT-4o output image) gives aligned real/fake pairs, the same
structure as Pico-Banana.

Column names are discovered defensively (the hub schema is not guaranteed):
we look for two image-typed columns and assign source/output by name.

Outputs:
  data/processed/pairs/gpt4o/{real,fake}/<idx>.jpg          (train)
  data/processed/pairs/gpt4o_val/{real,fake}/<idx>.jpg      (5% val)

    python -m piksign.download.sharegpt4o --n 16000
"""
from __future__ import annotations

import argparse
import random
import re

from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, processed_dir

Image.MAX_IMAGE_PIXELS = None

SOURCE_HINTS = re.compile(r"input|source|orig|before|condition", re.I)
OUTPUT_HINTS = re.compile(r"output|edit|result|after|target|gpt", re.I)


def find_image_columns(ds) -> list[str]:
    from datasets import Image as HFImage
    cols = []
    for name, feat in ds.features.items():
        if isinstance(feat, HFImage):
            cols.append(name)
    return cols


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="FreedomIntelligence/ShareGPT-4o-Image")
    ap.add_argument("--config", default=None, help="dataset config name (auto-detected otherwise)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=16000, help="max pairs")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    cfg = pick_config(args.repo_id, args.config)
    print(f"loading {args.repo_id} config={cfg} split={args.split} ...")
    ds = load_dataset(args.repo_id, cfg, split=args.split) if cfg else load_dataset(args.repo_id, split=args.split)
    img_cols = find_image_columns(ds)
    print(f"columns: {list(ds.features)}  image columns: {img_cols}")

    if len(img_cols) < 2:
        raise SystemExit(
            "ERROR: could not find (input image, output image) columns - this looks like "
            "the text-to-image subset. Re-run with --config pointing at the editing subset. "
            f"Available image columns: {img_cols}. Full features: {ds.features}"
        )

    src_col = next((c for c in img_cols if SOURCE_HINTS.search(c)), None)
    out_col = next((c for c in img_cols if OUTPUT_HINTS.search(c) and c != src_col), None)
    if src_col is None or out_col is None:
        src_col, out_col = img_cols[0], img_cols[1]
        print(f"WARNING: assigning by position: source={src_col} output={out_col} - verify with audit!")
    else:
        print(f"source column: {src_col}   output column: {out_col}")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: args.n]
    n_val = max(1, int(len(indices) * args.val_frac))
    val_set = set(indices[:n_val])

    roots = {
        False: processed_dir("pairs", "gpt4o"),
        True: processed_dir("pairs", "gpt4o_val"),
    }
    for r in roots.values():
        ensure(r / "real")
        ensure(r / "fake")

    written = 0
    for idx in tqdm(indices, desc="gpt4o pairs"):
        root = roots[idx in val_set]
        real_dst = root / "real" / f"{idx:07d}.jpg"
        fake_dst = root / "fake" / f"{idx:07d}.jpg"
        if real_dst.exists() and fake_dst.exists():
            written += 1
            continue
        try:
            row = ds[idx]
            real, fake = row[src_col], row[out_col]
            if real is None or fake is None:
                continue
            real, fake = real.convert("RGB"), fake.convert("RGB")
            if real.size != fake.size:
                real = real.resize(fake.size, Image.LANCZOS)
            normalize_save(fake, fake_dst)
            normalize_save(real, real_dst)
            written += 1
        except Exception as e:  # noqa: BLE001
            print(f"[skip {idx}] {e}")
    print(f"done: {written} pairs ({n_val} in val)")
    print("CAVEAT: audit whether the 'real' inputs are genuine photos; if the audit or "
          "eval looks off, fall back to COCO reals for this expert (see README).")


if __name__ == "__main__":
    main()
