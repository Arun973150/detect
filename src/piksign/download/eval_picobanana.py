"""Held-out Nano Banana eval fakes from Pico-Banana-400K (no training overlap).

Free alternative to fresh Gemini generation, which is blocked on the API free
tier (image gen has 0 quota without billing). These are real Nano Banana edits
the detector never saw: we re-derive the exact IDs A3 trained on (seed 42,
8000-sample) and EXCLUDE them, so the eval set is genuinely held out.

    python -m piksign.download.eval_picobanana --n 500

Output: data/eval/clean/nanobanana_heldout/fake/<id>.jpg
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from . import normalize_save
from .pico_banana import CDN_BASE, download_many, load_records, sft_jsonl_path
from ..paths import ensure, eval_dir

Image.MAX_IMAGE_PIXELS = None


def training_ids(records: dict, seed: int, n: int) -> set[int]:
    """Reproduce the training-set id sample so we can exclude it."""
    rng = random.Random(seed)
    all_ids = sorted(records)
    n = min(n, len(all_ids))
    return set(rng.sample(all_ids, n))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--exclude-seed", type=int, default=42, help="seed used at training time")
    ap.add_argument("--exclude-n", type=int, default=8000, help="training sample size to exclude")
    ap.add_argument("--seed", type=int, default=2025, help="eval sampling seed")
    ap.add_argument("--cdn-workers", type=int, default=8)
    ap.add_argument("--source-name", default="nanobanana_heldout")
    args = ap.parse_args()

    records = load_records(sft_jsonl_path())
    exclude = training_ids(records, args.exclude_seed, args.exclude_n)
    pool = [i for i in sorted(records) if i not in exclude]
    rng = random.Random(args.seed)
    eval_ids = sorted(rng.sample(pool, min(args.n, len(pool))))
    print(f"{len(eval_ids)} held-out eval ids (excluded {len(exclude)} training ids "
          f"from a pool of {len(records)})")

    raw = ensure(eval_dir("_raw", "nanobanana"))
    jobs = [(CDN_BASE + records[i]["output_image"], raw / f"{i}.png") for i in eval_ids]
    download_many(jobs, args.cdn_workers, "nanobanana eval (Apple CDN)", sleep=0.05)

    out = ensure(eval_dir("clean", args.source_name, "fake"))
    n_ok = 0
    for i in tqdm(eval_ids, desc="normalize"):
        src = raw / f"{i}.png"
        if not src.exists():
            continue
        dst = out / f"{i:06d}.jpg"
        if dst.exists():
            n_ok += 1
            continue
        try:
            with Image.open(src) as im:
                normalize_save(im.convert("RGB"), dst)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[skip {i}] {e}")
    print(f"done: {n_ok} held-out Nano Banana eval fakes -> {out}")


if __name__ == "__main__":
    main()
