"""Clean an aligned pair set by auditing its REAL class with a trusted expert.

Problem this solves: some datasets' "real" inputs are of mixed provenance
(e.g. ShareGPT-4o-Image edit inputs may themselves be AI-generated). Training
with polluted reals lowers the accuracy ceiling: the model is told "real"
about images carrying generator fingerprints.

Fix: score every real with an already-trained, independent expert (A1, the
VAE-recon generalist). Pairs whose real scores in the top --drop-frac are
dropped ENTIRELY (fake goes too, keeping the set aligned). The same score
cutoff is applied to the _val sibling directory if present.

    python -m piksign.train.filter_pairs \
        --expert checkpoints/a1_sd21 \
        --pairs data/processed/pairs/gpt4o \
        --out data/processed/pairs/gpt4o_clean \
        --drop-frac 0.25
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..models.expert import PixelExpert
from ..paths import ensure, list_images

Image.MAX_IMAGE_PIXELS = None


@torch.no_grad()
def score_dir(expert: PixelExpert, d: Path, n_crops: int, cache: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    if cache.exists():
        scores = json.loads(cache.read_text())
    todo = [p for p in list_images(d) if p.name not in scores]
    for p in tqdm(todo, desc=f"score {d.name}"):
        try:
            with Image.open(p) as im:
                scores[p.name] = expert.score_pil(im.convert("RGB"), n_crops=n_crops)
        except Exception as e:  # noqa: BLE001
            print(f"[skip {p.name}] {e}")
        if len(scores) % 500 == 0:
            cache.write_text(json.dumps(scores))
    cache.write_text(json.dumps(scores))
    return scores


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        import os
        os.link(src, dst)  # hardlink: instant, no extra space
    except OSError:
        shutil.copy2(src, dst)


def filter_split(root: Path, out_root: Path, scores: dict[str, float], cutoff: float) -> tuple[int, int]:
    kept = dropped = 0
    real_dir, fake_dir = root / "real", root / "fake"
    for rp in list_images(real_dir):
        fp = fake_dir / rp.name
        if not fp.exists():
            continue
        s = scores.get(rp.name)
        if s is None or s >= cutoff:
            dropped += 1
            continue
        link_or_copy(rp, out_root / "real" / rp.name)
        link_or_copy(fp, out_root / "fake" / rp.name)
        kept += 1
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert", type=Path, required=True, help="trusted expert checkpoint dir (e.g. a1_sd21)")
    ap.add_argument("--pairs", type=Path, required=True, help="pair set root containing real/ and fake/")
    ap.add_argument("--out", type=Path, required=True, help="output root for the cleaned pair set")
    ap.add_argument("--drop-frac", type=float, default=0.25, help="fraction of most-suspect reals to drop")
    ap.add_argument("--crops", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    expert = PixelExpert.load(args.expert, device=device)

    cache = args.pairs / "real_scores.json"
    scores = score_dir(expert, args.pairs / "real", args.crops, cache)
    vals = np.array(list(scores.values()))
    cutoff = float(np.quantile(vals, 1.0 - args.drop_frac))
    print(f"scored {len(vals)} reals: mean={vals.mean():.3f} median={np.median(vals):.3f}")
    print(f"cutoff at score {cutoff:.3f} (dropping top {args.drop_frac:.0%} most AI-looking reals)")

    kept, dropped = filter_split(args.pairs, ensure(args.out), scores, cutoff)
    print(f"train: kept {kept} pairs, dropped {dropped}")

    val_root = args.pairs.parent / (args.pairs.name + "_val")
    if val_root.is_dir():
        val_cache = val_root / "real_scores.json"
        val_scores = score_dir(expert, val_root / "real", args.crops, val_cache)
        out_val = ensure(Path(str(args.out) + "_val"))
        vk, vd = filter_split(val_root, out_val, val_scores, cutoff)
        print(f"val:   kept {vk} pairs, dropped {vd} (same cutoff)")

    print("next: retrain with the cleaned set, e.g.")
    print(f"  python -m piksign.train.train_expert --name a4_gpt4o_clean \\")
    print(f"      --real-dir {args.out}/real --fake-dir {args.out}/fake \\")
    print(f"      --val-real {args.out}_val/real --val-fake {args.out}_val/fake \\")
    print(f"      --lora-alpha 8.0 --launder-prob 0.5")


if __name__ == "__main__":
    main()
