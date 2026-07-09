"""COCO reals: sources for VAE reconstruction, semantic-branch reals, eval reals.

Downloads the official zips (resumable), extracts a seeded subset, optionally
deletes the zip afterwards.

    python -m piksign.download.coco --split train2017 --n 12000
    python -m piksign.download.coco --split val2017  --n 5000
"""
from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from ..paths import ensure, raw_dir

URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "unlabeled2017": "http://images.cocodataset.org/zips/unlabeled2017.zip",
}
# official image counts, used to resume-skip when the zip was already deleted
EXPECTED = {"train2017": 118287, "val2017": 5000, "unlabeled2017": 123403}


def download_resume(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    pos = dest.stat().st_size if dest.exists() else 0
    total = int(requests.head(url, timeout=60, allow_redirects=True).headers.get("content-length", 0))
    if pos and pos == total:
        print(f"[skip] {dest.name} already complete")
        return
    if pos:
        headers["Range"] = f"bytes={pos}-"
        mode = "ab"
        print(f"resuming {dest.name} at {pos / 1e9:.2f}GB")
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        with open(dest, mode) as f, tqdm(
            total=total, initial=pos, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
                bar.update(len(chunk))


def extract_subset(zip_path: Path, out_dir: Path, n: int, seed: int) -> int:
    ensure(out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        names = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
        if n and n < len(names):
            names = sorted(random.Random(seed).sample(names, n))
        count = 0
        for m in tqdm(names, desc=f"extract -> {out_dir.name}"):
            dest = out_dir / Path(m).name
            if dest.exists():
                count += 1
                continue
            with zf.open(m) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=sorted(URLS), default="train2017")
    ap.add_argument("--n", type=int, default=12000, help="0 = extract all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: data/raw/coco/<split>_<n|all>")
    ap.add_argument("--delete-zip", action="store_true")
    args = ap.parse_args()

    out = args.out or raw_dir("coco", f"{args.split}_{args.n or 'all'}")
    target = args.n or EXPECTED.get(args.split, 0)
    existing = len(list(out.glob("*.jpg"))) if out.exists() else 0
    if target and existing >= target:
        print(f"[skip] {out} already has {existing} images (target {target})")
        return
    if existing:
        print(f"resuming: {existing}/{target} images present, re-fetching zip to complete")

    zip_path = raw_dir("coco") / f"{args.split}.zip"
    download_resume(URLS[args.split], zip_path)
    n = extract_subset(zip_path, out, args.n, args.seed)
    print(f"extracted {n} images -> {out}")
    if args.delete_zip:
        zip_path.unlink()
        print(f"deleted {zip_path}")


if __name__ == "__main__":
    main()
