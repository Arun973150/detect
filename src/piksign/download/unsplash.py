"""Unsplash reals for the semantic branch (paper-faithful real class).

Why: the semantic corpus pairs polished Echo-4o fakes against real photos.
With mundane COCO snapshots as reals, "polished = fake" becomes a learnable
shortcut; Unsplash photos are polished professional photography, forcing the
VLM to judge content plausibility instead. This matches AlignGemini exactly.

Source: the official Unsplash Lite research dataset (25k photos; research
use) - a zip of TSVs including per-photo CDN URLs. We sample --n, fetch
server-resized JPEGs from the CDN, and push them through the standard
normalization funnel.

    python -m piksign.download.unsplash --n 5000 --replace
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, processed_dir, raw_dir

Image.MAX_IMAGE_PIXELS = None

DATASET_URLS = [
    "https://unsplash-datasets.s3.amazonaws.com/lite/latest/unsplash-research-dataset-lite-latest.zip",
    "https://unsplash.com/data/lite/latest",
]
UA = {"User-Agent": "piksign-research/0.1"}


def download_dataset_zip(dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 100 << 20:
        print(f"[skip] {dest.name} exists")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in DATASET_URLS:
        try:
            print(f"downloading Unsplash Lite dataset from {url} ...")
            with requests.get(url, stream=True, timeout=300, headers=UA, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            return dest
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  failed: {e}")
    raise SystemExit(f"could not download the dataset zip ({last_err}); "
                     "grab it manually from https://unsplash.com/data and place it at " + str(dest))


def read_photo_urls(zip_path: Path) -> list[str]:
    urls: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if Path(m).name.startswith("photos.tsv")]
        if not members:
            raise SystemExit(f"no photos.tsv* inside {zip_path}; members: {zf.namelist()[:10]}")
        for m in members:
            with zf.open(m) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
                if reader.fieldnames and "photo_image_url" not in reader.fieldnames:
                    raise SystemExit(f"unexpected TSV schema: {reader.fieldnames[:8]}")
                for row in reader:
                    u = (row.get("photo_image_url") or "").strip()
                    if u.startswith("http"):
                        urls.append(u)
    print(f"{len(urls)} photo urls in dataset")
    return urls


def fetch_one(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        # imgix params: server-side resized JPEG, plenty for a semantic corpus
        r = requests.get(url + "?w=1600&fit=max&fm=jpg&q=85", timeout=60, headers=UA)
        r.raise_for_status()
        with Image.open(io.BytesIO(r.content)) as im:
            normalize_save(im.convert("RGB"), dest)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: data/processed/semantic/real")
    ap.add_argument("--replace", action="store_true",
                    help="archive an existing non-empty out dir to <out>_coco_backup first")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.out or processed_dir("semantic", "real")
    if out.exists() and any(out.iterdir()):
        if not args.replace:
            raise SystemExit(f"{out} is not empty; re-run with --replace to swap it out")
        backup = out.parent / (out.name + "_coco_backup")
        if not backup.exists():
            out.rename(backup)
            print(f"archived old reals -> {backup}")
        else:
            shutil.rmtree(out)
    ensure(out)

    zip_path = raw_dir("unsplash") / "unsplash-lite-latest.zip"
    download_dataset_zip(zip_path)
    urls = read_photo_urls(zip_path)

    rng = random.Random(args.seed)
    rng.shuffle(urls)

    ok = 0
    idx = 0
    bar = tqdm(total=args.n, desc="unsplash reals")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pending = {}
        url_iter = iter(urls)
        while ok < args.n:
            while len(pending) < args.workers * 2:
                try:
                    u = next(url_iter)
                except StopIteration:
                    break
                pending[ex.submit(fetch_one, u, out / f"{idx:07d}.jpg")] = u
                idx += 1
            if not pending:
                break
            for fut in as_completed(list(pending)):
                pending.pop(fut, None)
                if fut.result():
                    ok += 1
                    bar.update(1)
                break
    bar.close()
    print(f"done: {ok} reals -> {out}")
    if ok < args.n:
        print(f"WARNING: only {ok}/{args.n} fetched; re-run to top up")
    print("license note: Unsplash Lite dataset is for research use.")


if __name__ == "__main__":
    main()
