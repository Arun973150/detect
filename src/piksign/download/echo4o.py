"""Semantic-branch corpus: Echo-4o surreal GPT-4o images (fake) + photo reals.

The semantic VLM must ONLY see semantic differences, so this corpus is:
  fake: semantically implausible GPT-4o generations (Echo-4o surreal subset)
  real: ordinary photographs (COCO subset by default)
Low-level cues are destroyed later by online pixel-scrambling during DPO
labeling/training, not here.

Repo layout of Yejy53/Echo-4o-Image (not loadable via `datasets`):
  Surrel-Fantasy-Image/images/*.tar.gz   [sic - "Surrel" is the repo's spelling]
We STREAM the tar.gz parts over HTTP in seeded-shuffled order, keeping every
--stride-th image until --n are saved; archives never touch the disk.

Outputs:
  data/processed/semantic/fake/<idx>.jpg
  data/processed/semantic/real/<idx>.jpg

    python -m piksign.download.echo4o --n 5000 --reals-dir data/raw/coco/train2017_12000
"""
from __future__ import annotations

import argparse
import random
import tarfile
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, list_images, processed_dir

Image.MAX_IMAGE_PIXELS = None

RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _headers() -> dict:
    h = {"User-Agent": "piksign-research/0.1"}
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok:
            h["Authorization"] = f"Bearer {tok}"
    except Exception:  # noqa: BLE001
        pass
    return h


def list_subset_tars(repo: str, prefix: str) -> list[str]:
    r = requests.get(f"https://huggingface.co/api/datasets/{repo}",
                     headers=_headers(), timeout=60)
    r.raise_for_status()
    files = [s["rfilename"] for s in r.json().get("siblings", [])]
    tars = [f for f in files if f.startswith(prefix) and (f.endswith(".tar.gz") or f.endswith(".tar"))]
    if not tars:
        raise SystemExit(f"no archives under '{prefix}' - repo files: {files[:20]} ...")
    return tars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="Yejy53/Echo-4o-Image")
    ap.add_argument("--subset-prefix", default="Surrel-Fantasy-Image/images/",
                    help="repo path prefix of the archive parts to stream")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=2, help="keep every k-th member (diversity vs bandwidth)")
    ap.add_argument("--reals-dir", type=Path, required=True,
                    help="directory of real photos to sample the real class from (e.g. COCO subset)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    fake_out = ensure(processed_dir("semantic", "fake"))
    have = len([p for p in fake_out.iterdir() if p.suffix == ".jpg"]) if fake_out.exists() else 0
    written = have
    print(f"{have} fakes already present, target {args.n}")

    if written < args.n:
        tars = list_subset_tars(args.repo_id, args.subset_prefix)
        rng.shuffle(tars)  # seeded order; typically only 2-3 parts get streamed
        print(f"{len(tars)} archive parts, streaming in seeded order...")
        for tname in tars:
            if written >= args.n:
                break
            url = RESOLVE.format(repo=args.repo_id, fname=tname)
            with requests.get(url, headers=_headers(), stream=True, timeout=600) as r:
                r.raise_for_status()
                with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
                    bar = tqdm(desc=Path(tname).name, unit=" members")
                    k = 0
                    for member in tf:
                        bar.update(1)
                        if written >= args.n:
                            break
                        if not member.isfile() or Path(member.name).suffix.lower() not in IMG_EXTS:
                            continue
                        k += 1
                        if k % args.stride:
                            continue
                        dst = fake_out / f"{written:07d}.jpg"
                        if dst.exists():
                            written += 1
                            continue
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            continue
                        try:
                            with Image.open(BytesIO(fobj.read())) as im:
                                normalize_save(im.convert("RGB"), dst)
                            written += 1
                        except Exception as e:  # noqa: BLE001
                            print(f"[skip {member.name}] {e}")
                    bar.close()
            print(f"  after {Path(tname).name}: {written}/{args.n} fakes")
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
