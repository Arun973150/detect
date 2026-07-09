"""GPT-4o aligned edit pairs from ShareGPT-4o-Image (HuggingFace) for expert A4.

Repo layout (not loadable via `datasets` - images live inside tar archives):
  text_and_image_to_image.json         metadata: input_prompt, input_image
                                       (LIST of paths), output_image (path)
  text_and_image_to_image_part_*.tar   image/v2v_*.png members

Strategy: parse the metadata, sample --n pairs, then STREAM each tar over
HTTP and extract only wanted members on the fly - the archives are never
stored on disk. Records with multiple input images (multi-reference edits)
are skipped since their alignment is ambiguous.

Outputs:
  data/processed/pairs/gpt4o/{real,fake}/<idx>.jpg          (train)
  data/processed/pairs/gpt4o_val/{real,fake}/<idx>.jpg      (val, every --val-every-th)

    python -m piksign.download.sharegpt4o --n 16000
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import tarfile
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, processed_dir, raw_dir

Image.MAX_IMAGE_PIXELS = None

RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{fname}"


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


def list_repo_files(repo: str) -> list[str]:
    r = requests.get(f"https://huggingface.co/api/datasets/{repo}",
                     headers=_headers(), timeout=60)
    r.raise_for_status()
    return [s["rfilename"] for s in r.json().get("siblings", [])]


def load_metadata(repo: str, meta_file: str) -> list[dict]:
    dest = ensure(raw_dir("sharegpt4o")) / meta_file
    if not dest.exists():
        print(f"downloading metadata {meta_file} ...")
        with requests.get(RESOLVE.format(repo=repo, fname=meta_file),
                          headers=_headers(), stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    with open(dest, encoding="utf-8") as f:
        return json.load(f)


def normalize_record(rec: dict) -> tuple[str, str] | None:
    """Returns (input_member, output_member) basenames, or None to skip."""
    inp = rec.get("input_image")
    out = rec.get("output_image")
    if isinstance(inp, list):
        if len(inp) != 1:
            return None  # multi-reference edit, alignment ambiguous
        inp = inp[0]
    if not isinstance(inp, str) or not isinstance(out, str):
        return None
    return Path(inp).name, Path(out).name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="FreedomIntelligence/ShareGPT-4o-Image")
    ap.add_argument("--meta-file", default="text_and_image_to_image.json")
    ap.add_argument("--tar-prefix", default="text_and_image_to_image_part_")
    ap.add_argument("--n", type=int, default=16000, help="max pairs")
    ap.add_argument("--val-every", type=int, default=20, help="every k-th pair goes to val (5%%)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_metadata(args.repo_id, args.meta_file)
    print(f"{len(records)} metadata records")

    usable: list[tuple[str, str]] = []
    for rec in records:
        pair = normalize_record(rec)
        if pair:
            usable.append(pair)
    print(f"{len(usable)} single-input edit pairs")

    rng = random.Random(args.seed)
    rng.shuffle(usable)
    sampled = usable[: args.n]

    roots = {
        False: processed_dir("pairs", "gpt4o"),
        True: processed_dir("pairs", "gpt4o_val"),
    }
    for r in roots.values():
        ensure(r / "real")
        ensure(r / "fake")
    staging = ensure(raw_dir("sharegpt4o", "staging"))

    # wanted member basename -> list of (pair_idx, role)
    wanted: dict[str, list[tuple[int, str]]] = {}
    remaining = 0
    for idx, (inp, out) in enumerate(sampled):
        root = roots[(idx % args.val_every) == 0]
        if (root / "real" / f"{idx:07d}.jpg").exists() and (root / "fake" / f"{idx:07d}.jpg").exists():
            continue
        wanted.setdefault(inp, []).append((idx, "real"))
        wanted.setdefault(out, []).append((idx, "fake"))
        remaining += 1
    print(f"{remaining} pairs to build ({len(sampled) - remaining} already done)")
    if not wanted:
        print("nothing to do")
        return

    def stage_path(idx: int, role: str) -> Path:
        return staging / f"{idx:07d}.{role}.png"

    def try_finalize(idx: int) -> bool:
        rp, fp = stage_path(idx, "real"), stage_path(idx, "fake")
        if not (rp.exists() and fp.exists()):
            return False
        root = roots[(idx % args.val_every) == 0]
        try:
            with Image.open(fp) as fk, Image.open(rp) as rl:
                fk = fk.convert("RGB")
                rl = rl.convert("RGB")
                if rl.size != fk.size:
                    rl = rl.resize(fk.size, Image.LANCZOS)
                normalize_save(fk, root / "fake" / f"{idx:07d}.jpg")
                normalize_save(rl, root / "real" / f"{idx:07d}.jpg")
        except Exception as e:  # noqa: BLE001
            print(f"[skip pair {idx}] {e}")
        rp.unlink(missing_ok=True)
        fp.unlink(missing_ok=True)
        return True

    tars = sorted(f for f in list_repo_files(args.repo_id)
                  if f.startswith(args.tar_prefix) and f.endswith(".tar"))
    print(f"streaming {len(tars)} tar archives (extracting only wanted members)...")
    built = 0
    for tname in tars:
        if not wanted:
            break
        url = RESOLVE.format(repo=args.repo_id, fname=tname)
        with requests.get(url, headers=_headers(), stream=True, timeout=600) as r:
            r.raise_for_status()
            r.raw.decode_content = True
            with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
                bar = tqdm(desc=tname, unit=" members")
                for member in tf:
                    bar.update(1)
                    if not member.isfile():
                        continue
                    base = Path(member.name).name
                    targets = wanted.pop(base, None)
                    if not targets:
                        continue
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    data = fobj.read()
                    for idx, role in targets:
                        stage_path(idx, role).write_bytes(data)
                        if try_finalize(idx):
                            built += 1
                bar.close()
        print(f"  after {tname}: {built} pairs built, {len(wanted)} members still wanted")

    # leftover stages (partner never found) - clean up
    leftovers = list(staging.glob("*.png"))
    if leftovers:
        print(f"cleaning {len(leftovers)} unmatched staged files")
        shutil.rmtree(staging, ignore_errors=True)

    print(f"done: {built} new pairs (target {remaining})")
    print("CAVEAT: many inputs are uniform 1024x1024 PNGs of unknown provenance; "
          "run the audit, and if a4 evals poorly on gpt4o_fresh retrain it with "
          "--real-dir pointed at COCO reals (see README).")


if __name__ == "__main__":
    main()
