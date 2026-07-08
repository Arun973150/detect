"""Re-download clean Pico-Banana-400K pairs for the Nano Banana expert (A3).

Pipeline:
  1. Load Apple's official sft.jsonl metadata (from the user's Drive copy or
     the Apple CDN). Each record i (1-based, matching the CDN filename
     `images/positive-edit/{i}.png`) holds:
       open_image_input_url : Flickr URL of the real OpenImages source
       output_image         : relative CDN path of the Nano Banana edit
       text / edit_type     : edit instruction metadata
  2. Choose pair ids. By default, reuse the piksign split
     (manifests/manifest_pico_train.csv + _val.csv, ids parsed from filenames)
     so the existing train/val partition is preserved; otherwise sample --n.
  3. Download edited images from Apple's CDN (per-image PNG URLs).
  4. Resolve each Flickr source URL to an OpenImages ImageID by streaming the
     official OpenImages metadata CSV once (matching on the Flickr basename,
     which is stable across host variants), then fetch the original from the
     public S3 mirror - Flickr originals themselves are often dead links.
  5. Build normalized training pairs: the real source is Lanczos-resized to
     its paired fake's exact WxH (Nano Banana re-renders at its own working
     resolution, so without this the pair leaks a resolution shortcut), then
     BOTH sides go through the same JPEG q95 funnel.

Outputs:
  data/raw/pico_banana/{edited,source}/     raw downloads
  data/processed/pairs/nanobanana/{real,fake}/<id>.jpg
  data/processed/pairs/nanobanana_val/{real,fake}/<id>.jpg
  data/processed/pairs/nanobanana/pairs.csv

License note: Pico-Banana-400K is CC BY-NC-ND 4.0 (research / non-commercial).

    python -m piksign.download.pico_banana --workers 16
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, processed_dir, raw_dir

CDN_BASE = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
SFT_JSONL_URL = CDN_BASE + "jsonl/sft.jsonl"
OPENIMAGES_META_URL = (
    "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
)
S3_IMG = "https://open-images-dataset.s3.amazonaws.com/train/{img_id}.jpg"

UA = {"User-Agent": "piksign-research/0.1"}


# ------------------------------------------------------------------ metadata

def sft_jsonl_path() -> Path:
    drive_copy = raw_dir("piksign_drive", "sft.jsonl")
    if drive_copy.exists():
        return drive_copy
    dest = ensure(raw_dir("pico_banana")) / "sft.jsonl"
    if not dest.exists():
        print(f"downloading official metadata {SFT_JSONL_URL} ...")
        with requests.get(SFT_JSONL_URL, stream=True, timeout=120, headers=UA) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return dest


def load_records(path: Path) -> dict[int, dict]:
    """Map pair id (int in the CDN filename) -> metadata record."""
    records: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m = re.search(r"/(\d+)\.\w+$", rec.get("output_image", ""))
            if m:
                records[int(m.group(1))] = rec
    print(f"loaded {len(records)} sft records from {path}")
    return records


def ids_from_piksign_manifest(manifest: Path) -> list[int]:
    ids = set()
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"(\d+)\.\w+$", row["path"].replace("\\", "/"))
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def choose_ids(records: dict[int, dict], args: argparse.Namespace) -> tuple[list[int], list[int]]:
    """Returns (train_ids, val_ids)."""
    tr_manifest = args.manifest_dir / "manifest_pico_train.csv"
    va_manifest = args.manifest_dir / "manifest_pico_val.csv"
    if not args.random_split and tr_manifest.exists() and va_manifest.exists():
        tr_all = ids_from_piksign_manifest(tr_manifest)
        va_all = ids_from_piksign_manifest(va_manifest)
        train_ids = [i for i in tr_all if i in records]
        val_ids = [i for i in va_all if i in records]
        rate = (len(train_ids) + len(val_ids)) / max(1, len(tr_all) + len(va_all))
        # The piksign zips appear to be numbered by line order, not CDN id -
        # verified empirically: ~34% of manifest ids (including 0) have no CDN
        # record. Only trust the split when it actually maps.
        if rate >= 0.98:
            print(f"using piksign split: {len(train_ids)} train / {len(val_ids)} val pairs")
            if args.n and args.n < len(train_ids):
                train_ids = sorted(random.Random(args.seed).sample(train_ids, args.n))
                print(f"subsampled train to {len(train_ids)}")
            return train_ids, val_ids
        print(f"WARNING: piksign manifest ids match only {rate:.0%} of CDN records - "
              "they are not CDN ids. Falling back to a clean random split.")
    rng = random.Random(args.seed)
    all_ids = sorted(records)
    n = min(args.n or 8000, len(all_ids))
    picked = sorted(rng.sample(all_ids, n))
    n_val = max(1, int(0.05 * n))
    val_ids = sorted(rng.sample(picked, n_val))
    train_ids = [i for i in picked if i not in set(val_ids)]
    print(f"random split: {len(train_ids)} train / {len(val_ids)} val pairs")
    return train_ids, val_ids


# ------------------------------------------------------------------ download
#
# The CDN tolerates modest parallel bursts (verified empirically), but at
# multi-thousand-file scale it may start throttling (429/403/503) or dropping
# connections. This downloader therefore:
#   - reuses one HTTP session per thread (connection pooling),
#   - honors Retry-After and applies a GLOBAL backoff shared by all threads,
#   - retries with exponential backoff + jitter,
#   - is fully resumable (existing files are skipped) - if a run dies,
#     just re-run the same command and it continues where it stopped.

_tls = threading.local()
_throttle_lock = threading.Lock()
_throttle_until = 0.0


def _session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(UA)
        _tls.session = s
    return s


def _global_pause(seconds: float) -> None:
    global _throttle_until
    with _throttle_lock:
        _throttle_until = max(_throttle_until, time.time() + seconds)


def _wait_for_throttle() -> None:
    while True:
        wait = _throttle_until - time.time()
        if wait <= 0:
            return
        time.sleep(min(wait, 5.0))


def _fetch(url: str, dest: Path, retries: int = 5, sleep: float = 0.0) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for attempt in range(retries):
        _wait_for_throttle()
        if sleep:
            time.sleep(sleep * (0.5 + random.random()))
        try:
            r = _session().get(url, timeout=90)
            if r.status_code == 404:
                return False
            if r.status_code in (403, 408, 429, 500, 502, 503):
                retry_after = float(r.headers.get("Retry-After") or 0)
                pause = max(retry_after, min(120.0, 5.0 * 2**attempt))
                print(f"[{r.status_code} on {dest.name}] backing off {pause:.0f}s (all threads)")
                _global_pause(pause)
                continue
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.rename(dest)
            return True
        except requests.RequestException:
            _global_pause(min(60.0, 2.0 * 2**attempt))
    return False


def download_many(jobs: list[tuple[str, Path]], workers: int, desc: str,
                  sleep: float = 0.0) -> int:
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, url, dest, 5, sleep): url for url, dest in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
            ok += bool(fut.result())
    missed = len(jobs) - ok
    print(f"{desc}: {ok}/{len(jobs)} fetched"
          + (f" - {missed} missing; RE-RUN this command to retry them" if missed else ""))
    return ok


def flickr_basename(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1].lower()


def resolve_openimages_ids(needed_urls: set[str], cache: Path) -> dict[str, str]:
    """Map Flickr basename -> OpenImages ImageID, streaming the 9M-row official
    metadata CSV once and caching the (tiny) result."""
    mapping: dict[str, str] = {}
    if cache.exists():
        with open(cache, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                mapping[row[0]] = row[1]
        missing = {flickr_basename(u) for u in needed_urls} - set(mapping)
        if not missing:
            print(f"url map cache hit: {len(mapping)} entries")
            return mapping

    needed = {flickr_basename(u) for u in needed_urls}
    print(f"resolving {len(needed)} source urls against OpenImages metadata (one-time, ~9M rows)...")
    with requests.get(OPENIMAGES_META_URL, stream=True, timeout=300, headers=UA) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        lines = io.TextIOWrapper(r.raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(lines)
        header = next(reader)
        i_id = header.index("ImageID")
        i_url = header.index("OriginalURL")
        remaining = set(needed) - set(mapping)
        for row in tqdm(reader, total=9_178_275, desc="scan openimages csv"):
            try:
                base = flickr_basename(row[i_url])
            except IndexError:
                continue
            if base in remaining:
                mapping[base] = row[i_id]
                remaining.discard(base)
                if not remaining:
                    break
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in sorted(mapping.items()):
            w.writerow([k, v])
    print(f"resolved {len(mapping)}/{len(needed)} source urls")
    return mapping


# ---------------------------------------------------------------- pair build

def build_pairs(
    ids: list[int], records: dict[int, dict],
    edited_dir: Path, source_dir: Path, out_root: Path,
) -> list[dict]:
    real_out = ensure(out_root / "real")
    fake_out = ensure(out_root / "fake")
    rows: list[dict] = []
    for i in tqdm(ids, desc=f"pairs -> {out_root.name}"):
        fake_src = edited_dir / f"{i}.png"
        real_src = source_dir / f"{i}.jpg"
        fake_dst = fake_out / f"{i:06d}.jpg"
        real_dst = real_out / f"{i:06d}.jpg"
        if not (fake_src.exists() and real_src.exists()):
            continue
        if not (fake_dst.exists() and real_dst.exists()):
            try:
                with Image.open(fake_src) as fk, Image.open(real_src) as rl:
                    fk = fk.convert("RGB")
                    rl = rl.convert("RGB")
                    if rl.size != fk.size:
                        rl = rl.resize(fk.size, Image.LANCZOS)
                    normalize_save(fk, fake_dst)
                    normalize_save(rl, real_dst)
            except Exception as e:  # noqa: BLE001
                print(f"[skip pair {i}] {e}")
                continue
        rec = records[i]
        rows.append({
            "id": i,
            "real": str(real_dst),
            "fake": str(fake_dst),
            "edit_type": rec.get("edit_type", ""),
            "instruction": rec.get("summarized_text", ""),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="cap train pairs (0 = all in split / 8000 random)")
    ap.add_argument("--random-split", action="store_true", help="ignore piksign manifests, sample randomly")
    ap.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    ap.add_argument("--workers", type=int, default=16, help="parallel downloads (OpenImages S3)")
    ap.add_argument("--cdn-workers", type=int, default=8,
                    help="parallel downloads against the Apple CDN (be polite)")
    ap.add_argument("--cdn-sleep", type=float, default=0.05,
                    help="mean per-request sleep vs the Apple CDN, seconds")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_records(sft_jsonl_path())
    train_ids, val_ids = choose_ids(records, args)
    all_ids = sorted(set(train_ids) | set(val_ids))

    edited_dir = ensure(raw_dir("pico_banana", "edited"))
    source_dir = ensure(raw_dir("pico_banana", "source"))

    # 1) Nano Banana edited images from Apple CDN (throttle-aware, resumable)
    jobs = [(CDN_BASE + records[i]["output_image"], edited_dir / f"{i}.png") for i in all_ids]
    download_many(jobs, args.cdn_workers, "edited (Apple CDN)", sleep=args.cdn_sleep)

    # 2) real sources: Flickr URL -> OpenImages ImageID -> S3 mirror
    needed_urls = {records[i]["open_image_input_url"] for i in all_ids}
    mapping = resolve_openimages_ids(needed_urls, raw_dir("pico_banana", "url_map.csv"))
    src_jobs, fallback = [], []
    for i in all_ids:
        url = records[i]["open_image_input_url"]
        dest = source_dir / f"{i}.jpg"
        img_id = mapping.get(flickr_basename(url))
        if img_id:
            src_jobs.append((S3_IMG.format(img_id=img_id), dest))
        else:
            fallback.append((url, dest))
    download_many(src_jobs, args.workers, "sources (OpenImages S3)")
    if fallback:
        download_many(fallback, args.workers, "sources (Flickr fallback)")

    # 3) normalized aligned pairs
    train_rows = build_pairs(train_ids, records, edited_dir, source_dir,
                             processed_dir("pairs", "nanobanana"))
    val_rows = build_pairs(val_ids, records, edited_dir, source_dir,
                           processed_dir("pairs", "nanobanana_val"))

    pairs_csv = processed_dir("pairs", "nanobanana") / "pairs.csv"
    with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "real", "fake", "edit_type", "instruction"])
        w.writeheader()
        w.writerows(train_rows + val_rows)
    print(f"done: {len(train_rows)} train pairs, {len(val_rows)} val pairs -> {pairs_csv}")
    print("next: python -m piksign.audit --real data/processed/pairs/nanobanana/real "
          "--fake data/processed/pairs/nanobanana/fake")


if __name__ == "__main__":
    main()
