"""GPT-4o aligned edit pairs v2, from GPT-Image-Edit-1.5M (UCSC-VLAA).

Why v2: ShareGPT-4o-Image's "real" inputs turned out ~half synthetic
(measured by A1 at mean score 0.499, visually confirmed), capping A4 at ~83%.
GPT-Image-Edit-1.5M's ultraedit subset has real-photo inputs with
gpt-image-1 outputs, license CC-BY-4.0.

Repo layout: gpt-edit/ultraedit.tar.gz.part001..004 (one gzip stream split
into parts; ~78GB total). Inside: <subset>/<task>/<role>/<id>.png,
paired by identical <id>. We chain the
part URLs into one HTTP stream, stage inputs (bounded per task), finalize a
pair when its output arrives, and stop as soon as --n pairs exist - archives
never touch the disk.

Outputs:
  data/processed/pairs/gpt4o_v2/{real,fake}/<task>_<id>.jpg
  data/processed/pairs/gpt4o_v2_val/{real,fake}/...   (stable hash split)

    python -m piksign.download.gptimageedit --n 16000
    python -m piksign.download.gptimageedit \
        --part-prefix gpt-edit/hqedit.tar.gz.part \
        --include-task edit --out-name gpt4o_hqedit_edit --n 16000
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import zlib
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from . import normalize_save
from ..paths import ensure, processed_dir, raw_dir

Image.MAX_IMAGE_PIXELS = None

REPO = "UCSC-VLAA/GPT-Image-Edit-1.5M"
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


def list_parts(repo: str, prefix: str) -> list[str]:
    r = requests.get(f"https://huggingface.co/api/datasets/{repo}",
                     headers=_headers(), timeout=60)
    r.raise_for_status()
    files = [s["rfilename"] for s in r.json().get("siblings", [])]
    parts = sorted(f for f in files if f.startswith(prefix))
    if not parts:
        raise SystemExit(f"no archive parts matching '{prefix}'")
    return parts


class ChainedHTTPStream:
    """Read a sequence of HTTP URLs as one continuous byte stream."""

    def __init__(self, urls: list[str], headers: dict) -> None:
        self.urls = urls
        self.headers = headers
        self.idx = 0
        self.raw = None
        self.bar = tqdm(total=len(urls), desc="archive parts", unit=" part")

    def _advance(self) -> bool:
        if self.raw is not None:
            self.bar.update(1)
            self.raw = None
        if self.idx >= len(self.urls):
            return False
        r = requests.get(self.urls[self.idx], headers=self.headers,
                         stream=True, timeout=900)
        r.raise_for_status()
        r.raw.decode_content = False  # gzip belongs to the tar stream, not HTTP
        self.raw = r.raw
        self.idx += 1
        return True

    def read(self, n: int = -1) -> bytes:
        chunks: list[bytes] = []
        remaining = n
        while remaining != 0:
            if self.raw is None and not self._advance():
                break
            data = self.raw.read(remaining if remaining > 0 else 1 << 20)
            if not data:
                self.raw = None
                continue
            chunks.append(data)
            if remaining > 0:
                remaining -= len(data)
        return b"".join(chunks)

    def close(self) -> None:
        self.bar.close()


def parse_member(name: str) -> tuple[str, str, str] | None:
    """<subset>/<task>/<role>/<id>.png -> (task, role, id) or None."""
    parts = name.strip("/").split("/")
    if len(parts) != 4:
        return None
    _, task, role, fname = parts
    p = Path(fname)
    if role not in ("input", "output") or p.suffix.lower() not in IMG_EXTS:
        return None
    return task, role, p.stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=REPO)
    ap.add_argument("--part-prefix", default="gpt-edit/ultraedit.tar.gz.part",
                    help="ultraedit only by default; omniedit is 3.7TB - do not")
    ap.add_argument("--n", type=int, default=16000, help="pairs to collect")
    ap.add_argument("--per-task-stage", type=int, default=4000,
                    help="max inputs staged per task while waiting for outputs")
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--out-name", default="gpt4o_v2")
    ap.add_argument("--include-task", action="append", default=[],
                    help="only keep this task; repeatable. Example: --include-task edit")
    ap.add_argument("--exclude-task", action="append", default=[],
                    help="drop this task; repeatable")
    args = ap.parse_args()
    include_tasks = set(args.include_task)
    exclude_tasks = set(args.exclude_task)

    train_root = processed_dir("pairs", args.out_name)
    val_root = processed_dir("pairs", args.out_name + "_val")
    for r in (train_root, val_root):
        ensure(r / "real")
        ensure(r / "fake")
    staging = ensure(raw_dir("gptimageedit", "staging"))

    def pair_paths(task: str, pid: str) -> tuple[Path, Path]:
        root = val_root if (zlib.crc32(f"{task}/{pid}".encode()) % args.val_every == 0) else train_root
        return root / "real" / f"{task}_{pid}.jpg", root / "fake" / f"{task}_{pid}.jpg"

    built = sum(1 for _ in (train_root / "fake").glob("*.jpg")) + \
        sum(1 for _ in (val_root / "fake").glob("*.jpg"))
    print(f"{built} pairs already present, target {args.n}")
    if built >= args.n:
        print("nothing to do")
        return

    parts = list_parts(args.repo_id, args.part_prefix)
    urls = [RESOLVE.format(repo=args.repo_id, fname=p) for p in parts]
    print(f"{len(parts)} parts to stream (early-stop at {args.n} pairs)")

    staged: dict[str, dict[str, Path]] = {}  # task -> id -> staged input path
    stream = ChainedHTTPStream(urls, _headers())
    members = tqdm(desc="members", unit=" files")
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tf:
            for member in tf:
                members.update(1)
                if built >= args.n:
                    break
                if not member.isfile():
                    continue
                parsed = parse_member(member.name)
                if parsed is None:
                    continue
                task, role, pid = parsed
                if include_tasks and task not in include_tasks:
                    continue
                if task in exclude_tasks:
                    continue
                real_dst, fake_dst = pair_paths(task, pid)
                if real_dst.exists() and fake_dst.exists():
                    continue
                if role == "input":
                    bucket = staged.setdefault(task, {})
                    if len(bucket) >= args.per_task_stage:
                        continue
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    sp = staging / f"{task}_{pid}.png"
                    sp.write_bytes(fobj.read())
                    bucket[pid] = sp
                else:  # output
                    sp = staged.get(task, {}).pop(pid, None)
                    if sp is None:
                        continue
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    try:
                        from io import BytesIO
                        with Image.open(BytesIO(fobj.read())) as fk, Image.open(sp) as rl:
                            fk = fk.convert("RGB")
                            rl = rl.convert("RGB")
                            if rl.size != fk.size:
                                rl = rl.resize(fk.size, Image.LANCZOS)
                            normalize_save(fk, fake_dst)
                            normalize_save(rl, real_dst)
                        built += 1
                        if built % 500 == 0:
                            print(f"  {built}/{args.n} pairs")
                    except Exception as e:  # noqa: BLE001
                        print(f"[skip {task}/{pid}] {e}")
                    finally:
                        sp.unlink(missing_ok=True)
    finally:
        members.close()
        stream.close()
        shutil.rmtree(staging, ignore_errors=True)

    print(f"done: {built} pairs -> {train_root} (+ _val)")
    print("preflight audit before training (should print a LOW mean, ~0.1):")
    print("  python -m piksign.train.filter_pairs --expert $PIKSIGN_CKPT/a1_sd21 \\")
    print(f"      --pairs {train_root} --out {train_root}_audit --drop-frac 0.0 --crops 3")


if __name__ == "__main__":
    main()
