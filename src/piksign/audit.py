"""Dataset audit: compression / format / resolution parity between classes.

If real and fake classes differ systematically in format, JPEG quality, or
resolution, a detector will learn that difference instead of the generator
fingerprint and silently fail in the wild. Run this on every pair set BEFORE
training an expert on it.

CLI:
    python -m piksign.audit --real data/processed/pairs/nanobanana/real --fake data/processed/pairs/nanobanana/fake
    python -m piksign.audit --manifest manifests/manifest_pico_train.csv --root data
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

from PIL import Image
from tqdm import tqdm

from .paths import IMG_EXTS, list_images

# Standard IJG luminance quantization table, used to estimate JPEG quality.
STD_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]


def estimate_jpeg_quality(img: Image.Image) -> float | None:
    """Approximate libjpeg quality from the luminance quantization table."""
    q = getattr(img, "quantization", None)
    if not q:
        return None
    table = q.get(0) or next(iter(q.values()), None)
    if not table or len(table) != 64:
        return None
    ratios = [t / s for t, s in zip(table, STD_LUMA) if s > 0]
    scale = 100.0 * median(ratios)
    if scale <= 100.0:
        quality = (200.0 - scale) / 2.0
    else:
        quality = 5000.0 / scale
    return max(1.0, min(100.0, quality))


def scan_file(p: Path) -> dict | None:
    try:
        with Image.open(p) as img:
            w, h = img.size
            fmt = (img.format or "?").upper()
            q = estimate_jpeg_quality(img) if fmt == "JPEG" else None
    except Exception:  # noqa: BLE001
        return None
    return {
        "path": str(p),
        "format": fmt,
        "w": w,
        "h": h,
        "megapixels": w * h / 1e6,
        "kb": p.stat().st_size / 1024,
        "jpeg_q": q,
    }


def summarize(name: str, rows: list[dict]) -> dict:
    fmts: dict[str, int] = {}
    for r in rows:
        fmts[r["format"]] = fmts.get(r["format"], 0) + 1
    qs = [r["jpeg_q"] for r in rows if r["jpeg_q"] is not None]
    mps = [r["megapixels"] for r in rows]
    kbs = [r["kb"] for r in rows]
    s = {
        "name": name,
        "n": len(rows),
        "formats": fmts,
        "median_mp": median(mps) if mps else 0.0,
        "median_kb": median(kbs) if kbs else 0.0,
        "median_jpeg_q": median(qs) if qs else None,
        "kb_per_mp": (median(kbs) / max(median(mps), 1e-6)) if mps else 0.0,
    }
    print(
        f"[{name}] n={s['n']}  formats={s['formats']}  "
        f"median {s['median_mp']:.2f}MP {s['median_kb']:.0f}KB "
        f"({s['kb_per_mp']:.0f}KB/MP)  est. JPEG q={s['median_jpeg_q']}"
    )
    return s


def compare(real: dict, fake: dict) -> list[str]:
    warnings: list[str] = []

    def frac(s: dict, fmt: str) -> float:
        return s["formats"].get(fmt, 0) / max(s["n"], 1)

    for fmt in set(real["formats"]) | set(fake["formats"]):
        if abs(frac(real, fmt) - frac(fake, fmt)) > 0.10:
            warnings.append(
                f"FORMAT MISMATCH: '{fmt}' is {frac(real, fmt):.0%} of real vs "
                f"{frac(fake, fmt):.0%} of fake -> normalize both classes identically."
            )
    if real["median_jpeg_q"] and fake["median_jpeg_q"]:
        if abs(real["median_jpeg_q"] - fake["median_jpeg_q"]) > 5:
            warnings.append(
                f"JPEG QUALITY MISMATCH: real q~{real['median_jpeg_q']:.0f} vs "
                f"fake q~{fake['median_jpeg_q']:.0f} -> compression becomes the label."
            )
    ratio = real["median_mp"] / max(fake["median_mp"], 1e-6)
    if ratio > 1.5 or ratio < 1 / 1.5:
        warnings.append(
            f"RESOLUTION MISMATCH: real ~{real['median_mp']:.2f}MP vs fake "
            f"~{fake['median_mp']:.2f}MP -> resize reals to their paired fake size."
        )
    kb_ratio = real["kb_per_mp"] / max(fake["kb_per_mp"], 1e-6)
    if kb_ratio > 1.6 or kb_ratio < 1 / 1.6:
        warnings.append(
            f"BITRATE MISMATCH: real ~{real['kb_per_mp']:.0f}KB/MP vs fake "
            f"~{fake['kb_per_mp']:.0f}KB/MP -> one class is much more compressed."
        )
    return warnings


def load_from_manifest(manifest: Path, root: Path) -> tuple[list[Path], list[Path]]:
    reals, fakes = [], []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = root / Path(row["path"].replace("\\", "/"))
            (fakes if str(row["label"]).strip() == "1" else reals).append(p)
    return reals, fakes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", type=Path, help="directory of real images")
    ap.add_argument("--fake", type=Path, help="directory of fake images")
    ap.add_argument("--manifest", type=Path, help="CSV with path,label columns")
    ap.add_argument("--root", type=Path, default=Path("."), help="root for manifest paths")
    ap.add_argument("--limit", type=int, default=0, help="sample at most N files per class")
    args = ap.parse_args()

    if args.manifest:
        real_paths, fake_paths = load_from_manifest(args.manifest, args.root)
    elif args.real and args.fake:
        real_paths, fake_paths = list_images(args.real), list_images(args.fake)
    else:
        ap.error("provide either --manifest or both --real and --fake")
        return

    if args.limit:
        real_paths, fake_paths = real_paths[: args.limit], fake_paths[: args.limit]

    real_rows = [r for p in tqdm(real_paths, desc="scan real") if (r := scan_file(p))]
    fake_rows = [r for p in tqdm(fake_paths, desc="scan fake") if (r := scan_file(p))]
    missing = (len(real_paths) - len(real_rows)) + (len(fake_paths) - len(fake_rows))
    if missing:
        print(f"WARNING: {missing} files missing/unreadable")

    real_s = summarize("real", real_rows)
    fake_s = summarize("fake", fake_rows)
    warnings = compare(real_s, fake_s)
    if warnings:
        print("\n=== PARITY WARNINGS (fix before training!) ===")
        for w in warnings:
            print(" -", w)
        raise SystemExit(1)
    print("\nOK: classes look compression/format/resolution-balanced.")


if __name__ == "__main__":
    main()
