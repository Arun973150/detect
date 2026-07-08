"""Central path resolution.

All data lives under a single root selected by the PIKSIGN_DATA env var
(default ./data), and all model outputs under PIKSIGN_CKPT (default ./checkpoints).
On RunPod both should point into the network volume, e.g.:
    export PIKSIGN_DATA=/workspace/data
    export PIKSIGN_CKPT=/workspace/checkpoints
"""
from __future__ import annotations

import os
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def data_root() -> Path:
    return Path(os.environ.get("PIKSIGN_DATA", "data")).resolve()


def ckpt_root() -> Path:
    return Path(os.environ.get("PIKSIGN_CKPT", "checkpoints")).resolve()


def raw_dir(*parts: str) -> Path:
    return data_root().joinpath("raw", *parts)


def processed_dir(*parts: str) -> Path:
    return data_root().joinpath("processed", *parts)


def eval_dir(*parts: str) -> Path:
    return data_root().joinpath("eval", *parts)


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)
