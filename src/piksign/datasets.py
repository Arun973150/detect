"""Torch datasets for pixel-expert training.

Design rules baked in here (do not "optimize" them away):
  - Images are NEVER resized to fit the network. Low-level generator
    fingerprints live in native-resolution pixel statistics, so we take
    random square crops at native resolution instead.
  - Laundering augmentation is drawn from the SAME distribution for both
    classes, so degradation can never become the label.
"""
from __future__ import annotations

import random
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .launder import TRAIN_AUG, LaunderConfig, random_launder
from .paths import list_images

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def _ensure_min_side(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    if m >= size:
        return img
    s = size / m
    return img.resize((max(size, int(w * s + 0.5)), max(size, int(h * s + 0.5))), Image.BICUBIC)


def random_crop(img: Image.Image, size: int, rng: random.Random) -> Image.Image:
    img = _ensure_min_side(img, size)
    w, h = img.size
    x = rng.randint(0, w - size)
    y = rng.randint(0, h - size)
    return img.crop((x, y, x + size, y + size))


def multi_crops(img: Image.Image, size: int, n: int, seed: int = 0) -> list[Image.Image]:
    """Deterministic crop set for inference: center + 4 corners + random fill."""
    img = _ensure_min_side(img, size)
    w, h = img.size
    xs, ys = w - size, h - size
    anchors = [
        (xs // 2, ys // 2),
        (0, 0), (xs, 0), (0, ys), (xs, ys),
    ]
    rng = random.Random(seed)
    while len(anchors) < n:
        anchors.append((rng.randint(0, xs) if xs else 0, rng.randint(0, ys) if ys else 0))
    return [img.crop((x, y, x + size, y + size)) for x, y in anchors[:n]]


def _stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


class PairImageDataset(Dataset):
    """Binary real/fake dataset over two directories (label: real=0, fake=1)."""

    def __init__(
        self,
        real_paths: list[Path],
        fake_paths: list[Path],
        crop: int = 224,
        train: bool = True,
        launder_prob: float = 0.9,
        launder_cfg: LaunderConfig = TRAIN_AUG,
        seed: int = 42,
    ) -> None:
        self.items = [(p, 0.0) for p in real_paths] + [(p, 1.0) for p in fake_paths]
        self.crop = crop
        self.train = train
        self.launder_prob = launder_prob
        self.launder_cfg = launder_cfg
        self.seed = seed
        self.epoch = 0  # bump externally for fresh augmentation draws per epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[idx]
        rng = random.Random(f"{self.seed}:{self.epoch}:{idx}")
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.train:
            if rng.random() < self.launder_prob:
                img = random_launder(img, rng, self.launder_cfg)
            img = random_crop(img, self.crop, rng)
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = multi_crops(img, self.crop, 1)[0]  # center crop
        return pil_to_tensor(img), torch.tensor(label, dtype=torch.float32)


def split_paths(
    real_dir: Path, fake_dir: Path, val_frac: float, seed: int = 42
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """Deterministic per-filename split so resumed runs keep the same val set."""

    def split(paths: list[Path]) -> tuple[list[Path], list[Path]]:
        tr, va = [], []
        for p in paths:
            h = _stable_hash(f"{seed}:{p.name}") % 10_000
            (va if h < val_frac * 10_000 else tr).append(p)
        return tr, va

    rt, rv = split(list_images(real_dir))
    ft, fv = split(list_images(fake_dir))
    return rt, ft, rv, fv


def make_loaders(
    real_dir: Path,
    fake_dir: Path,
    crop: int,
    batch_size: int,
    workers: int,
    val_frac: float = 0.05,
    val_real_dir: Path | None = None,
    val_fake_dir: Path | None = None,
    launder_prob: float = 0.9,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, PairImageDataset]:
    if val_real_dir and val_fake_dir:
        rt, ft = list_images(real_dir), list_images(fake_dir)
        rv, fv = list_images(val_real_dir), list_images(val_fake_dir)
    else:
        rt, ft, rv, fv = split_paths(real_dir, fake_dir, val_frac, seed)
    if not rt or not ft:
        raise FileNotFoundError(f"empty training class: real={len(rt)} fake={len(ft)}")

    train_ds = PairImageDataset(rt, ft, crop=crop, train=True, launder_prob=launder_prob, seed=seed)
    # Val uses clean center crops; laundered robustness is measured separately
    # by the eval harness on preset-degraded twins.
    val_ds = PairImageDataset(rv, fv, crop=crop, train=False, seed=seed + 1)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=True, persistent_workers=workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=max(2, workers // 2),
        pin_memory=True, persistent_workers=workers > 0,
    )
    print(f"train: {len(rt)} real / {len(ft)} fake   val: {len(rv)} real / {len(fv)} fake")
    return train_dl, val_dl, train_ds
