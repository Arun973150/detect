"""Image laundering / degradation operations.

Used three ways:
  1. Online training augmentation for the pixel experts. The SAME random
     distribution is applied to real and fake images, so degradation can never
     become the label shortcut - the only remaining separable signal is the
     generator fingerprint that survives degradation.
  2. Heavy "pixel scrambling" for the semantic-VLM corpus: destroys low-level
     generator traces so the only learnable signal is semantics (the
     Task-Model Alignment recipe).
  3. Offline eval-twin generation: WhatsApp / double-JPEG / screenshot /
     double-resize presets, mirroring a clean eval tree into degraded copies.

CLI:
    python -m piksign.launder --input data/eval/clean --output data/eval/whatsapp --preset whatsapp
"""
from __future__ import annotations

import argparse
import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from .paths import IMG_EXTS

RESAMPLE_CHOICES = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]


# ---------------------------------------------------------------- primitives

def jpeg_cycle(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def webp_cycle(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def scale_cycle(img: Image.Image, factor: float, rng: random.Random | None = None) -> Image.Image:
    """Downsample then upsample back to the original size.

    This is the single most destructive realistic operation for pixel
    artifacts (it rewrites neighborhood pixel relationships), so it is central
    both to the semantic-branch scrambling and to robustness training.
    """
    rng = rng or random
    w, h = img.size
    nw, nh = max(8, int(w * factor)), max(8, int(h * factor))
    down = img.resize((nw, nh), rng.choice(RESAMPLE_CHOICES))
    return down.resize((w, h), rng.choice(RESAMPLE_CHOICES))


def resize_to_max_side(img: Image.Image, max_side: int, resample=Image.LANCZOS) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    s = max_side / m
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), resample)


def gaussian_noise(img: Image.Image, sigma: float, rng: random.Random | None = None) -> Image.Image:
    seed = (rng or random).randrange(2**31)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0.0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius))


# ------------------------------------------------------- random augmentation

@dataclass
class LaunderConfig:
    p_scale: float = 0.7
    scale: tuple[float, float] = (0.25, 1.0)
    p_blur: float = 0.1
    blur_r: tuple[float, float] = (0.4, 1.5)
    p_noise: float = 0.2
    noise_sigma: tuple[float, float] = (1.0, 8.0)
    p_jpeg: float = 0.85
    jpeg_q: tuple[int, int] = (30, 95)
    p_webp: float = 0.1
    webp_q: tuple[int, int] = (40, 90)


# Default: robustness augmentation for pixel experts.
TRAIN_AUG = LaunderConfig()

# Semantic-branch scrambling: always destroy low-level cues, hard.
SEMANTIC_SCRAMBLE = LaunderConfig(
    p_scale=1.0, scale=(0.25, 0.5),
    p_blur=0.15, blur_r=(0.4, 1.2),
    p_noise=0.5, noise_sigma=(2.0, 10.0),
    p_jpeg=1.0, jpeg_q=(30, 70),
    p_webp=0.0,
)


def random_launder(img: Image.Image, rng: random.Random, cfg: LaunderConfig = TRAIN_AUG) -> Image.Image:
    img = img.convert("RGB")
    if rng.random() < cfg.p_scale:
        img = scale_cycle(img, rng.uniform(*cfg.scale), rng)
    if rng.random() < cfg.p_blur:
        img = gaussian_blur(img, rng.uniform(*cfg.blur_r))
    if rng.random() < cfg.p_noise:
        img = gaussian_noise(img, rng.uniform(*cfg.noise_sigma), rng)
    # lossy re-encode last, like real distribution pipelines
    if rng.random() < cfg.p_webp:
        img = webp_cycle(img, rng.randint(*cfg.webp_q))
    elif rng.random() < cfg.p_jpeg:
        img = jpeg_cycle(img, rng.randint(*cfg.jpeg_q))
    return img


# ------------------------------------------------------------- eval presets

def preset_whatsapp(img: Image.Image, rng: random.Random) -> Image.Image:
    img = resize_to_max_side(img.convert("RGB"), 1280, Image.BICUBIC)
    return jpeg_cycle(img, 70)


def preset_double_jpeg(img: Image.Image, rng: random.Random) -> Image.Image:
    q = rng.randint(60, 85)
    return jpeg_cycle(jpeg_cycle(img, q), q)


def preset_screenshot(img: Image.Image, rng: random.Random) -> Image.Image:
    # rendered smaller on a screen, screenshotted, then re-encoded on share
    img = img.convert("RGB")
    s = rng.uniform(0.55, 0.9)
    w, h = img.size
    img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BICUBIC)
    return jpeg_cycle(img, 90)


def preset_double_resize(img: Image.Image, rng: random.Random) -> Image.Image:
    # the paper's hardest robustness setting: 0.4x down, back up
    return scale_cycle(img.convert("RGB"), 0.4, rng)


PRESETS = {
    "whatsapp": preset_whatsapp,
    "double_jpeg": preset_double_jpeg,
    "screenshot": preset_screenshot,
    "double_resize": preset_double_resize,
}


def process_tree(input_dir: Path, output_dir: Path, preset: str, seed: int = 0) -> int:
    """Mirror input_dir into output_dir with the preset applied. Returns count."""
    fn = PRESETS[preset]
    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    n = 0
    for p in tqdm(files, desc=f"launder[{preset}]"):
        rel = p.relative_to(input_dir)
        out = (output_dir / rel).with_suffix(".jpg")
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(f"{seed}:{rel.as_posix()}")
        try:
            img = Image.open(p)
            img.load()
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {p}: {e}")
            continue
        fn(img, rng).save(out, format="JPEG", quality=95, subsampling=0)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = process_tree(args.input, args.output, args.preset, args.seed)
    print(f"wrote {n} images to {args.output}")


if __name__ == "__main__":
    main()
