"""Build VAE-reconstruction pair sets for the generalization experts (A1/A2).

For each real image we produce its encode-decode roundtrip through a
generator-family VAE. The pair is semantically identical, so the only
learnable signal is the VAE decoder fingerprint - the core of the
Task-Model Alignment recipe, and the part that transfers to unseen
commercial generators.

VAEs:
  sd21  stabilityai/stable-diffusion-2-1 (subfolder vae) - classic LDM family
  flux  black-forest-labs/FLUX.1-schnell (subfolder vae) - 16-ch modern family
        (schnell is Apache-2.0 and ungated, same autoencoder as FLUX dev)

Both the reconstruction AND its source go through the identical crop +
JPEG q95 funnel, so no format/resolution shortcut exists.

    python -m piksign.recon.vae_reconstruct --vae sd21 --input data/raw/coco/train2017_12000
    python -m piksign.recon.vae_reconstruct --vae flux --input data/raw/coco/train2017_12000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from ..download import normalize_save
from ..paths import ensure, list_images, processed_dir

Image.MAX_IMAGE_PIXELS = None

VAES = {
    "sd21": ("stabilityai/stable-diffusion-2-1", "vae"),
    "flux": ("black-forest-labs/FLUX.1-schnell", "vae"),
}


def prepare(img: Image.Image, max_side: int, rng: random.Random) -> Image.Image:
    """Random-crop (never resize) to bound VRAM, then crop to a /16 grid."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        nw, nh = min(w, max_side), min(h, max_side)
        x = rng.randint(0, w - nw)
        y = rng.randint(0, h - nh)
        img = img.crop((x, y, x + nw, y + nh))
    w, h = img.size
    w16, h16 = (w // 16) * 16, (h // 16) * 16
    if w16 < 64 or h16 < 64:
        return None  # too small to be useful
    if (w16, h16) != (w, h):
        img = img.crop(((w - w16) // 2, (h - h16) // 2,
                        (w - w16) // 2 + w16, (h - h16) // 2 + h16))
    return img


@torch.no_grad()
def reconstruct(vae, img: Image.Image, device: str, dtype) -> Image.Image:
    import numpy as np
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    x = torch.from_numpy(arr.transpose(2, 0, 1))[None].to(device=device, dtype=dtype)
    posterior = vae.encode(x).latent_dist
    rec = vae.decode(posterior.mode()).sample
    rec = ((rec.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(rec[0].permute(1, 2, 0).cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae", choices=sorted(VAES), required=True)
    ap.add_argument("--input", type=Path, required=True, help="directory of real images")
    ap.add_argument("--n", type=int, default=0, help="cap (0 = all)")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from diffusers import AutoencoderKL

    repo, sub = VAES[args.vae]
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"loading VAE {repo}/{sub} ...")
    vae = AutoencoderKL.from_pretrained(repo, subfolder=sub, torch_dtype=dtype)
    vae = vae.to(args.device).eval()

    out_root = processed_dir("pairs", f"{args.vae}_recon")
    real_out = ensure(out_root / "real")
    fake_out = ensure(out_root / "fake")

    paths = list_images(args.input)
    if args.n and args.n < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.n))
    print(f"{len(paths)} source images")

    n_ok = 0
    for p in tqdm(paths, desc=f"recon[{args.vae}]"):
        stem = p.stem
        real_dst = real_out / f"{stem}.jpg"
        fake_dst = fake_out / f"{stem}.jpg"
        if real_dst.exists() and fake_dst.exists():
            n_ok += 1
            continue
        rng = random.Random(f"{args.seed}:{stem}")
        try:
            with Image.open(p) as im:
                img = prepare(im, args.max_side, rng)
            if img is None:
                continue
            rec = reconstruct(vae, img, args.device, dtype)
            normalize_save(rec, fake_dst)
            normalize_save(img, real_dst)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[skip {p.name}] {e}")
    print(f"done: {n_ok} pairs -> {out_root}")


if __name__ == "__main__":
    main()
