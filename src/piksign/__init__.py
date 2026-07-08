"""PikSign: two-branch AI-generated image detector.

Branches:
  A. Pixel-artifact experts (DINOv2 + LoRA) trained on aligned real/fake pairs:
     A1 SD2.1-VAE reconstructions, A2 FLUX-VAE reconstructions,
     A3 Nano Banana edit pairs (Pico-Banana-400K), A4 GPT-4o edit pairs (ShareGPT-4o-Image).
  B. Semantic VLM (Qwen2.5-VL-7B + DPO LoRA) trained on pixel-scrambled
     real vs. semantically-implausible images.
  C. Provenance heuristics (C2PA / EXIF), rule-based.

Fusion: per-branch calibrated thresholds under a global false-positive budget, OR rule.
"""

__version__ = "0.1.0"
