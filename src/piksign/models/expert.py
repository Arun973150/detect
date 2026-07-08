"""Pixel-artifact expert: DINOv2 backbone + LoRA + linear head, single logit.

Conventions:
  label 0 = real, 1 = fake; score = sigmoid(logit) = P(fake).
  Inference never resizes the image: it scores a deterministic set of
  native-resolution crops and aggregates (default: mean of top-3, a
  compromise between max's subtle-edit sensitivity and mean's FP control).

Checkpoint layout (one directory per expert):
  <dir>/adapter/            PEFT LoRA weights
  <dir>/head.pt             linear head state dict
  <dir>/expert_config.json  backbone id, crop size, lora params
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from ..datasets import multi_crops, pil_to_tensor

DEFAULT_BACKBONE = "facebook/dinov2-large"


class PixelExpert(nn.Module):
    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        lora_r: int = 8,
        lora_alpha: float = 8.0,
        crop: int = 224,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        self.config = {
            "backbone": backbone, "lora_r": lora_r,
            "lora_alpha": lora_alpha, "crop": crop,
        }
        base = AutoModel.from_pretrained(backbone)
        lora = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
            target_modules=["query", "value"], bias="none",
        )
        self.backbone = get_peft_model(base, lora)
        self.head = nn.Linear(base.config.hidden_size, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------- inference

    @torch.no_grad()
    def score_pil(self, img: Image.Image, n_crops: int = 10, agg: str = "top3") -> float:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        crops = multi_crops(img.convert("RGB"), self.config["crop"], n_crops)
        batch = torch.stack([pil_to_tensor(c) for c in crops]).to(device=device, dtype=dtype)
        probs = torch.sigmoid(self.forward(batch).float())
        if agg == "max":
            return probs.max().item()
        if agg == "mean":
            return probs.mean().item()
        k = min(3, probs.numel())
        return probs.topk(k).values.mean().item()

    # ------------------------------------------------------------ persistence

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out_dir / "adapter")
        torch.save(self.head.state_dict(), out_dir / "head.pt")
        (out_dir / "expert_config.json").write_text(json.dumps(self.config, indent=2))

    @classmethod
    def load(cls, ckpt_dir: Path, device: str = "cuda", dtype=torch.float32) -> "PixelExpert":
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / "expert_config.json").read_text())
        model = cls(**cfg)
        from peft import PeftModel
        from transformers import AutoModel

        base = AutoModel.from_pretrained(cfg["backbone"])
        model.backbone = PeftModel.from_pretrained(base, ckpt_dir / "adapter")
        model.head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))
        return model.to(device=device, dtype=dtype).eval()


def is_expert_checkpoint(d: Path) -> bool:
    return (Path(d) / "expert_config.json").exists()
