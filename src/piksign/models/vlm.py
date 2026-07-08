"""Semantic VLM branch: Qwen2.5-VL wrapper for scoring + rationale.

Scoring is done by comparing the teacher-forced log-probability of the two
fixed verdict sentences (a calibrated two-way choice), which is far more
stable than parsing free-form generations. A short rationale is then
generated with the winning verdict prefilled.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ..launder import resize_to_max_side

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "Look carefully at this image and judge ONLY its semantic plausibility: "
    "object shapes and counts, anatomy, hands, text and signage, physics, "
    "lighting and shadow logic, reflections, and overall scene coherence. "
    "Ignore compression, noise, blur or resolution. Start your answer with "
    "exactly one of these sentences: 'This is an AI-generated image.' or "
    "'This is an authentic image.' Then briefly explain the semantic evidence."
)
VERDICT_FAKE = "This is an AI-generated image."
VERDICT_REAL = "This is an authentic image."


class SemanticVLM:
    def __init__(
        self,
        base: str = BASE_MODEL,
        adapter: str | Path | None = None,
        device: str = "cuda",
        max_side: int = 1024,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.max_side = max_side
        self.processor = AutoProcessor.from_pretrained(
            base, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=device,
        )
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model.eval()

    # ------------------------------------------------------------- internals

    def _inputs(self, img: Image.Image, assistant_text: str | None):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if assistant_text is not None:
            text = text + assistant_text
        return self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)

    @torch.no_grad()
    def _verdict_logprob(self, img: Image.Image, verdict: str) -> float:
        """Sum of token log-probs of `verdict` continuing the chat prompt."""
        base = self._inputs(img, None)
        full = self._inputs(img, verdict)
        n_prompt = base["input_ids"].shape[1]
        out = self.model(**full)
        logits = out.logits[0, n_prompt - 1: -1].float()
        targets = full["input_ids"][0, n_prompt:]
        logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs.gather(1, targets[:, None]).sum().item()

    # -------------------------------------------------------------- scoring

    @torch.no_grad()
    def score(self, img: Image.Image, with_rationale: bool = False) -> dict:
        img = resize_to_max_side(img.convert("RGB"), self.max_side)
        lp_fake = self._verdict_logprob(img, VERDICT_FAKE)
        lp_real = self._verdict_logprob(img, VERDICT_REAL)
        prob_fake = float(torch.sigmoid(torch.tensor(lp_fake - lp_real)))
        result = {
            "prob_fake": prob_fake,
            "verdict": VERDICT_FAKE if prob_fake >= 0.5 else VERDICT_REAL,
        }
        if with_rationale:
            prefix = result["verdict"] + " "
            inputs = self._inputs(img, prefix)
            gen = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            new_tokens = gen[0, inputs["input_ids"].shape[1]:]
            result["rationale"] = prefix + self.processor.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
        return result
