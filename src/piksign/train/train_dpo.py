"""DPO fine-tune Qwen2.5-VL-7B on the semantic preference pairs (branch B).

Hyperparameters follow the AlignGemini paper: LoRA r=16 alpha=32,
lr 1e-6, beta 0.05, one epoch. Needs ~60GB VRAM with bf16 + gradient
checkpointing on an A100-80GB.

Input:  data/processed/dpo_pairs.jsonl  (from label_dpo.py; images already scrambled)
Output: checkpoints/vlm_dpo/  (LoRA adapter)

    python -m piksign.train.train_dpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..paths import ckpt_root, processed_dir


def load_pairs(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not Path(rec["image"]).exists():
                continue
            rows.append({
                "images": [rec["image"]],
                "prompt": [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": rec["prompt"]},
                    ],
                }],
                "chosen": [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": rec["chosen"]}],
                }],
                "rejected": [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": rec["rejected"]}],
                }],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--pairs", type=Path, default=None, help="default data/processed/dpo_pairs.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="default checkpoints/vlm_dpo")
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=1, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--num-proc", type=int, default=2,
                    help="dataset map workers; vision tokenization is RAM-hungry, "
                         "raise only if the container has headroom")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs_path = args.pairs or processed_dir("dpo_pairs.jsonl")
    out_dir = args.out or (ckpt_root() / "vlm_dpo")

    from datasets import Dataset
    from datasets import Image as HFImage
    from datasets import Sequence
    from peft import LoraConfig
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import DPOConfig, DPOTrainer

    rows = load_pairs(pairs_path)
    print(f"{len(rows)} preference pairs")
    ds = Dataset.from_list(rows)
    ds = ds.cast_column("images", Sequence(HFImage()))  # lazy image loading

    processor = AutoProcessor.from_pretrained(
        args.base, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = DPOConfig(
        output_dir=str(out_dir),
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=args.seed,
        max_length=1600,
        max_prompt_length=None,
        dataset_num_proc=args.num_proc,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"saved LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()
