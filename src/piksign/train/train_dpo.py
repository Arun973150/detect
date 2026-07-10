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
import inspect
import json
import os
from importlib.metadata import version
from pathlib import Path

import torch

from ..paths import ckpt_root, processed_dir

# Fork-deadlock avoidance: the Qwen2.5-VL fast image processor uses Rust
# (rayon) + OpenMP thread pools; when HF datasets .map forks after those pools
# are live, tokenization wedges at 0% with low CPU. Capping every native pool
# to 1 thread before those libs initialize prevents the deadlock. Verified on
# RunPod: without these, tokenization hangs indefinitely.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# force-disable the flaky fast downloaders (RunPod templates enable hf_transfer,
# which errors when the package is absent and stalls silently when present)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"


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
    ap.add_argument("--num-proc", type=int, default=1,
                    help="dataset map workers; 1 = in-process (no fork, no deadlock, "
                         "continuous progress bar). >1 risks fork-deadlock after the "
                         "model/tokenizer are loaded - verified on RunPod.")
    ap.add_argument("--max-pixels", type=int, default=768 * 28 * 28,
                    help="vision token budget per image (smaller = faster tokenize/train)")
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

    trl_major_minor = tuple(int(p) for p in version("trl").split("+")[0].split(".")[:2])
    if trl_major_minor < (0, 29):
        raise SystemExit(
            "Qwen2.5-VL DPO needs trl>=0.29 so image_grid_thw is kept in "
            "vision batches. Run: pip install -U 'transformers>=4.56.2' "
            "'trl[vlm]>=0.29,<0.30'"
        )

    rows = load_pairs(pairs_path)
    print(f"{len(rows)} preference pairs")
    ds = Dataset.from_list(rows)
    ds = ds.cast_column("images", Sequence(HFImage()))  # lazy image loading

    processor = AutoProcessor.from_pretrained(
        args.base, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels
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

    dpo_kwargs = {
        "output_dir": str(out_dir),
        "beta": args.beta,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.bs,
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": [],
        "seed": args.seed,
        # For VLM-DPO, truncating can remove image tokens while keeping
        # pixel_values, which breaks Qwen2.5-VL's vision/text alignment.
        "max_length": None,
        "max_prompt_length": None,
        "dataset_num_proc": args.num_proc,
        "remove_unused_columns": False,
    }
    supported_dpo_args = set(inspect.signature(DPOConfig).parameters)
    dropped = sorted(set(dpo_kwargs) - supported_dpo_args)
    if dropped:
        print(f"Skipping unsupported DPOConfig args for trl {version('trl')}: {', '.join(dropped)}")
    cfg = DPOConfig(**{k: v for k, v in dpo_kwargs.items() if k in supported_dpo_args})

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
