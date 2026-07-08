"""Train one pixel-artifact expert (A1-A4).

    python -m piksign.train.train_expert --name a1_sd21 \
        --real-dir data/processed/pairs/sd21_recon/real \
        --fake-dir data/processed/pairs/sd21_recon/fake

    python -m piksign.train.train_expert --name a3_nanobanana \
        --real-dir data/processed/pairs/nanobanana/real \
        --fake-dir data/processed/pairs/nanobanana/fake \
        --val-real data/processed/pairs/nanobanana_val/real \
        --val-fake data/processed/pairs/nanobanana_val/fake
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..datasets import make_loaders
from ..eval.metrics import balanced_accuracy, roc_auc, tpr_at_fpr
from ..models.expert import DEFAULT_BACKBONE, PixelExpert
from ..paths import ckpt_root


@torch.no_grad()
def evaluate(model: PixelExpert, val_dl, device: str) -> dict:
    model.eval()
    scores, labels = [], []
    for x, y in val_dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    return {
        "bacc@0.5": balanced_accuracy(y, s, 0.5),
        "auc": roc_auc(y, s),
        "tpr@5fpr": tpr_at_fpr(y, s, 0.05),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="checkpoint name, e.g. a1_sd21")
    ap.add_argument("--real-dir", type=Path, required=True)
    ap.add_argument("--fake-dir", type=Path, required=True)
    ap.add_argument("--val-real", type=Path, default=None)
    ap.add_argument("--val-fake", type=Path, default=None)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=1.0,
                    help="paper value 1.0 (conservative, best generalization); "
                         "raise (e.g. 8.0) for target-specific experts")
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--launder-prob", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ckpt_root() / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, train_ds = make_loaders(
        args.real_dir, args.fake_dir, crop=args.crop, batch_size=args.bs,
        workers=args.workers, val_frac=args.val_frac,
        val_real_dir=args.val_real, val_fake_dir=args.val_fake,
        launder_prob=args.launder_prob, seed=args.seed,
    )

    model = PixelExpert(args.backbone, args.lora_r, args.lora_alpha, args.crop).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {n_train / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    warmup = max(20, total_steps // 20)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.BCEWithLogitsLoss()

    best = -1.0
    step = 0
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (x, y) in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            step += 1
            if (i + 1) % 50 == 0:
                ips = (i + 1) * args.bs / (time.time() - t0)
                print(f"epoch {epoch} step {i + 1}/{len(train_dl)} "
                      f"loss {running / 50:.4f} lr {sched.get_last_lr()[0]:.2e} {ips:.0f} img/s")
                running = 0.0

        metrics = evaluate(model, val_dl, device)
        print(f"epoch {epoch} val: {json.dumps(metrics)}")
        if metrics["bacc@0.5"] > best:
            best = metrics["bacc@0.5"]
            model.save(out_dir)
            (out_dir / "val_metrics.json").write_text(
                json.dumps({"epoch": epoch, **metrics, "args": vars(args)}, indent=2, default=str)
            )
            print(f"saved best -> {out_dir} (bacc {best:.4f})")

    print(f"finished. best val bacc={best:.4f}  checkpoint: {out_dir}")


if __name__ == "__main__":
    main()
