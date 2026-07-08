"""Evaluation harness: score eval sets with every branch, report per source/variant.

Eval tree convention:
    data/eval/<variant>/<source>/{real,fake}/*.jpg
      variant: clean | whatsapp | double_jpeg | screenshot | double_resize
      source:  nanobanana_fresh | gpt4o_fresh | reals_phone | reals_coco | ...

Degraded variants are produced from `clean` with the launder CLI, e.g.:
    python -m piksign.launder --input data/eval/clean --output data/eval/whatsapp --preset whatsapp

Subcommands:
    assemble  copy eval reals into the tree (COCO val + your phone photos)
    run       score all images with experts (+ optional VLM) into scores.jsonl
    report    aggregate scores.jsonl into a metrics table

Balanced accuracy per (source, variant) uses that variant's fake TPR and the
global real TNR of the same variant (reals are shared across fake sources).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..fusion import fuse, load_calibration
from ..models.provenance import check_file
from ..paths import ckpt_root, ensure, eval_dir, list_images
from .metrics import balanced_accuracy, roc_auc, tpr_at_fpr

Image.MAX_IMAGE_PIXELS = None


# ------------------------------------------------------------------ assemble

def cmd_assemble(args: argparse.Namespace) -> None:
    from ..download import normalize_save

    jobs = []
    if args.coco_val:
        jobs.append(("reals_coco", args.coco_val, args.n_coco))
    if args.phone:
        jobs.append(("reals_phone", args.phone, 0))
    if not jobs:
        raise SystemExit("provide --coco-val and/or --phone")
    rng = random.Random(args.seed)
    for source, src_dir, cap in jobs:
        paths = list_images(src_dir)
        if cap and cap < len(paths):
            paths = rng.sample(paths, cap)
        out = ensure(eval_dir("clean", source, "real"))
        for p in tqdm(paths, desc=f"assemble {source}"):
            dst = out / (p.stem + ".jpg")
            if dst.exists():
                continue
            try:
                with Image.open(p) as im:
                    normalize_save(im.convert("RGB"), dst)
            except Exception as e:  # noqa: BLE001
                print(f"[skip {p}] {e}")
        print(f"{source}: {len(list_images(out))} reals")


# ----------------------------------------------------------------- manifest

def scan_manifest(root: Path) -> list[dict]:
    rows = []
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for source_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            for cls, label in (("real", 0), ("fake", 1)):
                for img in list_images(source_dir / cls):
                    rows.append({
                        "path": str(img),
                        "label": label,
                        "source": source_dir.name,
                        "variant": variant_dir.name,
                    })
    return rows


# ---------------------------------------------------------------------- run

def cmd_run(args: argparse.Namespace) -> None:
    import torch
    from ..models.expert import PixelExpert, is_expert_checkpoint

    rows = scan_manifest(eval_dir())
    if not rows:
        raise SystemExit(f"no eval images under {eval_dir()}")
    print(f"{len(rows)} eval images")

    scores_path = args.scores or (eval_dir() / "scores.jsonl")
    done: dict[str, dict] = {}
    if scores_path.exists() and not args.fresh:
        with open(scores_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["path"]] = r
                except Exception:  # noqa: BLE001
                    pass
        print(f"resuming: {len(done)} already scored")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts = {}
    root = args.ckpt_root or ckpt_root()
    for d in sorted(Path(root).iterdir()) if Path(root).is_dir() else []:
        if is_expert_checkpoint(d):
            if args.experts and d.name not in args.experts:
                continue
            print(f"loading expert {d.name} ...")
            experts[d.name] = PixelExpert.load(d, device=device)
    if not experts:
        print("WARNING: no pixel experts found - provenance"
              + (" + vlm" if args.vlm else "") + " only")

    vlm = None
    if args.vlm:
        from ..models.vlm import SemanticVLM
        adapter = Path(root) / "vlm_dpo"
        vlm = SemanticVLM(adapter=adapter if adapter.exists() else None, device=device)
        print(f"loaded VLM (adapter={'yes' if adapter.exists() else 'BASE ONLY'})")

    with open(scores_path, "a", encoding="utf-8") as out_f:
        for row in tqdm(rows, desc="score"):
            prev = done.get(row["path"])
            needed = set(experts) | {"provenance"} | ({"vlm"} if vlm else set())
            if prev and needed.issubset(prev["scores"]):
                continue
            scores = dict(prev["scores"]) if prev else {}
            try:
                with Image.open(row["path"]) as im:
                    img = im.convert("RGB")
            except Exception as e:  # noqa: BLE001
                print(f"[skip {row['path']}] {e}")
                continue
            for name, expert in experts.items():
                if name not in scores:
                    scores[name] = expert.score_pil(img, n_crops=args.crops, agg=args.agg)
            if "provenance" not in scores:
                scores["provenance"] = check_file(row["path"])["score"]
            if vlm and "vlm" not in scores:
                scores["vlm"] = vlm.score(img)["prob_fake"]
            out_f.write(json.dumps({**row, "scores": scores}) + "\n")
            out_f.flush()
    print(f"scores -> {scores_path}")


# ------------------------------------------------------------------- report

def cmd_report(args: argparse.Namespace) -> None:
    scores_path = args.scores or (eval_dir() / "scores.jsonl")
    rows = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # resumed runs append updated rows for the same path - keep the last one
    rows = list({r["path"]: r for r in rows}.values())
    calib = load_calibration(args.calibration)
    if calib is None:
        branches = sorted({b for r in rows for b in r["scores"]})
        from ..fusion import default_calibration
        calib = default_calibration([b for b in branches if b != "provenance"])
        print("NOTE: no calibration.json - using naive 0.5 thresholds")

    by_variant_reals: dict[str, list[dict]] = defaultdict(list)
    by_group_fakes: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["label"] == 0:
            by_variant_reals[r["variant"]].append(r)
        else:
            by_group_fakes[(r["variant"], r["source"])].append(r)

    def fused_pred(r: dict) -> bool:
        return fuse(r["scores"], calib)["is_ai"]

    print(f"\n{'variant':<14}{'source':<20}{'n_fake':>7}{'TPR':>7}{'TNR':>7}{'bAcc':>7}")
    print("-" * 62)
    report_rows = []
    for variant in sorted(by_variant_reals):
        reals = by_variant_reals[variant]
        tnr = float(np.mean([not fused_pred(r) for r in reals])) if reals else float("nan")
        for (v, source), fakes in sorted(by_group_fakes.items()):
            if v != variant:
                continue
            tpr = float(np.mean([fused_pred(r) for r in fakes]))
            bacc = np.nanmean([tpr, tnr])
            print(f"{variant:<14}{source:<20}{len(fakes):>7}{tpr:>7.3f}{tnr:>7.3f}{bacc:>7.3f}")
            report_rows.append({"variant": variant, "source": source, "n_fake": len(fakes),
                                "tpr": tpr, "tnr": tnr, "bacc": float(bacc)})

    # per-branch diagnostics over everything
    branches = sorted({b for r in rows for b in r["scores"]})
    print(f"\nper-branch (all variants pooled): {'branch':<16}{'AUC':>7}{'TPR@5%FPR':>11}")
    labels = np.array([r["label"] for r in rows])
    for b in branches:
        s = np.array([r["scores"].get(b, np.nan) for r in rows])
        mask = ~np.isnan(s)
        if mask.sum() < 10:
            continue
        auc = roc_auc(labels[mask], s[mask])
        t5 = tpr_at_fpr(labels[mask], s[mask], 0.05)
        b5 = balanced_accuracy(labels[mask], s[mask], 0.5)
        print(f"  {b:<16} auc={auc:.3f}  tpr@5fpr={t5:.3f}  bacc@0.5={b5:.3f}")

    out = args.out or (eval_dir() / "report.json")
    Path(out).write_text(json.dumps(report_rows, indent=2))
    print(f"\nreport -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("assemble", help="copy eval reals into the tree")
    a.add_argument("--coco-val", type=Path, default=None)
    a.add_argument("--n-coco", type=int, default=1500)
    a.add_argument("--phone", type=Path, default=None, help="folder of your own phone photos")
    a.add_argument("--seed", type=int, default=7)
    a.set_defaults(fn=cmd_assemble)

    r = sub.add_parser("run", help="score all eval images")
    r.add_argument("--experts", nargs="*", default=None, help="subset of expert names")
    r.add_argument("--ckpt-root", type=Path, default=None)
    r.add_argument("--vlm", action="store_true")
    r.add_argument("--crops", type=int, default=10)
    r.add_argument("--agg", choices=["max", "mean", "top3"], default="top3")
    r.add_argument("--scores", type=Path, default=None)
    r.add_argument("--fresh", action="store_true", help="ignore existing scores")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="aggregate scores into metrics")
    p.add_argument("--scores", type=Path, default=None)
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
