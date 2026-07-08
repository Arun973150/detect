"""Branch fusion and calibration.

Decision rule: OR - an image is flagged fake if ANY branch fires above its
calibrated threshold. Real images should be benign along every dimension.

Calibration: each branch's threshold is set on validation REALS so the
branches share a global false-positive budget. With K branches and global
budget g, each branch gets per-branch FPR of 1-(1-g)^(1/K) (independence
approximation - conservative in practice because branch FPs correlate).

CLI (fit from an eval-harness scores file, using its clean+laundered reals):
    python -m piksign.fusion --scores data/eval/scores.jsonl --global-fpr 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .eval.metrics import threshold_at_fpr
from .paths import ckpt_root

# provenance is rule-based and near-zero-FP by construction; it keeps a fixed
# threshold and does not consume the statistical FPR budget.
PROVENANCE_THRESHOLD = 0.85


def default_calibration(branches: list[str]) -> dict:
    return {"global_fpr": None, "thresholds": {b: 0.5 for b in branches},
            "provenance_threshold": PROVENANCE_THRESHOLD}


def fit_calibration(rows: list[dict], global_fpr: float = 0.05) -> dict:
    """rows: [{label: 0|1, scores: {branch: float, ...}}, ...] from the harness."""
    branches = sorted({b for r in rows for b in r["scores"] if b != "provenance"})
    if not branches:
        raise ValueError("no branch scores found")
    per_branch_fpr = 1.0 - (1.0 - global_fpr) ** (1.0 / len(branches))
    thresholds = {}
    for b in branches:
        real_scores = np.array([r["scores"][b] for r in rows
                                if r["label"] == 0 and b in r["scores"]])
        if len(real_scores) < 50:
            print(f"WARNING: only {len(real_scores)} real scores for {b}; threshold unstable")
        thresholds[b] = threshold_at_fpr(real_scores, per_branch_fpr) if len(real_scores) else 0.5
    return {
        "global_fpr": global_fpr,
        "per_branch_fpr": per_branch_fpr,
        "thresholds": thresholds,
        "provenance_threshold": PROVENANCE_THRESHOLD,
    }


def fuse(scores: dict[str, float], calib: dict) -> dict:
    """Returns fused verdict + per-branch breakdown."""
    fired = []
    margins = []
    for branch, score in scores.items():
        thr = (calib["provenance_threshold"] if branch == "provenance"
               else calib["thresholds"].get(branch, 0.5))
        if score >= thr:
            fired.append(branch)
        margins.append((score - thr) / max(1e-6, 1.0 - thr) if thr < 1 else score - thr)
    strength = max(margins) if margins else 0.0
    return {
        "is_ai": bool(fired),
        "fired_branches": fired,
        "strength": round(float(np.clip(strength, -1, 1)), 4),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "thresholds": {k: round(float(calib["provenance_threshold"] if k == "provenance"
                                      else calib["thresholds"].get(k, 0.5)), 4)
                       for k in scores},
    }


def load_calibration(path: Path | None = None) -> dict | None:
    path = path or (ckpt_root() / "calibration.json")
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True, help="scores.jsonl from the eval harness")
    ap.add_argument("--global-fpr", type=float, default=0.05)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="restrict calibration to these degradation variants (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="default checkpoints/calibration.json")
    args = ap.parse_args()

    rows = []
    with open(args.scores, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = list({r.get("path", i): r for i, r in enumerate(rows)}.values())
    if args.variants:
        rows = [r for r in rows if r.get("variant") in set(args.variants)]
    print(f"calibrating on {len(rows)} rows "
          f"({sum(r['label'] == 0 for r in rows)} real / {sum(r['label'] == 1 for r in rows)} fake)")

    calib = fit_calibration(rows, args.global_fpr)
    out = args.out or (ckpt_root() / "calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib, indent=2))
    print(json.dumps(calib, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
