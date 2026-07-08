"""Detection metrics. Labels: 0=real, 1=fake. Scores: higher = more fake."""
from __future__ import annotations

import numpy as np


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    labels = np.asarray(labels)
    pred = np.asarray(scores) >= threshold
    real, fake = labels == 0, labels == 1
    tnr = (~pred[real]).mean() if real.any() else np.nan
    tpr = pred[fake].mean() if fake.any() else np.nan
    return float(np.nanmean([tnr, tpr]))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ties
    all_scores = np.concatenate([pos, neg])
    for v in np.unique(all_scores):
        mask = all_scores == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r_pos = ranks[: len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def threshold_at_fpr(real_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold such that at most target_fpr of reals score >= it."""
    real_scores = np.sort(np.asarray(real_scores, dtype=np.float64))
    n = len(real_scores)
    if n == 0:
        return 0.5
    k = int(np.ceil(n * (1.0 - target_fpr)))
    k = min(max(k, 0), n - 1)
    thr = real_scores[k]
    return float(np.nextafter(thr, np.inf))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    thr = threshold_at_fpr(scores[labels == 0], target_fpr)
    fake = scores[labels == 1]
    if len(fake) == 0:
        return float("nan")
    return float((fake >= thr).mean())
