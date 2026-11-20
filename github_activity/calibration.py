"""Activity-specific probability blending, F1 evaluation and threshold calibration."""

import numpy as np
from scipy.optimize import minimize

ANCHOR = 0.25
N_LABELS = 6


def _score_counts(tp, fp, fn):
    if not len(tp):
        return 0.0
    denominator = 2 * tp + fp + fn
    per_activity = np.divide(
        2 * tp, denominator, out=np.ones(len(tp), dtype=float), where=denominator != 0
    )
    micro = 2 * tp.sum() / denominator.sum() if denominator.sum() else 1.0
    return float((per_activity.mean() + micro) / 2)


def masked_f1_score(y, predictions, mask):
    truth = np.asarray(y, dtype=bool)
    valid = np.asarray(mask, dtype=bool)
    pred = np.asarray(predictions, dtype=bool)
    present = valid.any(axis=0)
    tp = (truth & pred & valid).sum(axis=0)[present].astype(float)
    fp = (~truth & pred & valid).sum(axis=0)[present]
    fn = (truth & ~pred & valid).sum(axis=0)[present]
    return _score_counts(tp, fp, fn)


def _eligible(target):
    return len(target) >= 100 and target.sum() >= 10 and (1 - target).sum() >= 10


def fit_blend(member_probabilities, y, mask):
    """Fit nonnegative weights summing to one, separately for each activity."""
    p = np.asarray(member_probabilities, dtype=np.float64)
    truth = np.asarray(y, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    uniform = np.full(p.shape[0], 1 / p.shape[0])
    weights = np.tile(uniform, (N_LABELS, 1))
    for k in range(N_LABELS):
        target = truth[valid[:, k], k]
        if not _eligible(target):
            continue
        x = p[:, valid[:, k], k].T

        def objective(w):
            q = np.clip(x @ w, 1e-7, 1 - 1e-7)
            loss = -(target * np.log(q) + (1 - target) * np.log1p(-q)).mean()
            loss += 0.02 * np.square(w - uniform).sum()
            grad = x.T @ ((q - target) / (q * (1 - q))) / len(q)
            return loss, grad + 0.04 * (w - uniform)

        result = minimize(
            objective, uniform, jac=True, method="SLSQP", bounds=[(0, 1)] * len(uniform),
            constraints=[dict(type="eq", fun=lambda w: w.sum() - 1, jac=lambda w: np.ones_like(w))],
            options=dict(maxiter=150, ftol=1e-10),
        )
        if result.success and np.isfinite(result.x).all():
            weights[k] = np.maximum(result.x, 0)
            weights[k] /= weights[k].sum()
    return weights


def blend(member_probabilities, weights):
    p = np.asarray(member_probabilities, dtype=np.float64)
    return np.einsum("mnk,km->nk", p, weights).astype(np.float32)


def fit_thresholds(y, p, mask):
    """Optimize the mean of macro- and micro-F1, then shrink toward 0.25."""
    truth = np.asarray(y, dtype=bool)
    valid = np.asarray(mask, dtype=bool)
    p = np.asarray(p, dtype=float)
    grid = np.unique(np.r_[np.arange(0.025, 0.801, 0.025), ANCHOR])
    order = np.argsort(abs(grid - ANCHOR), kind="stable")
    tp = np.zeros((N_LABELS, len(grid)))
    fp, fn = np.zeros_like(tp), np.zeros_like(tp)
    eligible = []
    for k in range(N_LABELS):
        target = truth[valid[:, k], k, None]
        pred = p[valid[:, k], k, None] >= grid[None, :]
        tp[k] = (pred & target).sum(axis=0)
        fp[k] = (pred & ~target).sum(axis=0)
        fn[k] = (~pred & target).sum(axis=0)
        if _eligible(target):
            eligible.append(k)

    columns = np.arange(N_LABELS)
    present = valid.any(axis=0)

    def score(idx):
        return _score_counts(tp[columns, idx][present], fp[columns, idx][present], fn[columns, idx][present])

    chosen = np.full(N_LABELS, np.flatnonzero(grid == ANCHOR)[0], dtype=int)
    best = score(chosen)
    for _ in range(4):
        changed = False
        for k in eligible:
            for candidate in order:
                trial = chosen.copy()
                trial[k] = candidate
                value = score(trial)
                if value > best + 1e-8:
                    chosen, best, changed = trial, value, True
        if not changed:
            break
    return 0.5 * ANCHOR + 0.5 * grid[chosen]
