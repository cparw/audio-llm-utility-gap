"""PART 23 independent verifier library (written from scratch, imports no code from the run scripts).

AUC: explicit Mann-Whitney over every positive x negative pair, ties count 0.5.
Cross-check: trapezoid area under the empirical ROC built from all distinct thresholds.
Bootstrap: 2000 draws, fresh numpy default_rng(0) per cell, speakers resampled with
replacement (speakers in sorted order, one clip per speaker), percentiles 2.5 / 97.5.
Paired: one speaker draw per replicate, both quantities recomputed on it.
Mean of five: all five per-repeat AUCs recomputed inside every draw, then averaged.
"""
import csv
import numpy as np

NB = 2000


def auc_mw(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    p = s[y == 1]
    n = s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(p) * len(n)))


def auc_trap(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    P = (y == 1).sum()
    N = (y == 0).sum()
    th = np.unique(s)[::-1]
    tpr = [0.0]
    fpr = [0.0]
    for t in th:
        tpr.append(((s >= t) & (y == 1)).sum() / P)
        fpr.append(((s >= t) & (y == 0)).sum() / N)
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def draws(speakers, nb=NB):
    """Yield clip index arrays. speakers: per-clip speaker ids (one clip per speaker)."""
    spk = np.asarray([str(x) for x in speakers])
    assert len(set(spk)) == len(spk), "expected one clip per speaker"
    order = sorted(set(spk))
    pos = {s: i for i, s in enumerate(spk)}
    map_idx = np.array([pos[s] for s in order])
    rng = np.random.default_rng(0)
    for _ in range(nb):
        pick = rng.choice(len(order), size=len(order), replace=True)
        yield map_idx[pick]


def pct(v):
    v = np.asarray(v, float)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_single(y, s, speakers):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    out = []
    for ii in draws(speakers):
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        out.append(auc_mw(yy, s[ii]))
    lo, hi = pct(out)
    return auc_mw(y, s), lo, hi, len(out)


def boot_mean5(y, S5, speakers):
    """S5: list of five score vectors. Value = mean of five per-repeat AUCs."""
    y = np.asarray(y).astype(int)
    S5 = [np.asarray(v, float) for v in S5]
    out = []
    for ii in draws(speakers):
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        out.append(np.mean([auc_mw(yy, v[ii]) for v in S5]))
    lo, hi = pct(out)
    return float(np.mean([auc_mw(y, v) for v in S5])), lo, hi, len(out)


def _val(y, A, ii=None):
    """A is either a vector (single AUC) or list of vectors (mean of per-repeat AUCs)."""
    if isinstance(A, (list, tuple)):
        if ii is None:
            return float(np.mean([auc_mw(y, v) for v in A]))
        return float(np.mean([auc_mw(y[ii], np.asarray(v, float)[ii]) for v in A]))
    A = np.asarray(A, float)
    return auc_mw(y, A) if ii is None else auc_mw(y[ii], A[ii])


def boot_paired(y, A, B, speakers, yB=None):
    """AUC(A) - AUC(B), A/B vector or list of five vectors; one draw per replicate."""
    y = np.asarray(y).astype(int)
    out = []
    for ii in draws(speakers):
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        out.append(_val(y, A, ii) - _val(y, B, ii))
    lo, hi = pct(out)
    return _val(y, A) - _val(y, B), lo, hi, len(out)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def agree4(a, b):
    return round(float(a), 4) == round(float(b), 4) or abs(float(a) - float(b)) < 5e-5
