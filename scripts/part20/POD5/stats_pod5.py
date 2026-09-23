"""PART20 POD5 statistics. AUC by the rank formula (scipy rankdata, ties averaged). Speaker bootstrap: 2000 draws,
fresh numpy.random.default_rng(0) per cell, speakers (np.unique order) resampled with replacement, percentile 2.5/97.5;
paired = one speaker draw per replicate for both sides. For multi-repeat cells the statistic is the mean over the
repeat columns of their AUCs, recomputed inside each draw."""
import numpy as np, hashlib, json, os
from scipy.stats import rankdata
B = 2000
def rauc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(np.asarray(s, float)); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def stat(y, S):  # S: (n, R)
    S = np.asarray(S, float).reshape(len(y), -1)
    return float(np.mean([rauc(y, S[:, j]) for j in range(S.shape[1])]))
def _groups(spk):
    u, inv = np.unique(np.asarray(spk).astype(str), return_inverse=True)
    return u, [np.where(inv == i)[0] for i in range(len(u))]
def boot(y, S, spk, S2=None):
    """returns point, lo, hi, valid; with S2: paired difference stat(S) - stat(S2)."""
    y = np.asarray(y).astype(int); u, idx = _groups(spk)
    f = (lambda r: stat(y[r], S[r]) - stat(y[r], S2[r])) if S2 is not None else (lambda r: stat(y[r], S[r]))
    S = np.asarray(S, float).reshape(len(y), -1)
    if S2 is not None: S2 = np.asarray(S2, float).reshape(len(y), -1)
    point = f(np.arange(len(y)))
    rng = np.random.default_rng(0); vals = []
    for _ in range(B):
        d = rng.integers(0, len(u), size=len(u)); r = np.concatenate([idx[i] for i in d])
        if len(np.unique(y[r])) < 2: vals.append(np.nan); continue
        vals.append(f(r))
    v = np.array(vals); ok = v[~np.isnan(v)]
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return float(point), float(lo), float(hi), int(len(ok))
def sha1mb(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh: h.update(fh.read(1 << 20))
    return h.hexdigest()
def paste(id_, what, v, lo, hi, n, nspk, f, diff=False):
    if diff:
        vs = "interval excludes zero" if (lo > 0 or hi < 0) else "interval includes zero"
    else:
        vs = "above 0.5" if v > 0.5 else "below 0.5"
    line = f"{id_} | {what} | {v:.4f} [{lo:.4f}, {hi:.4f}] | n={n} n_spk={nspk} | {f} | {v:.2f} [{lo:.2f}, {hi:.2f}] {vs}"
    row = "\t".join(map(str, [id_, what, f"{v:.4f}", f"{lo:.4f}", f"{hi:.4f}", n, nspk, f, f"{v:.2f} [{lo:.2f}, {hi:.2f}]", vs]))
    return line, row
