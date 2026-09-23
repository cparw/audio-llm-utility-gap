"""POD1 independent verification. Deliberately does NOT import lib_auc.
AUC is recomputed as the Mann-Whitney U statistic by explicit pairwise comparison
(ties counted as 0.5) instead of the sort/rank formula. The bootstrap is written
separately but follows the same protocol: 2000 draws, numpy default_rng(0) reseeded
per cell, SPEAKERS resampled with replacement, paired (one speaker draw per replicate,
both arms recomputed inside it), percentile 2.5/97.5."""
import sys, json, numpy as np, pandas as pd

def auc_pairwise(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    tot = 0.0
    for v in pos:                      # explicit pairwise count, independent of any rank sort
        tot += np.count_nonzero(v > neg) + 0.5 * np.count_nonzero(v == neg)
    return tot / (len(pos) * len(neg))

def paired_ci(y, s, spk, arm, draws=2000, seed=0):
    y = np.asarray(y); s = np.asarray(s, float); spk = np.asarray(spk); arm = np.asarray(arm)
    us = np.unique(spk); loc = {u: np.flatnonzero(spk == u) for u in us}
    g = np.random.default_rng(seed); C = []; A = []
    for _ in range(draws):
        sel = g.choice(us, size=us.size, replace=True)
        ix = np.concatenate([loc[u] for u in sel])
        yy = y[ix]; ss = s[ix]; aa = arm[ix]
        c = auc_pairwise(yy[aa == 'conflict'], ss[aa == 'conflict'])
        a = auc_pairwise(yy[aa == 'agreement'], ss[aa == 'agreement'])
        if np.isfinite(c) and np.isfinite(a): C.append(c); A.append(a)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(np.array(C)), q(np.array(A)), len(C)

def ci_single(y, s, spk, draws=2000, seed=0):
    y = np.asarray(y); s = np.asarray(s, float); spk = np.asarray(spk)
    us = np.unique(spk); loc = {u: np.flatnonzero(spk == u) for u in us}
    g = np.random.default_rng(seed); V = []
    for _ in range(draws):
        sel = g.choice(us, size=us.size, replace=True)
        ix = np.concatenate([loc[u] for u in sel])
        a = auc_pairwise(y[ix], s[ix])
        if np.isfinite(a): V.append(a)
    V = np.array(V)
    return float(np.percentile(V, 2.5)), float(np.percentile(V, 97.5)), len(V)
