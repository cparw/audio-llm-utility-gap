#!/usr/bin/env python3
"""POD6 independent verifier (part 20).

Written from the job text only. Nothing is imported or executed from the
compute side; every number is rebuilt from per-clip files or refitted here.

AUC      explicit Mann-Whitney: every positive against every negative, a tie
         counts 0.5. Cross-checked by a hand-built trapezoid ROC.
CI       2000 draws, a fresh numpy default_rng(0) per cell, SPEAKERS resampled
         with replacement (rng.choice over np.unique(speaker)), a draw is used
         only if both classes are present, percentiles 2.5 / 97.5.
MEAN5    one speaker draw per replicate, all five per-repeat AUCs recomputed
         inside that draw and averaged.
PAIRED   one speaker draw per replicate, both AUCs recomputed inside it.
FOLDS    pod rule: speakers renumbered by their position in
         default_rng(seed).permutation(np.unique(str ids)), then GroupKFold(5)
         with numpy argsort(kind="stable") reversed (own implementation, and
         sklearn GroupKFold as installed, both checked).
"""
import csv, os, sys, json
import numpy as np

NB = 2000


# ----------------------------------------------------------------- AUC
def auc_mw(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    tot = 0.0
    for i in range(0, len(pos), 512):
        b = pos[i:i + 512][:, None]
        tot += float((b > neg[None, :]).sum()) + 0.5 * float((b == neg[None, :]).sum())
    return tot / (len(pos) * len(neg))


def auc_trap(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    P = float((y == 1).sum()); N = float((y == 0).sum())
    o = np.argsort(-s, kind="mergesort"); ss = s[o]; yy = y[o]
    tp = fp = 0.0; x0 = y0 = 0.0; area = 0.0; i = 0
    while i < len(ss):
        j = i
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        tp += float((yy[i:j] == 1).sum()); fp += float((yy[i:j] == 0).sum())
        x1, y1 = fp / N, tp / P
        area += (x1 - x0) * (y1 + y0) / 2.0
        x0, y0 = x1, y1; i = j
    return area


def auc_both(y, s):
    a = auc_mw(y, s); t = auc_trap(y, s)
    assert abs(a - t) < 1e-9, ("MW and trapezoid disagree", a, t)
    return a


def _groups(spk):
    u = np.unique(spk)
    return u, {p: np.where(spk == p)[0] for p in u}


def boot_ci(spk, y, s, nb=NB, seed=0):
    rng = np.random.default_rng(seed); u, idx = _groups(spk); v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        v.append(auc_mw(y[ii], s[ii]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def boot_mean5(spk, y, S, nb=NB, seed=0):
    rng = np.random.default_rng(seed); u, idx = _groups(spk); v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        v.append(np.mean([auc_mw(y[ii], s[ii]) for s in S]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def boot_paired(spk, y, s1, s2, nb=NB, seed=0):
    rng = np.random.default_rng(seed); u, idx = _groups(spk); v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        v.append(auc_mw(y[ii], s1[ii]) - auc_mw(y[ii], s2[ii]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


# ----------------------------------------------------------------- folds
def gkf_stable(groups, k=5):
    """GroupKFold assignment with a stable tie order. Returns fold per sample."""
    ug, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(inv)
    order = np.argsort(cnt, kind="stable")[::-1]
    load = np.zeros(k); g2f = np.zeros(len(ug), int)
    for gi in order:
        f = int(np.argmin(load)); load[f] += cnt[gi]; g2f[gi] = f
    return g2f[inv]


def renumber(spk_str, seed):
    rng = np.random.default_rng(seed); u = np.unique(spk_str)
    perm = {s: i for i, s in enumerate(rng.permutation(u))}
    return np.array([perm[s] for s in spk_str])


def pod_folds(spk_str, seed):
    return gkf_stable(renumber(np.asarray(spk_str).astype(str), seed))


# ----------------------------------------------------------------- tf-idf
def tfidf_oof(texts, y, fold, vec_kw=None, C=1.0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    texts = np.array(["" if (t is None or (isinstance(t, float) and np.isnan(t))) else str(t) for t in texts], dtype=object)
    oof = np.full(len(y), np.nan)
    for f in sorted(set(fold)):
        te = fold == f; tr = ~te
        v = TfidfVectorizer(**(vec_kw or {}))
        Xtr = v.fit_transform(texts[tr]); Xte = v.transform(texts[te])
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    assert not np.isnan(oof).any()
    return oof


def cell(spk, y, s):
    a = auc_both(y, s); lo, hi, nu = boot_ci(spk, y, s)
    return dict(auc=a, lo=lo, hi=hi, draws=nu, n=int(len(y)), n_spk=int(len(np.unique(spk))),
                n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()))


if __name__ == "__main__":
    print("library module; see the driver scripts in this folder")
