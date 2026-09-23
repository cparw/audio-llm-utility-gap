#!/usr/bin/env python3
"""4a: AUC of each quiet-frame channel feature alone, and of all five together
with a GroupKFold(5)-by-speaker logistic probe. BEFORE and AFTER normalisation.

A single feature has a direction, so both the raw AUC and the direction-free
max(auc, 1-auc) are reported; the acceptance test is about SEPARATION, so the
direction-free value is the one that must fall below 0.60.

usage: feat_probe.py channel_norm_<tag>.csv <tag> <out.json>
"""
import csv, json, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

FEATS = ["noise_floor", "snr", "centroid", "rolloff", "tilt"]
CSV, TAG, OUT = sys.argv[1:4]

def auc_rank(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    rr = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        rr[i:j + 1] = rr[i:j + 1].mean(); i = j + 1
    r = np.empty(len(s)); r[o] = rr
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def boot_ci(spk, y, s, nb=2000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(spk)
    idx = {p: np.where(spk == p)[0] for p in u}; v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(np.unique(y[ii])) < 2: continue
        v.append(auc_rank(y[ii], s[ii]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

rows = list(csv.DictReader(open(CSV)))
y = np.array([int(r["label"]) for r in rows])
spk = np.array([r["spk"] for r in rows])
seen = {}
for s in spk:
    if s not in seen: seen[s] = len(seen)
g = np.array([seen[s] for s in spk])
print(f"{TAG}: n={len(rows)} speakers={len(seen)} pos={int(y.sum())}", flush=True)

res = {"tag": TAG, "n": len(rows), "n_speakers": len(seen), "single": {}, "combined": {}}
for when in ("before", "after"):
    res["single"][when] = {}
    for f in FEATS:
        x = np.array([float(r[f"{f}_{when}"]) for r in rows])
        a = float(auc_rank(y, x)); ab = max(a, 1 - a)
        lo, hi, nu = boot_ci(spk, y, x if a >= 0.5 else -x)
        res["single"][when][f] = dict(auc_raw=round(a, 4), auc_dirfree=round(ab, 4),
                                      lo=round(lo, 4), hi=round(hi, 4), usable_draws=nu)
        print(f"  {when:6s} {f:12s} AUC {a:.4f}  dir-free {ab:.4f}  [{lo:.4f}, {hi:.4f}]", flush=True)
    # all five together
    X = np.column_stack([[float(r[f"{f}_{when}"]) for r in rows] for f in FEATS])
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    a = float(auc_rank(y, oof)); lo, hi, nu = boot_ci(spk, y, oof)
    res["combined"][when] = dict(auc=round(a, 4), lo=round(lo, 4), hi=round(hi, 4), usable_draws=nu)
    print(f"  {when:6s} FIVE TOGETHER (GroupKFold5 by speaker) AUC {a:.4f}  [{lo:.4f}, {hi:.4f}]  draws={nu}", flush=True)
    with open(CSV.replace(".csv", f"_5feat_{when}_oof.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_probe"])
        for i, r in enumerate(rows): w.writerow([r["clip"], r["spk"], int(y[i]), float(oof[i])])

af = res["combined"]["after"]["auc"]
res["acceptance_test"] = dict(threshold=0.60, five_feature_auc_after=af, passed=bool(af < 0.60))
print(f"\n{TAG}: ACCEPTANCE TEST -- five-feature AUC after normalisation = {af:.4f}; "
      f"threshold 0.60; {'PASSED' if af < 0.60 else 'FAILED'}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("FEATPROBE DONE ->", OUT, flush=True)
