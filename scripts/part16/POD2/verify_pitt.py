"""SECOND, independent recomputation. Different code path from stats.py:
AUC by explicit Mann-Whitney U from a merge over sorted unique score values (no rank array),
bootstrap by building an index matrix rather than concatenating per draw.
Same documented protocol: 2000 draws, default_rng(0) per cell, speakers resampled, 2.5/97.5 pct.
usage: verify_pitt.py <csv> <label_col> <score_col> <spk_col> <set_col|NONE>"""
import sys, csv, json, os
import numpy as np

def auc_u(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    vals = np.unique(s)
    # for each unique value: how many negatives are strictly below, and how many equal
    neg_sorted = np.sort(neg)
    below = np.searchsorted(neg_sorted, vals, side='left')
    equal = np.searchsorted(neg_sorted, vals, side='right') - below
    idx = np.searchsorted(vals, pos)
    u = (below[idx] + 0.5 * equal[idx]).sum()
    return float(u) / (len(pos) * len(neg))

def draws(spk, n, seed):
    rng = np.random.default_rng(seed)
    u = np.unique(np.asarray(spk))
    return [rng.choice(u, size=len(u), replace=True) for _ in range(n)]

def ci_of(y, s, spk, n=2000, seed=0, sub=None):
    y = np.asarray(y); s = np.asarray(s); spk = np.asarray(spk)
    pos_of = {g: np.flatnonzero(spk == g) for g in np.unique(spk)}
    acc = []
    for pick in draws(spk, n, seed):
        ii = np.hstack([pos_of[g] for g in pick])
        if sub is not None: ii = ii[np.asarray(sub, bool)[ii]]
        if len(ii) == 0: continue
        a = auc_u(y[ii], s[ii])
        if a == a: acc.append(a)
    acc = np.asarray(acc)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5)), int(len(acc))

f, lc, sc, kc, setc = sys.argv[1:6]
R = list(csv.DictReader(open(f)))
y = np.array([int(float(r[lc])) for r in R]); s = np.array([float(r[sc]) for r in R])
spk = np.array([r[kc] for r in R])
o = {"file": os.path.abspath(f), "n": len(R), "n_spk": int(len(np.unique(spk))),
     "auc_all": round(auc_u(y, s), 6)}
lo, hi, k = ci_of(y, s, spk); o["auc_all_ci"] = [round(lo, 6), round(hi, 6)]; o["auc_all_draws"] = k
if setc != "NONE":
    st = np.array([r[setc] for r in R])
    for arm in ["conflict", "agreement"]:
        m = st == arm
        o[f"auc_{arm}"] = round(auc_u(y[m], s[m]), 6)
        # primary protocol: resample within the arm's own speaker universe
        l2, h2, k2 = ci_of(y[m], s[m], spk[m])
        o[f"auc_{arm}_ci"] = [round(l2, 6), round(h2, 6)]; o[f"auc_{arm}_draws"] = k2
        # sensitivity: resample all speakers then subset to the arm
        l3, h3, k3 = ci_of(y, s, spk, sub=m)
        o[f"auc_{arm}_ci_allspk"] = [round(l3, 6), round(h3, 6)]
    o["diff_conflict_minus_agreement"] = round(o["auc_conflict"] - o["auc_agreement"], 6)
    # paired: one speaker draw per replicate, both arms inside it
    pos_of = {g: np.flatnonzero(spk == g) for g in np.unique(spk)}
    mc = st == "conflict"; ma = st == "agreement"; d = []
    for pick in draws(spk, 2000, 0):
        ii = np.hstack([pos_of[g] for g in pick])
        ic = ii[mc[ii]]; ia = ii[ma[ii]]
        if len(ic) == 0 or len(ia) == 0: continue
        x1 = auc_u(y[ic], s[ic]); x2 = auc_u(y[ia], s[ia])
        if x1 == x1 and x2 == x2: d.append(x1 - x2)
    d = np.asarray(d)
    o["diff_ci_paired"] = [round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)]
    o["diff_draws"] = int(len(d))
print(json.dumps(o, indent=1))
