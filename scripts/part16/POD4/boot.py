#!/usr/bin/env python3
"""AUC + speaker bootstrap CI, and PAIRED differences between arms.

AUC is the rank formula, recomputed from the per-clip scores every time.
BOOTSTRAP: 2000 draws, numpy.random.default_rng(0) reseeded per cell, resample
SPEAKERS with replacement (never clips), percentile 2.5/97.5.
PAIRED: the speaker list is drawn ONCE per replicate and both quantities are
recomputed inside that same draw.
A draw is usable only if the resampled clip set contains both classes.
"""
import csv, sys, json
import numpy as np

NB = 2000

def auc_rank(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(np.unique(y)) < 2: return np.nan
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

def load(path, score_col, key_col="clip"):
    rows = list(csv.DictReader(open(path)))
    return {r[key_col]: (r["speaker"], int(r["label"]), float(r[score_col])) for r in rows}

def arrays(d, keys):
    spk = np.array([d[k][0] for k in keys])
    y   = np.array([d[k][1] for k in keys])
    s   = np.array([d[k][2] for k in keys])
    return spk, y, s

def boot_ci(spk, y, s, nb=NB, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(spk)
    idx_by = {sp: np.where(spk == sp)[0] for sp in u}
    vals = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx_by[p] for p in pick])
        a = auc_rank(y[ii], s[ii])
        if not np.isnan(a): vals.append(a)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def boot_paired(spk, y, s1, s2, nb=NB, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(spk)
    idx_by = {sp: np.where(spk == sp)[0] for sp in u}
    d, a1l, a2l = [], [], []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx_by[p] for p in pick])
        a1 = auc_rank(y[ii], s1[ii]); a2 = auc_rank(y[ii], s2[ii])
        if np.isnan(a1) or np.isnan(a2): continue
        d.append(a2 - a1); a1l.append(a1); a2l.append(a2)
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)

if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    out = []
    cells = {}
    for c in spec["cells"]:
        d = load(c["file"], c["col"])
        cells[c["id"]] = d
        keys = sorted(d.keys())
        if "restrict" in c:
            keys = [k for k in keys if k in set(c["restrict"])]
        spk, y, s = arrays(d, keys)
        a = auc_rank(y, s)
        lo, hi, nu = boot_ci(spk, y, s, seed=c.get("seed", 0))
        out.append(dict(id=c["id"], what=c["what"], value=round(float(a), 4),
                        lo=round(lo, 4), hi=round(hi, 4), n=len(keys),
                        n_spk=int(len(np.unique(spk))), usable_draws=nu, file=c["file"]))
        print(f"{c['id']:34s} AUC {a:.4f}  [{lo:.4f}, {hi:.4f}]  n={len(keys)} spk={len(np.unique(spk))} draws={nu}", flush=True)
    for p in spec.get("paired", []):
        d1, d2 = cells[p["a"]], cells[p["b"]]
        keys = sorted(set(d1) & set(d2))
        if "restrict" in p: keys = [k for k in keys if k in set(p["restrict"])]
        spk, y, s1 = arrays(d1, keys)
        _, _, s2 = arrays(d2, keys)
        a1, a2 = auc_rank(y, s1), auc_rank(y, s2)
        lo, hi, nu = boot_paired(spk, y, s1, s2, seed=p.get("seed", 0))
        out.append(dict(id=p["id"], what=p["what"], value=round(float(a2 - a1), 4),
                        lo=round(lo, 4), hi=round(hi, 4), n=len(keys),
                        n_spk=int(len(np.unique(spk))), usable_draws=nu,
                        file=f"{p['a']}->{p['b']}", a_auc=round(float(a1), 4), b_auc=round(float(a2), 4)))
        print(f"{p['id']:34s} PAIRED d {a2-a1:+.4f}  [{lo:+.4f}, {hi:+.4f}]  ({a1:.4f} -> {a2:.4f})  n={len(keys)} draws={nu}", flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print("BOOT DONE ->", sys.argv[2], flush=True)
