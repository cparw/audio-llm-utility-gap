#!/usr/bin/env python3
"""M11 robustness: the cross-system disagreement mean is dominated by clips whose
30 s window caught almost no speech (tiny WER denominators). Recompute (a) the
median, (b) the mean restricted to clips with >= 20 words in the window, and
(c) a pooled rate sum(S+D+I)/sum(N), which weights by words not by clips."""
import numpy as np, pandas as pd, scipy.stats as st
CSV  = "scores/part16/M/M11_kcl_wer.csv"
ROWS = "reports/part16/rows/M11.tsv"
NB = 2000
d = pd.read_csv(CSV)
n, nspk = len(d), d.spk.nunique()
spks = np.array(sorted(d.spk.unique()))
rowsof = {s: np.flatnonzero(d.spk.values == s) for s in spks}
lines = []

def draws(seed=0):
    rng = np.random.default_rng(seed)
    return [np.concatenate([rowsof[s] for s in rng.choice(spks, len(spks), True)]) for _ in range(NB)]

def report(tag, sub, col, stat):
    D = draws()
    out = {}
    for g in ("PD", "control"):
        m = (sub.group.values == g)
        v = sub[col].values.astype(float)[m]
        pt = float(stat(v))
        b = []
        for i in D:
            k = np.intersect1d(i, np.flatnonzero((d.group.values == g)))
            k = np.array([x for x in i if d.group.values[x] == g and x in set(sub.index)])
            if len(k): b.append(float(stat(d[col].values.astype(float)[k])))
        lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) if b else (np.nan, np.nan)
        out[g] = pt
        s = (f"M11 {tag} {g:<8} = {pt:.4f} [{lo:.4f}, {hi:.4f}]  n={int(m.sum())} "
             f"n_speakers={int(m.sum())} usable_draws={len(b)}/{NB}  file={CSV}")
        print(s)
        lines.append(f"M11\t{tag}_{g}\t{pt:.4f}\t{lo:.4f}\t{hi:.4f}\t{int(m.sum())}\t{int(m.sum())}\t{CSV}")
    print(f"M11 {tag} PD minus control = {out['PD']-out['control']:.4f}")
    lines.append(f"M11\t{tag}_diff_PD_minus_control\t{out['PD']-out['control']:.4f}\t\t\t{n}\t{nspk}\t{CSV}")

print("[B-robust] cross-system disagreement, large-v3 vs medium")
report("B_xsys_disagree_MEDIAN", d, "xsys_disagree", np.median)
sub = d[d.xsys_n_ref_words >= 20]
print(f"\n  restricted to clips with >= 20 words in the window: n={len(sub)} "
      f"(PD={int((sub.group=='PD').sum())}, control={int((sub.group=='control').sum())})")
report("B_xsys_disagree_ge20w_MEAN", sub, "xsys_disagree", np.mean)

print("\n  pooled, word-weighted: sum(S+D+I)/sum(N)")
for g in ("PD", "control"):
    s = d[d.group == g]
    p = float((s.xsys_S + s.xsys_D + s.xsys_I).sum() / s.xsys_n_ref_words.sum())
    D = draws(); b = []
    for i in D:
        k = np.array([x for x in i if d.group.values[x] == g])
        if len(k):
            b.append(float((d.xsys_S.values[k]+d.xsys_D.values[k]+d.xsys_I.values[k]).sum()
                           / d.xsys_n_ref_words.values[k].sum()))
    lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
    print(f"M11 B_xsys_disagree_POOLED {g:<8} = {p:.4f} [{lo:.4f}, {hi:.4f}]  n={len(s)} "
          f"n_speakers={len(s)} usable_draws={len(b)}/{NB}  file={CSV}")
    lines.append(f"M11\tB_xsys_disagree_POOLED_{g}\t{p:.4f}\t{lo:.4f}\t{hi:.4f}\t{len(s)}\t{len(s)}\t{CSV}")

print("\n  worst inflators (smallest denominators):")
print(d.nsmallest(4, "xsys_n_ref_words")[["file","group","xsys_n_ref_words","xsys_S","xsys_D","xsys_I","xsys_disagree"]].to_string(index=False))

with open(ROWS, "a") as f: f.write("\n".join(lines) + "\n")
print(f"\nappended {len(lines)} rows to {ROWS}")
