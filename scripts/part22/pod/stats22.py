"""PART 22 stats: rank AUC + 2000-draw speaker bootstrap (probe_boot.py convention)."""
import csv, numpy as np
def auc_rank(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); sv = s[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j+1] == sv[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))
def boot(y, spk, vecA, vecB=None, NB=2000):
    y = np.asarray(y); spk = list(spk)
    uspk = np.array(sorted(set(spk))); idx_of = {s: i for i, s in enumerate(spk)}
    rng = np.random.default_rng(0)
    va, vd = [], []
    for _ in range(NB):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.array([idx_of[uspk[p]] for p in pick])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        a = auc_rank(yy, vecA[ii]); va.append(a)
        if vecB is not None: vd.append(a - auc_rank(yy, vecB[ii]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out = dict(n_draws=NB, usable=len(va), auc_lo=q(va)[0], auc_hi=q(va)[1])
    if vecB is not None:
        out.update(diff_lo=q(vd)[0], diff_hi=q(vd)[1], diff_point=float(np.mean(vd)))
    return out
if __name__ == "__main__":
    import sys
    R = list(csv.DictReader(open(sys.argv[1])))
    y = np.array([int(r["label"]) for r in R]); spk = [r["pid"] for r in R]
    p = np.array([float(r["p_yes_answer"]) for r in R])
    print("n", len(R), "pos", y.sum(), "AUC %.4f" % auc_rank(y, p), boot(y, spk, p))
