"""PART20 POD1b stats: rank AUC (scipy rankdata, ties averaged), speaker bootstrap 2000 draws,
fresh numpy.random.default_rng(0) per cell, percentile 2.5/97.5; arm cells resample that arm's own speakers;
paired difference = one speaker draw per replicate over all speakers, both arms recomputed in it.
usage: stats20b.py OOF_CSV SCORE_COL -> json on stdout"""
import sys, csv, json, numpy as np
from scipy.stats import rankdata
def auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def boot(y, s, spk, n=2000):
    rng = np.random.default_rng(0); u = np.unique(spk); idx = {g: np.where(spk == g)[0] for g in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(u, size=len(u), replace=True)])
        a = auc(y[ii], s[ii])
        if not np.isnan(a): out.append(a)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)
def boot_pair(y, s, spk, ma, mb, n=2000):
    rng = np.random.default_rng(0); u = np.unique(spk); idx = {g: np.where(spk == g)[0] for g in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(u, size=len(u), replace=True)])
        ia, ib = ii[ma[ii]], ii[mb[ii]]
        if len(ia) == 0 or len(ib) == 0: continue
        a, b = auc(y[ia], s[ia]), auc(y[ib], s[ib])
        if not (np.isnan(a) or np.isnan(b)): out.append(a - b)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)
if __name__ == "__main__":
    f, col = sys.argv[1], sys.argv[2]
    R = list(csv.DictReader(open(f)))
    y = np.array([int(r["label"]) for r in R]); s = np.array([float(r[col]) for r in R])
    spk = np.array([r["speaker"] for r in R]); arm = np.array([r["set"] for r in R])
    res = {"file": f, "score_col": col}
    for name, m in (("overall", np.ones(len(R), bool)), ("conflict", arm == "conflict"), ("agreement", arm == "agreement")):
        lo, hi, nd = boot(y[m], s[m], spk[m])
        res[name] = {"auc": auc(y[m], s[m]), "lo": lo, "hi": hi, "n": int(m.sum()),
                     "n_spk": int(len(set(spk[m]))), "draws": nd}
    lo, hi, nd = boot_pair(y, s, spk, arm == "conflict", arm == "agreement")
    res["conflict_minus_agreement"] = {"diff": res["conflict"]["auc"] - res["agreement"]["auc"], "lo": lo, "hi": hi,
                                       "n": len(R), "n_spk": int(len(set(spk))), "draws": nd}
    if "answer_mass" in R[0]:
        am = np.array([float(r["answer_mass"]) for r in R]); res["answer_mass"] = {"median": float(np.median(am)), "min": float(am.min())}
    print(json.dumps(res))
