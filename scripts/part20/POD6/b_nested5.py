"""PART20 POD6 job B: Qwen2.5-Omni ENCODER nested probe, five renumbered splits, POD tie rule.
Per seed s in 0..4: speakers renumbered with numpy.random.default_rng(s).permutation(numpy.unique(ids)); outer
GroupKFold(5) with numpy.argsort(counts, kind='stable') reversed (the pod tie rule, checked to reproduce every saved
_pod fold file); inner GroupKFold(4) (same rule) on the training speakers picks the encoder layer by mean inner AUC;
refit StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced') at that layer; out-of-fold scores.
Checkpoint after every outer fold. Also runs the single split (no renumbering) for the single-split comparison.
usage: b_nested5.py STATES.npz OUTPREFIX [KEY=enc]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import sys, json, time, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
NPZ, OUTP = sys.argv[1:3]; KEY = sys.argv[3] if len(sys.argv) > 3 else "enc"
z = np.load(NPZ, allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
X = z[KEY].astype(np.float32); nl = X.shape[1]
NJ = max(2, (os.cpu_count() or 8) - 4)

def auc_rank(yy, s):
    r = rankdata(s); n1 = int((yy == 1).sum()); n0 = int((yy == 0).sum())
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def gkf(groups, k):
    uniq, inv = np.unique(groups, return_inverse=True); counts = np.bincount(inv)
    order = np.argsort(counts, kind="stable")[::-1]; fg = np.zeros(len(uniq), int); load = np.zeros(k)
    for gi in order:
        f = int(np.argmin(load)); fg[gi] = f; load[f] += counts[gi]
    fold = fg[inv]
    return [(np.where(fold != j)[0], np.where(fold == j)[0]) for j in range(k)], fold

def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]

def inner(tr, l, g):
    s = 0.0; sp, _ = gkf(g[tr], 4)
    for itr, ite in sp:
        p = fitpred(X[tr][itr, l], y[tr][itr], X[tr][ite, l])
        s += auc_rank(y[tr][ite], p) if len(set(y[tr][ite])) > 1 else 0.5
    return s / 4

CK = OUTP + "_ckpt.json"
ck = json.load(open(CK)) if os.path.exists(CK) else {}
t0 = time.time()
for tag in ["single", "seed0", "seed1", "seed2", "seed3", "seed4"]:
    if tag == "single":
        g = spk
    else:
        seed = int(tag[4:]); rng = np.random.default_rng(seed); u = np.unique(spk)
        perm = {s: i for i, s in enumerate(rng.permutation(u))}; g = np.array([perm[s] for s in spk])
    splits, fold = gkf(g, 5)
    st = ck.get(tag, {"oof": [None] * len(y), "layers": [None] * 5, "fold": fold.tolist()})
    assert st["fold"] == fold.tolist(), f"fold mismatch on resume {tag}"
    for k, (tr, te) in enumerate(splits):
        if st["layers"][k] is not None: continue
        sc = Parallel(n_jobs=min(nl, NJ))(delayed(inner)(tr, l, g) for l in range(nl))
        best = int(np.argmax(sc))
        p = fitpred(X[tr][:, best], y[tr], X[te][:, best])
        for i, v in zip(te, p): st["oof"][int(i)] = float(v)
        st["layers"][k] = best; ck[tag] = st
        json.dump(ck, open(CK + ".tmp", "w")); os.replace(CK + ".tmp", CK)
        print(f"{tag} fold {k} layer {best} inner {max(sc):.4f}  {time.time()-t0:.0f}s", flush=True)
    a = auc_rank(y, np.array(st["oof"])); st["auc"] = a; ck[tag] = st
    json.dump(ck, open(CK + ".tmp", "w")); os.replace(CK + ".tmp", CK)
    print(f"RESULT {tag} {KEY} AUC {a:.4f} layers {st['layers']}", flush=True)
import csv
with open(OUTP + f"_{KEY}_nested5_oof.csv", "w", newline="") as fh:
    w = csv.writer(fh); tags = ["single", "seed0", "seed1", "seed2", "seed3", "seed4"]
    w.writerow(["clip", "speaker", "label"] + [f"p_{t}" for t in tags] + [f"fold_{t}" for t in tags])
    for i in range(len(y)):
        w.writerow([names[i], spk[i], int(y[i])] + [ck[t]["oof"][i] for t in tags] + [ck[t]["fold"][i] for t in tags])
res = {t: dict(auc=round(ck[t]["auc"], 6), layers=ck[t]["layers"]) for t in ck}
res["mean5"] = round(float(np.mean([ck[f"seed{s}"]["auc"] for s in range(5)])), 6)
json.dump(res, open(OUTP + f"_{KEY}_nested5.json", "w"), indent=1)
print("NESTED5 DONE", json.dumps(res), flush=True)
