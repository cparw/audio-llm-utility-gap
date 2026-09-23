"""PART20 POD5: nested layer selection exactly as nested_repeats_all.py (outer 5 folds, inner 4 folds on the training
speakers pick the layer by mean inner AUC, refit on the outer training fold at that layer, out-of-fold scores;
StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'), float32 features), except that the outer
AND inner splits are READ from pitt_folds_part20.json (outer = the saved fold files; inner = GroupKFold(4) with the
same tie rule, derived on the Mac by prep_folds.py). Rows are matched by clip_id (orig window id).
usage: nested_p20.py STATES.npz FOLDS.json OUTPREFIX STREAM:SETPREFIX [...]   e.g. enc:mac_seed llm:pod_seed"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import sys, json, time, csv, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
NPZ, FJ, OUTP = sys.argv[1:4]; JOBS = [a.split(":") for a in sys.argv[4:]]
NJ = int(os.environ.get("NJ", "136"))
z = np.load(NPZ, allow_pickle=True); y = z["label"].astype(int); names = z["name"].astype(str)
F = json.load(open(FJ)); idx = {c: i for i, c in enumerate(F["clip_id"])}
pos = np.array([idx[n] for n in names]); assert len(set(pos)) == len(names)
assert (np.array(F["speaker_id"])[pos] == z["spk"].astype(str)).all(), "speaker mismatch"
def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]
def rauc(yy, s):
    r = rankdata(s); n1 = int(yy.sum()); n0 = len(yy) - n1; return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
X = {k: z[k].astype(np.float32) for k in set(j[0] for j in JOBS)}
def inner(key, setname, k, l):
    S = F["sets"][setname]; outer = np.array(S["outer"])[pos]; inn = np.array(S["inner"][str(k)])[pos]
    tr = np.where(outer != k)[0]; Xs = X[key]; s = 0.0
    for j in range(4):
        itr, ite = tr[inn[tr] != j], tr[inn[tr] == j]
        p = fitpred(Xs[itr, l], y[itr], Xs[ite, l])
        s += roc_auc_score(y[ite], p) if len(set(y[ite])) > 1 else 0.5
    return key, setname, k, l, s / 4
def outerfit(key, setname, k, l):
    S = F["sets"][setname]; outer = np.array(S["outer"])[pos]
    tr, te = np.where(outer != k)[0], np.where(outer == k)[0]
    return key, setname, k, te, fitpred(X[key][tr][:, l], y[tr], X[key][te][:, l])
t0 = time.time()
tasks = []
for key, pref in JOBS:
    sets = [s for s in F["sets"] if s.startswith(pref)] if not pref.endswith("single") else [pref]
    for s in sets:
        nl = X[key].shape[1]
        for k in range(5):
            for l in range(nl): tasks.append((key, s, k, l))
print(f"{len(tasks)} inner tasks on {NJ} workers", flush=True)
res = Parallel(n_jobs=NJ, batch_size=1)(delayed(inner)(*t) for t in tasks)
score = {}
for key, s, k, l, v in res: score.setdefault((key, s), np.full((5, X[key].shape[1]), -1.0))[k, l] = v
chosen = {ks: [int(np.argmax(score[ks][k])) for k in range(5)] for ks in score}
print(f"inner done {time.time()-t0:.0f}s", flush=True)
res2 = Parallel(n_jobs=NJ)(delayed(outerfit)(key, s, k, chosen[(key, s)][k]) for (key, s) in chosen for k in range(5))
oof = {ks: np.full(len(y), np.nan) for ks in chosen}
for key, s, k, te, p in res2: oof[(key, s)][te] = p
summary = {}
for key, pref in JOBS:
    sets = sorted(s for (kk, s) in chosen if kk == key and (s.startswith(pref)))
    if not sets: continue
    cols = {s: oof[(key, s)] for s in sets}
    for s in sets: assert not np.isnan(cols[s]).any()
    aucs = [rauc(y, cols[s]) for s in sets]
    tag = f"{key}_{pref}"
    with open(f"{OUTP}_{tag}_oof.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip_id", "cut_name", "speaker", "label", "set"] + [f"p_{s}" for s in sets])
        for i in range(len(y)):
            w.writerow([names[i], str(z["cut_name"][i]) if "cut_name" in z.files else names[i], str(z["spk"][i]), int(y[i]),
                        str(z["set"][i]) if "set" in z.files else ""] + [repr(float(cols[s][i])) for s in sets])
    summary[tag] = {"stream": key, "fold_sets": sets, "fold_files": [F["sets"][s]["file"] for s in sets], "per_repeat_auc": [round(a, 4) for a in aucs],
                    "mean_auc": round(float(np.mean(aucs)), 4), "chosen_layers": {s: chosen[(key, s)] for s in sets}, "n": int(len(y)),
                    "n_layers": int(X[key].shape[1]), "oof_csv": os.path.abspath(f"{OUTP}_{tag}_oof.csv")}
    print(f"NESTED {tag}: mean {np.mean(aucs):.4f} per-repeat {[round(a,4) for a in aucs]} layers {[chosen[(key,s)] for s in sets]}", flush=True)
json.dump({"states": os.path.abspath(NPZ), "folds_json": os.path.abspath(FJ), "results": summary, "seconds": round(time.time() - t0)},
          open(f"{OUTP}_nested_summary.json", "w"), indent=1)
print("NESTED DONE", flush=True)
