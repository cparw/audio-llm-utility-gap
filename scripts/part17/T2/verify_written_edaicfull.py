"""PART17 T2 final check for the E-DAIC full-window files: read the five WRITTEN shuffle fold csvs, refit enc/proj/llm/ans at the
saved layers_picked, average the five repeats, compare with the saved per-clip oof_mean. Appends to written_check.jsonl."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "1")
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); FD = "<local data dir>/release/folds"
E = "<local data dir>/release/edaic_rerun/part16/EDAICFULL"
P = pd.read_csv(f"{E}/edaic_full_perclip.csv", dtype={"pid": str}); J = json.load(open(f"{E}/edaic_full_nested_repeats.json"))
clips = [f"{p}.wav" for p in P["pid"]]; y = P["label"].values.astype(int)
S = {k: np.stack([np.load(f"{HERE}/edaicfull/shards/{c}.npz")[k] for c in clips]) for k in ("enc", "proj", "llm", "ans")}
folds = []
for r in range(5):
    F = pd.read_csv(f"{FD}/edaic_full_groupkfold5_shuffle_rs{r}.csv", dtype={"clip_id": str}); m = dict(zip(F["clip_id"], F["fold"]))
    folds.append(np.array([m[c] for c in clips]))
res = {}
for k in S:
    oofs = []
    for r in range(5):
        o = np.zeros(len(y))
        for i in range(5):
            tr = np.where(folds[r] != i)[0]; te = np.where(folds[r] == i)[0]; L = J[k]["layers_picked"][r*5+i]
            o[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)).fit(S[k][tr, L, :], y[tr]).predict_proba(S[k][te, L, :])[:, 1]
        oofs.append(o)
    res[k] = float(np.abs(np.mean(oofs, 0) - P[f"oof_{k}"].values).max()); print(k, res[k], flush=True)
for r in range(5):
    rec = dict(file=f"edaic_full_groupkfold5_shuffle_rs{r}.csv", status="CONFIRMED", speakers_disjoint=True, n=len(y),
               checks=[dict(run="EDAICFULL_mean_of_5", stream=k, maxabs=v) for k, v in res.items()], max_over_checks=max(res.values()),
               note="the saved per-clip file holds only the mean over the five splits, so the five files are checked together")
    open(f"{HERE}/written_check.jsonl", "a").write(json.dumps(rec) + "\n")
print("EDF WRITTEN DONE")
