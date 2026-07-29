import os, csv, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def load(npz, tasks):
    d = np.load(npz, allow_pickle=True); task = d["task"]; lab = d["label"].astype(int)
    keep = set(tasks.split(",")); m = np.array([t in keep for t in task])
    return d, np.where(m)[0], lab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--tasks", default="read"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dtr, itr, ltr = load(a.train, a.tasks); dte, ite, lte = load(a.test, a.tasks)
    L = int(dtr["n_layers"]); ytr = ltr[itr]; yte = lte[ite]
    print(f"train n={len(itr)} (PD {int(ytr.sum())}/HC {int((ytr==0).sum())}) -> test n={len(ite)} (PD {int(yte.sum())}/HC {int((yte==0).sum())}) tasks={a.tasks}", flush=True)
    rows = []
    for li in range(L):
        Xtr = dtr[f"layer_{li:02d}"][itr]; Xte = dte[f"layer_{li:02d}"][ite]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(Xtr, ytr); p = clf.predict_proba(Xte)[:, 1]
        ba = balanced_accuracy_score(yte, (p >= 0.5).astype(int))
        try: auc = roc_auc_score(yte, p)
        except Exception: auc = float("nan")
        rows.append((li, ba, auc)); print(f"  layer {li:02d} bAcc {ba:.3f} AUC {auc:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["layer","balacc","auc"]); w.writerows(rows)
    valid = [r for r in rows if not np.isnan(r[2])]
    best = max(valid, key=lambda r: r[2])
    print(f"BEST layer {best[0]}: bAcc {best[1]:.3f} AUC {best[2]:.3f}  -> {a.out}", flush=True)

if __name__ == "__main__": main()
