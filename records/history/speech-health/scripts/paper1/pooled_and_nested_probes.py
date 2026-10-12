#!/usr/bin/env python3
"""Pooled and nested encoder probes on the patient-onset windows of ADReSS-2020 and ADReSSo.
Same recipe as run_adress2020.py / extract_probe_score_patient.py: StandardScaler fitted on the
training fold, balanced logistic regression, StratifiedKFold(5, shuffle, seed 42). All speakers are
unique in both sets, so the folds are speaker-disjoint. 'peak' picks the layer after seeing every fold
(the number the per-layer curves show); 'nested' picks the layer inside each outer training fold."""
import csv, os, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
R = f"{D}/speech-health-repo/results/health"
def fit_score(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return c.decision_function(sc.transform(Xte))
def per_layer_cv(F, y, seed=42):
    out = []
    for L in range(F.shape[1]):
        a = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F[:, L, :], y):
            a.append(roc_auc_score(y[te], fit_score(F[tr, L, :], y[tr], F[te, L, :])))
        out.append(float(np.mean(a)))
    return out
def nested(F, y, seed=42):
    oof = np.zeros(len(y)); chosen = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F[:, 0, :], y):
        inner = per_layer_cv(F[tr], y[tr], seed)
        L = int(np.argmax(inner)); chosen.append(L)
        oof[te] = fit_score(F[tr, L, :], y[tr], F[te, L, :])
    return float(roc_auc_score(y, oof)), chosen
for name, npz, outdir in [("adress2020", f"{D}/adress2020/adress2020_encoder_states_patient.npz", f"{R}/adress2020"),
                          ("adresso", f"{D}/adresso/adresso_encoder_states_patient.npz", f"{R}/adresso")]:
    z = np.load(npz, allow_pickle=True); F, y = z["feats"], z["label"].astype(int)
    curve = per_layer_cv(F, y)
    Lpk = int(np.argmax(curve))
    nest, chosen = nested(F, y)
    with open(f"{outdir}/{name}_pooled_patient_probe.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "cv_auc"])
        for L, a in enumerate(curve): w.writerow([L, f"{a:.3f}"])
        w.writerow(["peak_layer", Lpk]); w.writerow(["peak_auc", f"{curve[Lpk]:.3f}"])
        w.writerow(["nested_auc", f"{nest:.3f}"]); w.writerow(["nested_layers", " ".join(map(str, chosen))])
    print(f"{name} patient window, n={len(y)}: peak layer {Lpk} AUC {curve[Lpk]:.3f}; nested AUC {nest:.3f} (layers {chosen})", flush=True)
