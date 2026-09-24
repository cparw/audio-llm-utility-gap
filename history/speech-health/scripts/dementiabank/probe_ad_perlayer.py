import numpy as np, csv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

d = np.load("/Volumes/G-Drive Pro/DementiaBank/ad_encoder_states.npz", allow_pickle=True)
X, y, spk = d["feats"], d["label"], d["spk"]
print("segments", X.shape, "speakers", len(set(spk)), "pos", int(y.sum()))

gkf = GroupKFold(n_splits=5)
rows = []
for L in range(X.shape[1]):
    XL = X[:, L, :]
    aucs = []
    oof = np.zeros(len(y))
    for tr, te in gkf.split(XL, y, groups=spk):
        sc = StandardScaler().fit(XL[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(XL[tr]), y[tr])
        p = clf.predict_proba(sc.transform(XL[te]))[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
    rows.append((L, float(np.mean(aucs)), float(np.std(aucs)), float(roc_auc_score(y, oof))))
    print(f"layer {L:2d}  auc_mean {rows[-1][1]:.3f}  std {rows[-1][2]:.3f}  oof {rows[-1][3]:.3f}", flush=True)

with open("/Volumes/G-Drive Pro/DementiaBank/ad_encoder_perlayer.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layer", "auc_mean", "auc_std", "auc_oof"]); w.writerows(rows)

# model's own answer AUC on the same segments
import collections
scores = {}
with open("/Volumes/G-Drive Pro/DementiaBank/ad_scores_qwen2audio.csv") as f:
    r = csv.DictReader(f)
    cols = r.fieldnames
    print("score cols:", cols)
    for row in r:
        scores[row[cols[0]]] = row
