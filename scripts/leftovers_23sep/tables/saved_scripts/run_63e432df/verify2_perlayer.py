import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R = "<local data dir>/release_from_mac/scores/part23"
SHARD_DIR = f"{R}/pull/p23-a-probes/p23a/shards"

def strip_np(x):
    if isinstance(x, str) and x.startswith("np.float64("):
        return float(x[len("np.float64("):-1])
    return float(x)

A2 = pd.read_csv(f"{R}/pull/p23-a-probes/p23a/out/A2_edaic_whole_zeroshot_perclip.csv")
A2 = A2.sort_values("pid").reset_index(drop=True)
pids = A2.pid.values.astype(int)
labels = A2.label.values.astype(int)
N = len(pids)
print("N clips:", N)

import os
missing = [p for p in pids if not os.path.exists(f"{SHARD_DIR}/{p}.npz")]
print("missing shard files:", missing)

enc_feats = np.zeros((N, 32, 1280), dtype=np.float32)
llm_feats = np.zeros((N, 29, 3584), dtype=np.float32)

for i, p in enumerate(pids):
    d = np.load(f"{SHARD_DIR}/{p}.npz")
    enc_feats[i] = d["enc"]
    llm_feats[i] = d["llm"]

print("loaded features", enc_feats.shape, llm_feats.shape)

groups = pids  # speaker == pid, one clip per speaker

def stage_auc_oof(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        oof[te] = proba
    return roc_auc_score(y, oof)

enc_claim = pd.read_csv(f"{R}/pull/p23-a-probes/p23a/out/o25_whole_encoder_perlayer.csv")
llm_claim = pd.read_csv(f"{R}/pull/p23-a-probes/p23a/out/o25_whole_llm_perlayer.csv")

enc_results = {}
print("=== encoder stages (0..31) ===")
for s in range(32):
    a = stage_auc_oof(enc_feats[:, s, :], labels, groups)
    claimed = float(enc_claim.loc[enc_claim.layer == s, "auc_oof"].values[0])
    diff = abs(a - claimed)
    enc_results[s] = (a, claimed, diff)
    print(f"enc stage {s}: recomputed={a:.6f} claimed={claimed:.6f} diff={diff:.6f}")

llm_results = {}
print("=== llm stages (0..28) ===")
for s in range(29):
    a = stage_auc_oof(llm_feats[:, s, :], labels, groups)
    claimed = float(llm_claim.loc[llm_claim.stage == s, "auc_oof"].values[0])
    diff = abs(a - claimed)
    llm_results[s] = (a, claimed, diff)
    print(f"llm stage {s}: recomputed={a:.6f} claimed={claimed:.6f} diff={diff:.6f}")

max_diff_enc = max(v[2] for v in enc_results.values())
max_diff_llm = max(v[2] for v in llm_results.values())
max_diff = max(max_diff_enc, max_diff_llm)
print("MAX ABS DIFF encoder:", max_diff_enc)
print("MAX ABS DIFF llm:", max_diff_llm)
print("MAX ABS DIFF overall:", max_diff)

with open("<local data dir>/scratch/results_perlayer.json", "w") as f:
    json.dump({"enc": enc_results, "llm": llm_results, "max_diff_enc": max_diff_enc,
               "max_diff_llm": max_diff_llm, "max_diff": max_diff}, f, indent=1)

print("DONE ITEM11")
