#!/usr/local/bin/python3
"""Extra: Pitt AF2 arm difference from the 30 s file that reproduces the printed Table 2 (0.58 / 0.53). Same rule as holm3.py."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D = "scores/part20/POD4c"
P = pd.concat([pd.read_csv(f"{D}/af2_orig30_conflict.csv"), pd.read_csv(f"{D}/af2_orig30_agreement.csv")], ignore_index=True)
assert len(P) == 468 and P.orig_clip_id.is_unique
mc = (P.arm == "conflict").values; ma = (P.arm == "agreement").values
spk = P.speaker_id.astype(str).values; y = P.label.values.astype(int); s = P.p_yes.values.astype(float)
u = np.unique(spk); idx = {k: np.where(spk == k)[0] for k in u}; rng = np.random.default_rng(0); d = []
for _ in range(2000):
    ii = np.concatenate([idx[p] for p in rng.choice(u, size=len(u), replace=True)])
    c = ii[mc[ii]]; a = ii[ma[ii]]
    if y[c].min() == y[c].max() or y[a].min() == y[a].max(): continue
    d.append(roc_auc_score(y[c], s[c]) - roc_auc_score(y[a], s[a]))
d = np.array(d); B = len(d)
p = min(1.0, 2 * min((1 + (d <= 0).sum()) / (B + 1), (1 + (d >= 0).sum()) / (B + 1)))
print(f"AF2 Pitt orig30: conflict {roc_auc_score(y[mc], s[mc]):.4f} agreement {roc_auc_score(y[ma], s[ma]):.4f} diff {roc_auc_score(y[mc], s[mc])-roc_auc_score(y[ma], s[ma]):+.4f} [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}] p={p:.4f} B={B} spk={len(u)}")
