#!/usr/local/bin/python3
"""Extra, not the spec: the three borderline cells re-run with 20000 draws (default_rng(0)) to size the Monte Carlo error of p."""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
NB = 20000
REL = "<local data dir>/release"
def run(spk, y, terms, sign):
    spk = np.asarray(spk).astype(str); y = np.asarray(y).astype(int); u = np.unique(spk)
    idx = {s: np.where(spk == s)[0] for s in u}; rng = np.random.default_rng(0); d = []
    for _ in range(NB):
        ii = np.concatenate([idx[p] for p in rng.choice(u, size=len(u), replace=True)]); v = 0.0; ok = True
        for (m, sc), sg in zip(terms, sign):
            jj = ii[m[ii]]
            if y[jj].min() == y[jj].max(): ok = False; break
            v += sg * roc_auc_score(y[jj], sc[jj])
        if ok: d.append(v)
    d = np.array(d); B = len(d)
    p = min(1.0, 2 * min((1 + (d <= 0).sum()) / (B + 1), (1 + (d >= 0).sum()) / (B + 1)))
    return B, np.percentile(d, 2.5), np.percentile(d, 97.5), p
P = pd.read_csv(f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv"); b = P["clip"].map(os.path.basename)
mc = b.str.startswith("conflict_").values; ma = b.str.startswith("agreement_").values; s = P.p_yes.values.astype(float)
print("Pitt Qwen2.5-Omni conflict minus agreement", run(P.speaker.values, P.label.values, [(mc, s), (ma, s)], [1, -1]))
for key, name in [("pg", "PC-GITA"), ("nv", "NeuroVoz")]:
    B_ = pd.read_csv(f"{REL}/edaic_rerun/part16/POD4/p16_{key}_before_enc_nested_oof.csv")
    A_ = pd.read_csv(f"{REL}/edaic_rerun/part16/POD4/p16_{key}_after_enc_nested_oof.csv")
    M = B_.merge(A_, on="clip", suffixes=("_b", "_a")); assert len(M) == len(B_) == len(A_)
    assert (M.label_b == M.label_a).all() and (M.speaker_b.astype(str) == M.speaker_a.astype(str)).all()
    al = np.ones(len(M), bool)
    print(f"{name} probe after minus before", run(M.speaker_b.values, M.label_b.values, [(al, M.p_probe_a.values), (al, M.p_probe_b.values)], [1, -1]))
