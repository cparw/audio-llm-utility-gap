#!/usr/bin/env python3
"""POD6 verifier, part C: recompute Pitt conflict / agreement / conflict-minus-agreement
cells from the raw per-clip score files (own code).
Arm from the clip basename prefix (conflict_ / agreement_). Arm cells resample the
speakers present in that arm; the difference draws once over all speakers per replicate
and recomputes both arm AUCs inside it.
usage: POD6_verify_C.py SCORES.csv SCORE_COL [SPK_COL] [CLIP_COL]
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from POD6_verify import auc_mw, auc_trap, boot_ci, _groups, NB


def arms(path, col, spk_col=None, clip_col=None):
    d = pd.read_csv(path)
    spk_col = spk_col or ("speaker_id" if "speaker_id" in d else ("speaker" if "speaker" in d else "spk"))
    clip_col = clip_col or next(c for c in ("clip", "clip_path", "path", "id") if c in d)
    d["b"] = d[clip_col].map(os.path.basename)
    d["arm"] = np.where(d["b"].str.startswith("conflict_"), "conflict", np.where(d["b"].str.startswith("agreement_"), "agreement", "?"))
    assert (d["arm"] != "?").all() and d["b"].is_unique
    return d.assign(spk=d[spk_col].astype(str), y=d["label"].astype(int), s=d[col].astype(float))


def diff_ci(d, nb=NB, seed=0):
    spk = d["spk"].to_numpy(); y = d["y"].to_numpy(); s = d["s"].to_numpy(); c = (d["arm"] == "conflict").to_numpy()
    rng = np.random.default_rng(seed); u, idx = _groups(spk); v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick]); ci = ii[c[ii]]; ai = ii[~c[ii]]
        if len(np.unique(y[ci])) < 2 or len(np.unique(y[ai])) < 2:
            continue
        v.append(auc_mw(y[ci], s[ci]) - auc_mw(y[ai], s[ai]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def cells(d):
    out = {}
    for arm in ("conflict", "agreement"):
        e = d[d["arm"] == arm]; y = e["y"].to_numpy(); s = e["s"].to_numpy()
        a = auc_mw(y, s); assert abs(a - auc_trap(y, s)) < 1e-9
        lo, hi, nu = boot_ci(e["spk"].to_numpy(), y, s)
        out[arm] = dict(auc=a, lo=lo, hi=hi, n=len(e), n_spk=int(e["spk"].nunique()), n_pos=int(y.sum()), draws=nu)
    lo, hi, nu = diff_ci(d)
    out["conflict_minus_agreement"] = dict(auc=out["conflict"]["auc"] - out["agreement"]["auc"], lo=lo, hi=hi,
                                           n=len(d), n_spk=int(d["spk"].nunique()), n_pos=int(d["y"].sum()), draws=nu)
    return out


if __name__ == "__main__":
    d = arms(sys.argv[1], sys.argv[2], *(sys.argv[3:5]))
    r = cells(d)
    for k, v in r.items():
        print(f"{k:26s} {v['auc']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] n={v['n']} spk={v['n_spk']} pos={v['n_pos']} draws={v['draws']}")
    print(json.dumps(r))
