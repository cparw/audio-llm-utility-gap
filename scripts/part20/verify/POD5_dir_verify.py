#!/usr/bin/env python3
"""PART 20 POD5 verifier, direction test, independent recomputation from the saved states.

Written from scratch; does not import or exec any compute-job script.
usage: POD5_dir_verify.py STATES.npz FOLD_FILE OUT.json [NJ]

X = ans[:, -1, :]  (final LM stage at the first answer position), float64, used as saved (norm verdict
re-checked against the stored p_yes).
Yes / No ids, two rules, both computed:
  rule   : every single-token encoding of Yes, yes, YES (bare and leading space); same for No
  legacy : first token of each of Yes, " Yes", yes, " yes", YES; same for No
lm_head rows and the final norm weight are read straight from the checkpoint safetensors.
d_weighted = sum_i w_yes_i W[yes_i] - sum_j w_no_j W[no_j], w = mean over clips of the softmax over the
Yes+No logits only, renormalised within Yes and within No.
Probe: StandardScaler + LogisticRegression(max_iter=2000, class_weight=balanced) on FOLD_FILE;
w_raw = mean over folds of coef / scale.
Removal: w_perp = w_fold - (w_fold.d / d.d) d; projection b on the HELD-OUT fold, standardised within
the held-out fold (b - b.mean()) / b.std(), pooled.
Floor: fresh default_rng(0), 2000 Gaussian directions, mean |cos| with d, percentile 2.5/97.5.
Cosine interval: speaker bootstrap, fresh default_rng(0), speakers from np.unique, 2000 draws, probe refit
on every draw (full data of the draw), cos(w_draw, d).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json, csv, glob, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

NPZ, FOLD, OUT = sys.argv[1:4]
NJ = int(sys.argv[4]) if len(sys.argv) > 4 else 32
SNAP = glob.glob("/workspace/hf/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/*")[0]

z = np.load(NPZ, allow_pickle=True)
X = z["ans"][:, -1, :].astype(np.float64)
y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
p_yes_stored = z["p_yes"].astype(np.float64)
fm = {r["clip_id"]: (r["speaker_id"], int(r["fold"])) for r in csv.DictReader(open(FOLD))}
fold = np.array([fm[n][1] for n in names])
assert all(fm[n][0] == s for n, s in zip(names, spk)), "speaker label mismatch vs fold file"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SNAP)
def legacy(ws):
    out = set()
    for w in ws:
        t = tok.encode(w, add_special_tokens=False)
        if t: out.add(t[0])
    return sorted(out)
def rule(ws):
    out = set()
    for w in ws:
        for f in (w, " " + w):
            t = tok.encode(f, add_special_tokens=False)
            if len(t) == 1: out.add(t[0])
    return sorted(out)
IDS = {"rule": (rule(["Yes", "yes", "YES"]), rule(["No", "no", "NO"])),
       "legacy": (legacy(["Yes", " Yes", "yes", " yes", "YES"]), legacy(["No", " No", "no", " no", "NO"]))}

from safetensors import safe_open
wm = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
def tensor(k):
    with safe_open(os.path.join(SNAP, wm[k]), framework="pt") as f:
        return f.get_tensor(k).float().numpy().astype(np.float64)
W = tensor("thinker.lm_head.weight"); gnorm = tensor("thinker.model.norm.weight")

def rms(S, eps=1e-6):
    return S / np.sqrt((S * S).mean(1, keepdims=True) + eps) * gnorm[None, :]
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
def auc(yy, s):
    s = np.asarray(s, float); pos = s[yy == 1]; neg = s[yy == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))

res = {"n": int(len(y)), "n_spk": int(len(np.unique(spk))), "fold_file": FOLD, "ids": IDS,
       "fold_sizes": np.bincount(fold).tolist()}
D = {}
for nm, (Y, N) in IDS.items():
    Wy, Wn = W[Y], W[N]
    def sm(S):
        L = np.concatenate([S @ Wy.T, S @ Wn.T], 1); L -= L.max(1, keepdims=True); e = np.exp(L)
        return e / e.sum(1, keepdims=True)
    chk = {}
    for vn, S in (("as_saved", X), ("rms_applied", rms(X))):
        ph = sm(S)[:, :len(Y)].sum(1)
        chk[vn] = {"mean_abs_vs_stored_p_yes": float(np.abs(ph - p_yes_stored).mean()),
                   "r": float(np.corrcoef(ph, p_yes_stored)[0, 1])}
    mp = sm(X).mean(0); k = len(Y)
    d = (mp[:k] / mp[:k].sum()) @ Wy - (mp[k:] / mp[k:].sum()) @ Wn
    D[nm] = d
    res[f"{nm}_norm_check"] = chk
    res[f"{nm}_mean_subset_mass"] = mp.tolist()
    res[f"{nm}_proj_auc"] = auc(y, X @ d)
    res[f"{nm}_gate_delta_vs_stored_pyes_auc"] = auc(y, X @ d) - auc(y, p_yes_stored)
res["cos_d_rule_vs_d_legacy"] = cos(D["rule"], D["legacy"])

oof = np.zeros(len(y)); ws = []; folds_fit = []
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
    oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    ws.append(lr.coef_.ravel() / sc.scale_); folds_fit.append((tr, te))
w_raw = np.mean(ws, 0)
res["probe_auc"] = auc(y, oof)
per = {}
for nm, d in D.items():
    perp_z = np.zeros(len(y)); keep_z = np.zeros(len(y))
    for (tr, te), wf in zip(folds_fit, ws):
        wp = wf - (wf @ d) / (d @ d) * d
        b = X[te] @ wp; a = X[te] @ wf
        perp_z[te] = (b - b.mean()) / b.std(); keep_z[te] = (a - a.mean()) / a.std()
    rng = np.random.default_rng(0); G = rng.standard_normal((2000, len(d)))
    rc = np.abs((G @ d) / (np.linalg.norm(G, axis=1) * np.linalg.norm(d)))
    res[nm] = {"cos_wraw_d": cos(w_raw, d), "after_removal_auc": auc(y, perp_z), "kept_control_auc": auc(y, keep_z),
               "floor_mean": float(rc.mean()), "floor_lo": float(np.percentile(rc, 2.5)), "floor_hi": float(np.percentile(rc, 97.5))}
    per[nm] = (perp_z, keep_z)

# cosine interval, probe refit per draw
u = np.unique(spk); idx = {s: np.where(spk == s)[0] for s in u}
rng = np.random.default_rng(0)
draws = [np.concatenate([idx[p] for p in rng.choice(u, size=len(u), replace=True)]) for _ in range(2000)]
def wdraw(r):
    warnings.filterwarnings("ignore"); np.seterr(all="ignore")
    if len(np.unique(y[r])) < 2: return None
    sc = StandardScaler().fit(X[r]); lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[r]), y[r])
    return lr.coef_.ravel() / sc.scale_
Wd = Parallel(n_jobs=NJ)(delayed(wdraw)(r) for r in draws)
for nm, d in D.items():
    c = np.array([cos(w, d) for w in Wd if w is not None])
    res[nm]["cos_lo"] = float(np.percentile(c, 2.5)); res[nm]["cos_hi"] = float(np.percentile(c, 97.5)); res[nm]["cos_valid"] = int(len(c))
np.savez(OUT.replace(".json", "_perclip.npz"), names=names, spk=spk, y=y, fold=fold, oof=oof,
         perp_z_rule=per["rule"][0], keep_z_rule=per["rule"][1], perp_z_legacy=per["legacy"][0],
         proj_rule=X @ D["rule"], proj_legacy=X @ D["legacy"], w_raw=w_raw)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
