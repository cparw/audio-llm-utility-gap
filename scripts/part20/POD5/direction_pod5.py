"""PART20 POD5: readout-direction test on the answer state (final LM stage at the first answer position), following
release/scripts/part10_readout_direction_v2.py (the paper's Omni run): states used as saved (norm verdict re-checked),
mass-weighted Yes-minus-No lm_head direction (weights = mean over clips of the Yes/No-subset softmax, normalised within
Yes and within No), gate |AUC(S.d) - zero-shot AUC| <= 0.01, probe = StandardScaler + LogisticRegression(max_iter=2000,
class_weight='balanced') on the SAVED Mac split pitt_groupkfold5_mac.csv (the split the paper's direction run used),
removal w_perp = w_fold - (w_fold.d/d.d) d, CANONICAL held-out-fold standardisation perp_z[te] = (b-b.mean())/b.std().
Random floor: fresh default_rng(0), 2000 Gaussian directions, mean |cos| with d. Intervals: speaker bootstrap 2000 draws,
fresh default_rng(0) per cell; cosine refits the probe on each draw (as v2 does); AUCs resample the fixed out-of-fold scores.
usage: direction_pod5.py STATES.npz READOUT.pt ZEROSHOT_CSV FOLDFILE OUTPREFIX TAG"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import sys, json, csv, time, numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_pod5 import rauc, boot, B
NPZ, RO, ZS, FOLDF, OUTP, TAG = sys.argv[1:7]
t0 = time.time()
z = np.load(NPZ, allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str)
S = z["ans"][:, -1, :].astype(np.float64)
zs = pd.read_csv(ZS)
key = "orig_clip_id" if "orig_clip_id" in zs.columns else "clip"
assert (zs[key].astype(str).values == names).all(), "zero-shot csv order mismatch"
p_stored = zs["p_yes"].values.astype(float); auc_zs = rauc(zs["label"].values, p_stored)
ro = torch.load(RO, map_location="cpu")
if "rows" in ro:
    ids = list(ro["ids"]); R = ro["rows"].double().numpy()
    yes_ids, no_ids = list(ro["yes_rule"]), list(ro["no_rule"])
    Wy = R[[ids.index(i) for i in yes_ids]]; Wn = R[[ids.index(i) for i in no_ids]]
    extra = {"legacy": (list(ro["yes_legacy"]), list(ro["no_legacy"]), R[[ids.index(i) for i in ro["yes_legacy"]]], R[[ids.index(i) for i in ro["no_legacy"]]])}
else:
    yes_ids, no_ids = [int(v) for v in ro["yes_ids"]], [int(v) for v in ro["no_ids"]]
    Wy, Wn = ro["lm_head_yes"].double().numpy(), ro["lm_head_no"].double().numpy(); extra = {}
gw = ro["final_norm_weight"].double().numpy(); eps = float(ro["norm_eps"]); nY = len(yes_ids)
def subset_probs(X, Wy, Wn):
    b = np.concatenate([X @ Wy.T, X @ Wn.T], 1); b -= b.max(1, keepdims=True); e = np.exp(b); return e / e.sum(1, keepdims=True)
def rms(X): return X / np.sqrt(np.mean(X * X, -1, keepdims=True) + eps) * gw[None]
verdict = {}
for nm, X in (("as_saved", S), ("rmsnorm_applied", rms(S))):
    ph = subset_probs(X, Wy, Wn)[:, :nY].sum(1)
    verdict[nm] = {"mean_abs_err_vs_stored_p_yes": float(np.abs(ph - p_stored).mean()), "r": float(np.corrcoef(ph, p_stored)[0, 1]), "auc": rauc(y, ph)}
use_norm = "as_saved" if verdict["as_saved"]["mean_abs_err_vs_stored_p_yes"] <= verdict["rmsnorm_applied"]["mean_abs_err_vs_stored_p_yes"] else "rmsnorm_applied"
if use_norm == "rmsnorm_applied": S = rms(S)
print("norm verdict", use_norm, json.dumps(verdict), flush=True)
def dirs_for(Wy, Wn):
    pr = subset_probs(S, Wy, Wn); mp = pr.mean(0); k = Wy.shape[0]
    wy, wn = mp[:k] / mp[:k].sum(), mp[k:] / mp[k:].sum()
    return {"d_weighted": wy @ Wy - wn @ Wn, "d_plain": Wy.mean(0) - Wn.mean(0), "d_top": Wy[int(np.argmax(mp[:k]))] - Wn[int(np.argmax(mp[k:]))]}, mp
D, mp = dirs_for(Wy, Wn)
gate = {k: {"proj_auc": rauc(y, S @ d), "delta": rauc(y, S @ d) - auc_zs, "pass": abs(rauc(y, S @ d) - auc_zs) <= 0.01} for k, d in D.items()}
print("gate", json.dumps(gate), flush=True)
used = "d_weighted" if gate["d_weighted"]["pass"] else next((k for k in ("d_plain", "d_top") if gate[k]["pass"]), "d_weighted")
fallback = "" if used == "d_weighted" and gate["d_weighted"]["pass"] else ("NO definition passed the gate; d_weighted reported anyway, flagged" if not any(g["pass"] for g in gate.values()) else f"d_weighted failed the gate, fell back to {used}")
d = D[used]
fd = pd.read_csv(FOLDF).set_index("clip_id").loc[names]; assert (fd.speaker_id.astype(str).values == spk).all(); fold = fd.fold.values.astype(int)
oof = np.zeros(len(y)); perp_z = np.zeros(len(y)); keep_z = np.zeros(len(y)); ws = []; pf = []; kf = []
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    sc = StandardScaler().fit(S[tr]); lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(S[tr]), y[tr])
    oof[te] = lr.predict_proba(sc.transform(S[te]))[:, 1]
    wf = lr.coef_.ravel() / sc.scale_; ws.append(wf)
    wp = wf - (wf @ d) / (d @ d) * d
    a, b = S[te] @ wf, S[te] @ wp
    keep_z[te] = (a - a.mean()) / a.std(); perp_z[te] = (b - b.mean()) / b.std()
    pf.append(rauc(y[te], b)); kf.append(rauc(y[te], a))
w_raw = np.mean(ws, 0); cos = float(w_raw @ d / (np.linalg.norm(w_raw) * np.linalg.norm(d)))
rng = np.random.default_rng(0); G = rng.standard_normal((2000, len(d)))
rc = np.abs((G @ d) / (np.linalg.norm(G, axis=1) * np.linalg.norm(d)))
floor, floor_lo, floor_hi = float(rc.mean()), float(np.percentile(rc, 2.5)), float(np.percentile(rc, 97.5))
print(f"probe {rauc(y,oof):.4f} after removal {rauc(y,perp_z):.4f} kept-control {rauc(y,keep_z):.4f} cos {cos:.4f} floor {floor:.4f} {time.time()-t0:.0f}s", flush=True)
u, inv = np.unique(spk, return_inverse=True); idx = [np.where(inv == i)[0] for i in range(len(u))]
rng = np.random.default_rng(0); draws = [np.concatenate([idx[i] for i in rng.integers(0, len(u), size=len(u))]) for _ in range(B)]
def cos_draw(r):
    warnings.filterwarnings("ignore"); np.seterr(all="ignore")
    if len(np.unique(y[r])) < 2: return np.nan
    sc = StandardScaler().fit(S[r]); lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(S[r]), y[r])
    w = lr.coef_.ravel() / sc.scale_; return float(w @ d / (np.linalg.norm(w) * np.linalg.norm(d)))
cs = np.array(Parallel(n_jobs=int(os.environ.get("NJ", "136")))(delayed(cos_draw)(r) for r in draws), float); ok = cs[~np.isnan(cs)]
cos_lo, cos_hi = [float(v) for v in np.percentile(ok, [2.5, 97.5])]
res = {"cos": (cos, cos_lo, cos_hi, len(ok))}
res["probe"] = boot(y, oof[:, None], spk); res["after_removal"] = boot(y, perp_z[:, None], spk)
res["probe_minus_after_removal"] = boot(y, oof[:, None], spk, S2=perp_z[:, None]); res["kept_control"] = boot(y, keep_z[:, None], spk)
res["auc_d"] = boot(y, (S @ d)[:, None], spk)
arms = {}
if "set" in z.files:
    st = z["set"].astype(str)
    for a in ("conflict", "agreement"):
        m = st == a
        arms[a] = {"probe": boot(y[m], oof[m][:, None], spk[m]), "after_removal": boot(y[m], perp_z[m][:, None], spk[m])}
extra_out = {}
for nm, (yi, ni, wy_, wn_) in extra.items():
    Dx, mpx = dirs_for(wy_, wn_); dx = Dx["d_weighted"]
    extra_out[nm] = {"yes_ids": yi, "no_ids": ni, "cos_w_raw_d": float(w_raw @ dx / (np.linalg.norm(w_raw) * np.linalg.norm(dx))),
                     "cos_d_rule_vs_d_legacy": float(d @ dx / (np.linalg.norm(d) * np.linalg.norm(dx))), "proj_auc": rauc(y, S @ dx)}
with open(f"{OUTP}_perclip.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip_id", "speaker", "label", "set", "fold", "p_probe_oof", "perp_z", "keep_z", "proj_d", "p_yes_zeroshot"])
    for i in range(len(y)): w.writerow([names[i], spk[i], int(y[i]), str(z["set"][i]) if "set" in z.files else "", int(fold[i]), repr(float(oof[i])), repr(float(perp_z[i])), repr(float(keep_z[i])), repr(float(S[i] @ d)), repr(float(p_stored[i]))])
out = {"tag": TAG, "states": os.path.abspath(NPZ), "readout": os.path.abspath(RO), "zeroshot_csv": os.path.abspath(ZS), "fold_file": os.path.abspath(FOLDF),
       "n": int(len(y)), "n_speakers": int(len(u)), "features": "ans[:, -1, :] final LM stage, first answer position", "norm_verdict": use_norm, "norm_checks": verdict,
       "yes_ids": yes_ids, "no_ids": no_ids, "mean_subset_mass": mp.tolist(), "auc_zeroshot_from_csv": auc_zs, "gate": gate, "direction_used": used, "fallback_note": fallback,
       "cosine": res["cos"], "random_floor_mean_abs_cos": floor, "random_floor_2.5_97.5": [floor_lo, floor_hi], "random_floor_n": 2000, "random_floor_seed": 0,
       "probe_auc": res["probe"], "after_removal_auc": res["after_removal"], "probe_minus_after_removal": res["probe_minus_after_removal"],
       "kept_control_auc": res["kept_control"], "auc_d": res["auc_d"], "per_fold_auc_removed": pf, "per_fold_auc_kept": kf, "arms": arms, "legacy_ids_check": extra_out,
       "format": "each stat = [point, lo2.5, hi97.5, valid_draws]", "bootstrap": {"draws": B, "seed": 0, "unit": "speaker"}, "seconds": round(time.time() - t0)}
json.dump(out, open(f"{OUTP}.json", "w"), indent=1)
print("DIRECTION", TAG, json.dumps({k: out[k] for k in ("cosine", "random_floor_mean_abs_cos", "probe_auc", "after_removal_auc", "probe_minus_after_removal", "direction_used")}), flush=True)
print("DIRECTION DONE", flush=True)
