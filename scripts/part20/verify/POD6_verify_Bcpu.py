#!/usr/bin/env python3
"""POD6 verifier, part B CPU cells: matched-subset zero-shot answer and channel features.
Recomputes from the compute-side per-clip csv, and independently from (a) the paper's
per-clip zero-shot file restricted to the subset and (b) own channel features measured
from the audio (POD6_verify_chan.py) with an own refit of the five-feature probe.
usage: POD6_verify_Bcpu.py B_DIR OWN_FEATS.csv PAPER_ZS.csv OUT.json"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from POD6_verify import auc_mw, auc_trap, boot_ci, gkf_stable
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BD, OWNF, ZS, OUT = sys.argv[1:5]
FE = ["noise_floor", "snr", "centroid", "rolloff", "tilt"]
res = []


def row(id_, what, comp, ver, n, nspk, file, extra=""):
    ag = all(round(float(a), 4) == round(float(b), 4) for a, b in zip(comp, ver))
    vs = "above" if ver[1] > 0.5 else "below" if ver[2] < 0.5 else "spans"
    res.append(dict(id=id_, what=what, value_computed=f"{comp[0]:.4f}", value_verified=f"{ver[0]:.4f}", lo=f"{ver[1]:.4f}",
                    hi=f"{ver[2]:.4f}", n=n, n_spk=nspk, file=file, two_dp=f"{ver[0]:.2f} [{ver[1]:.2f}, {ver[2]:.2f}]",
                    vs_half=vs, agree_4dp="yes" if ag else "NO", _comp=f"{comp[0]:.4f} [{comp[1]:.4f}, {comp[2]:.4f}]", _extra=extra))
    print(f"{'OK ' if ag else 'DISAGREE'} {id_}: comp {comp[0]:.4f} [{comp[1]:.4f}, {comp[2]:.4f}] ver {ver[0]:.4f} [{ver[1]:.4f}, {ver[2]:.4f}] {extra}", flush=True)


clips = set(pd.read_csv(os.path.join(BD, "nv_matched_clips.csv"))["clip"])
# ---------- zero-shot
sc = json.load(open(os.path.join(BD, "B_zeroshot_nvmatched.sidecar.json")))
c = pd.read_csv(os.path.join(BD, "B_zeroshot_nvmatched.csv"), dtype={"speaker": str})
z = pd.read_csv(ZS, dtype={"speaker": str}); zz = z[z["clip"].isin(clips)]
assert set(c["clip"]) == clips and len(c) == len(zz) == 907
m = c.merge(zz, on="clip"); assert (m.p_yes_x == m.p_yes_y).all() and (m.label_x == m.label_y).all() and (m.speaker_x == m.speaker_y).all()
assert c["prompt"].nunique() == 1 and c["prompt"].iloc[0] == sc["prompt"] == z["prompt"].iloc[0]
y = c.label.to_numpy(); s = c.p_yes.to_numpy(); spk = c.speaker.to_numpy()
a = auc_mw(y, s); assert abs(a - auc_trap(y, s)) < 1e-9
lo, hi, _ = boot_ci(spk, y, s)
row("B_zeroshot_nvmatched", "Qwen2.5-Omni zero-shot answer AUC, NeuroVoz age/sex matched subset (38 pairs; paper per-clip scores restricted)",
    (sc["auc"], sc["lo"], sc["hi"]), (a, lo, hi), len(c), len(np.unique(spk)), "scores/part20/POD6/B_neurovoz_matched/B_zeroshot_nvmatched.csv",
    f"n_pos={int(y.sum())} n_neg={int((1-y).sum())} prompt verbatim ok")
yf = z.label.to_numpy(); sf_ = z.p_yes.to_numpy(); af = auc_mw(yf, sf_); lof, hif, _ = boot_ci(z.speaker.to_numpy(), yf, sf_)
fs = sc["full_set_recomputed"]
row("B_zeroshot_nvfull_ref", "Qwen2.5-Omni zero-shot answer AUC, NeuroVoz full 1270 (paper reference, recomputed)",
    (fs["auc"], fs["lo"], fs["hi"]), (af, lof, hif), len(z), z.speaker.nunique(), "release/omni_final/omni_neurovoz_zeroshot_scores.csv")

# ---------- channel features
sc = json.load(open(os.path.join(BD, "B_channel5_nvmatched.sidecar.json")))["cells"]
c = pd.read_csv(os.path.join(BD, "B_channel5_nvmatched.csv"), dtype={"spk": str})
own = pd.read_csv(OWNF)
m = c.merge(own, on="clip"); assert len(m) == 907 and set(c["clip"]) == clips
fdiff = max(float(np.abs(m[f + "_before"] - m[f]).max()) for f in FE)
y = c.label.to_numpy(); spk = c.spk.to_numpy()
for f in FE:
    x = c[f + "_before"].to_numpy(); a = auc_mw(y, x); assert abs(a - auc_trap(y, x)) < 1e-9
    lo, hi, _ = boot_ci(spk, y, x)
    xo = m[f].to_numpy(); ao = auc_mw(m.label.to_numpy(), xo)
    k = sc[f]
    row(f"B_chan_{f}_raw_nvmatched", f"quiet-frame {f} alone, raw AUC (PD positive), NeuroVoz matched subset",
        (k["auc_raw"], k["raw_lo"], k["raw_hi"]), (a, lo, hi), len(c), len(np.unique(spk)),
        "scores/part20/POD6/B_neurovoz_matched/B_channel5_nvmatched.csv", f"own-audio AUC {ao:.4f}")
    d = max(a, 1 - a); lo2, hi2, _ = boot_ci(spk, y, x if a >= 0.5 else -x)
    row(f"B_chan_{f}_dirfree_nvmatched", f"quiet-frame {f} alone, direction-free max(AUC,1-AUC), NeuroVoz matched subset",
        (k["auc_dirfree"], k["dirfree_lo"], k["dirfree_hi"]), (d, lo2, hi2), len(c), len(np.unique(spk)),
        "scores/part20/POD6/B_neurovoz_matched/B_channel5_nvmatched.csv")
# five together: recompute from their oof, check fold file vs own stable-GroupKFold on speaker labels, refit with own features
fo = pd.read_csv(os.path.join(BD, "folds", "nvmatched_groupkfold5_pod_NEW.csv"), dtype={"speaker_id": str})
own_fold = gkf_stable(fo.speaker_id.to_numpy())
fold_ok = bool((own_fold == fo.fold.to_numpy()).all())
mm = c.merge(fo, left_on="clip", right_on="clip_id"); fold_same = bool((mm.fold_x == mm.fold_y).all())
o = c.p_five_feature_oof.to_numpy(); a = auc_mw(y, o); lo, hi, _ = boot_ci(spk, y, o)
X = m[FE].to_numpy(); ym = m.label.to_numpy(); fm = m.fold.to_numpy(); oo = np.zeros(len(ym))
for k_ in range(5):
    te = fm == k_; tr = ~te
    s_ = StandardScaler().fit(X[tr]); lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(s_.transform(X[tr]), ym[tr])
    oo[te] = lr.predict_proba(s_.transform(X[te]))[:, 1]
ao = auc_mw(ym, oo); omax = float(np.abs(oo - m.p_five_feature_oof.to_numpy()).max())
k = sc["five_together"]
row("B_chan_five_nvmatched", "five quiet-frame channel features together, LR out-of-fold AUC, NeuroVoz matched subset (NEW single split, stable GroupKFold on speaker labels)",
    (k["auc"], k["lo"], k["hi"]), (a, lo, hi), len(c), len(np.unique(spk)), "scores/part20/POD6/B_neurovoz_matched/B_channel5_nvmatched.csv",
    f"fold file == own stable GroupKFold: {fold_ok}; csv fold == fold file: {fold_same}; own-audio refit AUC {ao:.4f} max|dp| {omax:.1e}; own vs part16 feature max diff {fdiff:.1e}")
json.dump(res, open(OUT, "w"), indent=1)
