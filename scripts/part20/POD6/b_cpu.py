"""PART20 POD6 job B, CPU parts, on the NeuroVoz age and sex matched subset:
 1. NEW fold files for the subset (no saved file exists): single split and seeds 0..4, POD tie rule.
 2. Zero-shot answer AUC from the EXISTING omni_final/omni_neurovoz_zeroshot_scores.csv restricted to the subset.
 3. Five quiet-frame channel features (per-clip values from part16 POD4 channel_norm_neurovoz.csv, BEFORE columns,
    computed by chan4a.py) each alone and all five together (StandardScaler + LogisticRegression(max_iter=5000,
    class_weight='balanced'), as feat_probe.py). Method check: the same code on the full 1270 must reproduce
    featprobe_nv.json (0.7977 five-feature).
usage: b_cpu.py OUTDIR"""
import sys, os, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p20_common import auc_rank, boot_auc, paste, sidecar, pod_groupkfold
OUT = sys.argv[1]
R = "<local data dir>/release"
ZS = f"{R}/omni_final/omni_neurovoz_zeroshot_scores.csv"
CH = f"{R}/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv"
FPJ = f"{R}/edaic_rerun/part16/POD4/featprobe_nv.json"
SUB = f"{OUT}/nv_matched_clips.csv"; PAIRS = f"{OUT}/nv_matched_pairs.csv"
CMD = "/usr/local/bin/python3 " + " ".join(sys.argv)
FEATS = ["noise_floor", "snr", "centroid", "rolloff", "tilt"]
sub = pd.read_csv(SUB); keep = set(sub["clip"])
# ---- 1. fold files
os.makedirs(f"{OUT}/folds", exist_ok=True)
spk = sub.speaker.astype(str).values
_, f = pod_groupkfold(spk)
pd.DataFrame({"clip_id": sub["clip"], "speaker_id": spk, "fold": f}).to_csv(f"{OUT}/folds/nvmatched_groupkfold5_pod_NEW.csv", index=False)
for s in range(5):
    rng = np.random.default_rng(s); u = np.unique(spk); perm = {x: i for i, x in enumerate(rng.permutation(u))}
    _, fs = pod_groupkfold(np.array([perm[x] for x in spk]))
    pd.DataFrame({"clip_id": sub["clip"], "speaker_id": spk, "fold": fs}).to_csv(f"{OUT}/folds/nvmatched_groupkfold5_pod_seed{s}_NEW.csv", index=False)
print("NEW fold files written for the 907-clip subset:", sorted(os.listdir(f"{OUT}/folds")), flush=True)
# ---- 2. zero-shot on the subset from the existing per-clip file
z = pd.read_csv(ZS); zs = z[z["clip"].isin(keep)].copy(); assert len(zs) == len(sub) == 907
y = zs.label.values.astype(int); s = zs.p_yes.values; sp = zs.speaker.astype(str).values
a = auc_rank(y, s); lo, hi, nu = boot_auc(y, s, sp)
af = auc_rank(z.label.values, z.p_yes.values); lof, hif, _ = boot_auc(z.label.values, z.p_yes.values, z.speaker.astype(str).values)
per = f"{OUT}/B_zeroshot_nvmatched.csv"; zs.to_csv(per, index=False)
sidecar(per.replace(".csv", ".sidecar.json"), result="Qwen2.5-Omni zero-shot answer AUC, NeuroVoz age and sex matched subset (existing per-clip scores, not rerun)",
        auc=round(a, 4), lo=round(lo, 4), hi=round(hi, 4), usable_draws=nu, n=int(len(zs)), n_speakers=int(len(set(sp))),
        full_set_recomputed=dict(auc=round(af, 4), lo=round(lof, 4), hi=round(hif, 4), n=int(len(z)), paper="0.6951"),
        model_id="Qwen/Qwen2.5-Omni-7B", prompt=str(zs.prompt.iloc[0]), prompt_unique=int(z.prompt.nunique()),
        median_answer_mass=float(zs["mass"].median()), seed="default_rng(0), 2000 draws, speakers with replacement",
        command=CMD, sources=[ZS, SUB, PAIRS])
open(f"{OUT}/B_zeroshot_nvmatched.RESULT", "w").close()
paste("P20.POD6.B.nvmatched.zeroshot", "NeuroVoz matched subset Qwen2.5-Omni zero-shot answer AUC (existing scores)", a, lo, hi, len(zs), len(set(sp)), per)
paste("P20.POD6.B.nvfull.zeroshot_recomputed", "NeuroVoz full 1270 Qwen2.5-Omni zero-shot answer AUC recomputed (paper 0.6951)", af, lof, hif, len(z), z.speaker.nunique(), ZS)
# ---- 3. channel features
ch = pd.read_csv(CH)
def fivefeat(d, folds):
    X = d[[f"{k}_before" for k in FEATS]].values.astype(float); yy = d.label.values.astype(int); oof = np.zeros(len(d))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        oof[te] = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), yy[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    return oof
# method check on the full set: feat_probe.py numbered speakers by first appearance, GroupKFold(5) on a pod
seen = {}
for x in ch.spk.astype(str):
    if x not in seen: seen[x] = len(seen)
gfull = np.array([seen[x] for x in ch.spk.astype(str)])
fl, _ = pod_groupkfold(gfull)
oof_full = fivefeat(ch, fl); a_full = auc_rank(ch.label.values, oof_full)
ref = json.load(open(FPJ))
chk = {"five_feature_full_recomputed": round(a_full, 4), "paper": ref["combined"]["before"]["auc"]}
for k in FEATS:
    chk[f"{k}_raw_full_recomputed"] = round(auc_rank(ch.label.values, ch[f"{k}_before"].values), 4); chk[f"{k}_raw_paper"] = ref["single"]["before"][k]["auc_raw"]
print("METHOD CHECK full set:", chk, flush=True)
cs = ch[ch["clip"].isin(keep)].copy().reset_index(drop=True); assert len(cs) == 907
spc = cs.spk.astype(str).values; yc = cs.label.values.astype(int)
fsub, fold_sub = pod_groupkfold(spc)
chk["subset_fold_equals_saved_NEW_file"] = bool((pd.read_csv(f"{OUT}/folds/nvmatched_groupkfold5_pod_NEW.csv").set_index("clip_id").loc[cs["clip"], "fold"].values == fold_sub).all())
res = {}
for k in FEATS:
    x = cs[f"{k}_before"].values.astype(float); ar = auc_rank(yc, x); dfree = max(ar, 1 - ar)
    lo_, hi_, nu_ = boot_auc(yc, x if ar >= 0.5 else -x, spc)
    rlo, rhi, _ = boot_auc(yc, x, spc)
    res[k] = dict(auc_raw=ar, raw_lo=rlo, raw_hi=rhi, auc_dirfree=dfree, dirfree_lo=lo_, dirfree_hi=hi_, usable=nu_,
                  paper_full_raw=ref["single"]["before"][k]["auc_raw"], paper_full_dirfree=ref["single"]["before"][k]["auc_dirfree"])
oof = fivefeat(cs, fsub); a5 = auc_rank(yc, oof); lo5, hi5, nu5 = boot_auc(yc, oof, spc)
res["five_together"] = dict(auc=a5, lo=lo5, hi=hi5, usable=nu5, paper_full=ref["combined"]["before"]["auc"])
per = f"{OUT}/B_channel5_nvmatched.csv"
out = cs[["clip", "spk", "label"] + [f"{k}_before" for k in FEATS]].copy(); out["fold"] = fold_sub; out["p_five_feature_oof"] = oof; out.to_csv(per, index=False)
sidecar(per.replace(".csv", ".sidecar.json"), result="NeuroVoz matched subset: five quiet-frame channel features each alone and together",
        cells=res, method_check=chk, n=907, n_speakers=int(len(set(spc))), folds=f"{OUT}/folds/nvmatched_groupkfold5_pod_NEW.csv (NEW, POD tie rule on speaker labels)",
        feature_code=f"{R}/edaic_rerun/part16/POD4/chan4a.py (per-clip values reused, BEFORE normalisation)", model_id="none", prompt="none",
        seed="default_rng(0), 2000 draws, speakers with replacement", command=CMD, sources=[CH, FPJ, SUB, PAIRS])
open(f"{OUT}/B_channel5_nvmatched.RESULT", "w").close()
for k in FEATS:
    r = res[k]
    paste(f"P20.POD6.B.nvmatched.chan.{k}.raw", f"NeuroVoz matched subset channel feature {k} alone, raw AUC (full-set paper raw {r['paper_full_raw']})",
          r["auc_raw"], r["raw_lo"], r["raw_hi"], 907, len(set(spc)), per)
    paste(f"P20.POD6.B.nvmatched.chan.{k}.dirfree", f"NeuroVoz matched subset channel feature {k} alone, direction-free max(AUC,1-AUC) (full-set paper {r['paper_full_dirfree']})",
          r["auc_dirfree"], r["dirfree_lo"], r["dirfree_hi"], 907, len(set(spc)), per)
paste("P20.POD6.B.nvmatched.chan.five", "NeuroVoz matched subset five channel features together, out of fold (full-set paper 0.7977)", a5, lo5, hi5, 907, len(set(spc)), per)
json.dump(dict(check=chk, cells=res, zeroshot=dict(auc=a, lo=lo, hi=hi)), open(f"{OUT}/B_cpu_summary.json", "w"), indent=1, default=float)
