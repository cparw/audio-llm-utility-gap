import re, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = "<local data dir>/release_from_mac/scores/part23"

def strip_np(x):
    if isinstance(x, str) and x.startswith("np.float64("):
        return float(x[len("np.float64("):-1])
    return float(x)

def load_csv_stripped(path):
    df = pd.read_csv(path)
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].map(strip_np)
            except Exception:
                pass
    return df

# ---------- Load core files ----------
A2 = load_csv_stripped(f"{R}/pull/p23-a-probes/p23a/out/A2_edaic_whole_zeroshot_perclip.csv")
A3 = load_csv_stripped(f"{R}/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv")
Ctrunc = load_csv_stripped(f"{R}/C/C_trunc_275_derived.csv")

A2 = A2.sort_values("pid").reset_index(drop=True)
A3 = A3.sort_values("pid").reset_index(drop=True)
assert (A2.pid.values == A3.pid.values).all()
assert len(A2) == 275

FT_whole = []
FT_ctrl = []
for k in range(5):
    fw = pd.read_csv(f"{R}/pull/p23-ft{k}/out/FT_whole_fold{k}_oof.csv")
    fw["fold"] = k
    FT_whole.append(fw)
    fc = pd.read_csv(f"{R}/pull/p23-ft{k}/out/FT_ctrl300_fold{k}_oof.csv")
    fc["fold"] = k
    FT_ctrl.append(fc)
FT_whole = pd.concat(FT_whole, ignore_index=True)
FT_ctrl = pd.concat(FT_ctrl, ignore_index=True)
print("FT_whole n=", len(FT_whole), "FT_ctrl n=", len(FT_ctrl))
assert len(FT_whole) == 275 and len(FT_ctrl) == 275

# canonical clip order = sorted pid (matches A2/A3 order)
canon_pids = A2.pid.values
FT_whole_o = FT_whole.set_index("pid").loc[canon_pids].reset_index()
FT_ctrl_o = FT_ctrl.set_index("pid").loc[canon_pids].reset_index()
assert (FT_whole_o.pid.values == canon_pids).all()
assert (FT_ctrl_o.pid.values == canon_pids).all()
# label consistency
assert (FT_whole_o.label.values == A2.label.values).all()
assert (FT_ctrl_o.label.values == A2.label.values).all()

labels = A2.label.values.astype(int)
N = len(labels)
print("N=", N, "n_pos=", labels.sum())

def auc(y, s):
    return roc_auc_score(y, s)

def boot_ci(statfn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(N, size=N, replace=True)
        vals.append(statfn(idx))
    vals = np.array(vals)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def boot_ci_paired(statfn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(N, size=N, replace=True)
        vals.append(statfn(idx))
    vals = np.array(vals)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

results = []  # list of dicts: id, claimed, recomputed, lo, hi, verified, file

def check(id_, claimed, recomputed, lo, hi, file_, note=""):
    ok = abs(round(recomputed,4) - round(claimed,4)) < 1e-4 if claimed is not None else None
    results.append(dict(id=id_, claimed=claimed, recomputed=round(recomputed,4) if recomputed is not None else None,
                         lo=round(lo,4) if lo is not None else None, hi=round(hi,4) if hi is not None else None,
                         verified="yes" if ok else ("no" if ok is not None else "n/a"), file=file_, note=note))
    print(id_, "claimed=",claimed,"recomputed=",round(recomputed,4) if recomputed is not None else None,
          f"[{round(lo,4) if lo is not None else None},{round(hi,4) if hi is not None else None}]", "VER" if ok else "MISMATCH", note)

# =================== ITEM 1: zero-shot whole, 300s, diff ===================
p_whole = A2.p_yes_whole.values.astype(float)
p_300 = A2.p_yes_300s_T7b.values.astype(float)

auc_whole = auc(labels, p_whole)
auc_300 = auc(labels, p_300)
diff_w300 = auc_whole - auc_300

lo_w, hi_w = boot_ci(lambda idx: auc(labels[idx], p_whole[idx]))
lo_3, hi_3 = boot_ci(lambda idx: auc(labels[idx], p_300[idx]))
lo_d, hi_d = boot_ci_paired(lambda idx: auc(labels[idx], p_whole[idx]) - auc(labels[idx], p_300[idx]))

check("1_whole", 0.8366, auc_whole, lo_w, hi_w, "A2_edaic_whole_zeroshot_perclip.csv")
check("1_300s", 0.8278, auc_300, lo_3, hi_3, "A2_edaic_whole_zeroshot_perclip.csv")
check("1_whole_minus_300s", 0.0088, diff_w300, lo_d, hi_d, "A2_edaic_whole_zeroshot_perclip.csv")

# =================== ITEMS 2-5: mean-of-five stages ===================
def mean_of_five_stat(idx, cols):
    aucs = [auc(labels[idx], A3[c].values[idx]) for c in cols]
    return np.mean(aucs)

def mean_of_five_point(cols):
    aucs = [auc(labels, A3[c].values) for c in cols]
    return np.mean(aucs), aucs

stage_defs = {
    2: ("encoder", ["oof_enc_r0","oof_enc_r1","oof_enc_r2","oof_enc_r3","oof_enc_r4"], 0.5981, -0.2385),
    3: ("projector", ["oof_proj_r0","oof_proj_r1","oof_proj_r2","oof_proj_r3","oof_proj_r4"], 0.5866, -0.2500),
    4: ("LM", ["oof_llm_r0","oof_llm_r1","oof_llm_r2","oof_llm_r3","oof_llm_r4"], 0.7437, -0.0929),
    5: ("answer state", ["oof_ans_r0","oof_ans_r1","oof_ans_r2","oof_ans_r3","oof_ans_r4"], 0.8359, -0.0007),
}

for iid, (name, cols, claimed_val, claimed_diff) in stage_defs.items():
    pt, per_rep = mean_of_five_point(cols)
    lo, hi = boot_ci(lambda idx: mean_of_five_stat(idx, cols))
    check(f"{iid}_{name}_meanof5", claimed_val, pt, lo, hi, "A3_edaic_whole_meanof5_perclip.csv", note=f"per-rep={[round(x,4) for x in per_rep]}")

    diff_pt = pt - auc_whole
    lo_d, hi_d = boot_ci_paired(lambda idx: mean_of_five_stat(idx, cols) - auc(labels[idx], p_whole[idx]))
    check(f"{iid}_{name}_minus_zswhole", claimed_diff, diff_pt, lo_d, hi_d, "A3+A2", )

# =================== ITEM 6: FT whole pooled + per-fold + diff vs zswhole ===================
ft_whole_labels = FT_whole_o.label.values.astype(int)
ft_whole_p = FT_whole_o.p_yes.values.astype(float)
assert (ft_whole_labels == labels).all()

auc_ft_whole_pooled = auc(labels, ft_whole_p)
lo, hi = boot_ci(lambda idx: auc(labels[idx], ft_whole_p[idx]))
check("6_ft_whole_pooled", 0.7931, auc_ft_whole_pooled, lo, hi, "FT_whole_fold{k}_oof.csv pooled")

perfold_claimed = [0.9079, 0.8592, 0.8117, 0.4302, 0.8304]
for k in range(5):
    sub = FT_whole[FT_whole.fold == k]
    a = auc(sub.label.values, sub.p_yes.values)
    print(f"  ft_whole fold{k} auc={a:.4f} (claimed {perfold_claimed[k]}) n={len(sub)}")
    results.append(dict(id=f"6_ft_whole_fold{k}", claimed=perfold_claimed[k], recomputed=round(a,4), lo=None, hi=None,
                         verified="yes" if abs(round(a,4)-perfold_claimed[k])<1e-4 else "no", file=f"FT_whole_fold{k}_oof.csv"))

diff_ft_zs = auc_ft_whole_pooled - auc_whole
lo_d, hi_d = boot_ci_paired(lambda idx: auc(labels[idx], ft_whole_p[idx]) - auc(labels[idx], p_whole[idx]))
check("6_ft_whole_minus_zswhole", -0.0435, diff_ft_zs, lo_d, hi_d, "FT_whole pooled + A2")

# =================== ITEM 7: FT ctrl300 pooled + per-fold + diff vs FT whole ===================
ft_ctrl_labels = FT_ctrl_o.label.values.astype(int)
ft_ctrl_p = FT_ctrl_o.p_yes.values.astype(float)
assert (ft_ctrl_labels == labels).all()

auc_ft_ctrl_pooled = auc(labels, ft_ctrl_p)
lo, hi = boot_ci(lambda idx: auc(labels[idx], ft_ctrl_p[idx]))
check("7_ft_ctrl300_pooled", 0.8150, auc_ft_ctrl_pooled, lo, hi, "FT_ctrl300_fold{k}_oof.csv pooled")

perfold_ctrl_claimed = [0.9758, 0.8233, 0.8358, 0.7442, 0.8043]
for k in range(5):
    sub = FT_ctrl[FT_ctrl.fold == k]
    a = auc(sub.label.values, sub.p_yes.values)
    print(f"  ft_ctrl300 fold{k} auc={a:.4f} (claimed {perfold_ctrl_claimed[k]}) n={len(sub)}")
    results.append(dict(id=f"7_ft_ctrl300_fold{k}", claimed=perfold_ctrl_claimed[k], recomputed=round(a,4), lo=None, hi=None,
                         verified="yes" if abs(round(a,4)-perfold_ctrl_claimed[k])<1e-4 else "no", file=f"FT_ctrl300_fold{k}_oof.csv"))

diff_ftw_ftc = auc_ft_whole_pooled - auc_ft_ctrl_pooled
lo_d, hi_d = boot_ci_paired(lambda idx: auc(labels[idx], ft_whole_p[idx]) - auc(labels[idx], ft_ctrl_p[idx]))
check("7_ft_whole_minus_ctrl300", -0.0219, diff_ftw_ftc, lo_d, hi_d, "FT_whole pooled + FT_ctrl300 pooled")

# =================== ITEM 8: fold3 answer_mass collapse ===================
fold3 = FT_whole[FT_whole.fold == 3]
median_mass_fold3 = fold3.answer_mass.median()
count_lt_001 = int((fold3.answer_mass < 0.01).sum())
print("fold3 median answer_mass:", median_mass_fold3, "count<0.01:", count_lt_001, "n=", len(fold3))
results.append(dict(id="8_fold3_median_mass", claimed=0.0, recomputed=round(median_mass_fold3,4), lo=None, hi=None,
                     verified="yes" if round(median_mass_fold3,4)==0.0 else "no", file="FT_whole_fold3_oof.csv"))
results.append(dict(id="8_fold3_count_lt_0.01", claimed=None, recomputed=count_lt_001, lo=None, hi=None, verified="n/a", file="FT_whole_fold3_oof.csv"))

print("--- other folds / ctrl300 medians ---")
for k in range(5):
    subw = FT_whole[FT_whole.fold==k]
    subc = FT_ctrl[FT_ctrl.fold==k]
    print(f"whole fold{k} median_mass={subw.answer_mass.median():.6f}   ctrl300 fold{k} median_mass={subc.answer_mass.median():.6f}")

# =================== ITEM 9: NEW ===================
diff_ftctrl_zs300 = auc_ft_ctrl_pooled - auc_300
lo_d, hi_d = boot_ci_paired(lambda idx: auc(labels[idx], ft_ctrl_p[idx]) - auc(labels[idx], p_300[idx]))
check("9a_ft_ctrl300_minus_zs300", None, diff_ftctrl_zs300, lo_d, hi_d, "FT_ctrl300 pooled + A2 p_yes_300s_T7b")

mask220 = FT_whole_o.fold.values != 3
labels220 = labels[mask220]
ftw220 = ft_whole_p[mask220]
auc220 = auc(labels220, ftw220)
N220 = len(labels220)
def boot_ci_220(statfn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(N220, size=N220, replace=True)
        vals.append(statfn(idx))
    vals = np.array(vals)
    return np.percentile(vals,2.5), np.percentile(vals,97.5)
lo220, hi220 = boot_ci_220(lambda idx: auc(labels220[idx], ftw220[idx]))
check("9b_ft_whole_pooled_no_fold3", None, auc220, lo220, hi220, "FT_whole excl fold3 (220 clips)", note=f"n={N220}")

# =================== ITEM 10: truncation counts ===================
dur = Ctrunc.win_dur_s.values.astype(float)
q3o_cut_recount = int((dur > 900).sum())
af3_cut_recount = int((dur > 600).sum())
print("q3o cut recount (dur>900):", q3o_cut_recount, "  af3 cut recount (dur>600):", af3_cut_recount, "N=", len(dur))
results.append(dict(id="10_q3o_cut_count", claimed=0, recomputed=q3o_cut_recount, lo=None, hi=None,
                     verified="yes" if q3o_cut_recount==0 else "no", file="C_trunc_275_derived.csv"))
results.append(dict(id="10_af3_cut_count", claimed=136, recomputed=af3_cut_recount, lo=None, hi=None,
                     verified="yes" if af3_cut_recount==136 else "no", file="C_trunc_275_derived.csv"))

# sanity cross check against existing derived columns
print("cross-check q3o_cut col sum:", Ctrunc.q3o_cut.sum(), " af3_cut col sum:", Ctrunc.af3_cut.sum())

# save intermediate results as json for item11 script & final tsv writer
with open("<local data dir>/scratch/results_part1.json","w") as f:
    json.dump(results, f, indent=1)

print("DONE PART1")
