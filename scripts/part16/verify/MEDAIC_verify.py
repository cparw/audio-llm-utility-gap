#!/usr/bin/env python3
"""
INDEPENDENT verifier for PART16 / MEDAIC.
Written from scratch. Does not import or read the original MEDAIC_run.py.

AUC implementation 1: explicit Mann-Whitney U -- count concordant pairs over the
                      full positive x negative outer product, ties contribute 0.5.
AUC implementation 2: trapezoid over the full ROC curve swept at every distinct
                      threshold.
Both are computed for every point estimate and must agree.

Bootstrap: 2000 draws, numpy.random.default_rng(0) fresh per cell,
           speakers resampled with replacement, percentile 2.5 / 97.5.
           Paired = ONE speaker draw per replicate, applied to both arms.
"""
import csv, json, os, sys
import numpy as np

REL = "<local data dir>/Desktop/release"
OUT = os.path.join(REL, "edaic_rerun/part16/verify")
NDRAW = 2000

# ---------------------------------------------------------------- AUC impl 1
def auc_pairs(y, s):
    """Mann-Whitney U over the explicit positive x negative pair grid.
    Concordant = 1, tie = 0.5, discordant = 0."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    # outer comparison over every (positive, negative) pair
    d = pos[:, None] - neg[None, :]
    wins = (d > 0).sum()
    ties = (d == 0).sum()
    return (wins + 0.5 * ties) / (pos.size * neg.size)

# ---------------------------------------------------------------- AUC impl 2
def auc_trapz(y, s):
    """Trapezoid over the ROC swept at every distinct score threshold."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return None
    thr = np.unique(s)
    # sweep from high to low so FPR/TPR increase
    thr = np.concatenate(([np.inf], thr[::-1], [-np.inf]))
    tpr = []; fpr = []
    for t in thr:
        pred = s >= t
        tpr.append(((pred) & (y == 1)).sum() / P)
        fpr.append(((pred) & (y == 0)).sum() / N)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.trapezoid(tpr, fpr))

# --------------------------------------------------------------- bootstrap
def boot_ci(groups, y, s, n=NDRAW, seed=0):
    """Resample SPEAKERS with replacement. groups = speaker id per row."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, size=ug.size, replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        v = auc_pairs(y[ii], s[ii])
        if v is not None:
            vals.append(v)
    vals = np.array(vals)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)),
            len(vals), n)

def boot_paired(groups, y, sA, sB, n=NDRAW, seed=0):
    """Paired delta A - B. ONE speaker draw per replicate, shared by both arms."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, size=ug.size, replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        a = auc_pairs(y[ii], sA[ii]); b = auc_pairs(y[ii], sB[ii])
        if a is not None and b is not None:
            vals.append(a - b)
    vals = np.array(vals)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)),
            len(vals), n)

# --------------------------------------------------------------- loading
def load(path, keycol, labcol, scorecol):
    p = os.path.join(REL, path)
    rows = list(csv.DictReader(open(p)))
    out = {}
    for r in rows:
        k = r[keycol].strip()
        # sft uses a filesystem path in `path`; reduce to the basename stem
        if "/" in k:
            k = os.path.basename(k)
        k = k.replace(".wav", "")
        out[k] = (int(float(r[labcol])), float(r[scorecol]))
    return out, len(rows), p

def align(*dicts):
    keys = sorted(set(dicts[0]).intersection(*[set(d) for d in dicts[1:]]),
                  key=lambda x: (len(x), x))
    labs = [dicts[0][k][0] for k in keys]
    for d in dicts[1:]:
        for k in keys:
            assert d[k][0] == dicts[0][k][0], f"label mismatch at {k}"
    arrs = [np.array([d[k][1] for k in keys], float) for d in dicts]
    return np.array(keys), np.array(labs), arrs

R = {}
def rec(name, val, lo=None, hi=None, extra=None):
    R[name] = {"value": val, "lo": lo, "hi": hi}
    if extra: R[name].update(extra)
    print(f"{name:42s} {val if val is None else round(val,4)!s:>9s}"
          + (f"  [{lo:.4f}, {hi:.4f}]" if lo is not None else ""))

print("=" * 78)
print("LOADING")
print("=" * 78)

ans_full, n1, p1 = load("edaic_rerun/variants/o25_full_zeroshot_scores.csv",
                        "speaker", "label", "p_yes")
sft_full, n2, p2 = load("edaic_rerun/variants/sft_full_oof.csv",
                        "speaker", "label", "p_yes")
txt_full, n3, p3 = load("overnight2/text_new/o25_edaicfull_text.csv",
                        "speaker_id", "label", "p_yes")
ans_30,  n4, p4 = load("omni_final/omni_edaic_zeroshot_scores.csv",
                        "speaker", "label", "p_yes")
enc_30,  n5, p5 = load("omni_final/omni_edaic_enc_nested_oof.csv",
                        "speaker", "label", "p_probe")
llm_30,  n6, p6 = load("omni_final/omni_edaic_llm_nested_oof.csv",
                        "speaker", "label", "p_probe")
ansp_30, n7, p7 = load("omni_final/omni_edaic_ans_nested_oof.csv",
                        "speaker", "label", "p_probe")
proj_30, n8, p8 = load("omni_final/omni_edaic_proj_nested_oof.csv",
                        "speaker", "label", "p_probe")

for nm, n, d, p in [("ans_full",n1,ans_full,p1),("sft_full",n2,sft_full,p2),
                    ("txt_full",n3,txt_full,p3),("ans_30",n4,ans_30,p4),
                    ("enc_30",n5,enc_30,p5),("llm_30",n6,llm_30,p6),
                    ("ansp_30",n7,ansp_30,p7),("proj_30",n8,proj_30,p8)]:
    print(f"  {nm:9s} rows={n:5d} uniq_speakers={len(d):5d}  {p}")

print()
print("=" * 78)
print("SPEAKER OVERLAP  full (mf_full.csv) vs 30s (mf_edaic.csv)")
print("=" * 78)
sf = set(ans_full); s3 = set(ans_30)
print(f"  |full|={len(sf)}  |30s|={len(s3)}  inter={len(sf&s3)}  "
      f"full_only={len(sf-s3)}  30s_only={len(s3-sf)}")
lab_mismatch = [k for k in sf & s3 if ans_full[k][0] != ans_30[k][0]]
print(f"  label disagreements across the two sets: {len(lab_mismatch)}")
print(f"  positives full={sum(v[0] for v in ans_full.values())}  "
      f"30s={sum(v[0] for v in ans_30.values())}")
# are the two score vectors identical (i.e. same file under another name)?
common = sorted(sf & s3)
a = np.array([ans_full[k][1] for k in common])
b = np.array([ans_30[k][1] for k in common])
print(f"  identical p_yes vectors? {np.array_equal(a,b)}   "
      f"max|diff|={np.abs(a-b).max():.6f}  corr={np.corrcoef(a,b)[0,1]:.4f}")

print()
print("=" * 78)
print("POINT ESTIMATES  (two independent AUC implementations)")
print("=" * 78)
def pt(name, d):
    keys = sorted(d); y = np.array([d[k][0] for k in keys])
    s = np.array([d[k][1] for k in keys], float)
    v1 = auc_pairs(y, s); v2 = auc_trapz(y, s)
    flag = "OK " if abs(v1 - v2) < 1e-9 else "!!!"
    print(f"  {flag} {name:12s} pairs={v1:.6f}  trapz={v2:.6f}  n={len(keys)}")
    return v1
AUC = {}
for nm, d in [("ans_full",ans_full),("sft_full",sft_full),("txt_full",txt_full),
              ("ans_30",ans_30),("enc_30",enc_30),("llm_30",llm_30),
              ("ansp_30",ansp_30),("proj_30",proj_30)]:
    AUC[nm] = pt(nm, d)

print()
print("=" * 78)
print("CELLS  (2000 draws, rng(0) fresh per cell, speakers resampled)")
print("=" * 78)

# ---- full window, marginal
for nm, d, tag in [("ans_full",ans_full,"MEDAIC_full_answer"),
                   ("sft_full",sft_full,"MEDAIC_full_sft"),
                   ("txt_full",txt_full,"MEDAIC_full_text")]:
    keys = sorted(d); g = np.array(keys)
    y = np.array([d[k][0] for k in keys]); s = np.array([d[k][1] for k in keys], float)
    lo, hi, u, t = boot_ci(g, y, s)
    rec(tag, AUC[nm], lo, hi, {"usable": u, "total": t, "n": len(keys)})

# ---- full window, paired vs answer
for nm, d, tag in [("sft_full",sft_full,"MEDAIC_full_sft_minus_answer"),
                   ("txt_full",txt_full,"MEDAIC_full_text_minus_answer")]:
    k, y, (sA, sB) = align(d, ans_full)
    lo, hi, u, t = boot_paired(k, y, sA, sB)
    dv = auc_pairs(y, sA) - auc_pairs(y, sB)
    rec(tag, dv, lo, hi, {"usable": u, "total": t, "n": len(k),
                          "excludes_zero": bool(lo > 0 or hi < 0)})

# ---- 30 s window, marginal
for nm, d, tag in [("enc_30",enc_30,"MEDAIC_w30_enc"),
                   ("llm_30",llm_30,"MEDAIC_w30_llm"),
                   ("ansp_30",ansp_30,"MEDAIC_w30_ans"),
                   ("proj_30",proj_30,"MEDAIC_w30_proj"),
                   ("ans_30",ans_30,"MEDAIC_w30_answer")]:
    keys = sorted(d); g = np.array(keys)
    y = np.array([d[k][0] for k in keys]); s = np.array([d[k][1] for k in keys], float)
    lo, hi, u, t = boot_ci(g, y, s)
    rec(tag, AUC[nm], lo, hi, {"usable": u, "total": t, "n": len(keys)})

# ---- 30 s window, paired vs 30 s answer
for nm, d, tag in [("enc_30",enc_30,"MEDAIC_w30_enc_minus_answer"),
                   ("llm_30",llm_30,"MEDAIC_w30_llm_minus_answer"),
                   ("ansp_30",ansp_30,"MEDAIC_w30_ans_minus_answer"),
                   ("proj_30",proj_30,"MEDAIC_w30_proj_minus_answer")]:
    k, y, (sA, sB) = align(d, ans_30)
    lo, hi, u, t = boot_paired(k, y, sA, sB)
    dv = auc_pairs(y, sA) - auc_pairs(y, sB)
    rec(tag, dv, lo, hi, {"usable": u, "total": t, "n": len(k),
                          "excludes_zero": bool(lo > 0 or hi < 0)})

# ---- answer full minus answer 30 s, same speakers
k, y, (sF, sT) = align(ans_full, ans_30)
lo, hi, u, t = boot_paired(k, y, sF, sT)
dv = auc_pairs(y, sF) - auc_pairs(y, sT)
rec("MEDAIC_answer_full_minus_30s", dv, lo, hi,
    {"usable": u, "total": t, "n": len(k), "excludes_zero": bool(lo > 0 or hi < 0)})

print()
print("=" * 78)
print("MISSING-FILE AUDIT  (full-window nested probe per-clip data)")
print("=" * 78)
cands = ["edaic_rerun/variants/o25_full_enc_nested_oof.csv",
         "edaic_rerun/variants/o25_full_llm_nested_oof.csv",
         "edaic_rerun/variants/o25_full_ans_nested_oof.csv",
         "edaic_rerun/variants/o25_full_proj_nested_oof.csv",
         "edaic_rerun/variants/o25_full_states.npz",
         "edaic_rerun/variants/o25_full_nested.json"]
for c in cands:
    print(f"  {'EXISTS' if os.path.exists(os.path.join(REL,c)) else 'ABSENT'}  {c}")

nr = json.load(open(os.path.join(REL,"edaic_rerun/variants/o25_full_nested_repeats.json")))
print("\n  o25_full_nested_repeats.json means and recomputed means:")
for k2, v in nr.items():
    m = float(np.mean(v["per_repeat"]))
    ok = "OK " if abs(m - v["mean"]) < 5e-5 else "!!!"
    print(f"    {ok} {k2:5s} stored={v['mean']:.4f} recomputed_mean={m:.4f} "
          f"repeats={v['repeats']} n={v['n']}")
    R[f"json_{k2}_mean"] = {"value": v["mean"], "recomputed": round(m,4)}

print("\n  json point deltas vs recomputed full answer AUC "
      f"({AUC['ans_full']:.4f}):")
for k2 in ["llm","enc","ans","proj"]:
    print(f"    {k2:5s} {nr[k2]['mean']:.4f} - {AUC['ans_full']:.4f} = "
          f"{nr[k2]['mean']-AUC['ans_full']:+.4f}")
    R[f"point_{k2}_minus_answer"] = {"value": round(nr[k2]['mean']-AUC['ans_full'],4)}

json.dump(R, open(os.path.join(OUT,"MEDAIC_verify_results.json"),"w"), indent=1)
print(f"\nwrote {OUT}/MEDAIC_verify_results.json")
