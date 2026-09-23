"""
INDEPENDENT verifier for PART16 POD2.
Written from scratch. Does NOT import stats.py / verify_pitt.py / eval_pitt.py.

AUC implementations (two, both independent of any rank-sum one-liner):
  auc_bruteforce : explicit O(n_pos*n_neg) concordant-pair count, ties credited 0.5
  auc_trapezoid  : trapezoid area under the full empirical ROC built from sorted
                   unique thresholds (handles ties by grouping equal scores)

Bootstrap: 2000 draws, numpy.random.default_rng(0) fresh per cell,
           SPEAKERS resampled with replacement, percentile 2.5 / 97.5.
Paired:    one speaker draw per replicate, both quantities recomputed inside it.
"""
import csv, json, os, sys, hashlib
import numpy as np

REL   = "<local data dir>/Desktop/release"
POD2  = f"{REL}/edaic_rerun/part16/POD2"
OMNI  = f"{REL}/omni_final"
NDRAW = 2000

# ---------------------------------------------------------------- AUC #1
def auc_bruteforce(y, s):
    """Explicit pairwise concordance. No ranking, no argsort."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    if pos.size == 0 or neg.size == 0: return float("nan")
    tot = 0.0
    # chunk to keep memory sane but this is still a literal pairwise comparison
    for k in range(0, pos.size, 512):
        blk = pos[k:k+512][:, None]          # (b,1)
        d = blk - neg[None, :]               # (b, n_neg)
        tot += float((d > 0).sum()) + 0.5 * float((d == 0).sum())
    return tot / (pos.size * neg.size)

# ---------------------------------------------------------------- AUC #2
def auc_trapezoid(y, s):
    """Trapezoid over the full empirical ROC. Independent code path."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0: return float("nan")
    thr = np.unique(s)[::-1]                 # descending unique thresholds
    tpr = [0.0]; fpr = [0.0]
    for t in thr:
        sel = s >= t
        tpr.append(float(((y == 1) & sel).sum()) / P)
        fpr.append(float(((y == 0) & sel).sum()) / N)
    tpr = np.asarray(tpr); fpr = np.asarray(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

# ---------------------------------------------------------------- bootstrap
def spk_index(spk):
    spk = np.asarray(spk)
    u = np.unique(spk)
    return u, {g: np.flatnonzero(spk == g) for g in u}

def boot_ci(y, s, spk, ndraw=NDRAW, seed=0):
    y = np.asarray(y); s = np.asarray(s); spk = np.asarray(spk)
    u, where = spk_index(spk)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(ndraw):
        pick = rng.choice(u, size=u.size, replace=True)
        ii = np.concatenate([where[g] for g in pick])
        a = auc_bruteforce(y[ii], s[ii])
        if a == a: vals.append(a)
    v = np.asarray(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size)

def boot_paired_arms(y, s, spk, m_a, m_b, ndraw=NDRAW, seed=0):
    """one speaker draw per replicate; AUC(arm a) - AUC(arm b) inside that draw"""
    y = np.asarray(y); s = np.asarray(s); spk = np.asarray(spk)
    m_a = np.asarray(m_a, bool); m_b = np.asarray(m_b, bool)
    u, where = spk_index(spk)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(ndraw):
        pick = rng.choice(u, size=u.size, replace=True)
        ii = np.concatenate([where[g] for g in pick])
        ia = ii[m_a[ii]]; ib = ii[m_b[ii]]
        if ia.size == 0 or ib.size == 0: continue
        aa = auc_bruteforce(y[ia], s[ia]); bb = auc_bruteforce(y[ib], s[ib])
        if aa == aa and bb == bb: d.append(aa - bb)
    d = np.asarray(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), int(d.size)

def boot_paired_models(y, s1, s2, spk, ndraw=NDRAW, seed=0):
    """one speaker draw per replicate; AUC(score1) - AUC(score2) on the same rows"""
    y = np.asarray(y); s1 = np.asarray(s1); s2 = np.asarray(s2); spk = np.asarray(spk)
    u, where = spk_index(spk)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(ndraw):
        pick = rng.choice(u, size=u.size, replace=True)
        ii = np.concatenate([where[g] for g in pick])
        a1 = auc_bruteforce(y[ii], s1[ii]); a2 = auc_bruteforce(y[ii], s2[ii])
        if a1 == a1 and a2 == a2: d.append(a1 - a2)
    d = np.asarray(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), int(d.size)

# ---------------------------------------------------------------- io
def rd(p):
    with open(p) as fh: return list(csv.DictReader(fh))

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def base(p): return os.path.basename(p)

OUT = {}
def rec(k, v):
    OUT[k] = v
    print(f"{k}: {json.dumps(v)}", flush=True)

# ================================================================ load
p_lora = f"{POD2}/lora_pitt_oof.csv"
p_proj = f"{OMNI}/omnisft_pitt_oof.csv"
p_zs   = f"{OMNI}/omni_pitt_zeroshot_scores.csv"
p_abl  = f"{OMNI}/abl_pitt_oof.csv"
for p in (p_lora, p_proj, p_zs, p_abl):
    assert os.path.exists(p), f"MISSING {p}"

L = rd(p_lora); Pj = rd(p_proj); Z = rd(p_zs); A = rd(p_abl)
rec("file_rows", {"lora": len(L), "projector": len(Pj), "zeroshot": len(Z), "abl_lora": len(A)})
rec("lora_md5", md5(p_lora))

yL   = np.array([int(r["label"]) for r in L])
sL   = np.array([float(r["p_yes"]) for r in L])
sLmv = np.array([float(r["p_yes_multivariant"]) for r in L])
mass = np.array([float(r["answer_mass"]) for r in L])
kL   = np.array([r["spk"] for r in L])
arm  = np.array([r["arm"] for r in L])
nmL  = np.array([r["name"] for r in L])
fold = np.array([int(r["fold"]) for r in L])

rec("lora_counts", {"n": len(L), "n_spk": int(np.unique(kL).size),
                    "n_conflict": int((arm == "conflict").sum()),
                    "n_agreement": int((arm == "agreement").sum()),
                    "n_pos": int((yL == 1).sum()),
                    "nspk_conflict": int(np.unique(kL[arm == "conflict"]).size),
                    "nspk_agreement": int(np.unique(kL[arm == "agreement"]).size)})
rec("fold_sizes", np.bincount(fold, minlength=5).tolist())
rec("fold_positives", [int(yL[fold == k].sum()) for k in range(5)])
rec("speakers_split_across_folds",
    int(sum(1 for g in np.unique(kL) if np.unique(fold[kL == g]).size > 1)))

# ---- AUC implementation cross-check
a_bf = auc_bruteforce(yL, sL); a_tz = auc_trapezoid(yL, sL)
rec("impl_crosscheck_bruteforce_vs_trapezoid",
    {"bruteforce": a_bf, "trapezoid": a_tz, "absdiff": abs(a_bf - a_tz)})

# ================================================================ L cells
def cell(tag, y, s, spk, seed=0):
    a = auc_bruteforce(y, s)
    lo, hi, k = boot_ci(y, s, spk, seed=seed)
    rec(tag, {"auc": round(a, 6), "ci": [round(lo, 6), round(hi, 6)],
              "draws": k, "n": int(len(y)), "n_spk": int(np.unique(spk).size)})
    return a, lo, hi

cell("L1_lora_all", yL, sL, kL)
mC = arm == "conflict"; mA = arm == "agreement"
cell("L2_lora_conflict", yL[mC], sL[mC], kL[mC])
cell("L3_lora_agreement", yL[mA], sL[mA], kL[mA])
d = auc_bruteforce(yL[mC], sL[mC]) - auc_bruteforce(yL[mA], sL[mA])
lo, hi, k = boot_paired_arms(yL, sL, kL, mC, mA)
rec("L4_lora_paired_conflict_minus_agreement",
    {"diff": round(d, 6), "ci": [round(lo, 6), round(hi, 6)], "draws": k})

cell("L7_lora_all_multivariant", yL, sLmv, kL)
rec("L8_answer_mass", {"median": round(float(np.median(mass)), 6),
                       "min": round(float(mass.min()), 6),
                       "max": round(float(mass.max()), 6),
                       "n_below_0.9": int((mass < 0.9).sum())})

# ================================================================ projector / zeroshot / prior lora
def prep(rows, namecol, scorecol="p_yes", setcol=None):
    nm = np.array([base(r[namecol]) for r in rows])
    y  = np.array([int(r["label"]) for r in rows])
    s  = np.array([float(r[scorecol]) for r in rows])
    k  = np.array([r["speaker"] for r in rows])
    st = np.array([r[setcol] for r in rows]) if setcol else None
    return nm, y, s, k, st

nmP, yP, sP, kP, stP = prep(Pj, "path", setcol="set")
nmZ, yZ, sZ, kZ, _   = prep(Z, "clip")
nmA, yA, sA, kA, stA = prep(A, "path", setcol="set")

cell("V1_projector_all", yP, sP, kP)
cell("V3_projector_conflict", yP[stP == "conflict"], sP[stP == "conflict"], kP[stP == "conflict"])
cell("V4_projector_agreement", yP[stP == "agreement"], sP[stP == "agreement"], kP[stP == "agreement"])
dv = auc_bruteforce(yP[stP == "conflict"], sP[stP == "conflict"]) - \
     auc_bruteforce(yP[stP == "agreement"], sP[stP == "agreement"])
lo, hi, k = boot_paired_arms(yP, sP, kP, stP == "conflict", stP == "agreement")
rec("V5_projector_paired_conflict_minus_agreement",
    {"diff": round(dv, 6), "ci": [round(lo, 6), round(hi, 6)], "draws": k})

cell("V2_zeroshot_all", yZ, sZ, kZ)

cell("X1_prior_lora_all", yA, sA, kA)
cell("X2_prior_lora_conflict", yA[stA == "conflict"], sA[stA == "conflict"], kA[stA == "conflict"])
cell("X3_prior_lora_agreement", yA[stA == "agreement"], sA[stA == "agreement"], kA[stA == "agreement"])
dx = auc_bruteforce(yA[stA == "conflict"], sA[stA == "conflict"]) - \
     auc_bruteforce(yA[stA == "agreement"], sA[stA == "agreement"])
lo, hi, k = boot_paired_arms(yA, sA, kA, stA == "conflict", stA == "agreement")
rec("X4_prior_lora_paired_conflict_minus_agreement",
    {"diff": round(dx, 6), "ci": [round(lo, 6), round(hi, 6)], "draws": k})

# ================================================================ cross-file alignment + paired deltas
def align(nm_ref, y_ref, k_ref, nm_o, y_o, s_o, k_o, tag):
    pos = {n: i for i, n in enumerate(nm_o)}
    assert len(pos) == len(nm_o), f"{tag}: duplicate names"
    miss = [n for n in nm_ref if n not in pos]
    idx = np.array([pos[n] for n in nm_ref])
    rec(f"align_{tag}", {"missing": len(miss),
                         "label_mismatches": int((y_o[idx] != y_ref).sum()),
                         "speaker_mismatches": int((k_o[idx] != k_ref).sum())})
    return s_o[idx]

sP_al = align(nmL, yL, kL, nmP, yP, sP, kP, "projector")
sZ_al = align(nmL, yL, kL, nmZ, yZ, sZ, kZ, "zeroshot")
sA_al = align(nmL, yL, kL, nmA, yA, sA, kA, "prior_lora")

for tag, s1, s2 in [("D1_lora_minus_projector", sL, sP_al),
                    ("D2_lora_minus_zeroshot",  sL, sZ_al),
                    ("D3_projector_minus_zeroshot", sP_al, sZ_al)]:
    dd = auc_bruteforce(yL, s1) - auc_bruteforce(yL, s2)
    lo, hi, k = boot_paired_models(yL, s1, s2, kL)
    rec(tag, {"diff": round(dd, 6), "ci": [round(lo, 6), round(hi, 6)], "draws": k})

# ================================================================ fresh vs prior LoRA overlap
same = int((sL == sA_al).sum())
r_p = float(np.corrcoef(sL, sA_al)[0, 1])
rk1 = np.argsort(np.argsort(sL)); rk2 = np.argsort(np.argsort(sA_al))
r_s = float(np.corrcoef(rk1, rk2)[0, 1])
rec("fresh_vs_prior_lora", {"bit_identical_clips": same, "pearson": round(r_p, 4),
                            "spearman": round(r_s, 4), "auc_gap": round(a_bf - auc_bruteforce(yL, sA_al), 6)})

# ================================================================ arms cross-check vs manifest
MAN = "manifests/pitt_conflict_manifest_468.csv"
if os.path.exists(MAN):
    M = rd(MAN)
    mm = {base(r["segment_path"]): r for r in M}
    bad_arm = sum(1 for i, n in enumerate(nmL) if n not in mm or mm[n]["set"] != arm[i])
    bad_lab = sum(1 for i, n in enumerate(nmL) if n not in mm or int(mm[n]["label"]) != yL[i])
    rec("manifest_crosscheck", {"manifest_rows": len(M), "missing": int(sum(1 for n in nmL if n not in mm)),
                                "arm_mismatch": bad_arm, "label_mismatch": bad_lab})
else:
    rec("manifest_crosscheck", "MANIFEST NOT MOUNTED")

# ================================================================ param arithmetic
H, R_ = 3584, 8
KV = 512
per_layer = R_*(H+H) + R_*(H+KV) + R_*(H+KV) + R_*(H+H)
rec("param_arithmetic", {
    "lora_per_layer": per_layer, "lora_28_layers": per_layer*28,
    "claimed_lora": 5046272, "lora_match": per_layer*28 == 5046272,
    "projector_1280x3584_plus_bias": 1280*3584+3584,
    "claimed_projector": 4591104, "proj_match": (1280*3584+3584) == 4591104,
    "lora_pct_full": round(100*5046272/10732225408, 6),
    "lora_pct_thinker": round(100*5046272/8255263744, 6),
    "proj_pct_full": round(100*4591104/10732225408, 6),
    "proj_pct_thinker": round(100*4591104/8255263744, 6),
    "lora_over_projector_ratio": round(5046272/4591104, 4)})

with open(f"{os.path.dirname(os.path.abspath(__file__))}/POD2_verify_out.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("\nDONE")
