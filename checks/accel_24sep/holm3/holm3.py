#!/usr/local/bin/python3
"""holm3: Holm correction for three families of paired differences (ARMS, FINE TUNE, CONTROLS).

Independent code. Reads per-clip files only. Writes only into this folder.
AUC = sklearn roc_auc_score. Speaker bootstrap: 2000 draws, fresh numpy default_rng(0) per cell,
unique speakers (sorted) resampled with replacement, all their clips taken, 2.5 / 97.5 percentiles.
Paired = both terms computed on the same draw. A draw where any term has a single class is skipped.
Two-sided p = min(1, 2*min((1+#d<=0)/(B+1), (1+#d>=0)/(B+1))), B = usable draws (2000 when none skipped).
Holm at 0.05 within each family.
"""
import os, json, hashlib, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

OUT = "checks/accel_24sep/holm3"
NB = 2000
REL = "<local data dir>/release"
LR = "<local data dir>/paper1_local_runs"
S = "<local data dir>/release_from_mac/scores"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def base(x):
    return os.path.basename(str(x))


def fnum(x):
    x = str(x)
    if x.startswith("np.float64("):
        x = x[len("np.float64("):-1]
    return float(x)


def paired_boot(spk, y, terms, sign):
    """terms: list of (mask, score) ; statistic = sum(sign[k]*AUC_k). spk, y arrays over clips."""
    spk = np.asarray(spk).astype(str)
    y = np.asarray(y).astype(int)
    u = np.unique(spk)
    idx_by = {s: np.where(spk == s)[0] for s in u}

    def stat(ii):
        v = 0.0
        for (m, sc), sg in zip(terms, sign):
            jj = ii[m[ii]]
            yy = y[jj]
            if yy.min() == yy.max():
                return None
            v += sg * roc_auc_score(yy, sc[jj])
        return v

    allidx = np.arange(len(y))
    point = stat(allidx)
    pts = [roc_auc_score(y[m], sc[m]) for (m, sc) in terms]
    rng = np.random.default_rng(0)
    d = []
    skipped = 0
    for _ in range(NB):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx_by[p] for p in pick])
        v = stat(ii)
        if v is None:
            skipped += 1
            continue
        d.append(v)
    d = np.array(d)
    B = len(d)
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    p = min(1.0, 2 * min((1 + (d <= 0).sum()) / (B + 1), (1 + (d >= 0).sum()) / (B + 1)))
    return dict(value=point, lo=lo, hi=hi, p=p, B=B, skipped=skipped, n_clips=int(sum(m.sum() for m, _ in terms)),
                n_spk=len(u), term_aucs=pts, n_le0=int((d <= 0).sum()), n_ge0=int((d >= 0).sum()))


def holm(ps, alpha=0.05):
    ps = np.asarray(ps, float)
    m = len(ps)
    o = np.argsort(ps, kind="mergesort")
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj, adj < alpha


rows = []
files_used = {}


def reg(k, p):
    files_used[k] = (p, sha(p))


# ---------------- ARMS: E-DAIC ----------------
f = f"{REL}/edaic_rerun/part17/T1_perclip_all966.csv"
reg("edaic_arms", f)
E = pd.read_csv(f)
conf = E[E.set == "conflict"]
agr = E[(E.set == "agreement") & (E.in_distinct_set == 1)]
assert len(conf) == 483 and len(agr) == 311, (len(conf), len(agr))
EE = pd.concat([conf, agr], ignore_index=True)
mc = (EE.set == "conflict").values
ma = (EE.set == "agreement").values
tex_edaic = {"p_q2a": ("Qwen2-Audio", 0.23, 0.96), "p_o25": ("Qwen2.5-Omni", 0.22, 0.96), "p_q3o": ("Qwen3-Omni", 0.30, 0.97),
             "p_af2": ("Audio Flamingo 2", 0.53, 0.63), "p_af3": ("Audio Flamingo 3", 0.42, 0.84), "p_kimi": ("Kimi-Audio", 0.53, 0.89)}
for col, (name, tc, ta) in tex_edaic.items():
    sc = EE[col].values.astype(float)
    r = paired_boot(EE.speaker_id.values, EE.label.values, [(mc, sc), (ma, sc)], [1, -1])
    r.update(family="ARMS", claim=f"E-DAIC conflict minus agreement, {name}", tex_line=180, claimed=True,
             tex_value=f"Table 2 (tex l.168-173): {tc:.2f} / {ta:.2f}; l.180 'intervals ... exclude zero for all six models on depression'",
             file=f)
    rows.append(r)

# ---------------- ARMS: Pitt ----------------
pitt = [
    ("Qwen2-Audio", f"{LR}/probe2/pitt_zeroshot_scores.csv", "clip", "speaker", "p_yes", True, (0.41, 0.70)),
    ("Qwen2.5-Omni", f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv", "clip", "speaker", "p_yes", True, (0.53, 0.71)),
    ("Qwen3-Omni", f"{REL}/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv", "clip", "speaker", "p_yes", True, (0.61, 0.82)),
    ("Audio Flamingo 2", f"{LR}/af2_results/af2_pitt468.csv", "clip_path", "speaker_id", "p_yes", False, (0.58, 0.53)),
    ("Audio Flamingo 3", f"{REL}/overnight2/part3/af3_pitt.csv", "clip_path", "speaker_id", "p_yes", False, (0.61, 0.59)),
    ("Kimi-Audio", f"{REL}/overnight2/part3/kimi_pitt_zeroshot_scores.csv", "clip", "speaker", "p_yes", False, (0.72, 0.72)),
]
for name, f, cc, sc_, pc, claimed, (tc, ta) in pitt:
    reg(f"pitt_{name}", f)
    P = pd.read_csv(f)
    assert len(P) == 468, (name, len(P))
    b = P[cc].map(base)
    mc = b.str.startswith("conflict_").values
    ma = b.str.startswith("agreement_").values
    assert mc.sum() == 146 and ma.sum() == 322, (name, mc.sum(), ma.sum())
    s = P[pc].values.astype(float)
    r = paired_boot(P[sc_].values, P.label.values, [(mc, s), (ma, s)], [1, -1])
    r.update(family="ARMS", claim=f"Pitt conflict minus agreement, {name}", tex_line=180, claimed=claimed,
             tex_value=(f"Table 2: {tc:.2f} / {ta:.2f}; l.180 'exclude zero ... for the three Qwen models on Pitt'" if claimed
                        else f"Table 2: {tc:.2f} / {ta:.2f}; not claimed significant (family member only)"),
             file=f)
    rows.append(r)


# ---------------- FINE TUNE ----------------
def pair_files(fa, ka, sa, ya, pa, fb, kb, sb, yb, pb, conv_a=float, conv_b=float):
    A = pd.read_csv(fa)
    B = pd.read_csv(fb)
    A["_k"] = A[ka].map(base)
    B["_k"] = B[kb].map(base)
    assert A._k.is_unique and B._k.is_unique, (fa, fb)
    M = A.merge(B, on="_k", suffixes=("_a", "_b"))
    assert len(M) == len(A) == len(B), (fa, fb, len(M), len(A), len(B))

    def col(c, suf, df):
        return df[c + suf] if (c + suf) in df.columns else df[c]
    spa = col(sa, "_a", M).astype(str).values
    spb = col(sb, "_b", M).astype(str).values
    la = col(ya, "_a", M).astype(int).values
    lb = col(yb, "_b", M).astype(int).values
    assert (la == lb).all(), "label mismatch"
    assert (spa == spb).all(), "speaker mismatch"
    xa = np.array([conv_a(v) for v in col(pa, "_a", M).values])
    xb = np.array([conv_b(v) for v in col(pb, "_b", M).values])
    return spa, la, xa, xb


ft = [
    ("PC-GITA", "pcgita", 0.56, 0.89, True),
    ("NeuroVoz", "neurovoz", 0.70, 0.90, True),
    ("MDVR-KCL", "kcl", 0.71, 0.70, False),
    ("Pitt", "pitt", 0.66, 0.77, True),
    ("ADReSSo", "adresso", 0.62, 0.82, True),
    ("ADReSS-2020", "adress2020", 0.62, 0.77, True),
]
for name, key, tz, tf, claimed in ft:
    fz = f"{REL}/omni_final/omni_{key}_zeroshot_scores.csv"
    ff = f"{REL}/omni_final/omnisft_{key}_oof.csv"
    reg(f"zs_{key}", fz); reg(f"ft_{key}", ff)
    spk, y, z, t = pair_files(fz, "clip", "speaker", "label", "p_yes", ff, "path", "speaker", "label", "p_yes")
    allm = np.ones(len(y), bool)
    r = paired_boot(spk, y, [(allm, t), (allm, z)], [1, -1])
    r.update(family="FINE TUNE", claim=f"projector fine-tune minus zero-shot, {name}", tex_line=226, claimed=claimed,
             tex_value=(f"Table 1 (l.120-130): {tz:.2f} -> {tf:.2f}; l.226 'improves significantly on every dataset except MDVR-KCL ... and E-DAIC'"
                        if claimed else f"Table 1: {tz:.2f} -> {tf:.2f}; l.226 names it as the exception (not significant)"),
             file=f"{ff} vs {fz}")
    rows.append(r)

# E-DAIC whole interview (Table 1 0.84 -> 0.86), l.226 '+0.02 [-0.01, 0.05]'
fz = f"{S}/part23/A/A2_edaic_whole_zeroshot_perclip.csv"
ff = f"{S}/part24/FT/FT24_whole_seed0_perclip.csv"
reg("zs_edaic_whole", fz); reg("ft_edaic_whole_seed0", ff)
Z = pd.read_csv(fz); F = pd.read_csv(ff)
Z["_k"] = Z.pid.astype(str); F["_k"] = F.pid.astype(str)
M = Z.merge(F, on="_k", suffixes=("_z", "_f"))
assert len(M) == 275 and (M.label_z.values == M.label_f.values).all()
z = M.p_yes_whole.map(fnum).values; t = M.p_yes.astype(float).values
allm = np.ones(len(M), bool)
r = paired_boot(M._k.values, M.label_z.values, [(allm, t), (allm, z)], [1, -1])
r.update(family="FINE TUNE", claim="projector fine-tune minus zero-shot, E-DAIC whole interview (seed 0)", tex_line=226, claimed=False,
         tex_value="Table 1: 0.84 -> 0.86; l.226 '+0.02 [-0.01, 0.05]' (not significant)", file=f"{ff} vs {fz}")
rows.append(r)

# LoRA minus projector, Pitt
fl = f"{REL}/edaic_rerun/part16/POD2/lora_pitt_oof.csv"
fp = f"{REL}/omni_final/omnisft_pitt_oof.csv"
reg("lora_pitt", fl)
spk, y, l, pj = pair_files(fl, "name", "spk", "label", "p_yes", fp, "path", "speaker", "label", "p_yes")
allm = np.ones(len(y), bool)
r = paired_boot(spk, y, [(allm, l), (allm, pj)], [1, -1])
r.update(family="FINE TUNE", claim="LoRA minus projector fine-tune, Pitt", tex_line=226, claimed=True,
         tex_value="l.226 'reaches 0.825 on Pitt, 0.06 [0.02, 0.10] above the projector run'", file=f"{fl} vs {fp}")
rows.append(r)

# ---------------- CONTROLS ----------------
P4 = f"{REL}/edaic_rerun/part16/POD4"
for name, key, tex in [("PC-GITA", "pg", "l.149 'encoder probe falls to 0.83 (-0.07 [-0.12, -0.02])'"),
                       ("NeuroVoz", "nv", "l.149 'probe falls to 0.89 (-0.03 [-0.061, -0.001])'")]:
    fb = f"{P4}/p16_{key}_before_enc_nested_oof.csv"; fa = f"{P4}/p16_{key}_after_enc_nested_oof.csv"
    reg(f"ctrl_{key}_before", fb); reg(f"ctrl_{key}_after", fa)
    spk, y, b_, a_ = pair_files(fb, "clip", "speaker", "label", "p_probe", fa, "clip", "speaker", "label", "p_probe")
    allm = np.ones(len(y), bool)
    r = paired_boot(spk, y, [(allm, a_), (allm, b_)], [1, -1])
    r.update(family="CONTROLS", claim=f"encoder probe after minus before channel normalisation, {name}", tex_line=149, claimed=True,
             tex_value=tex, file=f"{fa} vs {fb}")
    rows.append(r)
# sensitivity members: zero-shot answer before/after (tex: 'the answer stays at 0.57' on PC-GITA; no claim of significance)
for name, key in [("PC-GITA", "pg"), ("NeuroVoz", "nv")]:
    fb = f"{P4}/p16_{key}_before_zeroshot_scores.csv"; fa = f"{P4}/p16_{key}_after_zeroshot_scores.csv"
    reg(f"ctrlzs_{key}_before", fb); reg(f"ctrlzs_{key}_after", fa)
    spk, y, b_, a_ = pair_files(fb, "clip", "speaker", "label", "p_yes", fa, "clip", "speaker", "label", "p_yes")
    allm = np.ones(len(y), bool)
    r = paired_boot(spk, y, [(allm, a_), (allm, b_)], [1, -1])
    r.update(family="CONTROLS", claim=f"zero-shot answer after minus before channel normalisation, {name}", tex_line=149, claimed=False,
             tex_value="not claimed significant (family member only, sensitivity)", file=f"{fa} vs {fb}")
    rows.append(r)

R = pd.DataFrame(rows)
# Holm: (a) over claimed-significant rows only (the job spec); (b) over every test in the family (stricter)
R["holm_claimed"] = np.nan; R["survives_claimed"] = None
R["holm_full"] = np.nan; R["survives_full"] = None
R["m_claimed"] = 0; R["m_full"] = 0
for fam in R.family.unique():
    ic = R.index[(R.family == fam) & (R.claimed)]
    adj, sv = holm(R.loc[ic, "p"].values)
    R.loc[ic, "holm_claimed"] = adj; R.loc[ic, "survives_claimed"] = sv; R.loc[ic, "m_claimed"] = len(ic)
    ia = R.index[R.family == fam]
    adj, sv = holm(R.loc[ia, "p"].values)
    R.loc[ia, "holm_full"] = adj; R.loc[ia, "survives_full"] = sv; R.loc[ia, "m_full"] = len(ia)

R["excl0"] = (R.lo > 0) | (R.hi < 0)
cols = ["family", "claim", "claimed", "tex_line", "tex_value", "value", "lo", "hi", "p", "holm_claimed", "survives_claimed",
        "m_claimed", "holm_full", "survives_full", "m_full", "excl0", "B", "skipped", "n_le0", "n_ge0", "n_clips", "n_spk", "term_aucs", "file"]
R[cols].to_csv(f"{OUT}/holm3_results.csv", index=False)
json.dump({k: {"path": v[0], "sha256_16": v[1]} for k, v in files_used.items()}, open(f"{OUT}/holm3_inputs.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
for _, r in R.iterrows():
    print(f"{r.family:9s} | {r.claim:70s} | l.{r.tex_line} | {r.value:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] | p={r.p:.4f} | "
          f"Holm(claimed)={r.holm_claimed if r.claimed else float('nan'):.4f} {r.survives_claimed} | Holm(full)={r.holm_full:.4f} {r.survives_full} | "
          f"B={r.B} skip={r.skipped} spk={r.n_spk} aucs={[round(a,4) for a in r.term_aucs]}")
