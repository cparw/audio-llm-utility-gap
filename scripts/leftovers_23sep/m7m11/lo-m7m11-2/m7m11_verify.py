"""Independent verifier for m7m11, run on a Linux CPU pod (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1, Python 3.12).
Deliberately different code from the Mac scripts: AUC from sklearn.metrics.roc_auc_score, speaker rows from
pandas groupby().indices, speaker list from Python sorted(set()), bootstrap index drawn as rng.choice(N, size=N,
replace=True) and mapped through that list, percentiles via np.quantile. Reads only the staged input files.
usage: python3 m7m11_verify.py data out/m7m11_verify.csv"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

D, OUTCSV = sys.argv[1], sys.argv[2]
NB = 2000
rows = []


def A(y, s):
    y = np.asarray(y, int)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, np.asarray(s, float)))


def spk_draws(spk, mask=None):
    s = pd.Series(np.asarray(spk).astype(str))
    if mask is not None:
        s = s[np.asarray(mask, bool)]
    ind = s.groupby(s).indices            # label -> positions within s
    lab = sorted(set(s.tolist()))
    pos = s.index.to_numpy()
    rng = np.random.default_rng(0)
    out = []
    for _ in range(NB):
        k = rng.choice(len(lab), size=len(lab), replace=True)
        out.append(np.concatenate([pos[ind[lab[j]]] for j in k]))
    return out, len(lab)


def q(v):
    v = np.array([x for x in v if not np.isnan(x)], float)
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)), len(v)


def put(i, v, lo=None, hi=None, extra=""):
    rows.append(dict(id=i, value=v, lo=lo, hi=hi, extra=extra))


def paired_ci(y, probes, z, spk):
    ds, nsp = spk_draws(spk)
    out = []
    for ii in ds:
        yy = y[ii]
        if yy.min() == yy.max():
            out.append(np.nan); continue
        out.append(np.mean([A(yy, p[ii]) for p in probes]) - A(yy, z[ii]))
    return q(out)


# ---------------- M7, five datasets
for ds in ["adresso", "adress2020", "pcgita", "neurovoz", "kcl"]:
    rep = json.load(open(f"{D}/omni_{ds}_nested_repeats.json"))["enc"]
    p = pd.read_csv(f"{D}/omni_{ds}_enc_nested_oof.csv"); z = pd.read_csv(f"{D}/omni_{ds}_zeroshot_scores.csv")
    m = pd.merge(p, z[["clip", "p_yes"]], on="clip", how="inner")
    y = m["label"].to_numpy(int); s1 = m["p_probe"].to_numpy(float); zz = m["p_yes"].to_numpy(float); sp = m["speaker"].astype(str).to_numpy()
    a_zs = A(y, zz); a1 = A(y, s1); mean5 = float(rep["mean"])
    lo1, hi1, _ = paired_ci(y, [s1], zz, sp)
    sh = mean5 - a1
    put(f"M7mo5_{ds}_gap", mean5 - a_zs, lo1 + sh, hi1 + sh)
    put(f"M7mo5_{ds}_probe_mean5", mean5, extra=f"mean of rounded per_repeat {np.mean(rep['per_repeat']):.6f}")
    ds_, _ = spk_draws(sp)
    zb = [A(y[ii], zz[ii]) for ii in ds_]
    zl, zh, _ = q(zb)
    put(f"M7mo5_{ds}_zeroshot", a_zs, zl, zh)
    put(f"M7mo5_{ds}_single_gap_ref", a1 - a_zs, lo1, hi1)


# ---------------- validation runs
def val(tag, y, reps, z, sp, single=None):
    y = np.asarray(y, int); reps = [np.asarray(r, float) for r in reps]; z = np.asarray(z, float)
    mean5 = float(np.mean([A(y, r) for r in reps])); g = mean5 - A(y, z)
    lo, hi, _ = paired_ci(y, reps, z, sp); put(f"VAL_{tag}_exact", g, lo, hi)
    prox = [("single", np.asarray(single, float))] if single is not None else [(f"repeat{r}", reps[r]) for r in range(len(reps))]
    for nm, pr in prox:
        l1, h1, _ = paired_ci(y, [pr], z, sp); sh = mean5 - A(y, pr)
        put(f"VAL_{tag}_shifted_{nm}", g, l1 + sh, h1 + sh)


zp = np.load(f"{D}/pitt_enc_nested5_oof.npz", allow_pickle=True)
zs_p = pd.read_csv(f"{D}/omni_pitt_zeroshot_scores.csv").set_index("clip")["p_yes"]
val("pitt_part10", zp["label"], list(zp["oof"]), zs_p.loc[[os.path.basename(str(n)) for n in zp["name"]]].to_numpy(float), zp["spk"])
t = pd.read_csv(f"{D}/T7b_edaic300_meanof5_perclip.csv")
val("edaic300_T7b", t["label"], [t[f"oof_enc_r{r}"] for r in range(5)], t["p_yes_answer"], t["pid"])
nv = pd.read_csv(f"{D}/omni_nvfull_enc_nested5_oof.csv").merge(pd.read_csv(f"{D}/omni_neurovoz_zeroshot_scores.csv")[["clip", "p_yes"]], on="clip")
val("neurovoz_pod6_reextract", nv["label"], [nv[f"p_seed{r}"] for r in range(5)], nv["p_yes"], nv["speaker"], nv["p_single"])
a5 = pd.read_csv(f"{D}/adress2020_pod4b_enc_rep5_oof.csv").merge(pd.read_csv(f"{D}/p16_adress2020_orig_enc_nested_oof.csv")[["clip", "p_probe"]], on="clip") \
    .merge(pd.read_csv(f"{D}/omni_adress2020_zeroshot_scores.csv")[["clip", "p_yes"]], on="clip")
val("adress2020_pod4b_reextract", a5["label"], [a5[f"p_seed{r}"] for r in range(5)], a5["p_yes"], a5["speaker"], a5["p_probe"])

# ---------------- M11
k = pd.read_csv(f"{D}/M11_kcl_wer.csv")
sp = k["spk"].astype(str).to_numpy(); G = k["group"].to_numpy()
ge20 = (k["xsys_n_ref_words"] >= 20).to_numpy()
def mean_of(c): v = k[c].to_numpy(float); return lambda ii: float(np.mean(v[ii])) if len(ii) else np.nan
def med_of(c): v = k[c].to_numpy(float); return lambda ii: float(np.median(v[ii])) if len(ii) else np.nan
def pooled(ii):
    if not len(ii): return np.nan
    return float((k["xsys_S"].to_numpy()[ii] + k["xsys_D"].to_numpy()[ii] + k["xsys_I"].to_numpy()[ii]).sum() / k["xsys_n_ref_words"].to_numpy()[ii].sum())
allm = np.ones(len(k), bool)
cells = [("B_xsys_disagree_mean", mean_of("xsys_disagree"), allm), ("C_mean_token_logprob_mean", mean_of("mean_token_logprob"), allm),
         ("C_compression_ratio_mean", mean_of("compression_ratio"), allm), ("D_n_words_30s_mean", mean_of("n_words_30s"), allm),
         ("F_wer_vs_local_copy_mean", mean_of("wer_local"), allm), ("B_xsys_disagree_MEDIAN", med_of("xsys_disagree"), allm),
         ("B_xsys_disagree_ge20w_MEAN", mean_of("xsys_disagree"), ge20), ("B_xsys_disagree_POOLED", pooled, allm)]
for key, f, sub in cells:
    pt = {}
    for g in ("PD", "control"):
        cell = sub & (G == g)
        pt[g] = f(np.flatnonzero(cell))
        dr, _ = spk_draws(sp, cell)                       # within the cell's own speakers (the rule)
        lo, hi, _ = q([f(ii) for ii in dr])
        da, _ = spk_draws(sp)                             # all 37 then keep the cell (the old verifier scheme)
        la, ha, _ = q([f(ii[cell[ii]]) if cell[ii].any() else np.nan for ii in da])
        put(f"M11fix_{key}_{g}", pt[g], lo, hi, extra=f"all37 [{la:.4f}, {ha:.4f}]")
    da, _ = spk_draws(sp)
    dd = []
    for ii in da:
        a = ii[(G[ii] == "PD") & sub[ii]]; b = ii[(G[ii] == "control") & sub[ii]]
        dd.append(f(a) - f(b) if len(a) and len(b) else np.nan)
    lo, hi, _ = q(dd)
    dk = key[:-5] + "_diff_PD_minus_control" if key.endswith("_mean") else key + "_diff_PD_minus_control"
    put(f"M11fix_{dk}", pt["PD"] - pt["control"], lo, hi)

pd.DataFrame(rows).to_csv(OUTCSV, index=False)
import sklearn, scipy, platform
json.dump(dict(sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, python=platform.python_version(),
               machine=platform.machine(), rows=len(rows)), open(OUTCSV.replace(".csv", "_env.json"), "w"), indent=1)
print("verify rows", len(rows))
