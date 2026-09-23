#!/usr/local/bin/python3
"""m7m11 (a): M7 gaps for the five non-Pitt datasets on the paper's probe convention.

Paper convention (Table 1): the Qwen2.5-Omni encoder probe value is the MEAN of the five per-repeat
nested AUCs (speakers renumbered with default_rng(seed).permutation, seeds 0..4, outer GroupKFold(5),
pod tie order). Gap = that mean minus the zero-shot answer AUC on the same clips.

What exists on disk for these five datasets (searched Mac release tree, G-Drive paper work, all session
transcripts; see sidecar): the five per-repeat AUCs (omni_final/omni_<ds>_nested_repeats.json) and the
single-split per-clip OOF (omni_final/omni_<ds>_enc_nested_oof.csv). The five per-repeat per-clip OOF
vectors were never saved, and the Table 1 states lived only on pod gv9jvm57b21mcz, which is deleted.
So a CPU refit on the Table 1 states is impossible and the exact paired interval (per draw: mean of the
five per-repeat AUCs minus the answer AUC) cannot be computed for these five datasets.

This script therefore writes:
  1. EXACT point gaps: Table 1 mean of five minus the zero-shot AUC recomputed from the per-clip file.
  2. An APPROXIMATE paired interval, labelled as such: the single-split paired bootstrap distribution
     (per draw AUC(single split OOF) minus AUC(answer), same speaker draw) shifted by
     (mean of five minus single-split AUC). It assumes the mean-of-five estimator has the same speaker
     sampling spread as one split.
  3. A validation of that approximation on every run where per-repeat per-clip OOF vectors DO exist:
     Pitt part10 (the Table 1 Pitt value 0.7706), E-DAIC 300 s T7b, NeuroVoz POD6 full re-extraction
     (same Table 1 outer splits), ADReSS-2020 POD4B re-extraction refit today on a Linux CPU pod
     (same Table 1 outer splits, 156/156 clips per seed). For each: exact paired interval vs the shifted
     single-split (or shifted single-repeat) interval, endpoint differences reported.
Bootstrap: 2000 draws; a fresh numpy default_rng(0) per cell; idx = rng.choice(N, size=N, replace=True)
over the unique sorted speaker ids (speaker column cast to str, as in part16/M7_gap_per_dataset.py);
percentiles 2.5 and 97.5; paired = the same idx for both AUCs.
"""
import os, sys, json, hashlib, platform
import numpy as np, pandas as pd, scipy, scipy.stats as st

OMNI = "<local data dir>/release/omni_final"
OUT = "<local data dir>/leftovers_23sep/m7m11"
PITT_NPZ = "<local data dir>/release/overnight2/part10/pitt_enc_nested5_oof.npz"
T7B = "<local data dir>/release/edaic_rerun/part17/T7b/T7b_edaic300_meanof5_perclip.csv"
NV6 = "<local data dir>/release/scores/part20/POD6/B_neurovoz_full_check/omni_nvfull_enc_nested5_oof.csv"
AD4B_SINGLE = "<local data dir>/release_from_mac/edaic_rerun/part16/POD4B/out/p16_adress2020_orig_enc_nested_oof.csv"
AD4B_REP5 = OUT + "/pull/lo-m7m11-1/out/adress2020_pod4b_enc_rep5_oof.csv"
AD4B_REP5_JSON = OUT + "/pull/lo-m7m11-1/out/adress2020_pod4b_enc_rep5.json"
OLD_M7 = "<local data dir>/release/edaic_rerun/part16/M7_gap_per_dataset.csv"
FOLDS = "<local data dir>/release/folds"
DATASETS = ["adresso", "adress2020", "pcgita", "neurovoz", "kcl"]
NB, SEED = 2000, 0


def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def draws(spk):
    """Fresh default_rng(0); idx over unique sorted speaker ids; returns list of row index arrays."""
    spk = np.asarray(spk).astype(str)
    u = np.unique(spk)
    rows = {s: np.flatnonzero(spk == s) for s in u}
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(NB):
        idx = rng.choice(len(u), size=len(u), replace=True)
        out.append(np.concatenate([rows[u[i]] for i in idx]))
    return out, len(u)


def pct(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))


def paired(y, probe_cols, zs, spk):
    """Per draw: mean over probe_cols of AUC minus AUC(zs). probe_cols: list of arrays."""
    D, nspk = draws(spk)
    diffs, probes, zss = [], [], []
    for ii in D:
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            diffs.append(np.nan); probes.append(np.nan); zss.append(np.nan); continue
        p = float(np.mean([auc(yy, c[ii]) for c in probe_cols]))
        q = auc(yy, zs[ii])
        diffs.append(p - q); probes.append(p); zss.append(q)
    return np.array(diffs), np.array(probes), np.array(zss), nspk


rows, side = [], {"task": "m7m11 (a): M7 non-Pitt gaps on the mean-of-five probe convention", "datasets": {},
                  "validation": {}, "inputs_sha256": {}}


def add(rid, what, value, lo, hi, n, nspk, kind, src, closes="STATUS_LEDGER 6.me.6 / 5.5 / spot check 5 (M7 non-Pitt gaps)", note=""):
    rows.append(dict(id=rid, what=what, value=round(value, 4) if value is not None else "",
                     lo="" if lo is None else round(lo, 4), hi="" if hi is None else round(hi, 4),
                     value_full=value, lo_full=lo, hi_full=hi, n=n, n_spk=nspk, kind=kind, source=src,
                     closes=closes, note=note))


# ------------------------------------------------------------------ 1 + 2: the five datasets
old = pd.read_csv(OLD_M7).set_index("dataset")
for ds in DATASETS:
    fj = f"{OMNI}/omni_{ds}_nested_repeats.json"; fp = f"{OMNI}/omni_{ds}_enc_nested_oof.csv"; fz = f"{OMNI}/omni_{ds}_zeroshot_scores.csv"
    for f in (fj, fp, fz):
        side["inputs_sha256"][f] = sha(f)
    rep = json.load(open(fj))["enc"]
    per = [float(v) for v in rep["per_repeat"]]; mean5 = float(rep["mean"])
    dp = pd.read_csv(fp); dz = pd.read_csv(fz)
    m = dp.merge(dz[["clip", "p_yes", "label"]], on="clip", how="inner", suffixes=("", "_z"))
    assert len(m) == len(dp) == len(dz) == rep["n"], (ds, len(m), len(dp), len(dz), rep["n"])
    assert (m["label"].values == m["label_z"].values).all()
    y = m["label"].values.astype(int); s1 = m["p_probe"].values.astype(float); zs = m["p_yes"].values.astype(float)
    spk = m["speaker"].astype(str).values
    a_single = auc(y, s1); a_zs = auc(y, zs)
    gap = mean5 - a_zs
    gap_from_rounded_repeats = float(np.mean(per)) - a_zs
    d1, p1, q1, nspk = paired(y, [s1], zs, spk)
    lo1, hi1, us1 = pct(d1)
    shift = mean5 - a_single
    alo, ahi = lo1 + shift, hi1 + shift
    zlo, zhi, _ = pct(q1)
    plo, phi, _ = pct(p1)
    n = len(y)
    add(f"M7mo5_{ds}_gap", f"Qwen2.5-Omni {ds}: encoder probe (Table 1 mean of five per-repeat AUCs) minus zero-shot answer AUC; "
        f"interval is the APPROXIMATE shifted single-split paired interval (exact per-draw mean-of-five interval impossible: per-repeat OOF never saved, states deleted)",
        gap, alo, ahi, n, nspk, "exact point, approximate interval", f"{fj} ; {fp} ; {fz}")
    add(f"M7mo5_{ds}_probe_mean5", f"Qwen2.5-Omni {ds}: Table 1 encoder probe, mean of five per-repeat AUCs {per}", mean5, None, None, n, nspk,
        "read from file (Table 1 value)", fj)
    add(f"M7mo5_{ds}_zeroshot", f"Qwen2.5-Omni {ds}: zero-shot answer AUC recomputed from per-clip p_yes, speaker bootstrap interval", a_zs, zlo, zhi, n, nspk,
        "exact", fz)
    add(f"M7mo5_{ds}_single_gap_ref", f"Qwen2.5-Omni {ds}: OLD single-split gap (single-split probe {a_single:.4f} minus zero-shot), exact paired interval, reference only (reproduces part16 M7)",
        a_single - a_zs, lo1, hi1, n, nspk, "exact, superseded convention", f"{fp} ; {fz}")
    o = old.loc[ds]
    side["datasets"][ds] = dict(n=n, n_speakers=nspk, n_pos=int(y.sum()), per_repeat_auc_table1=per, mean5_table1=mean5,
        mean_of_rounded_repeats=round(float(np.mean(per)), 6), zeroshot_auc=a_zs, single_split_auc=a_single,
        gap_exact=gap, gap_bounds_from_4dp_rounding=[gap - 0.00005, gap + 0.00005],
        gap_using_mean_of_rounded_repeats=gap_from_rounded_repeats,
        single_split_paired_ci=[lo1, hi1], shift=shift, approx_ci=[alo, ahi], usable_draws=us1,
        probe_mean5_single_shifted_ci=[plo + shift, phi + shift], zeroshot_ci=[zlo, zhi],
        old_part16_M7_row=dict(probe=float(o.probe_auc), zeroshot=float(o.zeroshot_auc), gap=float(o.paired_diff), lo=float(o.diff_lo), hi=float(o.diff_hi)),
        old_row_reproduced_4dp=bool(round(a_single - a_zs, 4) == round(float(o.paired_diff), 4) and round(lo1, 4) == round(float(o.diff_lo), 4) and round(hi1, 4) == round(float(o.diff_hi), 4)),
        table1_outer_split_files=[f"{FOLDS}/{ds}_groupkfold5_pod_seed{s}{'' if ds == 'pcgita' else '_UNVERIFIED'}.csv" for s in range(5)])
    print(f"{ds:11s} n={n:5d} spk={nspk:4d} mean5 {mean5:.4f} zs {a_zs:.4f} GAP {gap:+.4f} approx[{alo:+.4f},{ahi:+.4f}] | single {a_single:.4f} gap {a_single-a_zs:+.4f} [{lo1:+.4f},{hi1:+.4f}] usable {us1}/{NB} | old row reproduced {side['datasets'][ds]['old_row_reproduced_4dp']}")


# ------------------------------------------------------------------ 3: validation of the approximation
def validate(tag, y, reps, zs, spk, single=None, src="", note=""):
    y = np.asarray(y).astype(int); reps = [np.asarray(r, float) for r in reps]; zs = np.asarray(zs, float)
    per = [auc(y, r) for r in reps]; mean5 = float(np.mean(per)); a_zs = auc(y, zs)
    dex, pex, qex, nspk = paired(y, reps, zs, spk)
    elo, ehi, eus = pct(dex)
    res = dict(n=int(len(y)), n_speakers=nspk, per_repeat=per, mean5=mean5, zeroshot=a_zs, gap=mean5 - a_zs,
               exact_ci=[elo, ehi], usable=eus, note=note, source=src, proxies={})
    add(f"VAL_{tag}_exact", f"{tag}: EXACT paired interval, per draw mean of five per-repeat AUCs minus answer AUC", mean5 - a_zs, elo, ehi,
        int(len(y)), nspk, "exact (validation run)", src, closes="validation of the approximation used for M7 non-Pitt", note=note)
    proxies = []
    if single is not None:
        proxies.append(("single", np.asarray(single, float)))
    else:
        proxies += [(f"repeat{r}", reps[r]) for r in range(len(reps))]
    worst = 0.0
    for nm, pr in proxies:
        d1, _, _, _ = paired(y, [pr], zs, spk)
        lo1, hi1, _ = pct(d1)
        sh = mean5 - auc(y, pr)
        alo, ahi = lo1 + sh, hi1 + sh
        e = max(abs(alo - elo), abs(ahi - ehi)); worst = max(worst, e)
        res["proxies"][nm] = dict(proxy_auc=auc(y, pr), shifted_ci=[alo, ahi], lo_err=alo - elo, hi_err=ahi - ehi,
                                  width_exact=ehi - elo, width_shifted=ahi - alo)
        add(f"VAL_{tag}_shifted_{nm}", f"{tag}: shifted {nm} paired interval (the approximation), compare with the exact row", mean5 - a_zs, alo, ahi,
            int(len(y)), nspk, "approximation (validation run)", src, closes="validation of the approximation used for M7 non-Pitt",
            note=f"endpoint error lo {alo-elo:+.4f} hi {ahi-ehi:+.4f}")
    res["max_endpoint_error"] = worst
    side["validation"][tag] = res
    print(f"VAL {tag:22s} n={len(y):5d} mean5 {mean5:.4f} zs {a_zs:.4f} gap {mean5-a_zs:+.4f} exact[{elo:+.4f},{ehi:+.4f}] "
          + " ".join(f"{k}:[{v['shifted_ci'][0]:+.4f},{v['shifted_ci'][1]:+.4f}]" for k, v in res["proxies"].items())
          + f"  max endpoint err {worst:.4f}")
    return res


# Pitt part10 (Table 1 Pitt value 0.7706), zero-shot from omni_final
z = np.load(PITT_NPZ, allow_pickle=True); side["inputs_sha256"][PITT_NPZ] = sha(PITT_NPZ)
zsp = pd.read_csv(f"{OMNI}/omni_pitt_zeroshot_scores.csv"); side["inputs_sha256"][f"{OMNI}/omni_pitt_zeroshot_scores.csv"] = sha(f"{OMNI}/omni_pitt_zeroshot_scores.csv")
mp = dict(zip(zsp["clip"].astype(str), zsp["p_yes"]))
zpitt = np.array([mp[os.path.basename(str(n))] for n in z["name"]], float)
validate("pitt_part10", z["label"].astype(int), [z["oof"][r] for r in range(5)], zpitt, z["spk"].astype(str), None, PITT_NPZ,
         note="Table 1 Pitt encoder value 0.7706; exact row must equal part16 M7 fix +0.1128 [0.0455, 0.1768]")

# E-DAIC 300 s, T7b (encoder, answer 0.8278)
t = pd.read_csv(T7B); side["inputs_sha256"][T7B] = sha(T7B)
validate("edaic300_T7b", t["label"].values, [t[f"oof_enc_r{r}"].values for r in range(5)], t["p_yes_answer"].values, t["pid"].astype(str).values, None, T7B,
         note="exact row must equal master_lookup row 224: -0.2394 [-0.3188, -0.1572]")

# NeuroVoz POD6 full re-extraction, Table 1 outer splits (fold_check 1.0), paper zero-shot
nv = pd.read_csv(NV6); side["inputs_sha256"][NV6] = sha(NV6)
zn = pd.read_csv(f"{OMNI}/omni_neurovoz_zeroshot_scores.csv")
nvm = nv.merge(zn[["clip", "p_yes"]], on="clip", how="inner"); assert len(nvm) == 1270
validate("neurovoz_pod6_reextract", nvm["label"].values, [nvm[f"p_seed{r}"].values for r in range(5)], nvm["p_yes"].values,
         nvm["speaker"].astype(str).values, nvm["p_single"].values, NV6,
         note="re-extracted states (not Table 1 states), Table 1 outer splits; per-repeat 0.9287/0.9158/0.9253/0.9129/0.9262 vs Table 1 0.9275/0.9154/0.928/0.9124/0.9232")

# ADReSS-2020 POD4B re-extraction, refit today on lo-m7m11-1 (Linux sklearn 1.9.1), Table 1 outer splits
a5 = pd.read_csv(AD4B_REP5); a1 = pd.read_csv(AD4B_SINGLE); side["inputs_sha256"][AD4B_REP5] = sha(AD4B_REP5); side["inputs_sha256"][AD4B_SINGLE] = sha(AD4B_SINGLE)
za = pd.read_csv(f"{OMNI}/omni_adress2020_zeroshot_scores.csv")
am = a5.merge(a1[["clip", "p_probe"]], on="clip").merge(za[["clip", "p_yes"]], on="clip"); assert len(am) == 156
fold_match = {}
for s in range(5):
    f = pd.read_csv(f"{FOLDS}/adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv")
    mm = am.merge(f, left_on="clip", right_on="clip_id"); fold_match[f"seed{s}"] = int((mm[f"fold_seed{s}"] == mm["fold"]).sum())
side["adress2020_pod4b_fold_match_vs_table1_seed_files"] = fold_match
validate("adress2020_pod4b_reextract", am["label"].values, [am[f"p_seed{r}"].values for r in range(5)], am["p_yes"].values,
         am["speaker"].astype(str).values, am["p_probe"].values, AD4B_REP5,
         note=f"re-extracted states (PART 16 POD4B orig), refit on Linux CPU pod lo-m7m11-1 with sklearn 1.9.1; outer folds equal the Table 1 seed files {fold_match}")

# ------------------------------------------------------------------ write
df = pd.DataFrame(rows)
csv = OUT + "/m7_meanof5_gaps.csv"
df.to_csv(csv, index=False)
side.update(dict(
    command="/usr/local/bin/python3 '" + os.path.abspath(__file__) + "'",
    env=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, machine=platform.machine()),
    bootstrap="2000 draws; fresh numpy default_rng(0) per cell; idx = rng.choice(N, size=N, replace=True) over np.unique(speaker.astype(str)); "
              "rows of the drawn speakers concatenated; AUC by rank (scipy rankdata, ties averaged); percentiles 2.5/97.5; paired = same idx; "
              "draws with one class only are dropped (count in usable_draws)",
    approximation="approx_ci = single-split paired percentile interval shifted by (Table 1 mean of five minus single-split AUC)",
    why_not_exact="per-repeat per-clip OOF for adresso, adress2020, pcgita, neurovoz, kcl never saved (only json per-repeat AUCs); "
                  "Table 1 Omni states only on pod gv9jvm57b21mcz (omni-final, 18 Sep), which no longer exists in GET /v1/pods; "
                  "no omni_<ds>_states.npz for these five on the Mac release tree, G-Drive paper work, or in any session transcript",
    out_csv=csv))
json.dump(side, open(OUT + "/m7_meanof5_gaps.sidecar.json", "w"), indent=1, default=float)
print("wrote", csv)
