#!/usr/local/bin/python3
# p12mispair independent checker. Independent code, reads only.
import os, re, json, ast
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

R = os.path.expanduser("<local data dir>/release")
G = "<local data dir>"
P23 = G + "/release_from_mac/scores/part23"
P24 = G + "/release_from_mac/scores/part24"
OUT = G + "/checks/accel_24sep/p12mispair"
B = 2000

def auc(y, p):
    return float(roc_auc_score(np.asarray(y), np.asarray(p, dtype=float)))

def fnum(x):
    if isinstance(x, str):
        m = re.search(r"[-+0-9.eE]+", x.replace("np.float64(", ""))
        return float(m.group(0))
    return float(x)

def _groups(spk):
    spk = np.asarray(spk).astype(str)
    u, inv = np.unique(spk, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(u))]
    return idx

def boot(spk, y, fns):
    """fns: function(idx)->float. Fresh rng(0) per cell, unique speakers with replacement."""
    rng = np.random.default_rng(0)
    g = _groups(spk)
    y = np.asarray(y)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(g), len(g))
        ii = np.concatenate([g[k] for k in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(fns(ii))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)

def boot_auc(spk, y, p):
    y = np.asarray(y); p = np.asarray(p, float)
    return boot(spk, y, lambda ii: auc(y[ii], p[ii]))

def boot_diff(spk, y, pa, pb):
    y = np.asarray(y); pa = np.asarray(pa, float); pb = np.asarray(pb, float)
    return boot(spk, y, lambda ii: auc(y[ii], pa[ii]) - auc(y[ii], pb[ii]))

rows = []
def rec(i, item, tex, val, file, how, ok=None, note=""):
    if ok is None:
        try:
            ok = abs(round(abs(val), 2) - abs(float(tex))) < 1e-9
        except Exception:
            ok = None
    rows.append(dict(row=i, item_id=item, tex=tex, computed=val, match_2dp=("yes" if ok else "no") if ok is not None else "see note", file=file, how=how, note=note))
    print(i, item, tex, val, "OK" if ok else "CHECK", how, note)

bn = lambda s: os.path.basename(str(s))

# ---------- row 0: answer mass floor ----------
f = R + "/edaic_rerun/part16/M9_answer_mass.csv"
m9 = pd.read_csv(f)
zs = m9[m9.stream.str.contains("zero shot", case=False)]
k = zs.median_mass.idxmin()
a2 = pd.read_csv(P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv")
a2m = np.median([fnum(v) for v in a2.answer_mass_whole])
floor = min(zs.median_mass.min(), a2m)
# recompute min cell from its own file
src = zs.loc[k, "file"]
recomp = None
try:
    s = pd.read_csv(src)
    col = [c for c in s.columns if c in ("answer_mass", "mass")][0]
    recomp = float(np.median(s[col].astype(float)))
except Exception as e:
    recomp = str(e)
rec(0, "L67_321", "0.89", floor, f + " ; " + P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv",
    f"min median Yes+No mass over {len(zs)} zero-shot cells = {zs.median_mass.min():.4f} ({zs.loc[k,'model']} {zs.loc[k,'dataset']}); recomputed from {bn(src)} = {recomp}; E-DAIC whole median {a2m:.4f}",
    ok=floor >= 0.89, note="tex says 'at least 0.89'; true floor check (>=0.89)")

# ---------- rows 1-3: TF-IDF ----------
tf = {}
for ds in ["kcl", "neurovoz", "pcgita", "edaicfull", "pitt", "adresso", "adress2020"]:
    fp = f"{R}/scores/part20/POD6/A_language_saliency/A_tfidf_{ds}_oof.csv"
    d = pd.read_csv(fp)
    tf[ds] = (auc(d.label, d.p_tfidf), len(d), d.speaker.nunique())
pd_ = [tf[d][0] for d in ["kcl", "neurovoz", "pcgita"]]
ad_ = [tf[d][0] for d in ["pitt", "adresso", "adress2020"]]
tfp = f"{R}/scores/part20/POD6/A_language_saliency/A_tfidf_*_oof.csv"
rec(1, "L77_360", "0.43 to 0.62", f"{min(pd_):.4f} to {max(pd_):.4f}", tfp, "AUC(label,p_tfidf) KCL %.4f NeuroVoz %.4f PC-GITA %.4f" % tuple(pd_),
    ok=round(min(pd_), 2) == 0.43 and round(max(pd_), 2) == 0.62)
d = pd.read_csv(f"{R}/scores/part20/POD6/A_language_saliency/A_tfidf_edaicfull_oof.csv")
lo, hi, nb = boot_auc(d.speaker, d.label, d.p_tfidf)
rec(2, "L77_398", "0.68", tf["edaicfull"][0], tfp.replace("*", "edaicfull"), f"AUC [{lo:.4f}, {hi:.4f}] n={tf['edaicfull'][1]}")
rec(3, "L77_433", "0.84 to 0.93", f"{min(ad_):.4f} to {max(ad_):.4f}", tfp, "Pitt %.4f ADReSSo %.4f ADReSS-2020 %.4f" % tuple(ad_),
    ok=round(min(ad_), 2) == 0.84 and round(max(ad_), 2) == 0.93)

# ---------- row 4: wording shifts ----------
shifts = []
det = []
for ds in ["pcgita", "neurovoz", "kcl", "edaic", "pitt", "adresso", "adress2020"]:
    z = pd.read_csv(f"{R}/omni_final/omni_{ds}_zeroshot_scores.csv")
    az = auc(z.label, z.p_yes)
    for v in ["p2", "p3"]:
        w = pd.read_csv(f"{R}/omni_final/omni_{ds}_{v}.csv")
        aw = auc(w.label, w.p_yes)
        shifts.append(abs(aw - az)); det.append(f"{ds}/{v} {aw-az:+.4f}")
rec(4, "L92_538", "0.07", max(shifts), R + "/omni_final/omni_{ds}_{zeroshot_scores,p2,p3}.csv",
    f"max |wording - base| over {len(shifts)} runs = {max(shifts):.4f}; mean {np.mean(shifts):.4f} (tex 0.02 on average)",
    ok=round(max(shifts), 2) == 0.07 and round(np.mean(shifts), 2) == 0.02,
    note="E-DAIC wording runs are 30 s windows (omni_edaic_*), not the whole-interview Table 1 cell; " + "; ".join(det))

# ---------- row 5: durations ----------
mf = pd.read_csv("<local data dir>/DementiaBank/pitt_conflict_manifest.csv")
rec(5, "L97_803", "30 to 56", f"{mf.dur.min():.1f} to {mf.dur.max():.1f} (median {mf.dur.median():.1f})",
    "<local data dir>/DementiaBank/pitt_conflict_manifest.csv",
    f"dur column, n={len(mf)}, sets {mf.set.value_counts().to_dict()}",
    ok=round(mf.dur.min()) == 30 and round(mf.dur.max()) == 56 and round(mf.dur.median()) == 33)

# ---------- row 8: mid 30 ----------
fp = R + "/edaic_rerun/variants/o25_mid30_zeroshot_scores.csv"
d = pd.read_csv(fp)
a = auc(d.label, d.p_yes); lo, hi, nb = boot_auc(d.speaker, d.label, d.p_yes)
rec(8, "L140_549", "0.72", a, fp, f"AUC [{lo:.4f}, {hi:.4f}] n={len(d)}")

# ---------- rows 9-17: channel normalisation ----------
for ds in ["pcgita", "neurovoz"]:
    fp = f"{R}/edaic_rerun/part16/POD4/channel_norm_{ds}.csv"
    d = pd.read_csv(fp)
    vals = dict(ff_b=auc(d.label, d.fivefeat_oof_before), ff_a=auc(d.label, d.fivefeat_oof_after),
                ep_b=auc(d.label, d.enc_probe_oof_before), ep_a=auc(d.label, d.enc_probe_oof_after))
    lo, hi, nb = boot_diff(d.spk, d.label, d.enc_probe_oof_after, d.enc_probe_oof_before)
    vals["diff"] = vals["ep_a"] - vals["ep_b"]; vals["lo"] = lo; vals["hi"] = hi
    if ds == "pcgita":
        pc = (fp, vals)
    else:
        nv = (fp, vals)
fp, v = pc
rec(9, "L149_1096", "0.67", v["ff_b"], fp, "five-feature OOF AUC, PC-GITA, before")
rec(10, "L149_1224", "0.56", v["ff_a"], fp, "five-feature OOF AUC, PC-GITA, after")
rec(11, "L149_1283", "0.07", v["diff"], fp, f"enc probe after minus before {v['ep_a']:.4f}-{v['ep_b']:.4f} = {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] paired spk bootstrap", note="tex prints -0.07")
rec(12, "L149_1292", "0.12", v["lo"], fp, "lower bound of row 11 interval", note="tex prints -0.12")
rec(13, "L149_1301", "0.02", v["hi"], fp, "upper bound of row 11 interval", note="tex prints -0.02")
fp, v = nv
rec(14, "L149_1350", "0.60", v["ff_a"], fp, f"five-feature OOF AUC, NeuroVoz, after (before {v['ff_b']:.4f}, tex 0.80)")
rec(15, "L149_1402", "0.03", v["diff"], fp, f"enc probe after minus before {v['ep_a']:.4f}-{v['ep_b']:.4f} = {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]", note="tex prints -0.03")
rec(16, "L149_1411", "0.061", v["lo"], fp, "lower bound (tex gives 3 dp)", ok=abs(round(abs(v["lo"]), 3) - 0.061) < 1e-9, note="checked at 3 dp because tex prints -0.061")
rec(17, "L149_1421", "0.001", v["hi"], fp, "upper bound (tex gives 3 dp)", ok=abs(round(abs(v["hi"]), 3) - 0.001) < 1e-9, note="checked at 3 dp because tex prints -0.001")

# ---------- rows 18, 20: NeuroVoz matched ----------
fp = R + "/scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.csv"
d = pd.read_csv(fp)
seeds = [auc(d.label, d[f"p_seed{s}"]) for s in range(5)]
avgp = auc(d.label, d[[f"p_seed{s}" for s in range(5)]].mean(1))
sing = auc(d.label, d.p_single)
rec(18, "L149_1507", "0.90", np.mean(seeds), fp, f"mean of 5 seed AUCs {np.mean(seeds):.4f}; AUC of averaged p {avgp:.4f}; p_single {sing:.4f}; n={len(d)} spk={d.speaker.nunique()}",
    ok=round(np.mean(seeds), 2) == 0.90)
fp = R + "/scores/part20/POD6/B_neurovoz_matched/B_zeroshot_nvmatched.csv"
d = pd.read_csv(fp)
a = auc(d.label, d.p_yes); lo, hi, nb = boot_auc(d.speaker, d.label, d.p_yes)
rec(20, "L149_1527", "0.63", a, fp, f"AUC [{lo:.4f}, {hi:.4f}] n={len(d)} spk={d.speaker.nunique()}")

# ---------- row 19: AD encoder probe ----------
z = np.load(R + "/overnight2/part10/pitt_enc_nested5_oof.npz", allow_pickle=True)
pitt_r = [auc(z["label"], z["oof"][r]) for r in range(z["oof"].shape[0])]
ado = json.load(open(R + "/omni_final/omni_adresso_nested_repeats.json"))["enc"]
ad20 = json.load(open(R + "/omni_final/omni_adress2020_nested_repeats.json"))["enc"]
vals = [np.mean(pitt_r), ado["mean"], ad20["mean"]]
rec(19, "L149_152", "0.77 to 0.88", f"{min(vals):.4f} to {max(vals):.4f}",
    R + "/overnight2/part10/pitt_enc_nested5_oof.npz ; omni_final/omni_{adresso,adress2020}_nested_repeats.json",
    f"Pitt mean of 5 per-repeat AUCs {np.mean(pitt_r):.4f} (recomputed); ADReSSo json mean {ado['mean']} (per-repeat mean {np.mean(ado['per_repeat']):.4f}); ADReSS-2020 {ad20['mean']} ({np.mean(ad20['per_repeat']):.4f})",
    ok=round(min(vals), 2) == 0.77 and round(max(vals), 2) == 0.88,
    note="ADReSSo and ADReSS-2020 read from the repeats json summary, not recomputed per clip")

# ---------- rows 21-23: readout direction ----------
fp = R + "/omni/readout_direction.csv"
rd = pd.read_csv(fp).set_index("dataset").loc["pitt"]
rec(21, "L149_2192", "0.66", rd.auc_d, fp + " (row pitt)", "auc_d (summary csv)")
rec(22, "L149_2262", "0.77", rd.auc_probe, fp + " (row pitt)", "auc_probe (summary csv)")
rec(23, "L149_2283", "0.77", rd.auc_probe_without_d, fp + " (row pitt)", f"auc_probe_without_d (summary csv); cosine {rd.cosine}")

# ---------- row 25: Qwen2-Audio proj_d ----------
fp = R + "/edaic_rerun/part17/T3_q2a_direction.csv"
d = pd.read_csv(fp)
a = auc(d.label, d.proj_d)
rec(25, "L149_2463", "0.62", a, fp, f"AUC(label, proj_d) n={len(d)}; probe {auc(d.label, d.oof_probe):.4f}; perp_z {auc(d.label, d.perp_z):.4f}; nested_oof {auc(d.label, d.nested_oof):.4f}; nested_perp_z {auc(d.label, d.nested_perp_z):.4f}")

# ---------- rows 26-35: E-DAIC reversal ----------
fA3 = P23 + "/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv"
fA2 = P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv"
a3 = pd.read_csv(fA3)
a2 = pd.read_csv(fA2)
a3["p_ans"] = a3.p_yes_whole.map(fnum)
a2["p_ans2"] = a2.p_yes_whole.map(fnum)
m = a3.merge(a2[["pid", "p_ans2"]], on="pid")
assert np.allclose(m.p_ans, m.p_ans2)
y = m.label.values
def m5(pref, ii=None):
    ii = np.arange(len(m)) if ii is None else ii
    return float(np.mean([auc(y[ii], m[f"{pref}_r{r}"].values[ii]) for r in range(5)]))
A_ans = auc(y, m.p_ans)
L = m5("oof_llm"); E = m5("oof_enc"); S = m5("oof_ans")
def bd(pref):
    pa = m.p_ans.values
    return boot(m.speaker, y, lambda ii: m5(pref, ii) - auc(y[ii], pa[ii]))
lL = bd("oof_llm"); lE = bd("oof_enc"); lS = bd("oof_ans")
avgL = auc(y, m[[f"oof_llm_r{r}" for r in range(5)]].mean(1))
avgE = auc(y, m[[f"oof_enc_r{r}" for r in range(5)]].mean(1))
rec(26, "L149_2643", "0.74", L, fA3, f"LM probe mean of five per-repeat AUCs (AUC of averaged p {avgL:.4f})")
rec(27, "L149_2670", "0.60", E, fA3, f"encoder probe mean of five per-repeat AUCs (AUC of averaged p {avgE:.4f})")
rec(28, "L149_2701", "0.84", A_ans, fA2, f"AUC p_yes_whole n={len(m)}")
rec(29, "L149_2710", "0.09", L - A_ans, fA3 + " + A2", f"LM minus answer {L-A_ans:+.4f} [{lL[0]:+.4f}, {lL[1]:+.4f}] paired, mean-of-five inside each draw", note="tex prints -0.09")
rec(30, "L149_2719", "0.16", lL[0], fA3 + " + A2", "lower bound", note="tex prints -0.16")
rec(31, "L149_2728", "0.03", lL[1], fA3 + " + A2", "upper bound", note="tex prints -0.03")
rec(32, "L149_2741", "0.24", E - A_ans, fA3 + " + A2", f"encoder minus answer {E-A_ans:+.4f} [{lE[0]:+.4f}, {lE[1]:+.4f}]", note="tex prints -0.24")
rec(33, "L149_2750", "0.32", lE[0], fA3 + " + A2", "lower bound", note="tex prints -0.32")
rec(34, "L149_2759", "0.15", lE[1], fA3 + " + A2", "upper bound", note="tex prints -0.15")
rec(35, "L149_2821", "0.84", S, fA3, f"answer-state probe mean of five; minus answer {S-A_ans:+.4f} [{lS[0]:+.4f}, {lS[1]:+.4f}]")

# ---------- rows 36-37: Pitt gap ----------
fz = R + "/omni_final/omni_pitt_zeroshot_scores.csv"
zz = pd.read_csv(fz).set_index("clip")
names = [str(n) for n in z["name"]]
pz = zz.loc[names, "p_yes"].values
yl = z["label"]; assert (zz.loc[names, "label"].values == yl).all()
oof = z["oof"]
gap = np.mean(pitt_r) - auc(yl, pz)
lo, hi, nb = boot(z["spk"], yl, lambda ii: np.mean([auc(yl[ii], oof[r][ii]) for r in range(oof.shape[0])]) - auc(yl[ii], pz[ii]))
rec(36, "L149_333", "0.11", gap, R + "/overnight2/part10/pitt_enc_nested5_oof.npz + " + fz,
    f"mean-of-5 enc {np.mean(pitt_r):.4f} minus zero-shot {auc(yl,pz):.4f} = {gap:+.4f} [{lo:+.4f}, {hi:+.4f}] n={len(yl)} spk={len(np.unique(z['spk']))}")
rec(37, "L149_339", "0.05, 0.18", f"{lo:.4f}, {hi:.4f}", "same as row 36", "interval of the Pitt gap",
    ok=round(lo, 2) == 0.05 and round(hi, 2) == 0.18)

# ---------- row 38: PC-GITA gap ----------
pcj = json.load(open(R + "/omni_final/omni_pcgita_nested_repeats.json"))["enc"]
zp = pd.read_csv(R + "/omni_final/omni_pcgita_zeroshot_scores.csv")
azp = auc(zp.label, zp.p_yes)
e1 = pd.read_csv(R + "/omni_final/omni_pcgita_enc_nested_oof.csv")
e1a = auc(e1.label, e1.p_probe)
rec(38, "L149_362", "0.33", pcj["mean"] - azp, R + "/omni_final/omni_pcgita_nested_repeats.json + omni_pcgita_zeroshot_scores.csv",
    f"enc mean-of-5 {pcj['mean']} (per-repeat mean {np.mean(pcj['per_repeat']):.4f}) minus zero-shot {azp:.4f}; single-split enc OOF file gives {e1a:.4f} (gap {e1a-azp:.4f})",
    note="point only; no five-repeat per-clip PC-GITA file for an interval")

# ---------- rows 39-40: Qwen3-Omni ----------
fp = R + "/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv"
d = pd.read_csv(fp)
rec(39, "L149_593", "0.60", auc(d.label, d.p_yes), fp, f"AUC n={len(d)}")
fe = R + "/edaic_rerun/part16/POD3/q3o_pitt_encoder_nested_oof.csv"
fq = R + "/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"
if not os.path.exists(fq):
    fq = R + "/overnight2/part10/q3o_pitt_zeroshot_scores.csv"
e = pd.read_csv(fe); q = pd.read_csv(fq)
qe = np.mean([auc(e.label, e[f"p_probe_rep{r}"]) for r in range(5)])
qa = auc(q.label, q.p_yes)
rec(40, "L149_639", "0.05", qe - qa, fe + " + " + fq, f"Q3O Pitt enc mean-of-5 {qe:.4f} minus zero-shot {qa:.4f}")

# ---------- rows 41-43: eGeMAPS ----------
fg = R + "/overnight/egemaps_baseline.csv"
g = pd.read_csv(fg).set_index("dataset")
marg = (g.omni_encoder_probe - g.egemaps_auc).drop("edaic")
rec(41, "L149_796", "0.01 on MDVR-KCL to 0.18", f"{marg.min():+.4f} ({marg.idxmin()}) to {marg.max():+.4f} ({marg.idxmax()})", fg,
    "probe minus eGeMAPS per PD/AD set: " + ", ".join(f"{k} {v:+.4f}" for k, v in marg.items()),
    ok=marg.idxmin() == "kcl" and round(marg.min(), 2) == 0.01 and marg.idxmax() == "adresso" and round(marg.max(), 2) == 0.18,
    note="eGeMAPS AUC and probe read from the summary csv")
for i, it, ds, tex in [(42, "L149_911", "pcgita", "0.27"), (43, "L149_920", "neurovoz", "0.15")]:
    zf = f"{R}/omni_final/omni_{ds}_zeroshot_scores.csv"
    zd = pd.read_csv(zf); az = auc(zd.label, zd.p_yes)
    rec(i, it, tex, g.loc[ds, "egemaps_auc"] - az, fg + " + " + zf, f"eGeMAPS {g.loc[ds,'egemaps_auc']} minus recomputed zero-shot {az:.4f}")

# ---------- rows 48, 60: Qwen3-Omni transcript on pairs ----------
fp = R + "/scores/part20/POD3c/p14_q3o_text.csv"
d = pd.read_csv(fp)
c = d[d.set == "conflict"]; ag = d[d.set == "agreement"]
agd = ag[ag.in_agreement311_distinct.astype(str) == "True"]
agu = ag.drop_duplicates("seg_uid")
ca = auc(c.label, c.p_yes_text); clo, chi, _ = boot_auc(c.speaker_id, c.label, c.p_yes_text)
aa = auc(agd.label, agd.p_yes_text); alo, ahi, _ = boot_auc(agd.speaker_id, agd.label, agd.p_yes_text)
rec(48, "L180_1000", "0.95", aa, fp, f"agreement, {len(agd)} distinct (flag col; dedupe gives {len(agu)}): {aa:.4f} [{alo:.4f}, {ahi:.4f}]; all {len(ag)} rows {auc(ag.label, ag.p_yes_text):.4f}")
rec(60, "L180_991", "0.22", ca, fp, f"conflict n={len(c)}: {ca:.4f} [{clo:.4f}, {chi:.4f}]")

# ---------- rows 49-53: second sentiment scorer ----------
fp = R + "/scores/part20/POD3b/sst2_pairs_o25_audio.csv"
d = pd.read_csv(fp)
c = d[d.set == "conflict"]; ag = d[d.set == "agreement"].drop_duplicates("seg_uid")
ca = auc(c.label, c.p_yes); clo, chi, _ = boot_auc(c.speaker_id, c.label, c.p_yes)
aa = auc(ag.label, ag.p_yes); aall = auc(d[d.set == "agreement"].label, d[d.set == "agreement"].p_yes)
rec(49, "L180_1662", "0.46", ca, fp, f"conflict n={len(c)} spk={c.speaker_id.nunique()}")
rec(50, "L180_1668", "0.42, 0.51", f"{clo:.4f}, {chi:.4f}", fp, "conflict interval", ok=round(clo, 2) == 0.42 and round(chi, 2) == 0.51)
rec(51, "L180_1723", "0.39", aa - ca, fp, f"agreement distinct {len(ag)} segs {aa:.4f} minus conflict {ca:.4f}; all agreement rows {aall:.4f} (gap {aall-ca:.4f})")
fp2 = R + "/scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv"
d2 = pd.read_csv(fp2)
both = d2[(d2.set == "conflict") & (d2.rob_rule_arm == "conflict")]
neu = d2[(d2.set == "conflict") & (d2.rob_rule_arm == "none")]
rec(52, "L180_1824", "0.20", auc(both.label, both.p_yes), fp2, f"conflict & first scorer conflict, n={len(both)}", note="n check 420: " + str(len(both) == 420))
rec(53, "L180_1881", "0.62", auc(neu.label, neu.p_yes), fp2, f"conflict & first scorer neutral, n={len(neu)}", note="n check 855: " + str(len(neu) == 855))

# ---------- row 54: Flamingo Pitt arms ----------
vals = {}
for arm in ["conflict", "agreement"]:
    f2 = f"{R}/scores/part20/POD4c/af2_orig30_{arm}.csv"; dd = pd.read_csv(f2)
    vals[f"AF2 {arm}"] = auc(dd.label, dd.p_yes)
f3 = R + "/overnight2/part10/af3_pitt.csv"; d3 = pd.read_csv(f3)
d3["arm"] = d3.clip_path.map(lambda s: bn(s).split("_")[0])
for arm in ["conflict", "agreement"]:
    dd = d3[d3.arm == arm]; vals[f"AF3 {arm}"] = auc(dd.label, dd.p_yes)
rec(54, "L180_2394", "0.53 to 0.61", f"{min(vals.values()):.4f} to {max(vals.values()):.4f}", R + "/scores/part20/POD4c/af2_orig30_{conflict,agreement}.csv ; " + f3,
    "; ".join(f"{k} {v:.4f}" for k, v in vals.items()), ok=round(min(vals.values()), 2) == 0.53 and round(max(vals.values()), 2) == 0.61)

# ---------- rows 55-57: voice wording ----------
fv = R + "/scores/part20/POD4/o25_orig_voice.csv"
v = pd.read_csv(fv).merge(pd.read_csv(fz)[["clip", "p_yes", "label"]].rename(columns={"p_yes": "p_rec", "label": "lab2"}), on="clip")
assert (v.label == v.lab2).all() and len(v) == 468
dv = auc(v.label, v.p_yes) - auc(v.label, v.p_rec)
lo, hi, _ = boot_diff(v.speaker, v.label, v.p_yes, v.p_rec)
rec(55, "L180_2521", "0.01", dv, fv + " + " + fz, f"voice {auc(v.label,v.p_yes):.4f} minus recording {auc(v.label,v.p_rec):.4f} = {dv:+.4f} [{lo:+.4f}, {hi:+.4f}]", note="tex prints -0.01")
rec(56, "L180_2530", "0.04", lo, fv + " + " + fz, "lower bound", note="tex prints -0.04")
rec(57, "L180_2536", "0.02", hi, fv + " + " + fz, "upper bound")

# ---------- rows 58-59: Qwen2.5-Omni transcript on pairs ----------
fp = R + "/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv"
d = pd.read_csv(fp)
c = d[d.arm == "conflict"]; ag = d[d.arm == "agreement"]; agu = ag.drop_duplicates("seg")
ca = auc(c.label, c.p_yes_text); clo, chi, _ = boot_auc(c.speaker_id, c.label, c.p_yes_text)
rec(58, "L180_931", "0.15", ca, fp, f"conflict n={len(c)}: [{clo:.4f}, {chi:.4f}]")
rec(59, "L180_952", "0.96", auc(agu.label, agu.p_yes_text), fp, f"agreement distinct {len(agu)}: {auc(agu.label, agu.p_yes_text):.4f}; all {len(ag)} rows {auc(ag.label, ag.p_yes_text):.4f}")

# ---------- row 61: LoRA plus projector ----------
fp = R + "/omni_final/abl2_pitt_both_oof.csv"
d = pd.read_csv(fp)
rec(61, "L226_1084", "0.77", auc(d.label, d.p_yes), fp, f"AUC n={len(d)}")

# ---------- rows 62-65: E-DAIC FT ----------
f0 = P24 + "/FT/FT24_whole_seed0_perclip.csv"; f1 = P24 + "/FT/FT24_whole_seed1all_perclip.csv"
s0 = pd.read_csv(f0).merge(a2[["pid", "p_ans2", "label"]].rename(columns={"label": "l2"}), on="pid")
assert (s0.label == s0.l2).all()
dft = auc(s0.label, s0.p_yes) - auc(s0.label, s0.p_ans2)
lo, hi, _ = boot_diff(s0.pid, s0.label, s0.p_yes, s0.p_ans2)
rec(62, "L226_1154", "+0.02", dft, f0 + " + " + fA2, f"FT seed0 {auc(s0.label,s0.p_yes):.4f} minus zero-shot {auc(s0.label,s0.p_ans2):.4f} = {dft:+.4f} [{lo:+.4f}, {hi:+.4f}] n={len(s0)}")
rec(63, "L226_1164", "0.01", lo, f0 + " + A2", "lower bound", note="tex prints -0.01")
rec(64, "L226_1170", "0.05", hi, f0 + " + A2", "upper bound")
s1 = pd.read_csv(f1)
a1 = auc(s1.label, s1.p_yes); lo1, hi1, _ = boot_auc(s1.pid, s1.label, s1.p_yes)
rec(65, "L226_1181", "0.82", a1, f1, f"seed1 {a1:.4f} [{lo1:.4f}, {hi1:.4f}]")

# ---------- rows 66-67: FT on pairs ----------
fp = R + "/scores/part20/POD3/p14_ft966_oof.csv"
d = pd.read_csv(fp)
c = d[d.arm == "conflict"]; ag = d[d.arm == "agreement"].drop_duplicates("seg_uid")
ca = auc(c.label, c.p_yes); clo, chi, _ = boot_auc(c.speaker, c.label, c.p_yes)
aa = auc(ag.label, ag.p_yes); alo, ahi, _ = boot_auc(ag.speaker, ag.label, ag.p_yes)
fb = R + "/edaic_rerun/part14/p14_o25_zeroshot_scores.csv"
b = pd.read_csv(fb); bc = b[b["clip"].astype(str).str.startswith("conflict")]
fT = R + "/edaic_rerun/part17/T1_perclip_distinct_agreement.csv"
t = pd.read_csv(fT)
rec(66, "L226_1340", "0.59", ca, fp, f"after conflict {ca:.4f} [{clo:.4f}, {chi:.4f}] n={len(c)} (p_yes_sft col {auc(c.label,c.p_yes_sft):.4f}); before {auc(bc.label,bc.p_yes):.4f} from {bn(fb)} n={len(bc)} (tex 0.22)",
    ok=round(ca, 2) == 0.59 and round(auc(bc.label, bc.p_yes), 2) == 0.22)
rec(67, "L226_1379", "0.67", aa, fp, f"after agreement distinct {aa:.4f} [{alo:.4f}, {ahi:.4f}] n={len(ag)}; before {auc(t.label,t.p_o25):.4f} from {bn(fT)} n={len(t)} (tex 0.96)",
    ok=round(aa, 2) == 0.67 and round(auc(t.label, t.p_o25), 2) == 0.96)

# ---------- rows 68-71: Pitt FT runs ----------
runs = {
    "original omnisft": R + "/omni_final/omnisft_pitt_oof.csv",
    "seed1": R + "/scores/part20/POD1b/std_ft_seed1_pitt_oof.csv",
    "seed2": R + "/scores/part20/POD1b/std_ft_seed2_pitt_oof.csv",
    "seed3": R + "/scores/part20/POD1b/std_ft_seed3_pitt_oof.csv",
    "standard_rerun": R + "/scores/part20/POD1/standard_rerun_ft_pitt_oof.csv",
    "standard_rerun_r2": R + "/scores/part20/POD1/standard_rerun_ft_pitt_r2_oof.csv",
    "seed0 (extra)": R + "/scores/part20/POD1b/std_ft_seed0_pitt_oof.csv",
    "balanced": R + "/scores/part20/POD1/balanced_ft_pitt_oof.csv",
    "balanced_r2": R + "/scores/part20/POD1/balanced_ft_pitt_r2_oof.csv",
    "balanced_fold3rerun": R + "/scores/part20/POD1/balanced_ft_pitt_fold3rerun_oof.csv",
}
rs = {}
for k, fpp in runs.items():
    dd = pd.read_csv(fpp)
    cc = dd[dd.set == "conflict"]; aa_ = dd[dd.set == "agreement"]
    clo, chi, _ = boot_auc(cc.speaker, cc.label, cc.p_yes)
    rs[k] = dict(overall=auc(dd.label, dd.p_yes), conflict=auc(cc.label, cc.p_yes), clo=clo, chi=chi, agreement=auc(aa_.label, aa_.p_yes), n=len(dd))
    print(k, rs[k])
s3 = [rs[k]["overall"] for k in ["seed1", "seed2", "seed3"]]
rec(68, "L226_469", "0.74 to 0.81", f"{min(s3):.4f} to {max(s3):.4f}", R + "/scores/part20/POD1b/std_ft_seed{1,2,3}_pitt_oof.csv",
    "seeds 1-3 overall " + ", ".join(f"{x:.4f}" for x in s3), ok=round(min(s3), 2) == 0.74 and round(max(s3), 2) == 0.81)
six = ["original omnisft", "seed1", "seed2", "seed3", "standard_rerun", "standard_rerun_r2"]
cf = [rs[k]["conflict"] for k in six]; agv = [rs[k]["agreement"] for k in six]
rec(69, "L226_557", "0.45 to 0.57", f"{min(cf):.4f} to {max(cf):.4f}", "six standard runs (see note)",
    "conflict arm " + ", ".join(f"{k} {rs[k]['conflict']:.4f}" for k in six), ok=round(min(cf), 2) == 0.45 and round(max(cf), 2) == 0.57,
    note=f"seed0 file (not in the six) conflict {rs['seed0 (extra)']['conflict']:.4f}, agreement {rs['seed0 (extra)']['agreement']:.4f}, overall {rs['seed0 (extra)']['overall']:.4f}")
cont = {k: (rs[k]["clo"] <= 0.5 <= rs[k]["chi"]) for k in rs}
rec(70, "L226_596", "0.5", "all contain 0.5: " + str(all(cont.values())), "all runs above",
    "; ".join(f"{k} [{rs[k]['clo']:.4f}, {rs[k]['chi']:.4f}]" for k in rs), ok=all(cont[k] for k in six + ["balanced"]),
    note="0.5 is chance, a claim about intervals")
rec(71, "L226_634", "0.82 to 0.90", f"{min(agv):.4f} to {max(agv):.4f}", "six standard runs",
    "agreement arm " + ", ".join(f"{k} {rs[k]['agreement']:.4f}" for k in six), ok=round(min(agv), 2) == 0.82 and round(max(agv), 2) == 0.90)

# ---------- rows 72-73: parameter counts ----------
pj = json.load(open(R + "/edaic_rerun/part16/POD2/lora_pitt_params.json"))
# independent arithmetic: rank 8, q,k,v,o in 28 layers, hidden 3584, kv dim 512 (4 kv heads x 128)
lora = 28 * 8 * ((3584 + 3584) + (3584 + 512) * 2 + (3584 + 3584))
proj = 1280 * 3584 + 3584
rec(72, "L226_888", "5.0", pj["lora_trainable_params"] / 1e6, R + "/edaic_rerun/part16/POD2/lora_pitt_params.json",
    f"json {pj['lora_trainable_params']:,}; arithmetic {lora:,}", ok=round(pj['lora_trainable_params'] / 1e6, 1) == 5.0 and lora == pj['lora_trainable_params'],
    note="parameter count in millions, checked at 1 dp as printed")
rec(73, "L226_924", "4.6", pj["projector_trainable_params"] / 1e6, R + "/edaic_rerun/part16/POD2/lora_pitt_params.json",
    f"json {pj['projector_trainable_params']:,}; arithmetic 1280x3584+3584 = {proj:,}", ok=round(pj['projector_trainable_params'] / 1e6, 1) == 4.6 and proj == pj['projector_trainable_params'],
    note="parameter count in millions, checked at 1 dp as printed")

# ---------- rows 74-76: LoRA ----------
fl = R + "/edaic_rerun/part16/POD2/lora_pitt_oof.csv"
l = pd.read_csv(fl)
o = pd.read_csv(R + "/omni_final/omnisft_pitt_oof.csv"); o["name"] = o.path.map(bn)
lm = l.merge(o[["name", "p_yes", "label"]].rename(columns={"p_yes": "p_proj", "label": "l2"}), on="name")
assert len(lm) == 468 and (lm.label == lm.l2).all()
la = auc(lm.label, lm.p_yes); pa = auc(lm.label, lm.p_proj)
lo, hi, _ = boot_diff(lm.spk, lm.label, lm.p_yes, lm.p_proj)
lc = lm[lm.arm == "conflict"]
rec(74, "L226_958", "0.825", la, fl, f"LoRA AUC n={len(lm)} (tex 3 dp); conflict arm {auc(lc.label, lc.p_yes):.4f} (tex 0.55)", ok=abs(round(la, 3) - 0.825) < 1e-9, note="checked at 3 dp as printed")
rec(75, "L226_973", "0.06", la - pa, fl + " + omni_final/omnisft_pitt_oof.csv", f"LoRA {la:.4f} minus projector {pa:.4f} = {la-pa:+.4f} [{lo:+.4f}, {hi:+.4f}]")
rec(76, "L226_979", "0.02, 0.10", f"{lo:.4f}, {hi:.4f}", "as row 75", "interval", ok=round(lo, 2) == 0.02 and round(hi, 2) == 0.10)

out = pd.DataFrame(rows)
out.to_csv(OUT + "/mispair_checks.csv", index=False)
print("N rows", len(out), "yes", (out.match_2dp == "yes").sum())
