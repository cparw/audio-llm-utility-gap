#!/usr/local/bin/python3
# p12mispair: independent checks of the UNMATCHED rows that need a file.
import os, json, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = os.path.expanduser("<local data dir>/release")
G = "<local data dir>"
P23 = G + "/release_from_mac/scores/part23"
OUT = G + "/checks/accel_24sep/p12mispair"
B = 2000
def auc(y, p): return float(roc_auc_score(np.asarray(y), np.asarray(p, float)))
def fnum(x):
    if isinstance(x, str):
        return float(re.search(r"[-+0-9.eE]+", x.replace("np.float64(", "")).group(0))
    return float(x)
def boot_auc(spk, y, p):
    rng = np.random.default_rng(0)
    spk = np.asarray(spk).astype(str); y = np.asarray(y); p = np.asarray(p, float)
    u, inv = np.unique(spk, return_inverse=True); g = [np.where(inv == k)[0] for k in range(len(u))]
    v = []
    for _ in range(B):
        ii = np.concatenate([g[k] for k in rng.integers(0, len(g), len(g))])
        if len(np.unique(y[ii])) < 2: continue
        v.append(auc(y[ii], p[ii]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)
out = []
def rec(i, item, tex, val, file, how, ok):
    out.append(dict(row=i, item_id=item, tex=tex, computed=val, match=("yes" if ok else "no"), file=file, how=how))
    print(i, item, tex, val, "OK" if ok else "CHECK", how)

# rows 0,1: five of six PD/AD sets, five of seven datasets
f = R + "/edaic_rerun/part15/C_intervals.csv"
c = pd.read_csv(f)
print(c[["cell", "name", "quantity", "value", "ci_lo", "ci_hi", "excludes_zero"]].to_string())
out_c = c
# rows 2: 47 speakers in both Pitt arms
mf = pd.read_csv("<local data dir>/DementiaBank/pitt_conflict_manifest.csv", dtype={"spk": str})
mf["unit"] = mf.spk + "_" + mf.label.astype(str)  # speaker 172 has both labels, counted as two units as in the tex
both = int((mf.groupby("unit").set.nunique() == 2).sum())
both_raw = int((mf.groupby("spk").set.nunique() == 2).sum())
spk_arm = mf.groupby("set").spk.nunique().to_dict()
rec(2, "L74_739", "47", both, "<local data dir>/DementiaBank/pitt_conflict_manifest.csv", f"speaker-label units in both arms (raw speaker IDs: {both_raw}); per arm {spk_arm}; total spk {mf.spk.nunique()}; spk x label {mf.groupby(['spk','label']).ngroups}", both == 47)
# row 21 median 33 (and row 5 range)
rec(21, "L97_823", "33", round(mf.dur.median(), 1), "<local data dir>/DementiaBank/pitt_conflict_manifest.csv", "median dur", round(mf.dur.median()) == 33)
# row 23: 7.5 years per speaker
sp = mf.groupby(["spk", "label"], as_index=False).age.mean()
gap_s = sp[sp.label == 1].age.mean() - sp[sp.label == 0].age.mean()
gap_c = mf[mf.label == 1].age.mean() - mf[mf.label == 0].age.mean()
rec(23, "L101_219", "7.5", round(gap_s, 4), "<local data dir>/DementiaBank/pitt_conflict_manifest.csv", f"mean age dementia minus control per speaker label ({len(sp)} spk-label units); clip level {gap_c:.4f}", round(gap_s, 1) == 7.5)
# row 24: 81 pairs
am = pd.read_csv("<local data dir>/DementiaBank/pitt_agematched_manifest.csv", dtype={"speaker_id": str})
per = am.drop_duplicates(["speaker_id", "label"]).label.value_counts().to_dict()
rec(24, "L101_395", "81", per, "<local data dir>/DementiaBank/pitt_agematched_manifest.csv", f"speaker-label units per label in the whole-dataset matched manifest (rows {len(am)})", per.get(0) == 81 and per.get(1) == 81)
# rows 275 clips / 129 speakers (not in unmatched list but context)
oa = pd.read_csv(R + "/omni/pitt_agematched.csv") if os.path.exists(R + "/omni/pitt_agematched.csv") else None
if oa is not None:
    print("pitt_agematched.csv cols", list(oa.columns)[:12], len(oa))
# row 4: 20 participants label 0 with PHQ-8 >= 10
cand = []
for fp in [R + "/edaic_rerun/part16/M2_distilbert_sentiment.csv"]:
    d = pd.read_csv(fp)
    print("M2 cols", list(d.columns))
# row 6: 89 capped
a2 = pd.read_csv(P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv")
cap = int(a2.capped.astype(int).sum())
rec(6, "L81_570", "89", cap, P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv", f"sum capped; max win_dur {a2.win_dur_s.max()} s; n {len(a2)}", cap == 89 and abs(a2.win_dur_s.max() - 900) < 1e-3)
# row 15: 99 percent greedy
fp = R + "/edaic_rerun/part16/POD1/p14_o25_greedy.csv"
d = pd.read_csv(fp)
yn = d.is_yes_no.astype(int).mean()
cf = d[d.arm == "conflict"]
agree_c = (cf.first_word.str.strip().str.capitalize() == np.where(cf.p_yes > 0.5, "Yes", "No")).mean()
agree_col = cf.agrees.astype(str).eq("True").mean()
rec(15, "L92_681", "99", round(agree_c * 100, 2), fp, f"conflict clips where greedy first word equals the logit decision (p_yes>0.5): {agree_c:.4f}; file 'agrees' col {agree_col:.4f}; Yes/No first word on all {len(d)}: {yn:.4f}", round(agree_c * 100) == 99 and yn == 1.0)
# row 28, 31: E-DAIC windows
for i, it, tex, fp in [(28, "L140_493", "0.66", R + "/omni_final/omni_edaic_zeroshot_scores.csv"), (31, "L140_573", "0.81", R + "/edaic_rerun/variants/o25_mid300_zeroshot_scores.csv")]:
    d = pd.read_csv(fp); a = auc(d.label, d.p_yes); lo, hi = boot_auc(d.speaker, d.label, d.p_yes)
    rec(i, it, tex, round(a, 4), fp, f"AUC [{lo:.4f}, {hi:.4f}] n={len(d)}", round(a, 2) == float(tex))
d = a2; a = auc(d.label, d.p_yes_whole.map(fnum))
rec(32, "L140_607", "0.84", round(a, 4), P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv", "AUC p_yes_whole", round(a, 2) == 0.84)
# row 33: encoder probe never exceeds 0.63 on any window
encs = {}
for nm, fp in [("first30", R + "/omni_final/omni_edaic_nested_repeats.json"), ("mid30", R + "/edaic_rerun/variants/o25_mid30_nested_repeats.json"),
               ("mid300", R + "/edaic_rerun/variants/o25_mid300_nested_repeats.json"), ("full_json", R + "/edaic_rerun/variants/o25_full_nested_repeats.json")]:
    j = json.load(open(fp)); encs[nm] = (float(np.mean(j["enc"]["per_repeat"])), max(j["enc"]["per_repeat"]))
t7 = R + "/edaic_rerun/part17/T7b/T7b_edaic300_meanof5_perclip.csv"
if os.path.exists(t7):
    t = pd.read_csv(t7); cols = [c for c in t.columns if c.startswith("oof_enc")]
    encs["first300"] = (float(np.mean([auc(t.label, t[c]) for c in cols])), max(auc(t.label, t[c]) for c in cols))
a3 = pd.read_csv(P23 + "/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv")
wh = [auc(a3.label, a3[f"oof_enc_r{r}"]) for r in range(5)]
encs["whole"] = (float(np.mean(wh)), max(wh))
mx = max(v[0] for v in encs.values())
rec(33, "L140_674", "0.63", round(mx, 4), "nested_repeats json files + T7b + A3 (see how)", "mean-of-5 enc per window (mean, max single repeat): " + "; ".join(f"{k} {v[0]:.4f} ({v[1]:.4f})" for k, v in encs.items()), mx < 0.635)
# rows 34-36: NeuroVoz matched
fp = R + "/scores/part20/POD6/B_neurovoz_matched/nv_matched_pairs.csv"; p = pd.read_csv(fp)
rec(34, "L149_1433", "38", len(p), fp, f"pairs; max |age_diff| {p.age_diff.abs().max()}", len(p) == 38)
fp = R + "/scores/part20/POD6/B_neurovoz_matched/B_channel5_nvmatched.csv"; d = pd.read_csv(fp)
rec(35, "L149_1472", "907", len(d), fp, f"clips; speakers {d.spk.nunique()}", len(d) == 907)
a = auc(d.label, d.p_five_feature_oof)
rec(36, "L149_1595", "0.71", round(a, 4), fp, "AUC p_five_feature_oof", round(a, 2) == 0.71)
# row 37: eGeMAPS range
fp = R + "/overnight/egemaps_baseline.csv"; g = pd.read_csv(fp)
rec(37, "L149_735", "0.53 to 0.84", f"{g.egemaps_auc.min():.4f} to {g.egemaps_auc.max():.4f}", fp, "egemaps_auc over 7 sets (summary csv)", round(g.egemaps_auc.min(), 2) == 0.53 and round(g.egemaps_auc.max(), 2) == 0.84)
# row 39: 86 percent same decision
fp = R + "/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv"; d = pd.read_csv(fp)
same = ((d.p_yes_text > 0.5) == (d.p_yes_audio > 0.5)).mean()
rec(39, "L180_1096", "86", round(same * 100, 2), fp, f"text decision equals audio decision at 0.5 on {len(d)} clips", round(same * 100) == 86)
# rows 40-41: other wordings conflict arm
for i, it, tex, v in [(40, "L180_1309", "0.34", "p2"), (41, "L180_1318", "0.24", "p3")]:
    fp = R + f"/edaic_rerun/part16/POD1/p14_o25_audio_{v}.csv"; d = pd.read_csv(fp); cc = d[d.arm == "conflict"]
    a = auc(cc.label, cc.p_yes); lo, hi = boot_auc(cc.speaker, cc.label, cc.p_yes)
    rec(i, it, tex, round(a, 4), fp, f"conflict arm n={len(cc)} [{lo:.4f}, {hi:.4f}]", round(a, 2) == float(tex))
# rows 44-47, 51: thresholds and PHQ-10
for i, it, texn, texa, fp in [(44, "L180_1373/L180_1396", "646", "0.26", R + "/edaic_rerun/part16/POD1/p14_thr060_omni.csv"),
                              (45, "L180_1381/L180_1405", "281", "0.19", R + "/edaic_rerun/part16/POD1/p14_thr080_omni.csv"),
                              (51, "L180_1472/L180_1543", "924", "0.18", R + "/edaic_rerun/part16/POD1/p14_phq10_omni.csv")]:
    d = pd.read_csv(fp); cc = d[d.arm == "conflict"]
    a = auc(cc.label, cc.p_yes); lo, hi = boot_auc(cc.speaker, cc.label, cc.p_yes)
    rec(i, it, f"{texn} pairs, {texa}", f"{len(cc)} conflict rows, {cc.speaker.nunique()} spk, AUC {a:.4f}", fp, f"conflict arm [{lo:.4f}, {hi:.4f}]; total rows {len(d)}; all speakers {d.speaker.nunique()}", len(cc) == int(texn) and round(a, 2) == float(texa))
# rows 53-56: 1,334 / 420 / 855 (done in mispair script); recount
fp = R + "/scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv"; d = pd.read_csv(fp)
n1334 = (d.set == "conflict").sum(); n420 = ((d.set == "conflict") & (d.rob_rule_arm == "conflict")).sum(); n855 = ((d.set == "conflict") & (d.rob_rule_arm == "none")).sum()
rec(53, "L180_1616/1618", "1,334", int(n1334), fp, "conflict pairs", n1334 == 1334)
rec(55, "L180_1780", "420", int(n420), fp, "conflict & RoBERTa conflict", n420 == 420)
rec(56, "L180_1845", "855", int(n855), fp, "conflict & RoBERTa none", n855 == 855)
# row 57: AF2 Yes share 97 percent or more
fp = R + "/edaic_rerun/part14/p14_af2.csv"; d = pd.read_csv(fp)
d["arm"] = d.clip_path.map(lambda s: os.path.basename(s).split("_")[0])
ys = d.groupby("arm").p_yes.apply(lambda s: (s > 0.5).mean()).to_dict()
rec(57, "L180_2058", "97", {k: round(v, 4) for k, v in ys.items()}, fp, "share p_yes>0.5 per arm, E-DAIC pairs (rows incl. repeats)", min(ys.values()) >= 0.965)
# row 58: Qwen2-Audio Yes share 87 percent or more
fp = R + "/edaic_rerun/part14/p14_q2a_zeroshot_scores.csv"; d = pd.read_csv(fp)
d["arm"] = d["clip"].astype(str).str.split("_").str[0]
ys2 = d.groupby("arm").p_yes.apply(lambda s: (s > 0.5).mean()).to_dict()
rec(58, "L180_2110", "87", {k: round(v, 4) for k, v in ys2.items()}, fp, "share p_yes>0.5 per arm, E-DAIC pairs", round(min(ys2.values()) * 100) >= 87)
# row 59: LoRA conflict arm
fp = R + "/edaic_rerun/part16/POD2/lora_pitt_oof.csv"; d = pd.read_csv(fp); cc = d[d.arm == "conflict"]
a = auc(cc.label, cc.p_yes)
rec(59, "L226_1020", "0.55", round(a, 4), fp, f"LoRA conflict arm n={len(cc)}", round(a, 2) == 0.55)
pd.DataFrame(out).to_csv(OUT + "/unmatched_checks.csv", index=False)
