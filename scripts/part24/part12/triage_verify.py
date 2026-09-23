#!/usr/local/bin/python3
"""PART 24 item 4: hand triage of the 20 MISMATCH rows of the last Part 12 run.

For every MISMATCH row this script names the quantity the tex sentence actually states,
the per-clip file that holds it, and recomputes the value from that file here
(sklearn roc_auc_score; 2000 speaker draws, fresh numpy default_rng(0) per cell,
idx = rng.choice(S, size=S, replace=True) over the sorted unique speakers,
2.5 and 97.5 percentiles; paired = both terms on the same idx; draws with one class skipped).
Verdict per row:
  MISPAIRING  the matcher compared the paper number with a different quantity; the right
              file value, recomputed here, rounds to the paper number.
  REAL        the paper number disagrees with the result files for the quantity it names.
Reads only. Writes only triage_last_run.csv and its sidecar beside this script.
"""
import json, os, sys, platform, datetime, hashlib
import numpy as np, pandas as pd, sklearn
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "<local data dir>/release"
RUN = f"{REL}/edaic_rerun/part17/part12_prep/runs/AudioLLMHealthUtilityGap_20260923_063842"
OUT = os.path.join(HERE, "triage_last_run.csv")
NB = 2000
for p in (OUT,):
    assert "/omni_final/" not in p

def sha1m(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read(1 << 20)).hexdigest()

USED = {}
def rd(p, **kw):
    USED[p] = sha1m(p)
    return pd.read_csv(p, **kw)

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))

def boot(df, spk_col, stat):
    """stat(sub_df) -> float or None. Returns point, lo, hi, usable."""
    point = stat(df)
    spk = df[spk_col].astype(str).values
    uniq = np.unique(spk)
    groups = [np.flatnonzero(spk == u) for u in uniq]
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(NB):
        idx = rng.choice(len(uniq), size=len(uniq), replace=True)
        rows = np.concatenate([groups[i] for i in idx])
        v = stat(df.iloc[rows])
        if v is not None:
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), len(vals), len(uniq), len(df)

def f4(x):
    return f"{x:.4f}"

def cell(name, df, spk, stat):
    p, lo, hi, u, S, n = boot(df, spk, stat)
    return dict(name=name, value=p, lo=lo, hi=hi, usable=u, n=n, n_spk=S)

def a_stat(col, ycol="label"):
    return lambda d: auc(d[ycol], d[col])

def diff_stat(f1, f2):
    def s(d):
        a, b = f1(d), f2(d)
        return None if a is None or b is None else a - b
    return s

cells = {}
# ---------------------------------------------------------------- L244 claim (red span)
def paired_ft(tag, sft_path, zs_path):
    s = rd(sft_path); z = rd(zs_path)
    s["clip"] = s["path"].map(os.path.basename)
    m = z.merge(s[["clip", "p_yes"]].rename(columns={"p_yes": "p_ft"}), on="clip", how="inner")
    assert len(m) == len(z) == len(s), (tag, len(m), len(z), len(s))
    m = m.rename(columns={"p_yes": "p_zs"})
    c = cell(tag, m, "speaker", diff_stat(a_stat("p_ft"), a_stat("p_zs")))
    c["ft"] = auc(m.label, m.p_ft); c["zs"] = auc(m.label, m.p_zs)
    return c

OF = f"{REL}/omni_final"
cells["gain_edaic30"] = paired_ft("E-DAIC first 30 s: projector FT minus zero shot",
                                  f"{OF}/omnisft_edaic_oof.csv", f"{OF}/omni_edaic_zeroshot_scores.csv")
cells["gain_edaicfull"] = paired_ft("E-DAIC full window: projector FT minus zero shot",
                                    f"{REL}/edaic_rerun/variants/sft_full_oof.csv",
                                    f"{REL}/edaic_rerun/variants/o25_full_zeroshot_scores.csv")
for ds in ("pitt", "adresso", "adress2020"):
    cells[f"gain_{ds}"] = paired_ft(f"{ds}: projector FT minus zero shot",
                                    f"{OF}/omnisft_{ds}_oof.csv", f"{OF}/omni_{ds}_zeroshot_scores.csv")

# ---------------------------------------------------------------- L77 TF-IDF word regression
TF = f"{REL}/scores/part20/POD6/A_language_saliency"
for ds in ("kcl", "neurovoz", "pcgita", "edaicfull", "edaic30", "pitt", "adresso", "adress2020"):
    d = rd(f"{TF}/A_tfidf_{ds}_oof.csv")
    cells[f"tfidf_{ds}"] = cell(f"TF-IDF word LR {ds}", d, "speaker", a_stat("p_tfidf"))

# ---------------------------------------------------------------- L149 paired Delta
d = rd(f"{REL}/edaic_rerun/part16/M7_perclip/pcgita.csv")
cells["delta_pcgita"] = cell("PC-GITA Delta, single split probe minus zero shot", d, "speaker",
                             diff_stat(a_stat("p_probe"), a_stat("p_yes")))
cells["delta_pcgita"]["probe"] = auc(d.label, d.p_probe); cells["delta_pcgita"]["zs"] = auc(d.label, d.p_yes)
rep = json.load(open(f"{OF}/omni_pcgita_nested_repeats.json")); USED[f"{OF}/omni_pcgita_nested_repeats.json"] = "json"
cells["delta_pcgita"]["probe_mean5_json"] = rep["enc"]["mean"]
cells["delta_pcgita"]["delta_mean5_point"] = rep["enc"]["mean"] - cells["delta_pcgita"]["zs"]

npz_p = f"{REL}/overnight2/part10/pitt_enc_nested5_oof.npz"
npz = np.load(npz_p, allow_pickle=True); USED[npz_p] = sha1m(npz_p)
zp = rd(f"{OF}/omni_pitt_zeroshot_scores.csv")
pp = pd.DataFrame({"clip": npz["name"], "spk": npz["spk"], "label": npz["label"]})
for r in range(npz["oof"].shape[0]):
    pp[f"r{r}"] = npz["oof"][r]
pp = pp.merge(zp[["clip", "p_yes", "label"]].rename(columns={"label": "label_zs"}), on="clip")
assert len(pp) == 468 and (pp.label == pp.label_zs).all()
def pitt_delta(dd):
    a = [auc(dd.label, dd[f"r{r}"]) for r in range(5)]
    z = auc(dd.label, dd.p_yes)
    return None if z is None or any(x is None for x in a) else float(np.mean(a)) - z
cells["delta_pitt"] = cell("Pitt Delta, mean of 5 repeat probe AUCs minus zero shot", pp, "spk", pitt_delta)
cells["delta_pitt"]["probe"] = float(np.mean([auc(pp.label, pp[f"r{r}"]) for r in range(5)]))
cells["delta_pitt"]["zs"] = auc(pp.label, pp.p_yes)

# ---------------------------------------------------------------- L149 Qwen3-Omni PC-GITA zero shot
for tag, p in (("shipped", f"{REL}/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv"),
               ("pod3", f"{REL}/edaic_rerun/part16/POD3/q3o_pcgita_zeroshot_scores.csv")):
    d = rd(p)
    cells[f"q3o_pcgita_zs_{tag}"] = cell(f"Qwen3-Omni PC-GITA zero shot ({tag})", d, "speaker", a_stat("p_yes"))

# ---------------------------------------------------------------- L149 NeuroVoz channel features after normalisation
d = rd(f"{REL}/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv")
cells["nv_5feat_after"] = cell("NeuroVoz five channel features after normalisation", d, "spk", a_stat("fivefeat_oof_after"))
cells["nv_5feat_before"] = cell("NeuroVoz five channel features before normalisation", d, "spk", a_stat("fivefeat_oof_before"))

# ---------------------------------------------------------------- L149 NeuroVoz 38 matched pairs
NV = f"{REL}/scores/part20/POD6/B_neurovoz_matched"
d = rd(f"{NV}/B_encprobe_nested5_nvmatched_oof.csv")
def mean5(dd):
    a = [auc(dd.label, dd[f"p_seed{r}"]) for r in range(5)]
    return None if any(x is None for x in a) else float(np.mean(a))
cells["nvm_enc"] = cell("NeuroVoz matched encoder probe, mean of 5 seeds", d, "speaker", mean5)
pairs = rd(f"{NV}/nv_matched_pairs.csv")
cells["nvm_enc"]["n_pairs_file"] = len(pairs)
d = rd(f"{NV}/B_zeroshot_nvmatched.csv")
cells["nvm_zs"] = cell("NeuroVoz matched zero shot", d, "speaker", a_stat("p_yes"))

# ---------------------------------------------------------------- L180 transcript arms, E-DAIC Part 14 pairs
j = rd(f"{REL}/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv")
cells["o25t_conf"] = cell("Qwen2.5-Omni transcript, conflict 483", j[j.arm == "conflict"], "speaker_id", a_stat("p_yes_text"))
cells["o25t_agr483"] = cell("Qwen2.5-Omni transcript, agreement 483 rows", j[j.arm == "agreement"], "speaker_id", a_stat("p_yes_text"))
t = rd(f"{REL}/edaic_rerun/part17/T1_perclip_distinct_agreement.csv")
cells["o25t_agr311"] = cell("Qwen2.5-Omni transcript, agreement 311 distinct", t, "speaker_id", a_stat("p_o25t"))
q = rd(f"{REL}/scores/part20/POD3c/p14_q3o_text.csv")
cells["q3ot_conf"] = cell("Qwen3-Omni transcript, conflict 483", q[q.set == "conflict"], "speaker_id", a_stat("p_yes_text"))
cells["q3ot_agr483"] = cell("Qwen3-Omni transcript, agreement 483 rows", q[q.set == "agreement"], "speaker_id", a_stat("p_yes_text"))
qa = q[(q.set == "agreement") & (q.in_agreement311_distinct.astype(str) == "True")]
cells["q3ot_agr311"] = cell("Qwen3-Omni transcript, agreement 311 distinct", qa, "speaker_id", a_stat("p_yes_text"))
cells["q3ot_agr311"]["n_rows_check"] = len(qa)
cells["o25a_conf"] = cell("Qwen2.5-Omni AUDIO, conflict 483 (look-alike check)", j[j.arm == "conflict"], "speaker_id", a_stat("p_yes_audio"))
cells["o25a_agr483"] = cell("Qwen2.5-Omni AUDIO, agreement 483 (look-alike check)", j[j.arm == "agreement"], "speaker_id", a_stat("p_yes_audio"))
cells["q3oa_conf"] = cell("Qwen3-Omni AUDIO, conflict 483", q[q.set == "conflict"], "speaker_id", a_stat("p_yes_audio_q3o"))

# ---------------------------------------------------------------- L180 second sentiment scorer (DistilBERT SST-2), 1,334 pairs
s = rd(f"{REL}/scores/part20/POD3b/sst2_pairs_o25_audio.csv")
sc = s[s.set == "conflict"]; sa = s[s.set == "agreement"]
cells["sst2_conf"] = cell("SST-2 pairs, Qwen2.5-Omni audio, conflict", sc, "speaker_id", a_stat("p_yes"))
cells["sst2_conf"]["n_pairs"] = int(sc.pair_id.nunique())
sa_d = sa.drop_duplicates("seg_uid", keep="first")
both_d = pd.concat([sc.assign(arm="c"), sa_d.assign(arm="a")], ignore_index=True)
both_all = pd.concat([sc.assign(arm="c"), sa.assign(arm="a")], ignore_index=True)
def gap(dd):
    a = auc(dd[dd.arm == "a"].label, dd[dd.arm == "a"].p_yes)
    c = auc(dd[dd.arm == "c"].label, dd[dd.arm == "c"].p_yes)
    return None if a is None or c is None else a - c
cells["sst2_gap_distinct"] = cell("SST-2 pairs, agreement (distinct segments) minus conflict", both_d, "speaker_id", gap)
cells["sst2_gap_all"] = cell("SST-2 pairs, agreement (all rows) minus conflict", both_all, "speaker_id", gap)

# ---------------------------------------------------------------- L226 trainable parameters
cfgp = "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00/config.json"
cfg = json.load(open(cfgp)); USED[cfgp] = sha1m(cfgp)
tc = cfg["thinker_config"]["text_config"]; ac = cfg["thinker_config"]["audio_config"]
H = tc["hidden_size"]; L = tc["num_hidden_layers"]; kv = tc["num_key_value_heads"] * (H // tc["num_attention_heads"])
r = 8
lora = L * r * ((H + H) + (H + kv) + (H + kv) + (H + H))
proj = ac["d_model"] * ac["output_dim"] + ac["output_dim"]
pj = json.load(open(f"{REL}/edaic_rerun/part16/POD2/lora_pitt_params.json")); USED[f"{REL}/edaic_rerun/part16/POD2/lora_pitt_params.json"] = "json"
cells["params"] = dict(name="trainable parameters", lora_arith=lora, lora_json=pj["lora_trainable_params"],
                       proj_arith=proj, proj_json=pj["projector_trainable_params"])

# ---------------------------------------------------------------- L97 speaking rate alone
seg = rd(f"{REL}/edaic_rerun/pitt_all_segments.csv", dtype={"spk": str})
seg["sid"] = seg.grp.astype(str) + seg.spk.astype(str)
kept = seg[seg.orig_set.isin(["conflict", "agreement"])]
def dirfree(col):
    def s(dd):
        a = auc(dd.label, dd[col]); return None if a is None else max(a, 1 - a)
    return s
cells["rate_708"] = cell("speaking rate alone, all 708 segments, direction free", seg, "sid", dirfree("rate"))
cells["rate_708"]["raw"] = auc(seg.label, seg.rate)
cells["rate_468"] = cell("speaking rate alone, kept 468 segments, direction free", kept, "sid", dirfree("rate"))
cells["rate_468"]["raw"] = auc(kept.label, kept.rate)
cells["rate_468"]["n_kept"] = len(kept)
builder = f"{REL}/release_public/scripts/upstream/build_ad_conflict.py"
cells["rate_708"]["builder_comment"] = [ln.strip() for ln in open(builder) if "0.92" in ln]

# ---------------------------------------------------------------- L172 Audio Flamingo 3 Pitt arms, two copies
for tag, p in (("part3", f"{REL}/overnight2/part3/af3_pitt.csv"), ("part10", f"{REL}/overnight2/part10/af3_pitt.csv")):
    d = rd(p)
    d["arm"] = d.clip_path.map(lambda x: os.path.basename(x).split("_")[0])
    for arm in ("conflict", "agreement"):
        dd = d[d.arm == arm]
        cells[f"af3_{tag}_{arm}"] = cell(f"AF3 Pitt {arm} ({tag} copy)", dd, "speaker_id", a_stat("p_yes"))

# ---------------------------------------------------------------- assemble the triage table
mm = pd.read_csv(f"{RUN}/paper_vs_omni.csv")
mm = mm[mm.status == "MISMATCH"].copy()
C = cells
def ci(k):
    c = C[k]; return f"{f4(c['value'])} [{f4(c['lo'])}, {f4(c['hi'])}] n={c['n']} spk={c['n_spk']}"
def r2(x):
    return f"{x:.2f}"

T = {}
T["claim_L244"] = ("REAL", "no",
    "projector fine tune gain, paired, per dataset",
    "omni_final/omnisft_<ds>_oof.csv vs omni_final/omni_<ds>_zeroshot_scores.csv; edaic_rerun/variants/sft_full_oof.csv vs o25_full_zeroshot_scores.csv",
    f"AD: Pitt {ci('gain_pitt')}; ADReSSo {ci('gain_adresso')}; ADReSS {ci('gain_adress2020')}. MDD: E-DAIC 30 s {ci('gain_edaic30')}; E-DAIC full window {ci('gain_edaicfull')}",
    "The AD half holds (all three gains positive, intervals above zero). The MDD half is wrong: on E-DAIC 30 s the gain is negative with an interval spanning zero, and on the full window fine tuning lowers the AUC with an interval entirely below zero. The paper's own Section 3 sentence (line 226) says E-DAIC drops. Fix: say AD only.")
T["L77_312"] = ("MISPAIRING", "yes",
    "TF-IDF word logistic regression AUC, Parkinson's sets (min to max)",
    "scores/part20/POD6/A_language_saliency/A_tfidf_{kcl,neurovoz,pcgita}_oof.csv",
    f"MDVR-KCL {ci('tfidf_kcl')}; NeuroVoz {ci('tfidf_neurovoz')}; PC-GITA {ci('tfidf_pcgita')}",
    f"Matcher paired the range with the Qwen2.5-Omni transcript ANSWER set. The sentence is about the word regression: range {r2(C['tfidf_kcl']['value'])} to {r2(C['tfidf_pcgita']['value'])}. Wording note: the model is TF-IDF weighted words, not raw word counts.")
T["L77_350"] = ("MISPAIRING", "yes",
    "TF-IDF word logistic regression AUC, E-DAIC full interview",
    "scores/part20/POD6/A_language_saliency/A_tfidf_edaicfull_oof.csv",
    f"E-DAIC full {ci('tfidf_edaicfull')} (first 30 s would be {ci('tfidf_edaic30')})",
    "Matcher paired it with the Qwen2.5-Omni transcript answer (0.823). The word regression on the full interview gives 0.68.")
T["L77_369"] = ("MISPAIRING", "yes",
    "TF-IDF word logistic regression AUC, Alzheimer's sets (min to max)",
    "scores/part20/POD6/A_language_saliency/A_tfidf_{pitt,adresso,adress2020}_oof.csv",
    f"Pitt {ci('tfidf_pitt')}; ADReSSo {ci('tfidf_adresso')}; ADReSS {ci('tfidf_adress2020')}",
    f"Matcher reused the Parkinson's answer set. Word regression range {r2(min(C['tfidf_pitt']['value'], C['tfidf_adresso']['value']))} to {r2(C['tfidf_adress2020']['value'])}.")
dp = C["delta_pcgita"]
T["L149_308"] = ("MISPAIRING", "yes, with an estimator caveat",
    "paired Delta = probe AUC minus zero shot AUC, PC-GITA",
    "edaic_rerun/part16/M7_perclip/pcgita.csv (probe = omni_final/omni_pcgita_enc_nested_oof.csv, single split)",
    f"Delta {ci('delta_pcgita')} (probe {f4(dp['probe'])}, zero shot {f4(dp['zs'])}); with the five repeat probe mean {f4(dp['probe_mean5_json'])} the point is {f4(dp['delta_mean5_point'])}",
    f"Matcher paired 0.35 with the probe AUC 0.8941. The Delta file gives {r2(dp['value'])}. Caveat: it uses the single split probe ({r2(dp['probe'])}); the paper's PC-GITA probe is the five repeat mean {r2(dp['probe_mean5_json'])}, which gives {r2(dp['delta_mean5_point'])}. Pitt's 0.11 uses the five repeat mean, so the two Deltas use different probe estimators. No five repeat per-clip PC-GITA file exists on the Mac to put an interval on the {r2(dp['delta_mean5_point'])}.")
T["L149_328"] = ("MISPAIRING", "yes",
    "paired Delta, Pitt, five repeat probe mean minus zero shot",
    "overnight2/part10/pitt_enc_nested5_oof.npz + omni_final/omni_pitt_zeroshot_scores.csv (= M7_pitt_FIXED)",
    f"Delta {ci('delta_pitt')} (probe {f4(C['delta_pitt']['probe'])}, zero shot {f4(C['delta_pitt']['zs'])})",
    "Matcher paired 0.11 with the Qwen2-Audio Pitt probe AUC. The corrected M7 Delta gives 0.11.")
T["L149_556"] = ("MISPAIRING", "yes",
    "Qwen3-Omni zero shot AUC, PC-GITA",
    "overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv (shipped); edaic_rerun/part16/POD3/q3o_pcgita_zeroshot_scores.csv (pod rerun)",
    f"shipped {ci('q3o_pcgita_zs_shipped')}; pod rerun {ci('q3o_pcgita_zs_pod3')}",
    "Matcher paired the PC-GITA 0.60 with the Pitt zero shot (0.7619). Both PC-GITA files round to 0.60.")
T["L149_940"] = ("MISPAIRING", "yes",
    "NeuroVoz five quiet-frame channel features, logistic probe, after loudness and noise floor normalisation",
    "edaic_rerun/part16/POD4/channel_norm_neurovoz.csv column fivefeat_oof_after",
    f"after {ci('nv_5feat_after')} (before {ci('nv_5feat_before')})",
    "Matcher paired it with the NeuroVoz encoder probe (0.9213). 'Those features' are the channel features; after normalisation they give 0.60.")
T["L149_1063"] = ("MISPAIRING", "yes",
    "Qwen2.5-Omni encoder probe, NeuroVoz 38 age and sex matched pairs, mean of 5 seeds",
    "scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.csv",
    f"{ci('nvm_enc')}; pairs file rows {C['nvm_enc']['n_pairs_file']}",
    "Matcher paired it with the Pitt age matched probe (0.7953). The NeuroVoz matched run gives 0.90.")
T["L149_1096"] = ("MISPAIRING", "yes",
    "Qwen2.5-Omni zero shot, NeuroVoz 38 age and sex matched pairs",
    "scores/part20/POD6/B_neurovoz_matched/B_zeroshot_nvmatched.csv",
    ci("nvm_zs"),
    "Matcher paired it with the Pitt age matched zero shot (0.654). The NeuroVoz matched run gives 0.63.")
T["L180_791"] = ("MISPAIRING", "yes",
    "Qwen2.5-Omni transcript only, agreement arm, E-DAIC Part 14 pairs",
    "edaic_rerun/part17/T1_perclip_distinct_agreement.csv (p_o25t); edaic_rerun/part16/POD1/p14_o25_1a_joined.csv",
    f"311 distinct {ci('o25t_agr311')}; 483 rows {ci('o25t_agr483')} (conflict {ci('o25t_conf')})",
    "Matcher paired the agreement 0.96 with the conflict value 0.1498. Both agreement definitions round to 0.96.")
q3c, q3a, q3a4 = C["q3ot_conf"], C["q3ot_agr311"], C["q3ot_agr483"]
ok22 = r2(q3c["value"]) == "0.22"; ok95 = r2(q3a["value"]) == "0.95" or r2(q3a4["value"]) == "0.95"
T["L180_830"] = ("MISPAIRING" if ok22 else "REAL", "yes" if ok22 else "no",
    "Qwen3-Omni transcript only, conflict arm, E-DAIC Part 14 pairs",
    "scores/part20/POD3c/p14_q3o_text.csv (Part 20 POD3c, Qwen3-Omni-30B-A3B-Instruct)",
    f"{ci('q3ot_conf')}; look-alike: Qwen2.5-Omni AUDIO conflict {ci('o25a_conf')}",
    ("Matcher paired it with a Pitt Qwen2.5-Omni text file. " +
     (f"The Qwen3-Omni transcript file gives {r2(q3c['value'])}." if ok22 else
      f"The only Qwen3-Omni transcript run on these pairs gives {r2(q3c['value'])}, not 0.22. 0.22 equals the Qwen2.5-Omni AUDIO conflict value ({r2(C['o25a_conf']['value'])}), so the sentence likely carries the wrong cell.")))
T["L180_839"] = ("MISPAIRING" if ok95 else "REAL", "yes" if ok95 else "no",
    "Qwen3-Omni transcript only, agreement arm, E-DAIC Part 14 pairs",
    "scores/part20/POD3c/p14_q3o_text.csv",
    f"311 distinct {ci('q3ot_agr311')}; 483 rows {ci('q3ot_agr483')}; look-alike: Qwen2.5-Omni AUDIO agreement 483 {ci('o25a_agr483')}",
    ("Matcher paired it with a Pitt Qwen2.5-Omni text file. " +
     (f"The Qwen3-Omni transcript agreement arm gives {r2(q3a['value'])} (311 distinct) or {r2(q3a4['value'])} (483 rows)." if ok95 else
      f"The Qwen3-Omni transcript agreement arm gives {r2(q3a['value'])} (311 distinct) or {r2(q3a4['value'])} (483 rows), not 0.95. 0.95 equals the Qwen2.5-Omni AUDIO agreement value ({r2(C['o25a_agr483']['value'])}).")))
T["L180_1306"] = ("MISPAIRING", "yes",
    "Qwen2.5-Omni audio, conflict arm, 1,334 pairs chosen by the DistilBERT SST-2 scorer",
    "scores/part20/POD3b/sst2_pairs_o25_audio.csv",
    f"{ci('sst2_conf')}; pairs {C['sst2_conf']['n_pairs']}",
    "Matcher paired it with the Pitt conflict arm (0.5322). The second scorer run gives 0.46.")
T["L180_1321"] = ("MISPAIRING", "yes",
    "agreement minus conflict, paired, same 1,334 pairs",
    "scores/part20/POD3b/sst2_pairs_o25_audio.csv",
    f"distinct agreement segments {ci('sst2_gap_distinct')}; all agreement rows {ci('sst2_gap_all')}",
    f"Matcher paired it with the Pitt conflict arm. 0.39 is the gap with distinct agreement segments (the Table 2 convention); with all 1,334 agreement rows it is {r2(C['sst2_gap_all']['value'])}.")
pr = C["params"]
T["L226_670"] = ("MISPAIRING", "yes",
    "LoRA trainable parameters (r 8 on q, k, v, o of 28 LM layers)",
    "edaic_rerun/part16/POD2/lora_pitt_params.json; arithmetic from the Qwen2.5-Omni-7B config.json",
    f"json {pr['lora_json']:,}; arithmetic 28 x 8 x ((3584+3584)+(3584+512)x2+(3584+3584)) = {pr['lora_arith']:,}",
    "Matcher paired a parameter count (millions) with an AUC. 5,046,272 = 5.0 M.")
T["L226_706"] = ("MISPAIRING", "yes",
    "projector trainable parameters (Linear 1280 to 3584 with bias)",
    "edaic_rerun/part16/POD2/lora_pitt_params.json; arithmetic from config.json",
    f"json {pr['proj_json']:,}; arithmetic 1280 x 3584 + 3584 = {pr['proj_arith']:,}",
    "Matcher paired a parameter count with an AUC. 4,591,104 = 4.6 M.")
T["L97_361"] = ("REAL", "no",
    "speaking rate alone as a dementia predictor on the Pitt segments",
    "edaic_rerun/pitt_all_segments.csv (column rate); build_ad_conflict.py comment",
    f"all 708 {ci('rate_708')} (raw {f4(C['rate_708']['raw'])}); kept 468 {ci('rate_468')} (raw {f4(C['rate_468']['raw'])})",
    "No result file gives 0.92. Speaking rate alone gives 0.70 (all 708) and 0.71 (kept 468), direction free. 0.92 appears only in a code comment of build_ad_conflict.py ('caught by the validity battery at AUC 0.92'), with no output file behind it, and that comment describes a rate shortcut in the selected arm, not rate alone. Fix: 0.70, or drop the number.")
for k, arm, pv in (("L172_48", "conflict", "0.61"), ("L172_55", "agreement", "0.59")):
    a3, a10 = C[f"af3_part3_{arm}"], C[f"af3_part10_{arm}"]
    T[k] = ("MISPAIRING (file copy)", "yes, if part10 is the release copy",
        f"Audio Flamingo 3 zero shot, Pitt {arm} arm",
        "overnight2/part10/af3_pitt.csv (paper) vs overnight2/part3/af3_pitt.csv (matcher, master_lookup canonical)",
        f"part10 {ci(f'af3_part10_{arm}')}; part3 {ci(f'af3_part3_{arm}')}",
        f"Right quantity, different copy of the same run. The paper's {pv} is the part10 copy ({r2(a10['value'])}); the matcher used the part3 copy ({r2(a3['value'])}). Part 20 POD4c re-scored 10 Pitt windows and reproduced part10 exactly, not part3 (DISCREPANCIES.md). Decide which copy ships in the release; if part10, update master_lookup so the matcher stops flagging it.")

rows = []
for _, m in mm.iterrows():
    v = T.get(m.item_id)
    if v is None:
        v = ("UNTRIAGED", "", "", "", "", "no triage written for this item")
    rows.append(dict(order_group=m.order_group, line=int(m.line), item_id=m.item_id, paper_value=m.paper_value,
                     matcher_file_value=m.file_value, matcher_file=m.file_path,
                     triage=v[0], paper_ok=v[1], quantity_in_sentence=v[2], correct_file=v[3],
                     recomputed=v[4], reason=v[5], context=m.context))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
side = dict(result="hand triage of the MISMATCH rows of Part 12 run AudioLLMHealthUtilityGap_20260923_063842",
            per_row_csv=OUT, n=len(out), triage_counts=out.triage.value_counts().to_dict(),
            command=" ".join([sys.executable] + sys.argv), seed="numpy default_rng(0), fresh per cell",
            bootstrap="2000 draws, idx = rng.choice(S, size=S, replace=True) over sorted unique speakers, 2.5/97.5 percentiles, paired = same idx, one-class draws skipped",
            auc="sklearn.metrics.roc_auc_score", model_id="none run here (reads saved per-clip scores)", gpu="none (Mac CPU)",
            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
            inputs_sha256_first1mb=USED,
            cells={k: {kk: (vv if not isinstance(vv, (np.floating,)) else float(vv)) for kk, vv in c.items()} for k, c in cells.items()},
            date_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z")
json.dump(side, open(OUT.replace(".csv", ".sidecar.json"), "w"), indent=1, default=str)
print(out.triage.value_counts().to_string())
for _, r in out.iterrows():
    print(f"L{r.line:<4} {str(r.paper_value):<14} {r.triage:<22} {r.recomputed[:150]}")
