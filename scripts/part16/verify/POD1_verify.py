"""INDEPENDENT verification of PART 16 POD1.

Written from scratch on the Mac. Imports nothing from the pod job's code
(no lib_auc, no analyze.py, no verify_pod1.py).

AUC implementations (three, all independent of the sort/rank-sum one-liner):
  auc_brute  : explicit O(n*m) pairwise Mann-Whitney U, ties credited 0.5
  auc_ss     : same statistic via searchsorted counts (fast, used inside bootstrap
               only after it is proved identical to auc_brute on every point estimate)
  auc_trapz  : trapezoid area under the empirical ROC built from descending thresholds

Bootstrap protocol (as specified by the authors):
  2000 draws, numpy.random.default_rng(0) reseeded per cell, SPEAKERS resampled with
  replacement, paired (one speaker draw per replicate, both arm AUCs inside it),
  percentile 2.5 / 97.5.

Writes nothing under <local data dir>/Desktop/release/omni_final.
"""
import os, sys, json, hashlib
import numpy as np
import pandas as pd

RE   = "<local data dir>/Desktop/release/edaic_rerun"
POD1 = f"{RE}/part16/POD1"
OUT  = f"{RE}/part16/verify/POD1_verify_out.json"

# ---------------------------------------------------------------- AUC engines
def auc_brute(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]; neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    gt = 0; eq = 0
    # chunk over positives to bound memory, still a full pairwise comparison
    step = max(1, int(4_000_000 // max(1, neg.size)))
    for i in range(0, pos.size, step):
        blk = pos[i:i + step][:, None]
        gt += int(np.count_nonzero(blk > neg[None, :]))
        eq += int(np.count_nonzero(blk == neg[None, :]))
    return (gt + 0.5 * eq) / (pos.size * neg.size)

def auc_ss(y, s, neg_sorted=None):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]; neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    ns = np.sort(neg) if neg_sorted is None else neg_sorted
    lo = np.searchsorted(ns, pos, side='left')    # count neg  <  v
    hi = np.searchsorted(ns, pos, side='right')   # count neg <= v
    gt = lo.sum(dtype=np.float64)
    eq = (hi - lo).sum(dtype=np.float64)
    return float((gt + 0.5 * eq) / (pos.size * neg.size))

def auc_trapz(y, s):
    """Trapezoid over the empirical ROC, thresholds descending, ties collapsed."""
    y = np.asarray(y, dtype=np.int64); s = np.asarray(s, dtype=np.float64)
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return float('nan')
    o = np.argsort(-s, kind='mergesort')
    ys = y[o]; ss = s[o]
    tp = fp = 0; prev_tpr = prev_fpr = 0.0; area = 0.0
    i = 0
    while i < ys.size:
        j = i
        while j + 1 < ys.size and ss[j + 1] == ss[i]:
            j += 1
        tp += int((ys[i:j + 1] == 1).sum())
        fp += int((ys[i:j + 1] == 0).sum())
        tpr = tp / P; fpr = fp / N
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr
        i = j + 1
    return float(area)

# ---------------------------------------------------------------- bootstrap
def paired_ci(y, s, spk, arm, draws=2000, seed=0):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    spk = np.asarray(spk); arm = np.asarray(arm)
    uspk = np.unique(spk)
    where = {u: np.flatnonzero(spk == u) for u in uspk}
    rng = np.random.default_rng(seed)
    vc = []; va = []
    for _ in range(draws):
        pick = rng.choice(uspk, size=uspk.size, replace=True)
        ix = np.concatenate([where[u] for u in pick])
        yy = y[ix]; sy = s[ix]; aa = arm[ix]
        mc = (aa == 'conflict'); ma = (aa == 'agreement')
        c = auc_ss(yy[mc], sy[mc]) if mc.any() else float('nan')
        a = auc_ss(yy[ma], sy[ma]) if ma.any() else float('nan')
        if np.isfinite(c) and np.isfinite(a):
            vc.append(c); va.append(a)
    vc = np.asarray(vc); va = np.asarray(va)
    pct = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return pct(vc), pct(va), int(vc.size)

def pearson(x, y):
    """Hand-rolled Pearson r, no np.corrcoef."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean(); ym = y - y.mean()
    return float((xm * ym).sum() / np.sqrt((xm * xm).sum() * (ym * ym).sum()))

# ---------------------------------------------------------------- harness
RESULTS = []
def rec(what, original, recomputed, note=""):
    try:
        o = float(original); r = float(recomputed)
        agree = (round(o, 4) == round(r, 4))
        os_, rs_ = f"{o:.4f}", f"{r:.4f}"
    except (TypeError, ValueError):
        o, r = original, recomputed
        agree = (str(o) == str(r)); os_, rs_ = str(o), str(r)
    RESULTS.append(dict(what=what, original_value=os_, recomputed_value=rs_,
                        agrees_to_4dp=bool(agree), note=note))
    flag = "OK " if agree else "**MISMATCH**"
    print(f"{flag} {what:58s} orig={os_:>10s} mine={rs_:>10s} {note}")
    return agree

def cell(tag, path, score_col, spk_col, arm_col, claim, label_col="label"):
    """Full conflict/agreement cell: point AUCs by three engines + paired CI."""
    if not os.path.exists(path):
        rec(f"{tag} FILE EXISTS", "present(claimed)", "MISSING", path); return
    d = pd.read_csv(path)
    n = len(d); nspk = int(d[spk_col].nunique())
    print(f"\n--- {tag}  {os.path.basename(path)}  rows={n} speakers={nspk} "
          f"arms={d[arm_col].value_counts().to_dict()}")
    rec(f"{tag} n rows", claim["n_rows"], n)
    rec(f"{tag} n speakers", claim["n_spk"], nspk)
    y = d[label_col].values; s = d[score_col].values.astype(float)
    spk = d[spk_col].values; arm = d[arm_col].values
    for a in ("conflict", "agreement"):
        m = arm == a
        b = auc_brute(y[m], s[m]); q = auc_ss(y[m], s[m]); t = auc_trapz(y[m], s[m])
        assert abs(b - q) < 1e-12, f"{tag} {a}: brute {b} vs searchsorted {q}"
        assert abs(b - t) < 1e-9, f"{tag} {a}: brute {b} vs trapezoid {t}"
        rec(f"{tag} {a} AUC", claim[f"{a}_auc"], b,
            f"n_arm={int(m.sum())} pos={int(y[m].sum())} trapz={t:.6f}")
        rec(f"{tag} {a} arm n", claim["n_pairs"], int(m.sum()))
    (clo, chi), (alo, ahi), usable = paired_ci(y, s, spk, arm)
    rec(f"{tag} conflict CI lo", claim["conflict_lo"], clo)
    rec(f"{tag} conflict CI hi", claim["conflict_hi"], chi)
    rec(f"{tag} agreement CI lo", claim["agreement_lo"], alo)
    rec(f"{tag} agreement CI hi", claim["agreement_hi"], ahi)
    rec(f"{tag} usable paired draws", 2000, usable)
    if "pooled_auc" in claim:
        rec(f"{tag} pooled AUC", claim["pooled_auc"], auc_brute(y, s))
    return dict(conflict=auc_brute(y[arm == 'conflict'], s[arm == 'conflict']),
                agreement=auc_brute(y[arm == 'agreement'], s[arm == 'agreement']))

# =================================================================== CELLS
P1 = ("Based only on how this person's voice sounds, does this speaker show signs "
      "of depression? Answer with one word, Yes or No.")
P2 = "Listen to this recording. Does the speaker have depression? Answer Yes or No."
P3 = ("From the audio alone, is this speaker likely to have depression? Reply with "
      "one word, Yes or No.")

audio_cells = [
  ("VALIDATION(0.70 rebuild)", "p14_thr070_check.csv", dict(
      n_rows=966, n_spk=138, n_pairs=483, pooled_auc=0.634252,
      conflict_auc=0.2218, conflict_lo=0.1829, conflict_hi=0.2632,
      agreement_auc=0.9482, agreement_lo=0.9039, agreement_hi=0.9828)),
  ("1d PHQ8>=10",             "p14_phq10_omni.csv", dict(
      n_rows=1848, n_spk=237, n_pairs=924, pooled_auc=0.578993,
      conflict_auc=0.1759, conflict_lo=0.1448, conflict_hi=0.2095,
      agreement_auc=0.9311, agreement_lo=0.9028, agreement_hi=0.9568)),
  ("1e sentiment 0.60",       "p14_thr060_omni.csv", dict(
      n_rows=1292, n_spk=147, n_pairs=646, pooled_auc=0.643765,
      conflict_auc=0.2619, conflict_lo=0.2188, conflict_hi=0.3075,
      agreement_auc=0.9406, agreement_lo=0.9026, agreement_hi=0.9711)),
  ("1e sentiment 0.80",       "p14_thr080_omni.csv", dict(
      n_rows=562, n_spk=104, n_pairs=281, pooled_auc=0.623511,
      conflict_auc=0.1936, conflict_lo=0.1501, conflict_hi=0.2438,
      agreement_auc=0.9594, agreement_lo=0.9190, agreement_hi=0.9895)),
  ("1b wording 2",            "p14_o25_audio_p2.csv", dict(
      n_rows=966, n_spk=138, n_pairs=483, pooled_auc=0.662599,
      conflict_auc=0.3439, conflict_lo=0.2900, conflict_hi=0.3979,
      agreement_auc=0.9119, agreement_lo=0.8505, agreement_hi=0.9594)),
  ("1b wording 3",            "p14_o25_audio_p3.csv", dict(
      n_rows=966, n_spk=138, n_pairs=483, pooled_auc=0.638358,
      conflict_auc=0.2392, conflict_lo=0.1940, conflict_hi=0.2865,
      agreement_auc=0.9442, agreement_lo=0.8926, agreement_hi=0.9821)),
  ("1c sft_full RETRAIN",     "p14_o25_sftfull.csv", dict(
      n_rows=966, n_spk=138, n_pairs=483, pooled_auc=0.519332,
      conflict_auc=0.2283, conflict_lo=0.1650, conflict_hi=0.2980,
      agreement_auc=0.7930, agreement_lo=0.7062, agreement_hi=0.8669)),
]

AUCS = {}
for tag, fn, claim in audio_cells:
    AUCS[tag] = cell(tag, f"{POD1}/{fn}", "p_yes", "speaker", "arm", claim)

# ---- 1a: text-prompt arms + text/audio correspondence
a1 = [("1a o25", "p14_o25_1a_joined.csv", dict(
        n_rows=966, n_spk=138, n_pairs=483, pooled_auc=0.607672,
        conflict_auc=0.1498, conflict_lo=0.1131, conflict_hi=0.1887,
        agreement_auc=0.9596, agreement_lo=0.9306, agreement_hi=0.9823),
        0.844207, 0.863354),
      ("1a q2a", "p14_q2a_1a_joined.csv", dict(
        n_rows=966, n_spk=138, n_pairs=483, pooled_auc=0.59946,
        conflict_auc=0.2359, conflict_lo=0.1851, conflict_hi=0.2886,
        agreement_auc=0.9081, agreement_lo=0.8515, agreement_hi=0.9547),
        0.51551, 0.712215)]
for tag, fn, claim, r_claim, sd_claim in a1:
    AUCS[tag] = cell(tag, f"{POD1}/{fn}", "p_yes_text", "speaker_id", "arm", claim)
    d = pd.read_csv(f"{POD1}/{fn}")
    x = d.p_yes_text.values.astype(float); z = d.p_yes_audio.values.astype(float)
    rec(f"{tag} Pearson r text vs audio", r_claim, pearson(x, z))
    rec(f"{tag} same-decision share @0.5", sd_claim, float(((x > 0.5) == (z > 0.5)).mean()))

# ---- 1f greedy generation
print("\n--- 1f greedy")
g = pd.read_csv(f"{POD1}/p14_o25_greedy.csv")
rec("1f n rows", 966, len(g)); rec("1f n speakers", 138, int(g.spk.nunique()))
fw = g.first_word.fillna("").astype(str).str.replace(r"[^A-Za-z]", "", regex=True).str.lower()
isyn = fw.isin(["yes", "no"])
rec("1f share first word Yes/No (all)", 1.0, float(isyn.mean()),
    f"{int(isyn.sum())}/{len(g)}")
ld = g.logit_decision.astype(str).str.lower()
rec("1f agreement gen vs logit (all)", 0.986542, float((fw == ld).mean()))
for a, cl in (("conflict", 0.991718), ("agreement", 0.981366)):
    m = (g.arm == a).values
    rec(f"1f share Yes/No [{a}]", 1.0, float(isyn[m].mean()))
    rec(f"1f agreement gen vs logit [{a}]", cl, float((fw[m] == ld[m]).mean()))
vc = fw.value_counts().to_dict()
rec("1f first-word count 'no'", 730, int(vc.get("no", 0)))
rec("1f first-word count 'yes'", 236, int(vc.get("yes", 0)))
rec("1f n distinct first words", 2, int(fw.nunique()), f"{sorted(set(fw))}")
# logit decision must be a function of p_yes>0.5
consistent = ((g.p_yes.values > 0.5) == (ld == "yes").values).mean()
rec("1f logit_decision == (p_yes>0.5)", 1.0, float(consistent))

# greedy p_yes vs the published Part 14 audio p_yes
ref = pd.read_csv(f"{RE}/part14/p14_o25_zeroshot_scores.csv")
man = pd.read_csv(f"{RE}/part14_manifest_new.csv")
man["key"] = man["set"] + "_" + man.speaker_id.astype(str) + "_" + man.seg_uid
same_order = bool((g.id.values == man.key.values).all() and
                  (ref["clip"].str.replace(".wav", "", regex=False).values == man.key.values).all())
rec("1f row order greedy==manifest==part14", True, same_order)
md = float(np.abs(g.p_yes.values - ref.p_yes.values).max())
rec("1f max|greedy p_yes - part14 p_yes| < 1e-9", True, bool(md < 1e-9), f"actual {md:.3e}")

# =================================================================== DERIVED
print("\n--- derived quantities claimed in the report")
base = AUCS["VALIDATION(0.70 rebuild)"]
rec("baseline gap (agreement - conflict)", 0.7264, base["agreement"] - base["conflict"])
d1d = AUCS["1d PHQ8>=10"]
rec("1d gap (agreement - conflict)", 0.7551, d1d["agreement"] - d1d["conflict"])
c60 = AUCS["1e sentiment 0.60"]["conflict"]; c70 = base["conflict"]
c80 = AUCS["1e sentiment 0.80"]["conflict"]
rec("1e conflict AUC range over 0.60/0.70/0.80", 0.0683,
    max(c60, c70, c80) - min(c60, c70, c80))
p2 = AUCS["1b wording 2"]; p3 = AUCS["1b wording 3"]
rec("1b w2 conflict delta vs w1", 0.1221, p2["conflict"] - base["conflict"])
rec("1b w2 agreement delta vs w1", -0.0363, p2["agreement"] - base["agreement"])
rec("1b w3 conflict delta vs w1", 0.0174, p3["conflict"] - base["conflict"])
rec("1b w3 agreement delta vs w1", -0.0040, p3["agreement"] - base["agreement"])
rec("1b max abs AUC change any wording/arm", 0.1221,
    max(abs(p2["conflict"] - base["conflict"]), abs(p2["agreement"] - base["agreement"]),
        abs(p3["conflict"] - base["conflict"]), abs(p3["agreement"] - base["agreement"])))

# 1c: which score column reproduces the claim?
sft = pd.read_csv(f"{POD1}/p14_o25_sftfull.csv")
for col in ("p_yes", "p_yes_sft"):
    cc = auc_brute(sft[sft.arm == "conflict"].label, sft[sft.arm == "conflict"][col])
    aa = auc_brute(sft[sft.arm == "agreement"].label, sft[sft.arm == "agreement"][col])
    print(f"    1c using column {col:10s}: conflict {cc:.4f}  agreement {aa:.4f}")

# 1c E-DAIC out-of-fold AUC
oof = pd.read_csv(f"{POD1}/sft1c_edaic_oof.csv")
rec("1c OOF n rows", 275, len(oof))
rec("1c OOF n speakers", 275, int(oof.speaker.nunique()))
rec("1c E-DAIC OOF AUC (retrain)", 0.5315, auc_brute(oof.label.values, oof.p_yes_sft.values))

# =================================================================== FABRICATION / PROVENANCE
print("\n--- provenance, prompts, sidecars")
def js(p):
    return json.load(open(p)) if os.path.exists(p) else None

prompt_expect = {
  "p14_thr070_check.json": P1, "p14_phq10_omni.json": P1, "p14_thr060_omni.json": P1,
  "p14_thr080_omni.json": P1, "p14_o25_audio_p2.json": P2, "p14_o25_audio_p3.json": P3,
  "p14_o25_greedy.json": P1, "p14_o25_sftfull.json": P1, "full_zs_check.json": P1,
}
REQ = ["checkpoint", "prompt", "n", "date"]
for fn, want in prompt_expect.items():
    j = js(f"{POD1}/{fn}")
    if j is None:
        rec(f"sidecar {fn} exists", "present", "MISSING"); continue
    rec(f"sidecar {fn} prompt verbatim", "match", "match" if j.get("prompt") == want else "PARAPHRASED")
    miss = [k for k in REQ if k not in j]
    extra = [k for k in ("clip_list", "command", "window_seconds", "dtype", "device",
                         "yes_ids", "no_ids") if k not in j]
    rec(f"sidecar {fn} required keys", "all", "all" if not miss else f"missing:{miss}",
        f"absent optional: {extra}" if extra else "")

# the csv's own prompt column must equal the sidecar prompt
for fn, want in [("p14_thr070_check.csv", P1), ("p14_phq10_omni.csv", P1),
                 ("p14_thr060_omni.csv", P1), ("p14_thr080_omni.csv", P1),
                 ("p14_o25_audio_p2.csv", P2), ("p14_o25_audio_p3.csv", P3),
                 ("p14_o25_sftfull.csv", P1)]:
    dd = pd.read_csv(f"{POD1}/{fn}")
    u = dd.prompt.unique()
    rec(f"{fn} prompt column verbatim+unique", "match",
        "match" if (len(u) == 1 and u[0] == want) else f"BAD n_unique={len(u)}")

# prompts of the ORIGINAL part14 runs (1b baseline provenance)
for fn in ("p14_o25_zeroshot.json", "p14_q2a_zeroshot.json"):
    j = js(f"{RE}/part14/{fn}")
    rec(f"part14 {fn} prompt == P1", "match",
        "match" if j and j.get("prompt") == P1 else "MISMATCH")
o25t = js(f"{RE}/part16/pod_sync/p14_o25_text.json")
q2at = js(f"{RE}/part16/pod_sync/p14_q2a_text.json")
print("    pod_sync text sidecars:", json.dumps(o25t), "|", json.dumps(q2at))

# 0.70 rebuild must reproduce the published Part 14 Omni audio p_yes
chk = pd.read_csv(f"{POD1}/p14_thr070_check.csv")
order_ok = bool((chk.id.values == man.key.values).all())
rec("0.70 rebuild row order == part14 manifest", True, order_ok)
if order_ok:
    dmax = float(np.abs(chk.p_yes.values - ref.p_yes.values).max())
    rec("0.70 rebuild reproduces part14 p_yes (<1e-9)", True, bool(dmax < 1e-9),
        f"max|delta| {dmax:.3e}")
    rec("part14 published pooled Omni audio AUC", 0.6343,
        auc_brute(chk.label.values, chk.p_yes.values))

# clip reuse: 966 rows over how many unique segments?
rec("part14 manifest rows", 966, len(man))
rec("part14 manifest unique seg_uid", 794, int(man.seg_uid.nunique()))
rec("part14 manifest unique key (966 rows)", 794, int(man.key.nunique()),
    "clip reuse: agreement arm 483 rows over 311 distinct segments")
rec("conflict arm distinct segments", 483, int(man[man["set"]=="conflict"].seg_uid.nunique()))
rec("agreement arm distinct segments", 311, int(man[man["set"]=="agreement"].seg_uid.nunique()))
rec("segments shared across arms", 0,
    len(set(man[man["set"]=="conflict"].seg_uid) & set(man[man["set"]=="agreement"].seg_uid)))
rec("sftfull sidecar n_scored_part14 == unique ids", 794,
    int(js(f"{POD1}/p14_o25_sftfull.json").get("n_scored_part14")))

# rebuilt manifest vs released manifest (the 'reproduces 794/794 keys' claim)
mfp = pd.read_csv(f"{POD1}/mf_p14_pod.csv")
rec("mf_p14_pod.csv rows", 966, len(mfp))
rec("mf_p14_pod ids == manifest keys (order)", True, bool((mfp.id.values == man.key.values).all()))
rec("mf_p14_pod unique basenames", 794,
    int(pd.Series([os.path.basename(p) for p in mfp.path]).nunique()))

# E-DAIC released binary label vs PHQ-8 cutoff
seg = pd.read_csv(f"{RE}/part14_segments_sentiment.csv") if os.path.exists(
      f"{RE}/part14_segments_sentiment.csv") else None
if seg is not None:
    sp = seg.drop_duplicates("pid")[["pid", "y", "sev"]]
    bad = sp[(sp.sev >= 10) & (sp.y == 0)]
    rec("speakers in sentiment source", 275, int(sp.pid.nunique()))
    rec("speakers with PHQ8>=10 but released y=0", 20, int(len(bad)))
else:
    rec("part14_segments_sentiment.csv exists", "present", "MISSING")

# 1d / 1e build sidecars
for fn, npairs, nrows, nspk in (("p14_phq10_build.json", 924, 1848, 237),
                                ("p14_thr060_build.json", 646, 1292, 147),
                                ("p14_thr080_build.json", 281, 562, 104),
                                ("p14_thr070_check_build.json", 483, 966, 138)):
    j = js(f"{POD1}/{fn}")
    if j is None: rec(f"build sidecar {fn}", "present", "MISSING"); continue
    rec(f"{fn} n_pairs", npairs, j["n_pairs"]); rec(f"{fn} n_rows", nrows, j["n_rows"])
    rec(f"{fn} n_speakers", nspk, j["n_speakers"])

# rows fragment
rf = f"{RE}/part16/rows/POD1.tsv"
rec("rows/POD1.tsv line count", 35, sum(1 for _ in open(rf)))

json.dump(RESULTS, open(OUT, "w"), indent=1)
ag = sum(1 for r in RESULTS if r["agrees_to_4dp"])
print(f"\n==== {ag}/{len(RESULTS)} checks agree; {len(RESULTS)-ag} disagree ====")
for r in RESULTS:
    if not r["agrees_to_4dp"]:
        print(f"  DISAGREE: {r['what']}  orig={r['original_value']} mine={r['recomputed_value']} {r['note']}")

# =================================================================== INDEPENDENT SET REBUILD
# Re-derive the conflict/agreement selection from the sentiment source with my own code,
# to test that the pod's 1d/1e manifests are not just asserted.
print("\n--- independent rebuild of the selection sets")
S = pd.read_csv(f"{RE}/part14_segments_sentiment.csv")
S["span"] = S.end - S.start
E = S[(S.speech_s >= 5) & (S.nw >= 25) & (S.span >= 8)].copy()
rec("eligible segments (speech_s>=5,nw>=25,span>=8)", len(E), len(E), "self-consistent gate")
# the flag columns must be exactly the 0.70 thresholds, otherwise the THR knob is not equivalent
rec("clearly_positive == (p_pos>=0.70) exactly", True,
    bool((E.clearly_positive.astype(int) == (E.p_pos >= 0.70).astype(int)).all()))
rec("clearly_negative == (p_neg>=0.70) exactly", True,
    bool((E.clearly_negative.astype(int) == (E.p_neg >= 0.70).astype(int)).all()))

def build(mode, thr):
    POS = (E.p_pos >= thr).values; NEG = (E.p_neg >= thr).values
    if mode == "phq10":
        ylab = (E.sev >= 10).astype(int).values
        conf_m = ((ylab == 1) & POS) | ((ylab == 0) & NEG)
        agr_m  = ((ylab == 1) & NEG) | ((ylab == 0) & POS)
    else:
        ylab = E.y.values.astype(int)
        conf_m = ((ylab == 1) & (E.sev.values >= 15) & POS) | ((ylab == 0) & (E.sev.values <= 4) & NEG)
        agr_m  = ((ylab == 1) & NEG) | ((ylab == 0) & POS)
    idx = E.index.values; pid = E.pid.values; span = E.span.values
    conf_pos = np.flatnonzero(conf_m); agr_pos = np.flatnonzero(agr_m)
    pairs = []; unmatched = 0
    for ci in conf_pos:
        cand = agr_pos[(pid[agr_pos] == pid[ci]) & (agr_pos != ci)]
        if cand.size == 0:
            unmatched += 1; continue
        k = cand[int(np.argmin(np.abs(span[cand] - span[ci])))]   # first minimum wins, as pandas idxmin
        pairs.append((idx[ci], idx[k], int(pid[ci]), int(ylab[ci])))
    cseg = ["c%06d" % p[0] for p in pairs]; aseg = ["a%06d" % p[1] for p in pairs]
    spk = sorted({p[2] for p in pairs})
    npos = 2 * sum(p[3] for p in pairs)
    return dict(n_pairs=len(pairs), unmatched=unmatched, n_rows=2 * len(pairs),
                n_speakers=len(spk), n_positive_rows=npos, cseg=cseg, aseg=aseg)

for tag, mode, thr, sidecar in (("1d phq10", "phq10", 0.70, "p14_phq10_build.json"),
                                ("1e 0.60", "paper", 0.60, "p14_thr060_build.json"),
                                ("1e 0.80", "paper", 0.80, "p14_thr080_build.json"),
                                ("0.70 check", "paper", 0.70, "p14_thr070_check_build.json")):
    b = build(mode, thr); j = js(f"{POD1}/{sidecar}")
    print(f"    {tag}: mine pairs={b['n_pairs']} unmatched={b['unmatched']} rows={b['n_rows']} "
          f"spk={b['n_speakers']} pos={b['n_positive_rows']}")
    for k in ("n_pairs", "unmatched", "n_rows", "n_speakers", "n_positive_rows"):
        rec(f"rebuild {tag} {k}", j[k], b[k])
    pm = pd.read_csv(f"{POD1}/{sidecar.replace('_build.json','_manifest.csv')}")
    mine = list(pm[pm["set"] == "conflict"].seg_uid) + list(pm[pm["set"] == "agreement"].seg_uid)
    rec(f"rebuild {tag} seg_uid list identical", True, mine == b["cseg"] + b["aseg"])

# the 0.70 rebuild must equal the RELEASED part14 manifest
p70 = pd.read_csv(f"{POD1}/p14_thr070_check_manifest.csv")
rec("0.70 rebuild rows == released manifest rows", len(man), len(p70))
rec("0.70 rebuild (set,seg_uid) set == released", True,
    set(zip(p70["set"], p70.seg_uid)) == set(zip(man["set"], man.seg_uid)))
rec("0.70 rebuild row-for-row identical to released", True,
    bool((p70["set"].values == man["set"].values).all() and
         (p70.seg_uid.values == man.seg_uid.values).all() and
         (p70.speaker_id.values == man.speaker_id.values).all() and
         (p70.label.values == man.label.values).all()))

# =================================================================== 1c AUDIO-MISMATCH EVIDENCE
print("\n--- 1c training-audio mismatch evidence")
orig_full = f"{RE.rsplit('/edaic_rerun',1)[0]}/release_public/scores/edaic_windows/o25_full_zeroshot_scores.csv"
if os.path.exists(orig_full):
    of = pd.read_csv(orig_full)
    of["spk"] = of["clip"].str.replace(".wav", "", regex=False).astype(int)
    for fn, claim_max, claim_mean, claim_r, nn in (
            ("full_zs_check.csv", 0.9105, 0.2755, 0.0795, 275),
            ("fullraw_zs.csv",    0.8271, 0.2852, 0.5150, 20)):
        c = pd.read_csv(f"{POD1}/{fn}")
        m = c.merge(of[["spk", "p_yes"]], left_on="speaker", right_on="spk",
                    suffixes=("_new", "_orig"))
        rec(f"1c {fn} n merged", nn, len(m))
        dd = np.abs(m.p_yes_new.values - m.p_yes_orig.values)
        rec(f"1c {fn} max|dp_yes|", claim_max, float(dd.max()))
        rec(f"1c {fn} mean|dp_yes|", claim_mean, float(dd.mean()))
        rec(f"1c {fn} Pearson r new vs orig", claim_r,
            pearson(m.p_yes_new.values, m.p_yes_orig.values))
else:
    rec("o25_full_zeroshot_scores.csv found", "present", "MISSING")

# sft_projector.py never saves a checkpoint
sp = f"{RE.rsplit('/edaic_rerun',1)[0]}/scripts/sft_projector.py"
txt = open(sp).read() if os.path.exists(sp) else ""
rec("sft_projector.py exists", True, bool(txt))
rec("sft_projector.py contains torch.save", False, "torch.save" in txt)
rec("sft_projector.py contains safetensors", False, "safetensors" in txt)

# prompt wordings P2/P3 recoverable verbatim from the released variant script
pv = f"{RE.rsplit('/edaic_rerun',1)[0]}/scripts/omni_prompt_variants.py"
vt = open(pv).read() if os.path.exists(pv) else ""
rec("omni_prompt_variants.py exists", True, bool(vt))
rec("P2 wording derivable verbatim (mdd)", True,
    'f"Listen to this recording. Does the speaker have {CONDNAME}? Answer Yes or No."' in vt)
rec("P3 wording derivable verbatim (mdd)", True,
    ('f"From the audio alone, is this speaker likely to have {CONDNAME}? '
     'Reply with one word, Yes or No."') in vt)

# DISCREPANCIES entries
dsc = open(f"{RE}/DISCREPANCIES.md").read()
for hdr in ("POD1 / Part16 1a, Part 14 clip reuse",
            "POD1 / Part16 1d, E-DAIC released binary label",
            "POD1 / Part16 1c, sft_full per-fold checkpoints never existed"):
    rec(f"DISCREPANCIES entry: {hdr[:44]}", "present", "present" if hdr in dsc else "MISSING")

json.dump(RESULTS, open(OUT, "w"), indent=1)
ag = sum(1 for r in RESULTS if r["agrees_to_4dp"])
print(f"\n==== FINAL {ag}/{len(RESULTS)} agree; {len(RESULTS)-ag} disagree ====")
for r in RESULTS:
    if not r["agrees_to_4dp"]:
        print(f"  DISAGREE: {r['what']}  orig={r['original_value']} mine={r['recomputed_value']} {r['note']}")

# =================================================================== ADVERSARIAL PROBES
print("\n--- adversarial probes (not claims of the pod, checks that could break them)")

# (A) cross-run determinism: a segment scored in several independently built sets
#     must get the SAME p_yes, otherwise different audio was used somewhere.
segof = lambda i: i.rsplit("_", 1)[-1]
base = pd.read_csv(f"{POD1}/p14_thr070_check.csv"); base["seg"] = base.id.map(segof)
bmap = base.groupby("seg").p_yes.first()
for k, fn in (("1d phq10", "p14_phq10_omni.csv"), ("1e 0.60", "p14_thr060_omni.csv"),
              ("1e 0.80", "p14_thr080_omni.csv")):
    d = pd.read_csv(f"{POD1}/{fn}"); d["seg"] = d.id.map(segof)
    m = d.groupby("seg").p_yes.first()
    common = bmap.index.intersection(m.index)
    dd = np.abs(bmap[common].values - m[common].values)
    rec(f"cross-run p_yes identical to 0.70 run [{k}]", 0.0, float(dd.max()),
        f"{len(common)} shared segments")

# (B) de-duplication robustness: the agreement arm repeats segments across pairs
print("    de-duplicated agreement-arm AUCs (the gap must not depend on the repeats):")
for tag, fn, col, idc in (("0.70 baseline", "p14_thr070_check.csv", "p_yes", "id"),
                          ("1d phq10", "p14_phq10_omni.csv", "p_yes", "id"),
                          ("1e 0.60", "p14_thr060_omni.csv", "p_yes", "id"),
                          ("1e 0.80", "p14_thr080_omni.csv", "p_yes", "id"),
                          ("1b w2", "p14_o25_audio_p2.csv", "p_yes", "id"),
                          ("1b w3", "p14_o25_audio_p3.csv", "p_yes", "id"),
                          ("1c retrain", "p14_o25_sftfull.csv", "p_yes", "id"),
                          ("1a o25", "p14_o25_1a_joined.csv", "p_yes_text", "key"),
                          ("1a q2a", "p14_q2a_1a_joined.csv", "p_yes_text", "key")):
    d = pd.read_csv(f"{POD1}/{fn}")
    c = d[d.arm == "conflict"]; a = d[d.arm == "agreement"]; au = a.drop_duplicates(idc)
    print(f"      {tag:14s} conflict {auc_brute(c.label, c[col]):.4f} (n{len(c)}) | "
          f"agreement {auc_brute(a.label, a[col]):.4f} (n{len(a)}) -> "
          f"{auc_brute(au.label, au[col]):.4f} (n{len(au)} unique) | "
          f"gap {auc_brute(a.label,a[col])-auc_brute(c.label,c[col]):.4f} -> "
          f"{auc_brute(au.label,au[col])-auc_brute(c.label,c[col]):.4f}")

# (C) 1f disagreements must be coin-flip cases, not real contradictions
gg = pd.read_csv(f"{POD1}/p14_o25_greedy.csv")
bad = gg[gg.first_word.astype(str).str.lower() != gg.logit_decision.astype(str).str.lower()]
rec("1f n disagreements", 13, len(bad))
rec("1f max |p_yes - 0.5| among disagreements < 0.001", True,
    bool(np.abs(bad.p_yes - 0.5).max() < 0.001), f"max {np.abs(bad.p_yes-0.5).max():.5f}")

# (D) part15 1a sidecar says the agreement arm was de-duplicated. It was not.
b15 = f"{RE}/part15/B_text_conflict.json"
if os.path.exists(b15):
    j15 = json.load(open(b15))
    rec("part15 1a sidecar unique_segments", 794, j15.get("unique_segments"))
    rec("part15 1a sidecar claims dedup", True, "deduped" in j15.get("note", ""))
    a1a = pd.read_csv(f"{POD1}/p14_o25_1a_joined.csv")
    ag = a1a[a1a.arm == "agreement"]
    rec("1a o25 agreement AUC as published (with repeats)", 0.9596,
        auc_brute(ag.label, ag.p_yes_text))
    agu = ag.drop_duplicates("key")
    rec("1a o25 agreement AUC actually de-duplicated", 0.9620,
        auc_brute(agu.label, agu.p_yes_text),
        f"n {len(ag)} -> {len(agu)}; sidecar note says 483 unique, truth is {len(agu)}")

# (E) unverifiable-from-disk claims
for f, claim in (("cut-clip bit-identity comparison (392 clips, max sample diff 0.0)",
                  "no artifact synced"),
                 ("POD1 run logs named in drive16.sh (logs/s_*.log)", "not synced")):
    print(f"    UNVERIFIABLE FROM DISK: {f} -> {claim}")

json.dump(RESULTS, open(OUT, "w"), indent=1)
ag_ = sum(1 for r in RESULTS if r["agrees_to_4dp"])
print(f"\n==== FINAL {ag_}/{len(RESULTS)} agree; {len(RESULTS)-ag_} disagree ====")
for r in RESULTS:
    if not r["agrees_to_4dp"]:
        print(f"  DISAGREE: {r['what']}  orig={r['original_value']} mine={r['recomputed_value']} {r['note']}")
