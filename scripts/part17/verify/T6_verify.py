#!/usr/bin/env python3
"""PART17 T6 independent verifier.

Written from scratch. Does not import or read the main job's script or its intermediates for
any computed value; reads only the raw source files listed below. Read-only on every source.
Writes only to part17/verify/.

AUC: explicit Mann-Whitney over all positive x negative pairs, ties count 0.5.
Cross-check: own trapezoid ROC (thresholds = unique scores, descending).
Intervals: 2000 draws, fresh np.random.default_rng(0) per interval, speakers resampled with
replacement (sorted unique ids, rng.integers), 2.5 / 97.5 percentiles. Paired intervals use one
speaker draw per replicate for both quantities.
"""
import os, sys, json, hashlib
import numpy as np
import pandas as pd

S20 = "<local data dir>/release/overnight2/text_new/o25_pitt_text.csv"
S20J = "<local data dir>/release/overnight2/text_new/o25_pitt_text.json"
S15 = "<local data dir>/paper work/paper1_local_runs/omni_pitt_text.csv"
S15LOG = "<local data dir>/paper work/paper1_local_runs/logs/omni_pitt_text.log"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
MAN2 = "<local data dir>/paper work/paper1_local_runs/pitt_manifest.csv"
AUD = "<local data dir>/release/omni_final/omni_pitt_zeroshot_scores.csv"
T1CSV = "<local data dir>/release/master/table1_omni25.csv"
OUT = "<local data dir>/release/edaic_rerun/part17/verify"
B = 2000

LOG = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# ------------------------------------------------------------------ AUC implementations
def auc_mw(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

def auc_trap(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    npos = (y == 1).sum(); nneg = (y == 0).sum()
    thr = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in thr:
        tpr.append(((s >= t) & (y == 1)).sum() / npos)
        fpr.append(((s >= t) & (y == 0)).sum() / nneg)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

# ------------------------------------------------------------------ load raw sources
for p in [S20, S15, MAN, MAN2, AUD]:
    P(f"source {p} md5={md5(p)}")

man = pd.read_csv(MAN, dtype={"spk": str})
man["key"] = man["segment_path"].map(os.path.basename)
P(f"manifest rows={len(man)} unique keys={man['key'].nunique()} set={man['set'].value_counts().to_dict()} "
  f"label={man['label'].value_counts().to_dict()}")
man2 = pd.read_csv(MAN2)
man2["key"] = man2["path"].map(os.path.basename)

s20 = pd.read_csv(S20)
s20["key"] = s20["id"].map(os.path.basename)
P(f"S20 rows={len(s20)} unique keys={s20['key'].nunique()} speakers={s20['speaker_id'].nunique()} "
  f"labels={s20['label'].value_counts().to_dict()} p_yes NaN={int(s20['p_yes'].isna().sum())}")
prompts = s20["prompt"].unique()
P(f"S20 distinct prompts={len(prompts)}: {prompts[0]!r}")
s15 = pd.read_csv(S15)
s15["key"] = s15["id"].map(os.path.basename)
P(f"S15 rows={len(s15)} unique keys={s15['key'].nunique()} cols={list(s15.columns)} NaN={int(s15['p_yes'].isna().sum())}")
aud = pd.read_csv(AUD)
aud["key"] = aud["clip"].map(os.path.basename)
P(f"AUD rows={len(aud)} unique keys={aud['key'].nunique()} NaN={int(aud['p_yes'].isna().sum())}")

# ------------------------------------------------------------------ build join from the manifest
j = man[["key", "spk", "grp", "label", "set"]].rename(columns={"label": "y_man", "set": "arm"})
j["spk_full"] = j["grp"] + j["spk"]
j = j.merge(s20[["key", "speaker_id", "label", "p_yes", "answer_mass"]]
            .rename(columns={"label": "y20", "p_yes": "t20", "answer_mass": "m20"}), on="key", how="inner")
j = j.merge(s15[["key", "speaker", "label", "set", "p_yes", "mass"]]
            .rename(columns={"speaker": "spk15", "label": "y15", "set": "arm15", "p_yes": "t15", "mass": "m15"}),
            on="key", how="inner")
j = j.merge(aud[["key", "speaker", "label", "p_yes"]]
            .rename(columns={"speaker": "spkA", "label": "yA", "p_yes": "a"}), on="key", how="inner")
j = j.merge(man2[["key", "label", "speaker", "set"]]
            .rename(columns={"label": "y_m2", "speaker": "spk_m2", "set": "arm_m2"}), on="key", how="inner")
P(f"join rows={len(j)} arms={j['arm'].value_counts().to_dict()}")
checks = {
    "labels all agree (man,s20,s15,aud,man2)": int(((j.y_man == j.y20) & (j.y20 == j.y15) & (j.y15 == j.yA) & (j.yA == j.y_m2)).sum()),
    "s20 speaker_id == manifest grp+spk": int((j.speaker_id == j.spk_full).sum()),
    "s20 speaker_id == s15 speaker": int((j.speaker_id == j.spk15).sum()),
    "s20 speaker_id == audio speaker": int((j.speaker_id == j.spkA).sum()),
    "s20 speaker_id == pitt_manifest speaker": int((j.speaker_id == j.spk_m2).sum()),
    "manifest arm == s15 set": int((j.arm == j.arm15).sum()),
    "manifest arm == pitt_manifest set": int((j.arm == j.arm_m2).sum()),
    "grp consistent with label": int(((j.grp == "Dementia") == (j.y_man == 1)).sum()),
}
for k, v in checks.items():
    P(f"  check {k}: {v}/{len(j)}")
both = j.groupby("spk")["grp"].nunique()
P(f"bare spk numbers in both groups: {sorted(both[both > 1].index.tolist())}; "
  f"n speaker_id={j.speaker_id.nunique()} n bare spk={j.spk.nunique()}")
for arm in ["conflict", "agreement"]:
    d = j[j.arm == arm]
    P(f"arm {arm}: n={len(d)} speakers={d.speaker_id.nunique()} labels={d.y_man.value_counts().to_dict()}")

y = j["y_man"].values.astype(int)
t20 = j["t20"].values; t15 = j["t15"].values; a = j["a"].values
arm = j["arm"].values
isC = arm == "conflict"; isA = arm == "agreement"

# ------------------------------------------------------------------ point estimates, both AUC methods
results = []
def point(name, fn_mw, fn_tr):
    v1 = fn_mw(); v2 = fn_tr()
    ok = round(v1, 4) == round(v2, 4)
    P(f"POINT {name}: MW={v1:.6f} TRAP={v2:.6f} agree4dp={ok}")
    return v1, v2, ok

pts = {}
for runname, sc in [("s20", t20), ("s15", t15), ("aud", a)]:
    for armname, m in [("all", np.ones(len(y), bool)), ("conflict", isC), ("agreement", isA)]:
        pts[(runname, armname)] = point(f"{runname}_{armname}",
                                        lambda sc=sc, m=m: auc_mw(y[m], sc[m]),
                                        lambda sc=sc, m=m: auc_trap(y[m], sc[m]))

# sklearn third check for points
try:
    from sklearn.metrics import roc_auc_score
    for runname, sc in [("s20", t20), ("s15", t15), ("aud", a)]:
        for armname, m in [("all", np.ones(len(y), bool)), ("conflict", isC), ("agreement", isA)]:
            P(f"  sklearn {runname}_{armname}={roc_auc_score(y[m], sc[m]):.6f}")
except Exception as e:
    P(f"sklearn check skipped: {e}")

# ------------------------------------------------------------------ bootstrap machinery
def make_groups(spk_arr):
    ids = np.unique(spk_arr)
    idx = [np.where(spk_arr == s)[0] for s in ids]
    return ids, idx

def boot(mask, stat, cluster):
    """stat(idx) -> float over clip index array idx (global indices). Resample clusters within mask."""
    sub = np.where(mask)[0]
    ids, gidx = make_groups(cluster[sub])
    gidx = [sub[g] for g in gidx]
    rng = np.random.default_rng(0)
    vals = np.empty(B)
    for b in range(B):
        draw = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([gidx[d] for d in draw])
        vals[b] = stat(idx)
    nnan = int(np.isnan(vals).sum())
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return lo, hi, nnan, len(ids), len(sub)

def st_auc(sc, sub=None):
    def f(idx):
        if sub is not None:
            idx = idx[sub[idx]]
        return auc_mw(y[idx], sc[idx])
    return f

def st_diff(f1, f2):
    return lambda idx: f1(idx) - f2(idx)

spk = j["speaker_id"].values
bare = j["spk"].values
ALL = np.ones(len(y), bool)

rows = []
def cell(name, pointval, mask, stat, cluster=spk, file=""):
    lo, hi, nnan, nspk, n = boot(mask, stat, cluster)
    P(f"CELL {name}\t{pointval:.4f} [{lo:.4f}, {hi:.4f}]\tn={n} n_speakers={nspk} nan_draws={nnan}")
    rows.append(dict(id=name, value=round(pointval, 4), lo=round(lo, 4), hi=round(hi, 4), n=n,
                     n_speakers=nspk, nan_draws=nnan, file=file))

mw = lambda k: pts[k][0]
# Sep 20 transcript
cell("s20_all468", mw(("s20", "all")), ALL, st_auc(t20), file=S20)
cell("s20_conflict", mw(("s20", "conflict")), isC, st_auc(t20), file=S20)
cell("s20_agreement", mw(("s20", "agreement")), isA, st_auc(t20), file=S20)
cell("s20_conflict_minus_agreement", mw(("s20", "conflict")) - mw(("s20", "agreement")), ALL,
     st_diff(st_auc(t20, isC), st_auc(t20, isA)), file=S20)
# audio
cell("aud_all468", mw(("aud", "all")), ALL, st_auc(a), file=AUD)
cell("aud_conflict", mw(("aud", "conflict")), isC, st_auc(a), file=AUD)
cell("aud_agreement", mw(("aud", "agreement")), isA, st_auc(a), file=AUD)
cell("aud_conflict_minus_agreement", mw(("aud", "conflict")) - mw(("aud", "agreement")), ALL,
     st_diff(st_auc(a, isC), st_auc(a, isA)), file=AUD)
# Sep 20 transcript minus audio (paired)
cell("s20_text_minus_audio_all468", mw(("s20", "all")) - mw(("aud", "all")), ALL,
     st_diff(st_auc(t20), st_auc(a)))
cell("s20_text_minus_audio_conflict", mw(("s20", "conflict")) - mw(("aud", "conflict")), isC,
     st_diff(st_auc(t20), st_auc(a)))
cell("s20_text_minus_audio_agreement", mw(("s20", "agreement")) - mw(("aud", "agreement")), isA,
     st_diff(st_auc(t20), st_auc(a)))
gap_t20 = st_diff(st_auc(t20, isC), st_auc(t20, isA))
gap_a = st_diff(st_auc(a, isC), st_auc(a, isA))
cell("s20_gap_text_minus_gap_audio",
     (mw(("s20", "conflict")) - mw(("s20", "agreement"))) - (mw(("aud", "conflict")) - mw(("aud", "agreement"))),
     ALL, st_diff(gap_t20, gap_a))
# Sep 15 transcript
cell("s15_all468", mw(("s15", "all")), ALL, st_auc(t15), file=S15)
cell("s15_conflict", mw(("s15", "conflict")), isC, st_auc(t15), file=S15)
cell("s15_agreement", mw(("s15", "agreement")), isA, st_auc(t15), file=S15)
gap_t15 = st_diff(st_auc(t15, isC), st_auc(t15, isA))
cell("s15_conflict_minus_agreement", mw(("s15", "conflict")) - mw(("s15", "agreement")), ALL, gap_t15, file=S15)
cell("s15_text_minus_audio_all468", mw(("s15", "all")) - mw(("aud", "all")), ALL, st_diff(st_auc(t15), st_auc(a)))
cell("s15_text_minus_audio_conflict", mw(("s15", "conflict")) - mw(("aud", "conflict")), isC, st_diff(st_auc(t15), st_auc(a)))
cell("s15_text_minus_audio_agreement", mw(("s15", "agreement")) - mw(("aud", "agreement")), isA, st_diff(st_auc(t15), st_auc(a)))
# Sep 20 minus Sep 15 (paired)
cell("s20_minus_s15_all468", mw(("s20", "all")) - mw(("s15", "all")), ALL, st_diff(st_auc(t20), st_auc(t15)))
cell("s20_minus_s15_conflict", mw(("s20", "conflict")) - mw(("s15", "conflict")), isC, st_diff(st_auc(t20), st_auc(t15)))
cell("s20_minus_s15_agreement", mw(("s20", "agreement")) - mw(("s15", "agreement")), isA, st_diff(st_auc(t20), st_auc(t15)))
cell("s20_minus_s15_armgap",
     (mw(("s20", "conflict")) - mw(("s20", "agreement"))) - (mw(("s15", "conflict")) - mw(("s15", "agreement"))),
     ALL, st_diff(gap_t20, gap_t15))
# sensitivity: cluster on bare manifest spk (227)
cell("SENS227_s20_conflict", mw(("s20", "conflict")), isC, st_auc(t20), cluster=bare)
cell("SENS227_s20_agreement", mw(("s20", "agreement")), isA, st_auc(t20), cluster=bare)
cell("SENS227_s20_conflict_minus_agreement", mw(("s20", "conflict")) - mw(("s20", "agreement")), ALL, gap_t20, cluster=bare)

# ------------------------------------------------------------------ ratios and run-to-run stats
g20 = mw(("s20", "conflict")) - mw(("s20", "agreement"))
g15 = mw(("s15", "conflict")) - mw(("s15", "agreement"))
gA = mw(("aud", "conflict")) - mw(("aud", "agreement"))
P(f"ratio gap_text/gap_audio: s15={g15/gA:.4f} s20={g20/gA:.4f}")
d = t20 - t15
k = int(np.argmax(np.abs(d)))
def ranks(v):
    return pd.Series(v).rank(method="average").values
pear = np.corrcoef(t20, t15)[0, 1]
spear = np.corrcoef(ranks(t20), ranks(t15))[0, 1]
P(f"run-to-run: pearson={pear:.4f} spearman={spear:.4f} max_abs_diff={abs(d[k]):.4f} at {j['key'].iloc[k]} "
  f"(s20={t20[k]:.4f} s15={t15[k]:.4f}) mean_abs_diff={np.mean(np.abs(d)):.4f} n_gt_0.1={int((np.abs(d) > 0.1).sum())}")
P(f"mean p_yes s20={t20.mean():.4f} s15={t15.mean():.4f}; median mass s20={np.median(j['m20']):.4f} s15={np.median(j['m15']):.4f}")
P(f"Table1 check: s20 all AUC {mw(('s20','all')):.4f} -> 2dp {round(mw(('s20','all')), 2)}; s15 all AUC {mw(('s15','all')):.4f}")
t1 = pd.read_csv(T1CSV)
P(f"table1_omni25.csv Pitt row: {t1[t1.dataset == 'Pitt'][['text', 'text_source']].to_dict('records')}")
with open(S20J) as f:
    P(f"o25_pitt_text.json: {json.load(f)}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "T6_verify_cells.csv"), index=False)
with open(os.path.join(OUT, "T6_verify.json"), "w") as f:
    json.dump(dict(checks=checks, cells=rows,
                   points={f"{k[0]}_{k[1]}": dict(mw=v[0], trap=v[1], agree4dp=v[2]) for k, v in pts.items()},
                   run_to_run=dict(pearson=pear, spearman=spear, max_abs_diff=float(abs(d[k])),
                                   max_clip=j['key'].iloc[k], mean_abs_diff=float(np.mean(np.abs(d))),
                                   n_gt_01=int((np.abs(d) > 0.1).sum()))), f, indent=1, default=float)
with open(os.path.join(OUT, "T6_verify.log"), "w") as f:
    f.write("\n".join(LOG) + "\n")

# ------------------------------------------------------------------ extra: s15 gap minus audio gap (row in T6.tsv, not in report)
cell("s15_gap_text_minus_gap_audio", g15 - gA, ALL, st_diff(gap_t15, gap_a))

# ------------------------------------------------------------------ fabrication check on task outputs (read as DATA only, never used above)
TASK_TSV = "scores/part17/rows/T6.tsv"
TASK_CSV = "scores/part17/T6_pitt_text_sep20_arms.csv"
mine = {r["id"]: r for r in rows}
namemap = {"T6_s20_text_all468": "s20_all468", "T6_s20_text_conflict": "s20_conflict",
           "T6_s20_text_agreement": "s20_agreement", "T6_s20_text_conflict_minus_agreement": "s20_conflict_minus_agreement",
           "T6_s15_text_all468": "s15_all468", "T6_s15_text_conflict": "s15_conflict", "T6_s15_text_agreement": "s15_agreement",
           "T6_s15_text_conflict_minus_agreement": "s15_conflict_minus_agreement",
           "T6_audio_all468": "aud_all468", "T6_audio_conflict": "aud_conflict", "T6_audio_agreement": "aud_agreement",
           "T6_audio_conflict_minus_agreement": "aud_conflict_minus_agreement",
           "T6_s20_text_minus_audio_all468": "s20_text_minus_audio_all468",
           "T6_s20_text_minus_audio_conflict": "s20_text_minus_audio_conflict",
           "T6_s20_text_minus_audio_agreement": "s20_text_minus_audio_agreement",
           "T6_s20_gap_text_minus_gap_audio": "s20_gap_text_minus_gap_audio",
           "T6_s15_text_minus_audio_all468": "s15_text_minus_audio_all468",
           "T6_s15_text_minus_audio_conflict": "s15_text_minus_audio_conflict",
           "T6_s15_text_minus_audio_agreement": "s15_text_minus_audio_agreement",
           "T6_s15_gap_text_minus_gap_audio": "s15_gap_text_minus_gap_audio",
           "T6_s20_minus_s15_all468": "s20_minus_s15_all468", "T6_s20_minus_s15_conflict": "s20_minus_s15_conflict",
           "T6_s20_minus_s15_agreement": "s20_minus_s15_agreement", "T6_s20_minus_s15_armgap": "s20_minus_s15_armgap"}
tt = pd.read_csv(TASK_TSV, sep="\t")
nd = 0
for _, r in tt.iterrows():
    if r["id"] in namemap:
        m = mine[namemap[r["id"]]]
        for fld in ["value", "lo", "hi", "n", "n_speakers"]:
            tf = {"n_speakers": "n_spk"}.get(fld, fld)
            ok = abs(float(r[tf]) - float(m[fld])) < 5e-5
            if not ok:
                nd += 1
                P(f"DISAGREE {r['id']} {fld}: task={r[tf]} mine={m[fld]}")
P(f"T6.tsv interval rows compared: {sum(r in namemap for r in tt['id'])}, field disagreements={nd}")
tc = pd.read_csv(TASK_CSV)
tc["key"] = tc["id"].map(os.path.basename)
cmp = j[["key", "speaker_id", "y_man", "arm", "t20", "t15", "a"]].merge(tc, on="key")
P(f"task per-clip csv rows={len(tc)} matched={len(cmp)}; "
  f"max|p20 diff|={np.abs(cmp.t20 - cmp.p_yes_sep20).max():.2e} max|p15 diff|={np.abs(cmp.t15 - cmp.p_yes_sep15).max():.2e} "
  f"max|aud diff|={np.abs(cmp.a - cmp.p_yes_audio).max():.2e} label eq={int((cmp.y_man == cmp.label).sum())} "
  f"arm eq={int((cmp.arm_x == cmp.arm_y).sum())} spk eq={int((cmp.speaker_id == cmp.spk).sum())}")
with open(os.path.join(OUT, "T6_verify.log"), "w") as f:
    f.write("\n".join(LOG) + "\n")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "T6_verify_cells.csv"), index=False)
