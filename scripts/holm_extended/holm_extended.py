#!/usr/local/bin/python3
"""Holm correction over every difference the final Paper 1 tex (final_overleaf_v4_0813Z) calls significant.

Members:
  the 24 claimed rows of part25/B/B_holm.tsv (GAPS 7, ARMS 9, FINE TUNE 6, CONTROLS 2), saved draws reused;
  (a) Pitt interviewer removal answer change, POD4 o25_noinv_recording.csv minus the paper file (CONTROLS), regenerated;
  (b) Qwen2-Audio PC-GITA to KCL transfer probe minus the KCL answer, part27/A_q2a (GAPS), regenerated;
  (c) E-DAIC 400 Hz low-pass answer minus unfiltered answer, part27/B/mainpull vs part23 A2 (CONTROLS), regenerated;
  (d) the 14 of 18 part26 other-model probe-minus-answer cells whose intervals exclude zero (GAPS), saved draws reused
      and also regenerated here from each per-clip file as a check.
Bootstrap (regenerated cells): 2000 draws, fresh numpy default_rng(0) per cell, unique speakers in np.unique order of
string ids, idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker, every term on the same
draw (paired), 2.5 and 97.5 percentiles, sklearn roc_auc_score. Same code path as part25/B/scripts/B_holm_r2.py.
Two-sided p = min(1, 2*min((1+#d<=0)/(B+1), (1+#d>=0)/(B+1))), B = usable draws; |d| < 1e-12 counts as 0.
Holm step-down at 0.05 within each family.
Writes only under scores/holm_extended.
"""
import os, sys, json, hashlib, datetime, platform, tarfile, io, re
import numpy as np, pandas as pd, sklearn, scipy
from sklearn.metrics import roc_auc_score

NB = 2000
SC = "<local data dir>/release_from_mac/scores"
OUT = f"{SC}/holm_extended"
REL = "<local data dir>/release"
TEX = "<local data dir>/paper1_submission_23sep/final_overleaf_v4_0813Z/AudioLLMHealthUtilityGap.tex"
BH = f"{SC}/part25/B/B_holm.tsv"
BI = f"{SC}/part25/B/B_draws_index.tsv"
P26 = f"{SC}/part26/PART26_TABLE.tsv"
TIE = 1e-12
os.makedirs(f"{OUT}/draws", exist_ok=True)
os.makedirs(f"{OUT}/per_clip", exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def bn(x):
    return os.path.basename(str(x))


def fl(x):
    x = str(x).strip()
    m = re.match(r"^np\.float(32|64)\((.*)\)$", x)
    return float(m.group(2)) if m else float(x)


def auc(y, s):
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, s)


def make_stat(kind, y, S):
    if kind == "two":
        a, b = S

        def f(ii):
            t = [auc(y[ii], a[ii]), auc(y[ii], b[ii])]
            return t[0] - t[1], t
    elif kind == "mean5":
        P, z = S

        def f(ii):
            per = [auc(y[ii], P[r, ii]) for r in range(P.shape[0])]
            t = [float(np.mean(per)), auc(y[ii], z[ii])] + per
            return t[0] - t[1], t
    else:
        raise ValueError(kind)
    return f


def boot(spk, f):
    spk = np.asarray(spk).astype(str)
    u = np.unique(spk)
    rows = [np.flatnonzero(spk == s) for s in u]
    rng = np.random.default_rng(0)
    d = np.full(NB, np.nan)
    T = None
    sidx = np.empty((NB, len(u)), np.int32)
    for b in range(NB):
        idx = rng.choice(len(u), size=len(u), replace=True)
        sidx[b] = idx
        ii = np.concatenate([rows[k] for k in idx])
        v, t = f(ii)
        if T is None:
            T = np.full((NB, len(t)), np.nan)
        T[b] = t
        if not np.any(np.isnan(t)):
            d[b] = v
    return d, T, sidx, u


def pval(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    d = np.where(np.abs(d) < TIE, 0.0, d)
    Bu = len(d)
    le, ge = int((d <= 0).sum()), int((d >= 0).sum())
    return min(1.0, 2 * min((1 + le) / (Bu + 1), (1 + ge) / (Bu + 1))), Bu, le, ge


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    run_ = 0.0
    for r, i in enumerate(o):
        run_ = max(run_, min(1.0, (m - r) * p[i]))
        adj[i] = run_
    return adj


TEXT = open(TEX).read()
rows = []
checks = {}


def add(rid, family, source, claim, printed, d, point, lo_hi=None, n_clips=None, n_spk=None, draws=None,
        draws_source="", per_clip="", tex_quote=None, extra=None):
    if tex_quote is not None:
        assert tex_quote in TEXT, (rid, tex_quote)
    p, Bu, le, ge = pval(d)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    if lo_hi is not None:
        assert abs(lo - lo_hi[0]) < 5e-5 and abs(hi - lo_hi[1]) < 5e-5, (rid, lo, hi, lo_hi)
    rows.append(dict(family=family, id=rid, source=source, claim=claim, printed=printed, value=float(point),
                     lo=float(lo), hi=float(hi), p=float(p), usable_draws=Bu, n_le0=le, n_ge0=ge,
                     n_clips=n_clips, n_spk=n_spk, draws=draws, draws_sha256=sha(draws) if draws else "",
                     draws_source=draws_source, per_clip=per_clip, extra=extra or {}))
    print(f"{family:9s} {rid:28s} {point:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} B={Bu} le0={le} ge0={ge}", flush=True)


def regen(rid, kind, frame, score_cols):
    y = frame.label.values.astype(int)
    if kind == "mean5":
        S = [frame[score_cols[:-1]].values.T.astype(float), frame[score_cols[-1]].values.astype(float)]
    else:
        S = [frame[c].values.astype(float) for c in score_cols]
    f = make_stat(kind, y, S)
    point, pt = f(np.arange(len(y)))
    d, T, sidx, u = boot(frame.speaker.values, f)
    return point, pt, d, T, sidx, u


# ============================================================ the existing 24 (saved draws)
H = pd.read_csv(BH, sep="\t", dtype=str)
I = pd.read_csv(BI, sep="\t", dtype=str).set_index("id")
MAIN = ["GAPS", "ARMS", "FINE TUNE", "CONTROLS"]
H24 = H[(H.role == "claimed") & H.family.isin(MAIN)]
assert len(H24) == 24, len(H24)
for r in H24.itertuples():
    dp = r.draws
    assert sha(dp) == I.loc[r.id, "draws_sha256"], r.id
    Z = np.load(dp, allow_pickle=True)
    d = Z["diff"].astype(float)
    point = float(Z["point_diff"])
    assert abs(point - float(r.value)) < 5e-5
    p, _, _, _ = pval(d)
    assert abs(p - float(r.p)) < 5e-5, (r.id, p, r.p)
    fam_short = r.family
    add(r.id, fam_short, "part25/B B_holm.tsv (existing 24)", r.claim, r.value_ci, d, point,
        lo_hi=(float(r.lo), float(r.hi)), n_clips=int(r.n_clips), n_spk=int(r.n_spk), draws=dp,
        draws_source="saved draws, part25/B/draws (" + r.draws_source + ")", per_clip=r.per_clip)

# ============================================================ (b) Qwen2-Audio PC-GITA -> KCL transfer minus KCL answer
fq = f"{SC}/part27/A_q2a/q2a_kcl_perclip_scores.csv"
fkz = "<local data dir>/paper1_local_runs/probe2/kcl_zeroshot_scores.csv"
Q = pd.read_csv(fq)
KZ = pd.read_csv(fkz)
assert len(Q) == 37 and Q["clip"].is_unique
Mq = Q.merge(KZ[["clip", "speaker", "label", "p_yes"]], on="clip", suffixes=("", "_z"))
assert len(Mq) == 37 and (Mq.label == Mq.label_z).all() and (Mq.speaker.astype(str) == Mq.speaker_z.astype(str)).all()
assert np.array_equal(Mq.zeroshot_p_yes.values, Mq.p_yes.values)
assert np.array_equal(Q.pcgita_to_kcl_script.values, Q.pcgita_to_kcl_nested.values)
fr = pd.DataFrame({"clip": Mq["clip"], "speaker": Mq.speaker.astype(str), "label": Mq.label.astype(int),
                   "transfer_probe": Mq.pcgita_to_kcl_nested.astype(float), "zeroshot": Mq.zeroshot_p_yes.astype(float)})
point, pt, d, T, sidx, u = regen("b", "two", fr, ["transfer_probe", "zeroshot"])
pc = f"{OUT}/per_clip/b_q2a_pcgita_to_kcl_perclip.csv"
fr.to_csv(pc, index=False)
dfile = f"{OUT}/draws/b_q2a_pcgita_to_kcl_draws.npz"
np.savez_compressed(dfile, diff=d, terms=T, speaker_idx=sidx, speakers=u, point_diff=point, point_terms=np.array(pt))
# the original script's share of draws <= 0 was 0.0015
assert abs(np.nanmean(d <= 0) - 0.0015) < 1e-12, np.nanmean(d <= 0)
add("b_q2a_pcgita_to_kcl", "GAPS", "part27/A_q2a (regenerated)",
    "Qwen2-Audio encoder probe trained on PC-GITA, applied to MDVR-KCL, minus Qwen2-Audio KCL zero-shot answer",
    "+0.29 [0.09, 0.50]", d, point, lo_hi=(0.08647, 0.49631), n_clips=37, n_spk=len(u), draws=dfile,
    draws_source="regenerated here from the per-clip file (no saved draws existed)", per_clip=pc,
    tex_quote="0.29 [0.09, 0.50] above that model's own KCL answer of 0.59",
    extra=dict(terms=[float(x) for x in pt], sources=[dict(path=fq, sha256=sha(fq)), dict(path=fkz, sha256=sha(fkz))],
               note="pcgita_to_kcl_script and pcgita_to_kcl_nested columns are identical; zero-shot column equals kcl_zeroshot_scores.csv p_yes exactly"))

# ============================================================ (d) part26: 14 other-model cells (saved draws, regenerated as a check)
T26 = pd.read_csv(P26, sep="\t", dtype={"n_speakers": int})
sel = T26[(T26.dataset != "E-DAIC") & (T26.interval == "above 0")]
assert len(T26[T26.dataset != "E-DAIC"]) == 18 and len(sel) == 14, len(sel)
MSH = {"Qwen2-Audio": "q2a", "Qwen3-Omni-30B-A3B": "q3o", "Kimi-Audio": "kimi"}
DSH = {"PC-GITA": "pcgita", "NeuroVoz": "neurovoz", "Pitt": "pitt", "ADReSSo": "adresso", "ADReSS-2020": "adress2020"}
regen_check = {}
for r in sel.itertuples():
    rid = f"d_{MSH[r.model]}_{DSH[r.dataset]}"
    Z = np.load(r.draws, allow_pickle=True)
    dsv = Z["diff"].astype(float)
    assert dsv.shape == (NB,) and not np.isnan(dsv).any()
    X = pd.read_csv(r.perclip)
    assert X["clip"].is_unique
    fr = pd.DataFrame({"clip": X["clip"], "speaker": X.speaker.astype(str), "label": X.label.astype(int)})
    for k in range(5):
        fr[f"probe_r{k}"] = X[f"p_probe_seed{k}"].astype(float)
    fr["zeroshot"] = X.p_yes_zeroshot.astype(float)
    point, pt, d, T, sidx, u = regen(rid, "mean5", fr, [f"probe_r{k}" for k in range(5)] + ["zeroshot"])
    assert len(u) == r.n_speakers, (rid, len(u), r.n_speakers)
    maxdiff = float(np.max(np.abs(d - dsv)))
    same_idx = bool(np.array_equal(Z["speaker_idx"], sidx)) if "speaker_idx" in Z.files else None
    same_spk = [str(s) for s in Z["speakers"]] == [str(s) for s in u] if "speakers" in Z.files else None
    assert maxdiff < 1e-9 and same_idx is not False and same_spk is not False, (rid, maxdiff, same_idx, same_spk)
    assert abs(point - float(Z["point_diff"])) < 1e-9 and abs(point - float(r.gap)) < 5e-5
    regen_check[rid] = dict(max_abs_draw_diff_saved_vs_regenerated=maxdiff, same_speaker_idx=same_idx,
                            same_speaker_list=same_spk, point_saved=float(Z["point_diff"]), point_regenerated=float(point))
    mname = "Qwen3-Omni" if r.model.startswith("Qwen3") else r.model
    add(rid, "GAPS", "part26 (saved draws)", f"{mname} encoder probe (mean of five) minus zero-shot answer, {r.dataset}",
        f"{float(r.gap):+.4f} [{float(r.lo):+.4f}, {float(r.hi):+.4f}] (counted in '14 of the 18')", dsv, float(Z["point_diff"]),
        lo_hi=(float(r.lo), float(r.hi)), n_clips=int(r.n), n_spk=len(u), draws=r.draws,
        draws_source="saved draws, part26 cell npz (regenerated here and equal to 1e-9)", per_clip=r.perclip,
        tex_quote="with an interval above zero on 14 of the 18 Parkinson's and Alzheimer's cells",
        extra=dict(regen_check=regen_check[rid]))

# ============================================================ (a) Pitt interviewer removal, answer change
fn = f"{SC}/part20/POD4/o25_noinv_recording.csv"
fp = f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv"
fn_mac = f"{REL}/scores/part20/POD4/o25_noinv_recording.csv"
assert sha(fn) == sha(fn_mac)
A_ = pd.read_csv(fn)
P_ = pd.read_csv(fp)
A_["_k"] = A_["clip"].map(bn)
P_["_k"] = P_["clip"].map(bn)
assert A_._k.is_unique and P_._k.is_unique and len(A_) == len(P_) == 468
Ma = A_[["_k", "speaker", "label", "p_yes"]].merge(P_[["_k", "speaker", "label", "p_yes"]], on="_k", suffixes=("_n", "_p"))
assert len(Ma) == 468 and (Ma.speaker_n.astype(str) == Ma.speaker_p.astype(str)).all() and (Ma.label_n == Ma.label_p).all()
fr = pd.DataFrame({"clip": Ma._k, "speaker": Ma.speaker_n.astype(str), "label": Ma.label_n.astype(int),
                   "answer_noinv": Ma.p_yes_n.astype(float), "answer_paper": Ma.p_yes_p.astype(float)})
point, pt, d, T, sidx, u = regen("a", "two", fr, ["answer_noinv", "answer_paper"])
pc = f"{OUT}/per_clip/a_pitt_noinv_answer_perclip.csv"
fr.to_csv(pc, index=False)
dfile = f"{OUT}/draws/a_pitt_noinv_answer_draws.npz"
np.savez_compressed(dfile, diff=d, terms=T, speaker_idx=sidx, speakers=u, point_diff=point, point_terms=np.array(pt))
# POD5 cross-check (second extraction of the interviewer-free answer, same paper file)
f5 = f"{SC}/part20/POD5/POD5_zs_o25_noinv.csv"
F5 = pd.read_csv(f5)
F5["_k"] = F5.orig_clip_id.map(bn)
M5 = fr.merge(F5[["_k", "speaker", "label", "p_yes_noinv", "p_yes_original_paper"]], left_on="clip", right_on="_k")
assert len(M5) == 468 and (M5.speaker_x == M5.speaker_y.astype(str)).all() and np.allclose(M5.p_yes_original_paper, M5.answer_paper, atol=1e-9)
fr5 = pd.DataFrame({"clip": M5["clip"], "speaker": M5.speaker_x, "label": M5.label_x, "a": M5.p_yes_noinv.astype(float),
                    "b": M5.answer_paper.astype(float)})
pt5, _, d5, _, _, _ = regen("a5", "two", fr5, ["a", "b"])
p5 = pval(d5)
a5 = dict(file=f5, value=float(pt5), lo=float(np.nanpercentile(d5, 2.5)), hi=float(np.nanpercentile(d5, 97.5)), p=p5[0],
          n_le0=p5[2], note="POD5 p_yes_noinv against the same paper file; not used in Holm")
print("  (a) POD5 cross-check:", json.dumps(a5))
add("a_pitt_noinv_answer", "CONTROLS", "part20/POD4 (regenerated)",
    "Qwen2.5-Omni zero-shot answer, interviewer-free Pitt windows minus the paper's original windows",
    "+0.07 [0.02, 0.11]", d, point, lo_hi=(0.0228, 0.1081), n_clips=468, n_spk=len(u), draws=dfile,
    draws_source="regenerated here from the per-clip files (no saved draws existed)", per_clip=pc,
    tex_quote="raises the answer from 0.66 to 0.72 (+0.07 [0.02, 0.11])",
    extra=dict(terms=[float(x) for x in pt], pod4_sidecar="paper minus noinv = -0.0652 [-0.1081, -0.0228]",
               sources=[dict(path=fn, sha256=sha(fn)), dict(path=fp, sha256=sha(fp))], pod5_crosscheck=a5))

# ============================================================ (c) E-DAIC 400 Hz low-pass answer minus unfiltered answer
flp = f"{SC}/part27/B/mainpull/A_extract_perclip_lp400.csv"
fa2 = f"{SC}/part23/A/A2_edaic_whole_zeroshot_perclip.csv"
fa3 = f"{SC}/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv"
ftar = f"{SC}/part27/B/final_pod_pull.tar"
with tarfile.open(ftar) as tf:
    tb = tf.extractfile("p23a/A_extract_perclip_lp400.csv").read()
tar_same = hashlib.sha256(tb).hexdigest() == sha(flp)
assert tar_same, "mainpull lp400 csv differs from final_pod_pull.tar copy"
L4 = pd.read_csv(flp, dtype={"pid": str})
A2 = pd.read_csv(fa2, dtype={"pid": str, "speaker": str})
A3 = pd.read_csv(fa3, dtype={"pid": str, "speaker": str})
assert len(L4) == len(A2) == len(A3) == 275 and L4.pid.is_unique and A2.pid.is_unique
assert (L4.lp_hz.astype(float) == 400.0).all()
Mc = A2[["pid", "speaker", "label", "p_yes_whole"]].merge(L4[["pid", "p_yes"]], on="pid").merge(
    A3[["pid", "speaker", "label", "p_yes_whole"]], on="pid", suffixes=("", "_a3"))
assert len(Mc) == 275 and (Mc.label == Mc.label_a3).all() and (Mc.speaker == Mc.speaker_a3).all() and (Mc.speaker == Mc.pid).all()
assert np.array_equal(Mc.p_yes_whole.map(fl).values, Mc.p_yes_whole_a3.map(fl).values)
fr = pd.DataFrame({"clip": Mc.pid, "speaker": Mc.speaker.astype(str), "label": Mc.label.astype(int),
                   "answer_lp400": Mc.p_yes.astype(float), "answer_unfiltered": Mc.p_yes_whole.map(fl)})
point, pt, d, T, sidx, u = regen("c", "two", fr, ["answer_lp400", "answer_unfiltered"])
pc = f"{OUT}/per_clip/c_edaic_lp400_answer_perclip.csv"
fr.to_csv(pc, index=False)
dfile = f"{OUT}/draws/c_edaic_lp400_answer_draws.npz"
np.savez_compressed(dfile, diff=d, terms=T, speaker_idx=sidx, speakers=u, point_diff=point, point_terms=np.array(pt))
add("c_edaic_lp400_answer", "CONTROLS", "part27/B mainpull vs part23 A2 (regenerated)",
    "Qwen2.5-Omni zero-shot answer, E-DAIC whole interview, 400 Hz low-pass minus unfiltered",
    "-0.08 [-0.14, -0.01]", d, point, lo_hi=(-0.1416, -0.0144), n_clips=275, n_spk=len(u), draws=dfile,
    draws_source="regenerated here from the per-clip files (no saved draws existed)", per_clip=pc,
    tex_quote="yet the answer falls only from 0.84 to 0.76 ($-$0.08 [$-$0.14, $-$0.01])",
    extra=dict(terms=[float(x) for x in pt], fastcheck="part27/verify/B_fastcheck.tsv -0.0762 [-0.1416, -0.0144]",
               mainpull_equals_final_pod_pull_tar=tar_same,
               sources=[dict(path=flp, sha256=sha(flp)), dict(path=fa2, sha256=sha(fa2)), dict(path=fa3, sha256=sha(fa3))]))

# ============================================================ Holm within family
R = pd.DataFrame(rows)
FAM_ORDER = ["GAPS", "ARMS", "FINE TUNE", "CONTROLS"]
R["holm_p"] = np.nan
R["m_family"] = 0
for fam in FAM_ORDER:
    ix = R.index[R.family == fam]
    R.loc[ix, "holm_p"] = holm(R.loc[ix, "p"].values)
    R.loc[ix, "m_family"] = len(ix)
R["survives"] = np.where(R.holm_p <= 0.05, "yes", "no")
R["excludes_zero"] = (R.lo > 0) | (R.hi < 0)
assert R.excludes_zero.all()
R["value_ci"] = [f"{v:+.4f} [{l:+.4f}, {h:+.4f}]" for v, l, h in zip(R.value, R.lo, R.hi)]
R["_o"] = R.family.map({f: i for i, f in enumerate(FAM_ORDER)})
R = R.sort_values(["_o", "p"], kind="mergesort").drop(columns="_o")
cols = ["family", "claim", "value_ci", "p", "holm_p", "survives", "m_family", "id", "source", "printed", "value", "lo",
        "hi", "usable_draws", "n_le0", "n_ge0", "n_clips", "n_spk", "draws", "draws_sha256", "draws_source", "per_clip"]
out = R[cols].copy()
for c in ["value", "lo", "hi"]:
    out[c] = out[c].map(lambda x: f"{x:+.4f}")
for c in ["p", "holm_p"]:
    out[c] = out[c].map(lambda x: f"{x:.4f}")
tsv = f"{OUT}/holm_extended.tsv"
out.to_csv(tsv, sep="\t", index=False)

N = len(R)
fail = R[R.survives == "no"]
per_fam = {f: dict(m=int((R.family == f).sum()), survive=int(((R.family == f) & (R.survives == "yes")).sum()),
                   max_holm_p=float(R[R.family == f].holm_p.max())) for f in FAM_ORDER}
if len(fail) == 0:
    line = f"All {N} differences called significant in this paper survive Holm correction within their family (gaps, arms, fine-tuning, controls)."
else:
    line = (f"{N - len(fail)} of the {N} differences called significant in this paper survive Holm correction within their family "
            f"(gaps, arms, fine-tuning, controls); " + "; ".join(fail.claim) + " do not.")
side = dict(task="Holm over the 24 existing plus (a) Pitt interviewer removal, (b) Q2A PC-GITA to KCL transfer, "
                 "(c) E-DAIC 400 Hz low-pass answer drop, (d) 14 part26 other-model cells",
            date_utc=datetime.datetime.utcnow().isoformat() + "Z", tex=TEX, tex_sha256=sha(TEX),
            n_draws=NB, rng="numpy default_rng(0), fresh per cell",
            resample="unique speakers (np.unique order of string ids), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker; paired",
            p_rule="min(1, 2*min((1+#d<=0)/(B+1), (1+#d>=0)/(B+1))), B = usable draws; |d| < 1e-12 counts as 0",
            holm="step-down, adjusted p = running max of min(1,(m-rank+1)*p), survives when adjusted p <= 0.05, within family",
            families=per_fam, n_total=N, n_survive=int((R.survives == "yes").sum()), paper_line=line,
            failures=list(fail.claim), part26_regeneration_check=regen_check,
            rows=[{k: (v if not isinstance(v, (np.floating, np.integer, np.bool_)) else v.item()) for k, v in r.items()}
                  for r in R.drop(columns=["value_ci"]).to_dict("records")],
            versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                          pandas=pd.__version__, scipy=scipy.__version__, platform=platform.platform()),
            command=f"/usr/local/bin/python3 '{OUT}/holm_extended.py'",
            outputs=dict(tsv=tsv, tsv_sha256=sha(tsv)))
json.dump(side, open(f"{OUT}/holm_extended.sidecar.json", "w"), indent=1, default=str)

print()
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 110)
print(out[["family", "claim", "value_ci", "p", "holm_p", "survives"]].to_string(index=False))
print()
print(json.dumps(per_fam))
print(f"survive: {int((R.survives == 'yes').sum())} of {N}")
print(line)
