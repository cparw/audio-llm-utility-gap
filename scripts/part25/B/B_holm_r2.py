#!/usr/local/bin/python3
"""PART 25 track B (r2): Holm correction over every difference the final tex calls significant or
quotes with an interval that excludes zero.

Reads per-clip files and the Track A draw files; writes only under part25/B. Nothing under omni_final is written.
AUC: sklearn roc_auc_score.
Speaker bootstrap: 2000 draws, fresh numpy default_rng(0) per cell, unique speakers in np.unique order
(speaker ids as strings), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of every drawn
speaker, 2.5 / 97.5 percentiles.
Paired: every term of a difference is computed on the same draw.
Mean of five: in each draw the mean of the five per-repeat AUCs.
A draw where any term has one class only is NaN and left out (count reported).
Two-sided p = min(1, 2*min((1+#d<=0)/(B+1), (1+#d>=0)/(B+1))), B = usable draws; |d| < 1e-12 counts as 0.
Holm at 0.05 within four families (GAPS, ARMS, FINE TUNE, CONTROLS) over the claimed differences (primary).
Sensitivity: same families widened with the same-kind differences the tex reports as not significant.
Extra (not in the requested list, not in the paper line): the abstract's 'four of six models fall below
chance' on the E-DAIC conflict arm, as conflict AUC minus 0.5, Holm within its own family; and the Pitt
'none falls significantly below chance' statement as a check only.
"""
import os, sys, json, hashlib, datetime, platform
import numpy as np, pandas as pd, sklearn, scipy
from sklearn.metrics import roc_auc_score

NB = 2000
B = "<local data dir>/release_from_mac/scores/part25/B"
A = "<local data dir>/release_from_mac/scores/part25/A"
REL = "<local data dir>/release"
LR = "<local data dir>/paper1_local_runs"
SC = "<local data dir>/release_from_mac/scores"
TEX = "<local data dir>/paper1_submission_23sep/final_overleaf_2325Z/AudioLLMHealthUtilityGap.tex"
DB = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
ONLY = sys.argv[1:]  # optional ids to run (debug)
for sub in ("draws", "per_clip", "logs", "verify"):
    os.makedirs(f"{B}/{sub}", exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def bn(x):
    return os.path.basename(str(x))


def fl(x):
    x = str(x)
    return float(x[len("np.float64("):-1]) if x.startswith("np.float64(") else float(x)


def auc(y, s):
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, s)


# ---------------------------------------------------------------- statistics
def make_stat(kind, y, S, masks=None):
    """'two': S=[a,b] -> AUC(a)-AUC(b); 'arms': S=[s], masks=[mc,ma] -> AUC_conflict-AUC_agreement;
    'mean5': S=[P(5,n), z] -> mean_r AUC(P_r) - AUC(z); 'one': S=[s] -> AUC(s) - 0.5."""
    if kind == "two":
        a, b = S

        def f(ii):
            t = [auc(y[ii], a[ii]), auc(y[ii], b[ii])]
            return t[0] - t[1], t
    elif kind == "arms":
        s, = S
        mc, ma = masks

        def f(ii):
            c = ii[mc[ii]]
            g = ii[ma[ii]]
            t = [auc(y[c], s[c]), auc(y[g], s[g])]
            return t[0] - t[1], t
    elif kind == "mean5":
        P, z = S

        def f(ii):
            per = [auc(y[ii], P[r, ii]) for r in range(P.shape[0])]
            t = [float(np.mean(per)), auc(y[ii], z[ii])] + per
            return t[0] - t[1], t
    elif kind == "one":
        s, = S

        def f(ii):
            t = [auc(y[ii], s[ii]), 0.5]
            return t[0] - 0.5, t
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


TIE = 1e-12  # |draw| below this is a floating point tie at zero and counts as exactly 0 (both tails)


def pval(d):
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


# ---------------------------------------------------------------- loaders
def load_pair(fa, ka, spa, ya, sa, fb, kb, spb, yb, sb, conv=float):
    """merge two per-clip files on the clip basename; returns frame with clip,speaker,label,a,b"""
    X = pd.read_csv(fa)
    Y = pd.read_csv(fb)
    X["_k"] = X[ka].map(bn)
    Y["_k"] = Y[kb].map(bn)
    assert X._k.is_unique and Y._k.is_unique, (fa, fb)
    M = X[["_k", spa, ya, sa]].rename(columns={spa: "sa", ya: "ya", sa: "a"}).merge(
        Y[["_k", spb, yb, sb]].rename(columns={spb: "sb", yb: "yb", sb: "b"}), on="_k")
    assert len(M) == len(X) == len(Y), (fa, fb, len(M), len(X), len(Y))
    assert (M.ya.astype(int).values == M.yb.astype(int).values).all(), "label mismatch"
    assert (M.sa.astype(str).values == M.sb.astype(str).values).all(), "speaker mismatch"
    return pd.DataFrame({"clip": M._k, "speaker": M.sa.astype(str), "label": M.ya.astype(int),
                         "a": M.a.map(conv).astype(float), "b": M.b.map(conv).astype(float)})


def pitt_arm_map():
    D = pd.read_csv(DB, dtype={"spk": str})
    D["_k"] = D.segment_path.map(bn)
    assert D._k.is_unique and len(D) == 468
    D["speaker"] = D.grp.astype(str) + D.spk.astype(str)
    return D.set_index("_k")[["set", "speaker", "label"]]


# ---------------------------------------------------------------- claim table
TEXL = open(TEX).read().split("\n")
rows = []


def run(rid, family, role, claim, tex_line, tex_quote, kind, frame, score_cols, masks_cols=None,
        sources=(), tex_check=None, draws_from=None, note=""):
    if ONLY and rid not in ONLY:
        return
    assert tex_quote[:40] in TEXL[tex_line - 1], (rid, tex_line, tex_quote)
    y = frame.label.values.astype(int)
    if kind == "mean5":
        S = [frame[score_cols[:-1]].values.T.astype(float), frame[score_cols[-1]].values.astype(float)]
    else:
        S = [frame[c].values.astype(float) for c in score_cols]
    masks = [frame[c].values.astype(bool) for c in masks_cols] if masks_cols else None
    f = make_stat(kind, y, S, masks)
    point, pt_terms = f(np.arange(len(y)))
    d, T, sidx, u = boot(frame.speaker.values, f)
    reuse = None
    if draws_from is not None:  # Track A saved draws: must equal this regeneration
        Z = np.load(draws_from, allow_pickle=True)
        dA = Z["diff"].astype(float)
        same_idx = bool(np.array_equal(Z["speaker_idx"], sidx)) and [str(x) for x in Z["speakers"]] == [str(x) for x in u]
        maxdiff = float(np.nanmax(np.abs(dA - d)))
        nan_same = bool(np.array_equal(np.isnan(dA), np.isnan(d)))
        reuse = dict(file=draws_from, sha256=sha(draws_from), same_speaker_idx=same_idx, same_nan_pattern=nan_same,
                     max_abs_draw_diff_vs_regenerated=maxdiff, point_diff_saved=float(Z["point_diff"]),
                     point_diff_here=float(point))
        assert same_idx and nan_same and maxdiff < 1e-9 and abs(float(Z["point_diff"]) - point) < 1e-9, (rid, reuse)
        d_use = dA
    else:
        d_use = d
    p, Bu, le, ge = pval(d_use)
    lo, hi = np.nanpercentile(d_use, [2.5, 97.5])
    ndeg = int(np.isnan(d_use).sum())
    dfile = f"{B}/draws/B_{rid}_draws.npz"
    np.savez_compressed(dfile, diff=d_use, terms=T, speaker_idx=sidx, speakers=u, point_diff=point,
                        point_terms=np.array(pt_terms, float))
    pc = f"{B}/per_clip/B_{rid}_perclip.csv"
    frame.to_csv(pc, index=False)
    chk = None
    if tex_check:
        chk = {}
        for k, (val, nd) in tex_check.items():
            got = {"value": point, "lo": lo, "hi": hi, "t0": pt_terms[0], "t1": pt_terms[1]}[k]
            chk[k] = dict(tex=val, computed=round(float(got), 6), rounds_to=round(float(got), nd),
                          match=bool(abs(round(float(got), nd) - val) < 1e-9))
    rec = dict(id=rid, family=family, role=role, claim=claim, tex_line=tex_line, tex_quote=tex_quote,
               estimator=kind, value=float(point), lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0),
               p=float(p), usable_draws=Bu, degenerate_draws=ndeg, n_le0=le, n_ge0=ge,
               term_points=[float(x) for x in pt_terms], n_clips=int(len(y)), n_spk=int(len(u)),
               n_pos=int(y.sum()), per_clip=pc, per_clip_sha256=sha(pc), draws=dfile, draws_sha256=sha(dfile),
               draws_source=("Track A saved draws (regenerated here and equal to 1e-9)" if reuse else "regenerated here from the per-clip file"),
               trackA_reuse=reuse, sources=[dict(path=s, sha256=sha(s)) for s in sources], tex_check=chk, note=note,
               bootstrap=dict(n_draws=NB, rng="numpy default_rng(0), fresh per cell",
                              resample="np.unique string speaker ids; idx = rng.choice(n_spk, size=n_spk, replace=True); all clips of each drawn speaker",
                              percentiles=[2.5, 97.5], auc="sklearn.metrics.roc_auc_score"),
               written_utc=datetime.datetime.utcnow().isoformat() + "Z")
    json.dump(rec, open(pc.replace(".csv", ".sidecar.json"), "w"), indent=1)
    rows.append(rec)
    ok = "" if not chk else (" tex:" + ",".join(f"{k}={'OK' if v['match'] else 'MISMATCH'}" for k, v in chk.items()))
    print(f"{family:9s} {rid:26s} {role[:5]:5s} {point:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} B={Bu} deg={ndeg} "
          f"terms={[round(x, 4) for x in pt_terms[:2]]} spk={len(u)} n={len(y)}{ok}", flush=True)


Q_GAP = "The paired difference $\\Delta$ excludes zero on every Parkinson's and Alzheimer's set except MDVR-KCL"
Q_ARM = "Speaker-level bootstrap intervals for the difference between the two arms exclude zero for all six models on depression"
Q_FT = "After fine-tuning the projector, AUC improves significantly on every dataset except MDVR-KCL"

# ================================================================ GAPS
NAMES = {"pcgita": "PC-GITA", "neurovoz": "NeuroVoz", "adresso": "ADReSSo", "adress2020": "ADReSS-2020"}
for key in ["pcgita", "neurovoz", "adresso", "adress2020"]:
    f = f"{A}/per_clip/A_{key}_perclip.csv"
    X = pd.read_csv(f)
    fr = pd.DataFrame({"clip": X["clip"], "speaker": X.speaker.astype(str), "label": X.label.astype(int)})
    for r in range(5):
        fr[f"probe_r{r}"] = X[f"p_probe_seed{r}"].astype(float)
    fr["zeroshot"] = X.p_yes_zeroshot_saved.astype(float)
    tc = {"value": (0.33, 2)} if key == "pcgita" else None
    run(f"G_{key}", "GAPS", "claimed", f"encoder probe (mean of five) minus zero-shot answer, {NAMES[key]}", 149,
        Q_GAP, "mean5", fr, [f"probe_r{r}" for r in range(5)] + ["zeroshot"],
        sources=[f, f"{REL}/omni_final/omni_{key}_zeroshot_scores.csv"], tex_check=tc,
        draws_from=f"{A}/draws/A_{key}_draws.npz",
        note="also l.52 'the gap is significant on every Parkinson's and Alzheimer's set except MDVR-KCL'; "
             "probe vectors from the Track A re-extraction whose mean of five equals Table 1 to 4 dp")

fz = f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv"
fo = f"{REL}/overnight2/part10/pitt_enc_nested5_oof.npz"
Z = np.load(fo, allow_pickle=True)
Po = pd.DataFrame({"clip": [bn(x) for x in Z["name"]], "speaker": Z["spk"].astype(str), "label": Z["label"].astype(int)})
for r in range(5):
    Po[f"probe_r{r}"] = Z["oof"][r].astype(float)
Zs = pd.read_csv(fz)
Zs["clip"] = Zs["clip"].map(bn)
Pm = Po.merge(Zs[["clip", "speaker", "label", "p_yes"]], on="clip", suffixes=("", "_z"))
assert len(Pm) == 468 and (Pm.label == Pm.label_z).all() and (Pm.speaker == Pm.speaker_z.astype(str)).all()
Pm = Pm.drop(columns=["speaker_z", "label_z"]).rename(columns={"p_yes": "zeroshot"})
run("G_pitt", "GAPS", "claimed", "encoder probe (mean of five) minus zero-shot answer, Pitt", 149,
    "from 0.11 [0.05, 0.18] on Pitt to 0.33 on PC-GITA", "mean5", Pm, [f"probe_r{r}" for r in range(5)] + ["zeroshot"],
    sources=[fo, fz], tex_check={"value": (0.11, 2), "lo": (0.05, 2), "hi": (0.18, 2)},
    draws_from=f"{A}/draws/A_pitt_draws.npz")

fe = f"{SC}/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv"
fe2 = f"{SC}/part23/A/A2_edaic_whole_zeroshot_perclip.csv"
E3 = pd.read_csv(fe)
E2 = pd.read_csv(fe2)
Em = E3.merge(E2[["pid", "label", "p_yes_whole"]], on="pid", suffixes=("", "_a2"))
assert len(Em) == 275 and (Em.label == Em.label_a2).all()
assert np.allclose(Em.p_yes_whole.map(fl), Em.p_yes_whole_a2.map(fl))
for st, nm, tv in [("llm", "language model probe", (-0.09, -0.16, -0.03)), ("enc", "encoder probe", (-0.24, -0.32, -0.15))]:
    fr = pd.DataFrame({"clip": Em.pid.astype(str), "speaker": Em.speaker.astype(str), "label": Em.label.astype(int)})
    for r in range(5):
        fr[f"probe_r{r}"] = Em[f"oof_{st}_r{r}"].astype(float)
    fr["zeroshot"] = Em.p_yes_whole.map(fl)
    run(f"G_edaic_{st}", "GAPS", "claimed", f"{nm} (mean of five) minus zero-shot answer, E-DAIC whole interview", 149,
        "On E-DAIC the language model probe reaches 0.74 and the encoder probe 0.60, both below the answer of 0.84",
        "mean5", fr, [f"probe_r{r}" for r in range(5)] + ["zeroshot"], sources=[fe, fe2],
        tex_check={"value": (tv[0], 2), "lo": (tv[1], 2), "hi": (tv[2], 2)},
        note="reversal, interval below zero; also l.52 'reverses on depression' and abstract l.26 'On depression the answer beats every probe'")
fr = pd.DataFrame({"clip": Em.pid.astype(str), "speaker": Em.speaker.astype(str), "label": Em.label.astype(int)})
for r in range(5):
    fr[f"probe_r{r}"] = Em[f"oof_ans_r{r}"].astype(float)
fr["zeroshot"] = Em.p_yes_whole.map(fl)
run("Gs_edaic_ansstate", "GAPS", "sensitivity only", "answer-state probe (mean of five) minus zero-shot answer, E-DAIC whole interview", 149,
    "while a probe on the answer state matches the answer (0.84)", "mean5", fr,
    [f"probe_r{r}" for r in range(5)] + ["zeroshot"], sources=[fe, fe2],
    note="tex reports it as matching (not significant); it contradicts the abstract's 'the answer beats every probe'")
fk = f"{REL}/omni_final/omni_kcl_enc_nested_oof.csv"
fkz = f"{REL}/omni_final/omni_kcl_zeroshot_scores.csv"
K = load_pair(fk, "clip", "speaker", "label", "p_probe", fkz, "clip", "speaker", "label", "p_yes")
K = K.rename(columns={"a": "probe_single", "b": "zeroshot"})
run("Gs_kcl_single", "GAPS", "sensitivity only", "encoder probe (single nested split, stand-in) minus zero-shot answer, MDVR-KCL", 149,
    Q_GAP, "two", K, ["probe_single", "zeroshot"], sources=[fk, fkz],
    note="no per-repeat OOF vectors exist for the Omni KCL mean of five (only omni_kcl_nested_repeats.json, mean 0.7506), so the single nested split stands in; sensitivity only")

# ================================================================ ARMS
fT = f"{REL}/edaic_rerun/part17/T1_perclip_all966.csv"
T1 = pd.read_csv(fT)
conf = T1[T1.set == "conflict"]
agr = T1[(T1.set == "agreement") & (T1.in_distinct_set == 1)]
assert len(conf) == 483 and len(agr) == 311 and agr.seg_uid.is_unique and conf.seg_uid.is_unique
EE = pd.concat([conf, agr], ignore_index=True)
TAB2E = {"p_q2a": ("Qwen2-Audio", 0.23, 0.96, 168), "p_o25": ("Qwen2.5-Omni", 0.22, 0.96, 169),
         "p_q3o": ("Qwen3-Omni", 0.30, 0.97, 170), "p_af2": ("Audio Flamingo 2", 0.53, 0.63, 171),
         "p_af3": ("Audio Flamingo 3", 0.42, 0.84, 172), "p_kimi": ("Kimi-Audio", 0.53, 0.89, 173)}
for col, (nm, tc, ta, tl) in TAB2E.items():
    fr = pd.DataFrame({"clip": EE.clip_id, "speaker": EE.speaker_id.astype(str), "label": EE.label.astype(int),
                       "arm": EE.set, "is_conflict": EE.set == "conflict", "is_agreement": EE.set == "agreement",
                       "score": EE[col].astype(float)})
    run(f"R_edaic_{col[2:]}", "ARMS", "claimed", f"conflict minus agreement arm AUC, E-DAIC, {nm}", 180,
        Q_ARM, "arms", fr, ["score"], ["is_conflict", "is_agreement"], sources=[fT],
        tex_check={"t0": (tc, 2), "t1": (ta, 2)}, note=f"Table 2 row l.{tl}: {tc:.2f} / {ta:.2f}; agreement scored on the 311 distinct segments")

ARM = pitt_arm_map()
PITT = [("q2a", "Qwen2-Audio", f"{LR}/probe2/pitt_zeroshot_scores.csv", "clip", "speaker", True, 0.41, 0.70, 168),
        ("o25", "Qwen2.5-Omni", f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv", "clip", "speaker", True, 0.53, 0.71, 169),
        ("q3o", "Qwen3-Omni", f"{REL}/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv", "clip", "speaker", True, 0.61, 0.82, 170),
        ("af2", "Audio Flamingo 2", None, "orig_clip_id", "speaker_id", False, 0.58, 0.53, 171),
        ("af3", "Audio Flamingo 3", f"{REL}/overnight2/part10/af3_pitt.csv", "clip_path", "speaker_id", False, 0.61, 0.59, 172),
        ("kimi", "Kimi-Audio", f"{REL}/overnight2/part3/kimi_pitt_zeroshot_scores.csv", "clip", "speaker", False, 0.72, 0.72, 173)]
PITT_FR = {}
for key, nm, f, kc, sc_, claimed, tc, ta, tl in PITT:
    if key == "af2":
        f1, f2 = f"{SC}/part20/POD4c/af2_orig30_conflict.csv", f"{SC}/part20/POD4c/af2_orig30_agreement.csv"
        P = pd.concat([pd.read_csv(f1), pd.read_csv(f2)], ignore_index=True)
        srcs = [f1, f2]
    else:
        P = pd.read_csv(f)
        srcs = [f]
    P["_k"] = P[kc].map(bn)
    assert len(P) == 468 and P._k.is_unique
    M = P.join(ARM, on="_k", rsuffix="_m")
    assert M.set.notna().all()
    spk_arm = M["speaker_m"] if "speaker_m" in M.columns else M["speaker"]
    assert (M[sc_].astype(str).values == spk_arm.astype(str).values).all()
    assert (M.label.astype(int).values == M.label_m.astype(int).values).all()
    assert (M._k.str.startswith("conflict_") == (M.set == "conflict")).all()
    fr = pd.DataFrame({"clip": M._k, "speaker": M[sc_].astype(str), "label": M.label.astype(int), "arm": M.set,
                       "is_conflict": M.set == "conflict", "is_agreement": M.set == "agreement", "score": M.p_yes.astype(float)})
    assert fr.is_conflict.sum() == 146 and fr.is_agreement.sum() == 322
    PITT_FR[key] = (fr, srcs, nm, tc)
    if claimed:
        q = "and for the three Qwen models on Pitt"
        ql = 180
    else:
        q = "Kimi-Audio shows no drop, and the two Audio Flamingo models score 0.53 to 0.61 on both arms"
        ql = 180
    run(f"R_pitt_{key}" if claimed else f"Rs_pitt_{key}", "ARMS", "claimed" if claimed else "sensitivity only",
        f"conflict minus agreement arm AUC, Pitt, {nm}", ql, q, "arms", fr, ["score"], ["is_conflict", "is_agreement"],
        sources=srcs + [DB], tex_check={"t0": (tc, 2), "t1": (ta, 2)},
        note=f"Table 2 row l.{tl}: {tc:.2f} / {ta:.2f}; arm from pitt_conflict_manifest.csv set column" +
             ("" if claimed else "; sensitivity member (tex reports no significant drop)"))

# ================================================================ FINE TUNE
FT = [("pcgita", "PC-GITA", 0.56, 0.89, True), ("neurovoz", "NeuroVoz", 0.70, 0.90, True),
      ("pitt", "Pitt", 0.66, 0.77, True), ("adresso", "ADReSSo", 0.62, 0.82, True),
      ("adress2020", "ADReSS-2020", 0.62, 0.77, True), ("kcl", "MDVR-KCL", 0.71, 0.70, False)]
for key, nm, tz, tf, claimed in FT:
    ff = f"{REL}/omni_final/omnisft_{key}_oof.csv"
    fz = f"{REL}/omni_final/omni_{key}_zeroshot_scores.csv"
    fr = load_pair(ff, "path", "speaker", "label", "p_yes", fz, "clip", "speaker", "label", "p_yes")
    fr = fr.rename(columns={"a": "finetuned", "b": "zeroshot"})
    run(f"F_{key}" if claimed else f"Fs_{key}", "FINE TUNE", "claimed" if claimed else "sensitivity only",
        f"projector fine-tune minus zero-shot answer, {nm}", 226, Q_FT,
        "two", fr, ["finetuned", "zeroshot"], sources=[ff, fz], tex_check={"t0": (tf, 2), "t1": (tz, 2)},
        note=f"Table 1 l.120-130: {tz:.2f} -> {tf:.2f}; also abstract l.26, l.53, l.244" +
             ("" if claimed else "; sensitivity member (tex names it as the exception)"))
ff = f"{SC}/part24/FT/FT24_whole_seed0_perclip.csv"
F0 = pd.read_csv(ff)
Ew = E2[["pid", "speaker", "label", "p_yes_whole"]].merge(F0[["pid", "label", "p_yes"]], on="pid", suffixes=("", "_f"))
assert len(Ew) == 275 and (Ew.label == Ew.label_f).all()
fr = pd.DataFrame({"clip": Ew.pid.astype(str), "speaker": Ew.speaker.astype(str), "label": Ew.label.astype(int),
                   "finetuned": Ew.p_yes.astype(float), "zeroshot": Ew.p_yes_whole.map(fl)})
run("Fs_edaic_seed0", "FINE TUNE", "sensitivity only", "projector fine-tune (seed 0) minus zero-shot answer, E-DAIC whole interview", 226,
    "On E-DAIC the retrained projector stays at the zero-shot level (+0.02 [$-$0.01, 0.05]", "two", fr,
    ["finetuned", "zeroshot"], sources=[ff, fe2], tex_check={"value": (0.02, 2), "lo": (-0.01, 2), "hi": (0.05, 2)},
    note="sensitivity member (tex reports it as not significant)")
fl_ = f"{REL}/edaic_rerun/part16/POD2/lora_pitt_oof.csv"
fp = f"{REL}/omni_final/omnisft_pitt_oof.csv"
fr = load_pair(fl_, "name", "spk", "label", "p_yes", fp, "path", "speaker", "label", "p_yes").rename(columns={"a": "lora", "b": "projector"})
run("F_lora_minus_projector", "FINE TUNE", "claimed", "LoRA minus projector fine-tune, Pitt", 226,
    "reaches 0.825 on Pitt, 0.06 [0.02, 0.10] above the projector run", "two", fr, ["lora", "projector"],
    sources=[fl_, fp], tex_check={"value": (0.06, 2), "lo": (0.02, 2), "hi": (0.10, 2), "t0": (0.825, 3)})

# ================================================================ CONTROLS
P4 = f"{REL}/edaic_rerun/part16/POD4"
for key, nm, tv, nd, ta, q in [("pg", "PC-GITA", (-0.07, -0.12, -0.02), 2, 0.83, "where the encoder probe falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02])"),
                               ("nv", "NeuroVoz", (-0.03, -0.061, -0.001), 3, 0.89, "where the probe falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])")]:
    fa, fb = f"{P4}/p16_{key}_after_enc_nested_oof.csv", f"{P4}/p16_{key}_before_enc_nested_oof.csv"
    fr = load_pair(fa, "clip", "speaker", "label", "p_probe", fb, "clip", "speaker", "label", "p_probe").rename(
        columns={"a": "probe_after", "b": "probe_before"})
    run(f"C_{key}_probe_drop", "CONTROLS", "claimed", f"encoder probe after minus before loudness and noise-floor normalisation, {nm}", 149,
        q, "two", fr, ["probe_after", "probe_before"], sources=[fa, fb],
        tex_check={"value": (tv[0], 2), "lo": (tv[1], nd), "hi": (tv[2], nd), "t0": (ta, 2)},
        note="single nested split probe, as reported in the tex")
fa, fb = f"{P4}/p16_pg_after_zeroshot_scores.csv", f"{P4}/p16_pg_before_zeroshot_scores.csv"
fr = load_pair(fa, "clip", "speaker", "label", "p_yes", fb, "clip", "speaker", "label", "p_yes").rename(
    columns={"a": "answer_after", "b": "answer_before"})
run("Cs_pg_answer", "CONTROLS", "sensitivity only", "zero-shot answer after minus before normalisation, PC-GITA", 149,
    "and the answer stays at 0.57", "two", fr, ["answer_after", "answer_before"], sources=[fa, fb],
    tex_check={"t0": (0.57, 2)}, note="sensitivity member (tex says the answer stays)")

# ================================================================ EXTRA: below chance (not in the requested list)
for col, (nm, tc, ta, tl) in TAB2E.items():
    below = tc < 0.5
    fr = conf[["clip_id", "speaker_id", "label", col]].rename(columns={"clip_id": "clip", "speaker_id": "speaker", col: "score"}).copy()
    fr["speaker"] = fr.speaker.astype(str)
    run(f"X_edaic_conf_{col[2:]}", "EXTRA CHANCE", "claimed" if below else "sensitivity only",
        f"conflict arm AUC minus 0.5, E-DAIC, {nm}", 180,
        "On the depression segments four of the six models fall below chance when the words point away from the label",
        "one", fr, ["score"], sources=[fT], tex_check={"t0": (tc, 2)},
        note=f"not in the requested list; abstract l.26 'four of six models fall below chance'; Table 2 l.{tl} conflict {tc:.2f}")
for key in ["q2a", "o25", "q3o"]:
    frp, srcs, nm, tc = PITT_FR[key]
    fr = frp[frp.is_conflict][["clip", "speaker", "label", "score"]].copy()
    run(f"X_pitt_conf_{key}", "EXTRA CHANCE", "check only", f"conflict arm AUC minus 0.5, Pitt, {nm}", 180,
        "but none falls significantly below chance", "one", fr, ["score"], sources=srcs + [DB], tex_check={"t0": (tc, 2)},
        note="non-significance statement; checked, not Holm corrected")

if ONLY:
    sys.exit(0)

# ================================================================ HOLM + outputs
R = pd.DataFrame(rows)
for c in ["holm_p", "holm_p_full"]:
    R[c] = np.nan
for c in ["survives", "survives_full"]:
    R[c] = ""
R["m_family"] = 0
R["m_family_full"] = 0
for fam in R.family.unique():
    ic = R.index[(R.family == fam) & (R.role == "claimed")]
    adj = holm(R.loc[ic, "p"].values)
    R.loc[ic, "holm_p"] = adj
    R.loc[ic, "survives"] = np.where(adj <= 0.05, "yes", "no")
    R.loc[ic, "m_family"] = len(ic)
    ia = R.index[(R.family == fam) & R.role.isin(["claimed", "sensitivity only"])]
    adj = holm(R.loc[ia, "p"].values)
    R.loc[ia, "holm_p_full"] = adj
    R.loc[ia, "survives_full"] = np.where(adj <= 0.05, "yes", "no")
    R.loc[ia, "m_family_full"] = len(ia)


def ci4(v, l, h):
    return f"{v:+.4f} [{l:+.4f}, {h:+.4f}]"


R["value_ci"] = [ci4(v, l, h) for v, l, h in zip(R.value, R.lo, R.hi)]
cols = ["family", "claim", "tex_line", "value_ci", "p", "holm_p", "survives", "m_family", "role", "id",
        "value", "lo", "hi", "holm_p_full", "survives_full", "m_family_full", "usable_draws", "degenerate_draws",
        "n_le0", "n_ge0", "n_clips", "n_spk", "term_points", "excludes_zero", "draws", "per_clip", "draws_source"]
out = R[cols].copy()
out["term_points"] = out.term_points.map(lambda t: " ".join(f"{x:.4f}" for x in t))
for c in ["value", "lo", "hi"]:
    out[c] = out[c].map(lambda x: f"{x:+.4f}")
for c in ["p", "holm_p", "holm_p_full"]:
    out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
out.to_csv(f"{B}/B_holm.tsv", sep="\t", index=False)
idx = R[["id", "family", "role", "draws", "draws_sha256", "usable_draws", "degenerate_draws", "draws_source", "per_clip"]].copy()
idx["draws_mean"] = [float(np.nanmean(np.load(p)["diff"])) for p in R.draws]
idx["npz_keys"] = "diff terms speaker_idx speakers point_diff point_terms"
idx.to_csv(f"{B}/B_draws_index.tsv", sep="\t", index=False)

MAIN = ["GAPS", "ARMS", "FINE TUNE", "CONTROLS"]
cl = R[(R.role == "claimed") & R.family.isin(MAIN)]
surv = cl[cl.survives == "yes"]
fail = cl[cl.survives == "no"]
line = (f"{len(surv)} of {len(cl)} differences survive Holm correction within their family" +
        ("." if len(fail) == 0 else "; " + " and ".join(fail.claim) + " do not."))
fs = cl[cl.survives_full == "yes"]
ff_ = cl[cl.survives_full == "no"]
nsens = int(((R.role == "sensitivity only") & R.family.isin(MAIN)).sum())
line_full = (f"Sensitivity (families widened with the {nsens} same-kind differences the tex reports as not significant): "
             f"{len(fs)} of {len(cl)} claimed differences survive" + ("." if len(ff_) == 0 else "; " + " and ".join(ff_.claim) + " do not."))
ex = R[(R.family == "EXTRA CHANCE") & (R.role == "claimed")]
line_extra = (f"Extra, not in the paper line: {int((ex.survives == 'yes').sum())} of {len(ex)} E-DAIC conflict arms stay below chance after Holm (m={len(ex)}).")
tex_mism = [dict(id=r.id, check={k: v for k, v in r.tex_check.items() if not v["match"]}) for r in R.itertuples()
            if r.tex_check and any(not v["match"] for v in r.tex_check.values())]
side = dict(task="PART 25 B r2: Holm correction over the differences the final tex calls significant",
            date_utc=datetime.datetime.utcnow().isoformat() + "Z", tex=TEX, tex_sha256=sha(TEX), n_draws=NB,
            rng="numpy default_rng(0), fresh per cell",
            resample="unique speakers (np.unique order of string ids), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker",
            paired="all terms on the same draw", mean_of_five="mean of the five per-repeat AUCs in each draw",
            p_rule="min(1, 2*min((1+#d<=0)/(B+1), (1+#d>=0)/(B+1))), B = usable draws; a draw with |d| < 1e-12 is a floating point tie and counts as exactly 0 (sklearn and rank AUC differ by up to 1e-15 on exact ties; affects only sensitivity or check rows)",
            holm="step-down, adjusted p = running max of min(1,(m-rank+1)*p), survives when adjusted p <= 0.05, within family",
            families={f: dict(m=int(((R.family == f) & (R.role == "claimed")).sum()),
                              m_full=int(((R.family == f) & R.role.isin(["claimed", "sensitivity only"])).sum())) for f in R.family.unique()},
            paper_line=line, sensitivity_line=line_full, extra_line=line_extra, tex_value_mismatches=tex_mism,
            versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                          pandas=pd.__version__, scipy=scipy.__version__, platform=platform.platform()),
            command="/usr/local/bin/python3 scripts/B_holm_r2.py",
            outputs=dict(tsv=f"{B}/B_holm.tsv", index=f"{B}/B_draws_index.tsv", tsv_sha256=sha(f"{B}/B_holm.tsv"),
                         index_sha256=sha(f"{B}/B_draws_index.tsv")))
json.dump(side, open(f"{B}/B_holm.sidecar.json", "w"), indent=1)
print()
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print(out[["family", "role", "id", "tex_line", "value_ci", "p", "holm_p", "survives", "holm_p_full", "survives_full"]].to_string(index=False))
print()
print(line)
print(line_full)
print(line_extra)
print("tex mismatches:", json.dumps(tex_mism))
