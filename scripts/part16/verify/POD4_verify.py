#!/usr/bin/env python3
"""INDEPENDENT verifier for PART 16 POD4.

Written from scratch on the Mac. Nothing imported from the pod job's scripts.

AUC implementation: trapezoidal integration of the full empirical ROC curve,
built by walking the distinct score thresholds from high to low and grouping
tied scores into a single ROC vertex. This is NOT the rank-sum formula and NOT
scipy; ties produce a diagonal ROC segment whose trapezoid contribution is the
0.5-per-tied-pair credit, so it must equal the tie-corrected Mann-Whitney value
if both are right.

A brute-force O(n^2) pairwise counter is also included and asserted against the
trapezoid value on every cell, as a third opinion.

Bootstrap protocol as specified by the task: 2000 draws,
numpy.random.default_rng(0), SPEAKERS resampled with replacement, percentile
2.5/97.5. Paired contrasts draw one speaker list per replicate.
"""
import csv, json, os, sys
import numpy as np

BASE = "<local data dir>/Desktop/release/edaic_rerun/part16/POD4"
OMNI = "<local data dir>/Desktop/release/omni_final"
NB = 2000


# ----------------------------------------------------------------- AUC ------
def auc_roc_trapezoid(y, s):
    """Area under the empirical ROC by trapezoid over distinct thresholds."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")   # descending score
    ys = y[order]
    ss = s[order]
    tp = 0
    fp = 0
    prev_tpr = 0.0
    prev_fpr = 0.0
    area = 0.0
    i = 0
    n = len(ss)
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        block = ys[i:j + 1]
        tp += int((block == 1).sum())
        fp += int((block == 0).sum())
        tpr = tp / P
        fpr = fp / N
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr
        i = j + 1
    return float(area)


def auc_pairwise_bruteforce(y, s):
    """Literal definition: concordant pairs, ties 0.5. O(n^2), small sets only."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum()
    ties = (diff == 0).sum()
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def dirfree(a):
    """Direction-free AUC: max(a, 1-a)."""
    return a if a >= 0.5 else 1.0 - a


# ----------------------------------------------------------- bootstrap ------
def _speaker_index(spk):
    u = np.unique(spk)
    return u, {sp: np.where(spk == sp)[0] for sp in u}


def boot_ci(spk, y, s, nb=NB, seed=0):
    rng = np.random.default_rng(seed)
    u, idx_by = _speaker_index(spk)
    vals = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx_by[p] for p in pick])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        vals.append(auc_roc_trapezoid(yy, s[ii]))
    v = np.asarray(vals, dtype=float)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))


def boot_paired(spk, y, s_from, s_to, nb=NB, seed=0):
    """CI on AUC(s_to) - AUC(s_from), one speaker draw per replicate."""
    rng = np.random.default_rng(seed)
    u, idx_by = _speaker_index(spk)
    d = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx_by[p] for p in pick])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        a_from = auc_roc_trapezoid(yy, s_from[ii])
        a_to = auc_roc_trapezoid(yy, s_to[ii])
        d.append(a_to - a_from)
    d = np.asarray(d, dtype=float)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), int(len(d))


# ---------------------------------------------------------------- io --------
def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def col(rs, name, cast=float):
    return np.array([cast(r[name]) for r in rs])


RESULTS = []


def record(vid, what, theirs, mine, lo=None, hi=None, their_lo=None, their_hi=None, note=""):
    def q(x):
        return None if x is None else round(float(x), 4)
    agree = (theirs is not None) and abs(round(float(theirs), 4) - q(mine)) < 1e-9
    ci_ok = True
    if their_lo is not None:
        ci_ok = (abs(round(float(their_lo), 4) - q(lo)) < 1e-9
                 and abs(round(float(their_hi), 4) - q(hi)) < 1e-9)
    RESULTS.append(dict(id=vid, what=what, theirs=theirs, mine=q(mine),
                        mine_lo=q(lo), mine_hi=q(hi),
                        their_lo=their_lo, their_hi=their_hi,
                        agree=bool(agree), ci_agree=bool(ci_ok), note=note))
    flag = "OK " if agree and ci_ok else "XX "
    print(f"{flag}{vid:28s} theirs={theirs}  mine={q(mine)}  "
          f"CI mine=[{q(lo)},{q(hi)}] theirs=[{their_lo},{their_hi}] {note}")
    return q(mine)


# ============================== self-test of the AUC implementation =========
def selftest():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(6, 60))
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            continue
        s = np.round(rng.normal(size=n), 1)      # forces many ties
        a = auc_roc_trapezoid(y, s)
        b = auc_pairwise_bruteforce(y, s)
        assert abs(a - b) < 1e-12, (a, b)
    # degenerate: all identical scores must give exactly 0.5
    assert abs(auc_roc_trapezoid(np.array([0, 1, 0, 1]), np.array([3.0] * 4)) - 0.5) < 1e-12
    # perfect separation
    assert abs(auc_roc_trapezoid(np.array([0, 0, 1, 1]), np.array([1.0, 2, 3, 4])) - 1.0) < 1e-12
    print("AUC self-test passed: trapezoid ROC == brute-force pairwise on 200 tied-score cases\n")


# ===================================================================== 4b ===
def part_4b():
    print("=" * 78)
    print("4b  PITT, INTERVIEWER REMOVED")
    print("=" * 78)
    pc = rows(os.path.join(BASE, "pitt_no_interviewer.csv"))
    print(f"pitt_no_interviewer.csv rows={len(pc)} speakers={len(set(r['speaker'] for r in pc))}")

    # -- cross-check the per-clip csv against the raw arm score files ---------
    arm_files = {
        "orig":     ("p16_pitt_orig_zeroshot_scores.csv", "p16_pitt_orig_enc_nested_oof.csv"),
        "paronly":  ("p16_pitt_paronly_zeroshot_scores.csv", "p16_pitt_paronly_enc_nested_oof.csv"),
        "durmatch": ("p16_pitt_durmatch_zeroshot_scores.csv", "p16_pitt_durmatch_enc_nested_oof.csv"),
    }
    raw = {}
    for arm, (zf, ef) in arm_files.items():
        z = {r["clip"]: r for r in rows(os.path.join(BASE, zf))}
        e = {r["clip"]: r for r in rows(os.path.join(BASE, ef))}
        raw[arm] = (z, e)
        maxz = max(abs(float(z[r["clip"]]["p_yes"]) - float(r["zs_" + arm])) for r in pc)
        maxe = max(abs(float(e[r["clip"]]["p_probe"]) - float(r["enc_" + arm])) for r in pc)
        print(f"  arm {arm:9s} n_zs={len(z)} n_enc={len(e)} "
              f"max|csv-source| zs={maxz:.2e} enc={maxe:.2e}")

    spk = col(pc, "speaker", str)
    y = col(pc, "label", int)
    setcol = col(pc, "set", str)

    # full-precision scores from the source files, keyed in per-clip csv order
    S = {}
    for arm in arm_files:
        z, e = raw[arm]
        S["zs_" + arm] = np.array([float(z[r["clip"]]["p_yes"]) for r in pc])
        S["enc_" + arm] = np.array([float(e[r["clip"]]["p_probe"]) for r in pc])

    # published reference, read from omni_final (READ ONLY)
    pubz = {r["clip"]: float(r["p_yes"]) for r in rows(os.path.join(OMNI, "omni_pitt_zeroshot_scores.csv"))}
    pube = {r["clip"]: float(r["p_probe"]) for r in rows(os.path.join(OMNI, "omni_pitt_enc_nested_oof.csv"))}
    print(f"  published omni_final n_zs={len(pubz)} n_enc={len(pube)}  "
          f"clip overlap with 468: zs={len(set(pubz)&set(r['clip'] for r in pc))} "
          f"enc={len(set(pube)&set(r['clip'] for r in pc))}")
    S["zs_pub"] = np.array([pubz[r["clip"]] for r in pc])
    S["enc_pub"] = np.array([pube[r["clip"]] for r in pc])

    cells = [
        ("POD4_4b_pub_zs",  "Pitt published zero-shot",        "zs_pub",      0.6578, 0.6033, 0.7125),
        ("POD4_4b_pub_enc", "Pitt published encoder probe",    "enc_pub",     0.7969, 0.7412, 0.8512),
        ("POD4_4b_A_zs",    "arm A original cut zero-shot",    "zs_orig",     0.6497, 0.5959, 0.7056),
        ("POD4_4b_A_enc",   "arm A original cut encoder",      "enc_orig",    0.7949, 0.7431, 0.8446),
        ("POD4_4b_B_zs",    "arm B interviewer removed zs",    "zs_paronly",  0.7230, 0.6664, 0.7753),
        ("POD4_4b_B_enc",   "arm B interviewer removed enc",   "enc_paronly", 0.7497, 0.6940, 0.8035),
        ("POD4_4b_C_zs",    "arm C duration-matched zs",       "zs_durmatch", 0.6145, 0.5582, 0.6683),
        ("POD4_4b_C_enc",   "arm C duration-matched enc",      "enc_durmatch",0.7980, 0.7448, 0.8489),
    ]
    for vid, what, k, th, tl, thh in cells:
        a = auc_roc_trapezoid(y, S[k])
        lo, hi, nu = boot_ci(spk, y, S[k])
        record(vid, what, th, a, lo, hi, tl, thh, note=f"n=468 draws={nu}")

    # -- conflict / agreement split -----------------------------------------
    split = [
        ("POD4_4b_A_zs_conf",  "arm A zs conflict",  "zs_orig",     "conflict",  0.5298, 0.4189, 0.6386),
        ("POD4_4b_A_enc_conf", "arm A enc conflict", "enc_orig",    "conflict",  0.6161, 0.5060, 0.7281),
        ("POD4_4b_A_zs_agr",   "arm A zs agreement", "zs_orig",     "agreement", 0.6952, 0.6327, 0.7560),
        ("POD4_4b_A_enc_agr",  "arm A enc agreement","enc_orig",    "agreement", 0.8674, 0.8201, 0.9096),
        ("POD4_4b_B_zs_conf",  "arm B zs conflict",  "zs_paronly",  "conflict",  0.5408, 0.4245, 0.6622),
        ("POD4_4b_B_enc_conf", "arm B enc conflict", "enc_paronly", "conflict",  0.6295, 0.5111, 0.7345),
        ("POD4_4b_B_zs_agr",   "arm B zs agreement", "zs_paronly",  "agreement", 0.8007, 0.7500, 0.8501),
        ("POD4_4b_B_enc_agr",  "arm B enc agreement","enc_paronly", "agreement", 0.8037, 0.7452, 0.8539),
    ]
    for vid, what, k, which, th, tl, thh in split:
        m = setcol == which
        a = auc_roc_trapezoid(y[m], S[k][m])
        lo, hi, nu = boot_ci(spk[m], y[m], S[k][m])
        record(vid, what, th, a, lo, hi, tl, thh,
               note=f"n={int(m.sum())} spk={len(set(spk[m]))} draws={nu}")

    # -- paired deltas -------------------------------------------------------
    paired = [
        ("POD4_4b_pd_zs_AB",  "zs A->B",  "zs_orig",     "zs_paronly",   0.0732,  0.0312,  0.1165),
        ("POD4_4b_pd_zs_CB",  "zs C->B",  "zs_durmatch", "zs_paronly",   0.1084,  0.0638,  0.1550),
        ("POD4_4b_pd_enc_AB", "enc A->B", "enc_orig",    "enc_paronly",  -0.0452, -0.0857, -0.0083),
        ("POD4_4b_pd_enc_CB", "enc C->B", "enc_durmatch","enc_paronly",  -0.0483, -0.0907, -0.0062),
        ("POD4_4b_pd_zs_AC",  "zs A->C",  "zs_orig",     "zs_durmatch",  -0.0352, -0.0729, -0.0001),
        ("POD4_4b_pd_enc_AC", "enc A->C", "enc_orig",    "enc_durmatch",  0.0032, -0.0367,  0.0394),
    ]
    for vid, what, kf, kt, th, tl, thh in paired:
        d = auc_roc_trapezoid(y, S[kt]) - auc_roc_trapezoid(y, S[kf])
        lo, hi, nu = boot_paired(spk, y, S[kf], S[kt])
        record(vid, what, th, d, lo, hi, tl, thh, note=f"draws={nu}")

    # -- interviewer descriptive claims -------------------------------------
    print("\n-- interviewer cue --")
    inv = col(pc, "inv_ms_win")
    par = col(pc, "par_dur_s_recut")
    nwin = int((inv > 0).sum())
    print(f"  windows with timed INV > 0 : {nwin} / {len(pc)}  "
          f"({100*nwin/len(pc):.1f}%)   [claim 158 / 33.8%]")
    print(f"  mean INV ms dementia={inv[y==1].mean():.1f} control={inv[y==0].mean():.1f}  "
          f"[claim 852.5 / 186.4]")
    print(f"  windows with INV: dementia {int((inv[y==1]>0).sum())}/{int((y==1).sum())}  "
          f"control {int((inv[y==0]>0).sum())}/{int((y==0).sum())}  [claim 127/290, 31/178]")
    a_inv = auc_roc_trapezoid(y, inv)
    record("POD4_4b_invcue", "INV ms alone predicts dementia", 0.6427, a_inv,
           note="raw AUC, direction-free=%.4f" % dirfree(a_inv))
    a_dur = dirfree(auc_roc_trapezoid(y, par))
    record("POD4_4b_durcue", "participant duration alone, direction-free", 0.6648, a_dur,
           note="raw=%.4f" % auc_roc_trapezoid(y, par))
    print(f"  mean participant duration control={par[y==0].mean():.2f}s "
          f"dementia={par[y==1].mean():.2f}s  [claim 19.85 / 16.24]")
    print(f"  mean recut duration all={par.mean():.2f}s median={np.median(par):.2f}s "
          f"under10s={int((par<10).sum())} under5s={int((par<5).sum())} "
          f"min={par.min():.2f} max={par.max():.2f}  [claim 17.61 / 18.55 / 68 / 19]")

    # -- gap arithmetic ------------------------------------------------------
    gA = auc_roc_trapezoid(y, S["enc_orig"]) - auc_roc_trapezoid(y, S["zs_orig"])
    gB = auc_roc_trapezoid(y, S["enc_paronly"]) - auc_roc_trapezoid(y, S["zs_paronly"])
    record("POD4_4b_gap_orig", "gap on the original cut", 0.1452, gA)
    record("POD4_4b_gap_free", "gap interviewer-free", 0.0267, gB)

    # -- cha_overlap_468 consistency ----------------------------------------
    ov = rows(os.path.join(BASE, "cha_overlap_468.csv"))
    ovd = {r["clip"]: r for r in ov}
    mism = sum(1 for r in pc if abs(float(ovd[r["clip"]]["inv_ms_win"]) - float(r["inv_ms_win"])) > 1e-9)
    invsh = np.array([float(r["inv_share_win"]) for r in ov])
    parsh = np.array([float(r["par_share_win"]) for r in ov])
    print(f"\n  cha_overlap_468.csv rows={len(ov)} unresolved(win_span<=0)="
          f"{sum(1 for r in ov if float(r['win_span_ms'])<=0)}  "
          f"inv_ms mismatches vs per-clip csv={mism}")
    print(f"  mean INV share={invsh.mean():.4f} [claim 0.0200]  "
          f"mean PAR share={parsh.mean():.4f} [claim 0.5871]  "
          f"mean gap share={1-invsh.mean()-parsh.mean():.4f} [claim 0.3929]")
    print(f"  max INV share={invsh.max():.4f} [claim 0.2818]  "
          f"untimed INV utts total={sum(int(r['untimed_inv']) for r in ov)} [claim 788]")


# ===================================================================== 4a ===
FEATS = ["noise_floor", "snr", "centroid", "rolloff", "tilt"]


def part_4a(tag, csvname, zs_before, zs_after, enc_before, enc_after,
            claims, pub_zs_file):
    print("\n" + "=" * 78)
    print(f"4a  CHANNEL NORMALISATION, {tag}")
    print("=" * 78)
    rs = rows(os.path.join(BASE, csvname))
    spk = col(rs, "spk", str)
    y = col(rs, "label", int)
    print(f"{csvname} rows={len(rs)} speakers={len(set(spk))} "
          f"pos={int((y==1).sum())} neg={int((y==0).sum())}")

    # single-feature direction-free AUCs
    for f in FEATS:
        for when in ("before", "after"):
            key = f"{tag}_{f}_{when}"
            v = col(rs, f"{f}_{when}")
            raw = auc_roc_trapezoid(y, v)
            a = dirfree(raw)
            lo, hi, nu = boot_ci(spk, y, v)
            if raw < 0.5:
                lo, hi = 1 - hi, 1 - lo
            th, tl, thh = claims.get(key, (None, None, None))
            record(key, f"{f} alone {when}, direction-free", th, a, lo, hi, tl, thh,
                   note=f"raw={raw:.4f} draws={nu}")

    # five-feature OOF, as stored
    for when in ("before", "after"):
        v = col(rs, f"fivefeat_oof_{when}")
        a = auc_roc_trapezoid(y, v)
        lo, hi, nu = boot_ci(spk, y, v)
        th, tl, thh = claims[f"{tag}_5f_{when}"]
        record(f"{tag}_5f_{when}", f"five features together {when}", th, a, lo, hi, tl, thh,
               note=f"draws={nu}")

    # five-feature OOF, REFIT INDEPENDENTLY from the raw feature columns
    try:
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        seen = {}
        for s_ in spk:
            if s_ not in seen:
                seen[s_] = len(seen)
        gint = np.array([seen[s_] for s_ in spk])
        for when in ("before", "after"):
            X = np.column_stack([col(rs, f"{f}_{when}") for f in FEATS])
            stored = col(rs, f"fivefeat_oof_{when}")
            th, _, _ = claims[f"{tag}_5f_{when}"]
            for gname, gvec in (("intcode", gint), ("strspk", spk)):
                oof = np.zeros(len(y))
                for tr, te in GroupKFold(n_splits=5).split(X, y, groups=gvec):
                    pipe = make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=5000,
                                                            class_weight="balanced"))
                    pipe.fit(X[tr], y[tr])
                    oof[te] = pipe.predict_proba(X[te])[:, 1]
                a = auc_roc_trapezoid(y, oof)
                record(f"{tag}_5f_{when}_REFIT_{gname}",
                       f"five features {when}, MY OWN refit (groups={gname})", th, a,
                       note=f"max|mine-stored|={np.abs(oof-stored).max():.3e} "
                            f"corr={np.corrcoef(oof,stored)[0,1]:.6f}")
    except Exception as exc:                                    # pragma: no cover
        print("  refit skipped:", exc)

    # encoder probe + zero-shot, from the source score files
    zsb = rows(os.path.join(BASE, zs_before))
    zsa = rows(os.path.join(BASE, zs_after))
    eb = rows(os.path.join(BASE, enc_before))
    ea = rows(os.path.join(BASE, enc_after))
    print(f"  source rows: zs_before={len(zsb)} zs_after={len(zsa)} "
          f"enc_before={len(eb)} enc_after={len(ea)}")
    stem = lambda c: os.path.splitext(c)[0]
    dzb = {stem(r["clip"]): r for r in zsb}
    dza = {stem(r["clip"]): r for r in zsa}
    deb = {stem(r["clip"]): r for r in eb}
    dea = {stem(r["clip"]): r for r in ea}
    clips = [stem(r["clip"]) for r in rs]
    missing = [c for c in clips if c not in dzb or c not in dza or c not in deb or c not in dea]
    print(f"  clips in channel csv missing from a score file: {len(missing)}")

    zb = np.array([float(dzb[c]["p_yes"]) for c in clips])
    za = np.array([float(dza[c]["p_yes"]) for c in clips])
    pb = np.array([float(deb[c]["p_probe"]) for c in clips])
    pa = np.array([float(dea[c]["p_probe"]) for c in clips])
    # the channel csv also carries the encoder oof; cross-check
    print(f"  max|enc_probe_oof_before(csv) - p_probe(file)| = "
          f"{np.abs(col(rs,'enc_probe_oof_before')-pb).max():.3e}")
    print(f"  max|enc_probe_oof_after(csv)  - p_probe(file)| = "
          f"{np.abs(col(rs,'enc_probe_oof_after')-pa).max():.3e}")

    for key, vec in ((f"{tag}_zs_before", zb), (f"{tag}_zs_after", za),
                     (f"{tag}_enc_before", pb), (f"{tag}_enc_after", pa)):
        a = auc_roc_trapezoid(y, vec)
        lo, hi, nu = boot_ci(spk, y, vec)
        th, tl, thh = claims[key]
        record(key, key, th, a, lo, hi, tl, thh, note=f"draws={nu}")

    for key, vf, vt in ((f"{tag}_pd_zs", zb, za), (f"{tag}_pd_enc", pb, pa)):
        d = auc_roc_trapezoid(y, vt) - auc_roc_trapezoid(y, vf)
        lo, hi, nu = boot_paired(spk, y, vf, vt)
        th, tl, thh = claims[key]
        record(key, key, th, d, lo, hi, tl, thh, note=f"draws={nu}")

    # shaping residuals
    nf = col(rs, "nf_err_db")
    tl_ = col(rs, "tilt_err_db_per_khz")
    print(f"\n  residual noise floor mean={nf.mean():.4f} dB sd={nf.std(ddof=1):.4f}  "
          f"within 1 dB={100*np.mean(np.abs(nf)<=1.0):.1f}%")
    print(f"  residual tilt mean={tl_.mean():.5f} dB/kHz sd={tl_.std(ddof=1):.5f}  "
          f"within 0.1={100*np.mean(np.abs(tl_)<=0.1):.1f}%")

    # published reference from omni_final (READ ONLY)
    if os.path.exists(pub_zs_file):
        pr = rows(pub_zs_file)
        pd_ = {stem(r["clip"]): r for r in pr}
        ok = [c for c in clips if c in pd_]
        print(f"  omni_final published zero-shot file rows={len(pr)} "
              f"clip overlap with this manifest={len(ok)}/{len(clips)}")
        if len(ok) == len(clips):
            pv = np.array([float(pd_[c]["p_yes"]) for c in clips])
            print(f"  published zero-shot AUC on these same clips = "
                  f"{auc_roc_trapezoid(y, pv):.4f}  "
                  f"(POD4 re-run BEFORE = {auc_roc_trapezoid(y, zb):.4f})")


NV_CLAIMS = {
    "nv_noise_floor_before": (0.5410, 0.4731, 0.6049),
    "nv_noise_floor_after":  (0.5407, 0.4594, 0.6191),
    "nv_snr_before":         (0.5432, 0.4640, 0.6216),
    "nv_snr_after":          (0.5305, 0.4498, 0.6080),
    "nv_centroid_before":    (0.7782, 0.7109, 0.8396),
    "nv_centroid_after":     (0.5603, 0.5051, 0.6188),
    "nv_rolloff_before":     (0.8005, 0.7370, 0.8574),
    "nv_rolloff_after":      (0.5938, 0.5438, 0.6448),
    "nv_tilt_before":        (0.7020, 0.6451, 0.7552),
    "nv_tilt_after":         (0.5808, 0.5369, 0.6259),
    "nv_5f_before":          (0.7977, 0.7248, 0.8611),
    "nv_5f_after":           (0.6025, 0.5479, 0.6580),
    "nv_zs_before":          (0.7108, 0.6523, 0.7671),
    "nv_zs_after":           (0.6991, 0.6348, 0.7598),
    "nv_enc_before":         (0.9233, 0.8800, 0.9573),
    "nv_enc_after":          (0.8924, 0.8481, 0.9323),
    "nv_pd_zs":              (-0.0117, -0.0332, 0.0098),
    "nv_pd_enc":             (-0.0308, -0.0608, -0.0012),
}

PG_CLAIMS = {
    "pg_noise_floor_before": (0.6627, 0.5885, 0.7362),
    "pg_noise_floor_after":  (0.5734, 0.5113, 0.6341),
    "pg_snr_before":         (0.5873, 0.5261, 0.6464),
    "pg_snr_after":          (0.5735, 0.5158, 0.6286),
    "pg_centroid_before":    (0.5539, 0.4997, 0.6104),
    "pg_centroid_after":     (0.5591, 0.5029, 0.6136),
    "pg_rolloff_before":     (0.5533, 0.5000, 0.6081),
    "pg_rolloff_after":      (0.5657, 0.5102, 0.6189),
    "pg_tilt_before":        (0.5610, 0.5048, 0.6194),
    "pg_tilt_after":         (0.5526, 0.4987, 0.6030),
    "pg_5f_before":          (0.6657, 0.5851, 0.7452),
    "pg_5f_after":           (0.5590, 0.4922, 0.6213),
    "pg_zs_before":          (0.5728, 0.5002, 0.6440),
    "pg_zs_after":           (0.5674, 0.5054, 0.6304),
    "pg_enc_before":         (0.8959, 0.8339, 0.9448),
    "pg_enc_after":          (0.8298, 0.7546, 0.8968),
    "pg_pd_zs":              (-0.0054, -0.0437, 0.0357),
    "pg_pd_enc":             (-0.0661, -0.1181, -0.0180),
}


if __name__ == "__main__":
    selftest()
    part_4b()
    part_4a("nv", "channel_norm_neurovoz.csv",
            "p16_nv_before_zeroshot_scores.csv", "p16_nv_after_zeroshot_scores.csv",
            "p16_nv_before_enc_nested_oof.csv", "p16_nv_after_enc_nested_oof.csv",
            NV_CLAIMS, os.path.join(OMNI, "omni_neurovoz_zeroshot_scores.csv"))
    part_4a("pg", "channel_norm_pcgita.csv",
            "p16_pg_before_zeroshot_scores.csv", "p16_pg_after_zeroshot_scores.csv",
            "p16_pg_before_enc_nested_oof.csv", "p16_pg_after_enc_nested_oof.csv",
            PG_CLAIMS, os.path.join(OMNI, "omni_pcgita_zeroshot_scores.csv"))

    out = "reports/part16/verify/POD4_verify_results.json"
    with open(out, "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    checked = [r for r in RESULTS if r["theirs"] is not None]
    bad = [r for r in checked if not r["agree"]]
    badci = [r for r in checked if r["theirs"] is not None and r["their_lo"] is not None
             and not r["ci_agree"]]
    print("\n" + "=" * 78)
    print(f"point estimates checked: {len(checked)}   agree to 4dp: "
          f"{len(checked)-len(bad)}   DISAGREE: {len(bad)}")
    for r in bad:
        print(f"   XX {r['id']}: theirs={r['theirs']} mine={r['mine']}")
    print(f"CIs checked: {sum(1 for r in checked if r['their_lo'] is not None)}   "
          f"DISAGREE: {len(badci)}")
    for r in badci:
        print(f"   XX {r['id']}: theirs=[{r['their_lo']},{r['their_hi']}] "
              f"mine=[{r['mine_lo']},{r['mine_hi']}]")
    print("written:", out)
