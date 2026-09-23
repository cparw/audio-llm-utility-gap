#!/usr/bin/env python3
"""POD1 independent verifier, part 20: BALANCED projector fine-tune on Pitt 468.

Written from the task text only. Nothing from the compute side is imported or executed.

AUC     explicit Mann-Whitney over every positive x negative pair, ties count 0.5,
        cross-checked by a trapezoid ROC built by hand (one point per distinct score).
CI      2000 draws, a fresh numpy default_rng(0) for every cell, speakers resampled with
        replacement (rng.choice over the sorted unique speakers of that cell), a draw is
        kept only when both classes are present, percentiles 2.5 and 97.5.
PAIRED  balanced minus standard, joined on clip basename, one speaker draw per replicate,
        both AUCs recomputed inside the draw.
ARMS    conflict / agreement from the conflict manifest `set` column (by segment basename),
        checked against the file name prefix and the standard file's set column.
WEIGHTS recomputed per fold from the manifest and the fold file:
        w(label y, arm a) = (N_y / 2) / N_{y,a} over the training fold, then divided by the
        training-fold mean weight.

usage: POD1_verify.py BAL_OOF STD_OOF CONFLICT_MANIFEST FOLDS [FOLD_JSON_DIR|-] [SCORE_COL] [COMPUTE_MANIFEST]
"""
import csv, os, sys, json, glob, hashlib
import numpy as np

NB = 2000
PROMPT = ("Based only on this recording, does this speaker show signs of dementia? "
          "Answer with one word, Yes or No.")
ARMS = ("conflict", "agreement")


def auc_mw(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def auc_trap(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    P = float((y == 1).sum()); N = float((y == 0).sum())
    xs = [0.0]; ys = [0.0]
    for t in np.unique(s)[::-1]:
        m = s >= t
        ys.append((m & (y == 1)).sum() / P); xs.append((m & (y == 0)).sum() / N)
    xs = np.array(xs); ys = np.array(ys)
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0))


def spk_index(spk):
    u = np.unique(spk)
    return u, {p: np.flatnonzero(spk == p) for p in u}


def boot(spk, y, scores, nb=NB, seed=0):
    """scores: one array (cell) or two arrays (paired, first minus second)."""
    rng = np.random.default_rng(seed)
    u, idx = spk_index(spk)
    out = []; degen = 0
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(np.unique(y[ii])) < 2:
            degen += 1; continue
        if len(scores) == 1:
            out.append(auc_mw(y[ii], scores[0][ii]))
        else:
            out.append(auc_mw(y[ii], scores[0][ii]) - auc_mw(y[ii], scores[1][ii]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), int(len(out)), degen


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    bal_p, std_p, man_p, fold_p = sys.argv[1:5]
    fj_dir = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    col = sys.argv[6] if len(sys.argv) > 6 else "p_yes"
    pod_man = sys.argv[7] if len(sys.argv) > 7 else None
    pm = None
    if pod_man:
        pmr = list(csv.DictReader(open(pod_man)))
        pm = [os.path.basename(r["path"]) for r in pmr]
    rep = {"files": {"balanced": bal_p, "standard": std_p, "manifest": man_p, "folds": fold_p,
                     "folds_md5": md5(fold_p), "balanced_md5": md5(bal_p)}, "checks": {}, "cells": {}}
    C = rep["checks"]

    man = list(csv.DictReader(open(man_p)))
    mkey = "segment_path" if "segment_path" in man[0] else "path"
    M = {os.path.basename(r[mkey]): r for r in man}
    assert len(M) == len(man) == 468, ("manifest", len(M), len(man))
    arm = {k: M[k]["set"] for k in M}
    mlab = {k: int(M[k]["label"]) for k in M}
    for k in M:
        assert k.startswith(arm[k] + "_"), ("arm vs file prefix", k, arm[k])
    FO = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(fold_p))}
    assert set(FO) == set(M), "fold file clips != manifest clips"

    def load(p, c):
        rows = list(csv.DictReader(open(p)))
        h = rows[0].keys()
        kc = next(x for x in ("path", "clip_path", "clip_id", "clip", "wav") if x in h)
        sc = "speaker" if "speaker" in h else "speaker_id"
        d = {}
        for r in rows:
            k = os.path.basename(r[kc]); assert k not in d, ("dup", k)
            d[k] = r
        return d, rows, sc, list(h)
    B, brows, bsc, bh = load(bal_p, col)
    S, srows, ssc, sh = load(std_p, "p_yes")
    rep["balanced_columns"] = bh
    C["n_balanced_rows"] = len(brows)
    C["balanced_clips_eq_manifest"] = set(B) == set(M)
    C["standard_clips_eq_manifest"] = set(S) == set(M)
    C["label_bal_eq_manifest"] = all(int(float(B[k]["label"])) == mlab[k] for k in B)
    C["label_std_eq_manifest"] = all(int(float(S[k]["label"])) == mlab[k] for k in S)
    C["speaker_bal_eq_folds"] = all(B[k][bsc] == FO[k][1] for k in B)
    C["speaker_std_eq_folds"] = all(S[k][ssc] == FO[k][1] for k in S)
    if "set" in bh:
        C["set_bal_eq_manifest"] = all(B[k]["set"] == arm[k] for k in B)
    C["set_std_eq_manifest"] = all(S[k]["set"] == arm[k] for k in S)
    if "prompt" in bh:
        C["prompt_bal_verbatim_all_rows"] = all(B[k]["prompt"] == PROMPT for k in B)
    C["prompt_std_verbatim_all_rows"] = all(S[k]["prompt"] == PROMPT for k in S)
    if "fold" in bh:
        C["fold_col_eq_fold_file"] = all(int(float(B[k]["fold"])) == FO[k][0] for k in B)
    if "std_p_yes" in bh:   # the reference column carried inside the balanced file must equal the reference file
        C["std_p_yes_col_eq_reference_file_p_yes"] = all(float(B[k]["std_p_yes"]) == float(S[k]["p_yes"]) for k in B)

    # ---- weight table recomputed from manifest + folds
    keys = sorted(M)
    y_all = np.array([mlab[k] for k in keys]); a_all = np.array([arm[k] for k in keys])
    f_all = np.array([FO[k][0] for k in keys]); s_all = np.array([FO[k][1] for k in keys])
    wt = {}
    for f in range(5):
        tr = f_all != f
        assert not (set(s_all[tr]) & set(s_all[~tr])), "speaker leak in fold file"
        w = np.zeros(len(keys))
        for lab in (0, 1):
            ny = int((tr & (y_all == lab)).sum())
            for a in ARMS:
                m = tr & (y_all == lab) & (a_all == a)
                w[m] = (ny / 2.0) / m.sum()
        mean_before = float(w[tr].mean())
        w[tr] = w[tr] / w[tr].mean()
        tab = []
        for lab in (0, 1):
            for a in ARMS:
                m = tr & (y_all == lab) & (a_all == a)
                tab.append({"label": lab, "arm": a, "n": int(m.sum()), "weight": float(w[m][0]),
                            "total_weight": float(w[m].sum())})
        wt[f] = {"n_train": int(tr.sum()), "n_test": int((~tr).sum()), "mean_before_rescale": mean_before,
                 "table": tab}
    rep["weight_tables"] = wt

    # ---- compare with compute fold jsons if given
    if fj_dir:
        fjs = sorted(glob.glob(os.path.join(fj_dir, "*_fold[0-9].json")))
        rep["fold_jsons"] = fjs
        for p in fjs:
            J = json.load(open(p)); f = int(J["fold"])
            ok = J.get("n_train") == wt[f]["n_train"] and J.get("n_test") == wt[f]["n_test"]
            mx = 0.0
            ref = wt[f]["table"]
            if J.get("weights_mode") == "uniform":   # control: every weight must be exactly 1
                ref = [dict(t, weight=1.0, total_weight=float(t["n"])) for t in ref]
            for t, u in zip(sorted(J["weight_table"], key=lambda t: (t["label"], t["arm"])),
                            sorted(ref, key=lambda t: (t["label"], t["arm"]))):
                ok &= (t["label"], t["arm"], t["n"]) == (u["label"], u["arm"], u["n"])
                mx = max(mx, abs(t["weight"] - u["weight"]), abs(t["total_weight"] - u["total_weight"]))
            C[f"fold{f}_json_weight_table_match(max_abs_diff)"] = [bool(ok), mx]
            C[f"fold{f}_json_mode"] = J.get("weights_mode")
            # scores in the fold json are indexed by the compute manifest row; compare to the assembled OOF
            if pod_man and "scores" in J:
                d = [abs(float(B[pm[e["row"]]][col]) - e[col]) for e in J["scores"]]
                fo_ok = all(FO[pm[e["row"]]][0] == f for e in J["scores"])
                C[f"fold{f}_json_scores_eq_oof(max_abs_diff,n,test_clips_in_fold)"] = [max(d), len(d), fo_ok]
        rep["fold_json_count"] = len(fjs)

    # ---- AUC cells
    def arrays(D, sc, kk, c):
        return (np.array([D[k][sc] for k in kk]), np.array([int(float(D[k]["label"])) for k in kk]),
                np.array([float(D[k][c]) for k in kk]))
    for name, D, sc, c in (("balanced", B, bsc, col), ("standard", S, ssc, "p_yes")):
        for arm_name in ("all", "conflict", "agreement"):
            kk = sorted(k for k in D if arm_name == "all" or arm[k] == arm_name)
            spk, y, s = arrays(D, sc, kk, c)
            a = auc_mw(y, s); t = auc_trap(y, s)
            assert abs(a - t) < 1e-12, ("MW vs trapezoid", a, t)
            lo, hi, nu, dg = boot(spk, y, [s])
            rep["cells"][f"{name}_{arm_name}"] = dict(value=a, trapezoid=t, lo=lo, hi=hi, n=len(kk),
                                                      n_spk=int(len(np.unique(spk))), n_pos=int(y.sum()),
                                                      n_neg=int((y == 0).sum()), draws=nu, degenerate=dg)
    for arm_name in ("all", "conflict", "agreement"):
        kk = sorted(k for k in set(B) & set(S) if arm_name == "all" or arm[k] == arm_name)
        spk, y, sb = arrays(B, bsc, kk, col)
        spk2, y2, ss = arrays(S, ssc, kk, "p_yes")
        assert (spk == spk2).all() and (y == y2).all()
        ab = auc_mw(y, sb); as_ = auc_mw(y, ss)
        lo, hi, nu, dg = boot(spk, y, [sb, ss])
        rep["cells"][f"paired_bal_minus_std_{arm_name}"] = dict(value=ab - as_, a_bal=ab, a_std=as_, lo=lo, hi=hi,
                                                                n=len(kk), n_spk=int(len(np.unique(spk))),
                                                                draws=nu, degenerate=dg)
    print(json.dumps(rep, indent=1, default=str))


def _entry():
    if len(sys.argv) > 1 and sys.argv[1] == "--gap":
        gap_main(*sys.argv[2:5])
    else:
        main()


# ---------------------------------------------------------------------------
# extras: conflict-minus-agreement gap. One speaker draw over ALL speakers of the
# 468 per replicate; inside the draw AUC(conflict clips) - AUC(agreement clips);
# for the gap change, (bal gap) - (std gap) on the same draw.
# usage: POD1_verify.py --gap BAL_OOF STD_OOF CONFLICT_MANIFEST
def gap_main(bal_p, std_p, man_p):
    man = list(csv.DictReader(open(man_p)))
    arm = {os.path.basename(r["segment_path"]): r["set"] for r in man}
    B = {os.path.basename(r["path"]): r for r in csv.DictReader(open(bal_p))}
    S = {os.path.basename(r["path"]): r for r in csv.DictReader(open(std_p))}
    kk = sorted(set(B) & set(S))
    spk = np.array([B[k]["speaker"] for k in kk]); y = np.array([int(B[k]["label"]) for k in kk])
    a = np.array([arm[k] for k in kk]); sb = np.array([float(B[k]["p_yes"]) for k in kk])
    ss = np.array([float(S[k]["p_yes"]) for k in kk]); c = a == "conflict"; g = a == "agreement"

    def gap(ii, s):
        return auc_mw(y[ii][c[ii]], s[ii][c[ii]]) - auc_mw(y[ii][g[ii]], s[ii][g[ii]])
    allidx = np.arange(len(kk))
    pt = {"bal": gap(allidx, sb), "std": gap(allidx, ss)}
    pt["change"] = pt["bal"] - pt["std"]
    rng = np.random.default_rng(0); u, idx = spk_index(spk)
    v = {"bal": [], "std": [], "change": []}
    for _ in range(NB):
        ii = np.concatenate([idx[p] for p in rng.choice(u, size=len(u), replace=True)])
        gb, gs = gap(ii, sb), gap(ii, ss)
        v["bal"].append(gb); v["std"].append(gs); v["change"].append(gb - gs)
    out = {k: dict(value=pt[k], lo=float(np.percentile(v[k], 2.5)), hi=float(np.percentile(v[k], 97.5)),
                   n=len(kk), n_spk=int(len(u))) for k in pt}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    _entry()
