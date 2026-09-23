#!/usr/bin/env python3
"""POD1b independent verifier: three extra seeds of the standard Pitt projector fine-tune.

Written from scratch by the verifier. It never imports or execs the compute job's code.
Inputs are only per-clip csv files, the sidecar json, the fold file and the original run's oof csv.

AUC: explicit Mann-Whitney over every positive x negative pair, ties count 0.5,
cross-checked by a trapezoid ROC built from the unique score thresholds.
Intervals: 2000 draws, a fresh numpy default_rng(0) per cell, speakers resampled with
replacement (never clips), percentiles 2.5 / 97.5 (numpy linear). A draw missing a class
is skipped and counted. Paired: one speaker draw per replicate, both AUCs recomputed on it.
Arms: resampling pool reported both ways, all 228 speakers then subset to the arm, and
arm speakers only.

usage: POD1b_verify.py OOF_CSV [--sidecar JSON] [--orig ORIG_OOF] [--folds FOLDFILE] [--json OUT]
"""
import csv, sys, os, json, argparse, hashlib
import numpy as np

NB = 2000
PROMPT = ("Based only on this recording, does this speaker show signs of dementia? "
          "Answer with one word, Yes or No.")


def mw_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p = s[y == 1]; n = s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(p) * len(n)))


def trap_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    P = (y == 1).sum(); N = (y == 0).sum()
    th = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in th:
        tpr.append(((s >= t) & (y == 1)).sum() / P)
        fpr.append(((s >= t) & (y == 0)).sum() / N)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def first(cols, *names):
    for n in names:
        if n in cols:
            return n
    return None


def load(path, score=None):
    rows = list(csv.DictReader(open(path)))
    cols = rows[0].keys()
    kc = first(cols, "clip", "clip_id", "path", "file", "wav")
    sc = score or first(cols, "p_yes", "score", "oof", "prob")
    spc = first(cols, "speaker", "speaker_id", "spk")
    ac = first(cols, "set", "arm")
    fc = first(cols, "fold")
    d = {}
    for r in rows:
        k = os.path.basename(r[kc])
        if k in d:
            raise SystemExit(f"duplicate clip {k} in {path}")
        arm = r[ac] if ac and r.get(ac) else k.split("_")[0]
        d[k] = dict(spk=str(r[spc]), y=int(r["label"]), s=float(r[sc]), arm=arm,
                    fold=(r[fc] if fc else None), prompt=r.get("prompt"), raw=r)
    return d, dict(key=kc, score=sc, speaker=spc, arm=ac, fold=fc, cols=list(cols))


def boot_single(spk_all, y, s, mask, pool):
    rng = np.random.default_rng(0)
    u = np.unique(spk_all) if pool == "all" else np.unique(spk_all[mask])
    idx = {p: np.where((spk_all == p) & mask)[0] for p in u}
    vals = []; bad = 0
    for _ in range(NB):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        a = mw_auc(y[ii], s[ii])
        if np.isnan(a):
            bad += 1; continue
        vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), bad


def boot_paired(spk_all, y, s1, s2, mask, pool):
    rng = np.random.default_rng(0)
    u = np.unique(spk_all) if pool == "all" else np.unique(spk_all[mask])
    idx = {p: np.where((spk_all == p) & mask)[0] for p in u}
    vals = []; bad = 0
    for _ in range(NB):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        a1 = mw_auc(y[ii], s1[ii]); a2 = mw_auc(y[ii], s2[ii])
        if np.isnan(a1) or np.isnan(a2):
            bad += 1; continue
        vals.append(a1 - a2)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--orig", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--score", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no_paired", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args()

    D, meta = load(a.oof, a.score)
    keys = sorted(D)
    out = {"file": a.oof, "meta": meta, "checks": {}, "cells": []}
    ck = out["checks"]
    ck["n"] = len(D)
    ck["md5"] = hashlib.md5(open(a.oof, "rb").read()).hexdigest()
    ck["nan_scores"] = int(sum(np.isnan(D[k]["s"]) for k in keys))
    ck["score_min_max"] = [float(min(D[k]["s"] for k in keys)), float(max(D[k]["s"] for k in keys))]
    prompts = sorted({D[k]["prompt"] for k in keys if D[k]["prompt"] is not None})
    ck["prompts"] = prompts
    ck["prompt_verbatim_ok"] = (prompts == [PROMPT]) if prompts else None

    spk = np.array([D[k]["spk"] for k in keys])
    y = np.array([D[k]["y"] for k in keys])
    s = np.array([D[k]["s"] for k in keys])
    arm = np.array([D[k]["arm"] for k in keys])
    ck["n_spk"] = int(len(np.unique(spk)))
    ck["n_pos"] = int((y == 1).sum()); ck["n_neg"] = int((y == 0).sum())
    ck["arm_counts"] = {x: int((arm == x).sum()) for x in np.unique(arm)}
    ck["arm_label_counts"] = {x: {"pos": int(((arm == x) & (y == 1)).sum()), "neg": int(((arm == x) & (y == 0)).sum())}
                              for x in np.unique(arm)}
    ck["arm_spk"] = {x: int(len(np.unique(spk[arm == x]))) for x in np.unique(arm)}
    ck["arm_from_prefix_agrees"] = bool(all(D[k]["arm"] == k.split("_")[0] for k in keys))

    if a.folds:
        F = {r["clip_id"]: r for r in csv.DictReader(open(a.folds))}
        ck["fold_file"] = a.folds
        ck["fold_file_md5"] = hashlib.md5(open(a.folds, "rb").read()).hexdigest()
        ck["fold_file_same_clips"] = set(F) == set(D)
        ck["fold_file_speaker_mismatch"] = sum(F[k]["speaker_id"] != D[k]["spk"] for k in keys if k in F)
        if meta["fold"]:
            ck["fold_col_mismatch_vs_file"] = sum(str(F[k]["fold"]) != str(D[k]["fold"]) for k in keys if k in F)
        else:
            ck["fold_col_mismatch_vs_file"] = "no fold column in oof csv"

    if a.sidecar:
        S = json.load(open(a.sidecar))
        out["sidecar"] = S
        ck["sidecar_prompt_ok"] = S.get("prompt") == PROMPT if "prompt" in S else "no prompt key"
        ck["sidecar_n"] = S.get("n"); ck["sidecar_n_speakers"] = S.get("n_speakers")

    O = None
    if a.orig:
        O, _ = load(a.orig)
        ck["orig_same_clips"] = set(O) == set(D)
        ck["orig_speaker_mismatch"] = sum(O[k]["spk"] != D[k]["spk"] for k in keys if k in O)
        ck["orig_label_mismatch"] = sum(O[k]["y"] != D[k]["y"] for k in keys if k in O)
        ck["orig_arm_mismatch"] = sum(O[k]["arm"] != D[k]["arm"] for k in keys if k in O)
        so = np.array([O[k]["s"] for k in keys])
        ck["identical_scores_to_orig"] = int(np.sum(np.isclose(so, s, rtol=0, atol=1e-9)))
        ck["max_abs_diff_vs_orig"] = float(np.max(np.abs(so - s)))
        ck["pearson_vs_orig"] = float(np.corrcoef(so, s)[0, 1])

    def cell(cid, what, val, trap, lo, hi, n, nspk, bad):
        c = dict(id=cid, what=what, value=val, trap=trap, lo=lo, hi=hi, n=n, n_spk=nspk, degenerate=bad)
        out["cells"].append(c)
        t = "" if trap is None else f" trap {trap:.6f}"
        print(f"{cid}\t{what}\t{val:.6f}{t}\t[{lo:.4f}, {hi:.4f}]\tn={n} spk={nspk} degenerate={bad}", flush=True)

    allm = np.ones(len(y), bool)
    v = mw_auc(y, s); t = trap_auc(y, s)
    lo, hi, bad = boot_single(spk, y, s, allm, "all")
    cell("overall", "overall OOF AUC", v, t, lo, hi, int(len(y)), ck["n_spk"], bad)
    for x in ("conflict", "agreement"):
        m = arm == x
        v = mw_auc(y[m], s[m]); t = trap_auc(y[m], s[m])
        for pool in ("all", "arm"):
            lo, hi, bad = boot_single(spk, y, s, m, pool)
            npool = ck["n_spk"] if pool == "all" else ck["arm_spk"][x]
            cell(f"{x}_pool{pool}", f"{x} arm OOF AUC (pool {pool})", v, t, lo, hi, int(m.sum()), npool, bad)
    # paired conflict minus agreement within this seed: one speaker draw, both arms recomputed
    rng = np.random.default_rng(0); u = np.unique(spk)
    idx = {p: np.where(spk == p)[0] for p in u}
    vals = []; bad = 0
    for _ in range(NB):
        ii = np.concatenate([idx[p] for p in rng.choice(u, size=len(u), replace=True)])
        mc = arm[ii] == "conflict"; ma = arm[ii] == "agreement"
        c1 = mw_auc(y[ii][mc], s[ii][mc]); c2 = mw_auc(y[ii][ma], s[ii][ma])
        if np.isnan(c1) or np.isnan(c2):
            bad += 1; continue
        vals.append(c1 - c2)
    mc = arm == "conflict"; ma = arm == "agreement"
    cell("paired_conf_minus_agr", "paired conflict minus agreement (pool all)",
         mw_auc(y[mc], s[mc]) - mw_auc(y[ma], s[ma]), None,
         float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(len(y)), ck["n_spk"], bad)
    if O is not None and not a.no_paired:
        so = np.array([O[k]["s"] for k in keys])
        d = mw_auc(y, s) - mw_auc(y, so)
        lo, hi, bad = boot_paired(spk, y, s, so, allm, "all")
        cell("paired_vs_orig_overall", "paired this seed minus original, overall", d, None, lo, hi,
             int(len(y)), ck["n_spk"], bad)
        for x in ("conflict", "agreement"):
            m = arm == x
            d = mw_auc(y[m], s[m]) - mw_auc(y[m], so[m])
            lo, hi, bad = boot_paired(spk, y, s, so, m, "all")
            cell(f"paired_vs_orig_{x}", f"paired this seed minus original, {x} arm (pool all)", d, None,
                 lo, hi, int(m.sum()), ck["n_spk"], bad)
    # per-fold diagnostics straight from the csv (fold column, answer_mass column when present)
    if meta["fold"]:
        fo = np.array([int(D[k]["fold"]) for k in keys])
        am = np.array([float(D[k]["raw"]["answer_mass"]) for k in keys]) if "answer_mass" in meta["cols"] else None
        pf = {}
        for f in sorted(set(fo)):
            m = fo == f
            e = {"n": int(m.sum()), "auc_within_fold": mw_auc(y[m], s[m])}
            if am is not None:
                e.update(answer_mass_min=float(am[m].min()), answer_mass_median=float(np.median(am[m])),
                         n_answer_mass_below_half=int((am[m] < 0.5).sum()))
            pf[str(f)] = e
        out["per_fold"] = pf
        if am is not None:
            out["answer_mass"] = {"median": float(np.median(am)), "min": float(am.min()),
                                  "n_below_half": int((am < 0.5).sum())}
        for f, e in pf.items():
            print(f"fold {f}: n={e['n']} within-fold AUC {e['auc_within_fold']:.6f}" +
                  (f" mass min {e['answer_mass_min']:.3g} median {e['answer_mass_median']:.3g} below0.5 {e['n_answer_mass_below_half']}"
                   if am is not None else ""), flush=True)
    if a.state:
        St = json.load(open(a.state))
        rows_order = [os.path.basename(r[meta["key"]]) for r in csv.DictReader(open(a.oof))]
        so_ = np.array(St["oof"]); sc_ = np.array([D[k]["s"] for k in rows_order])
        ck["state_folds_done"] = sorted(St.get("done", []))
        ck["state_oof_max_abs_diff_vs_csv"] = float(np.max(np.abs(so_ - sc_)))
    print(json.dumps(ck, indent=1, default=str))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
