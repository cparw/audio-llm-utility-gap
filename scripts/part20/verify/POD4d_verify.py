#!/usr/bin/env python3
"""POD4d independent verifier: Kimi-Audio zero-shot on the 468 Pitt interviewer-free windows.

Written from scratch by the verifier. It never imports or execs the compute job's code.
Inputs are only per-clip csv files (clip, speaker, label, p_yes, ...).

AUC: explicit Mann-Whitney over every positive x negative pair, ties count 0.5,
cross-checked by a trapezoid ROC built from the unique score thresholds.
Intervals: 2000 draws, a fresh numpy default_rng(0) per cell, speakers resampled with
replacement (never clips), percentiles 2.5 / 97.5 (numpy linear). A draw missing a class
is skipped and counted. Paired: one speaker draw per replicate, both AUCs recomputed on it.
Arms: the resampling pool is reported both ways, all 228 speakers then subset to the arm
(the convention that reproduces the canonical arm intervals 0.6153-0.8159 / 0.6519-0.7875)
and arm speakers only.

usage: POD4d_verify.py CUT_CSV [ORIG_CSV] [--score p_yes] [--orig_score p_yes] [--cut_key orig_clip_id]
                       [--orig_key clip] [--json out.json]
"cut" is the interviewer-free file, "orig" the original-window file; paired = AUC(orig) - AUC(cut).
"""
import csv, sys, json, argparse
import numpy as np

NB = 2000


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


def load(path, score="p_yes", key="clip"):
    rows = list(csv.DictReader(open(path)))
    d = {}
    for r in rows:
        if r[key] in d:
            raise SystemExit(f"duplicate clip {r[key]} in {path}")
        d[r[key]] = (r["speaker"], int(r["label"]), float(r[score]), r)
    return d


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
    """returns interval of AUC(s1) - AUC(s2)"""
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
    ap.add_argument("cut")
    ap.add_argument("orig", nargs="?",
                    default="<local data dir>/release/overnight2/part3/kimi_pitt_zeroshot_scores.csv")
    ap.add_argument("--score", default="p_yes")
    ap.add_argument("--orig_score", default="p_yes")
    ap.add_argument("--cut_key", default="clip")
    ap.add_argument("--orig_key", default="clip")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    C = load(a.cut, a.score, a.cut_key)
    O = load(a.orig, a.orig_score, a.orig_key)
    keys = sorted(O.keys())
    out = {"cut_file": a.cut, "orig_file": a.orig, "checks": {}, "cells": []}
    ck = out["checks"]
    ck["n_cut"] = len(C); ck["n_orig"] = len(O)
    ck["same_clip_set"] = set(C) == set(O)
    ck["missing_in_cut"] = sorted(set(O) - set(C))[:10]
    ck["extra_in_cut"] = sorted(set(C) - set(O))[:10]
    common = [k for k in keys if k in C]
    ck["speaker_mismatch"] = sum(C[k][0] != O[k][0] for k in common)
    ck["label_mismatch"] = sum(C[k][1] != O[k][1] for k in common)
    if "prompt" in next(iter(C.values()))[3]:
        ck["cut_prompts"] = sorted({C[k][3]["prompt"] for k in C})
    ck["nan_scores_cut"] = int(sum(np.isnan(C[k][2]) for k in C))

    spk = np.array([O[k][0] for k in common])
    y = np.array([O[k][1] for k in common])
    so = np.array([O[k][2] for k in common])
    sc = np.array([C[k][2] for k in common])
    arm = np.array([k.split("_")[0] for k in common])
    ck["n_spk"] = int(len(np.unique(spk)))
    ck["arm_counts"] = {x: int((arm == x).sum()) for x in np.unique(arm)}
    ck["arm_spk"] = {x: int(len(np.unique(spk[arm == x]))) for x in np.unique(arm)}

    def cell(cid, what, val, trap, lo, hi, n, nspk, bad, extra=None):
        c = dict(id=cid, what=what, value=val, trap=trap, lo=lo, hi=hi, n=n, n_spk=nspk, degenerate=bad)
        if extra: c.update(extra)
        out["cells"].append(c)
        t = "" if trap is None else f" trap {trap:.6f}"
        print(f"{cid}\t{what}\t{val:.6f}{t}\t[{lo:.4f}, {hi:.4f}]\tn={n} spk={nspk} degenerate={bad}", flush=True)

    allm = np.ones(len(y), bool)
    for tag, s in (("cut", sc), ("orig", so)):
        v = mw_auc(y, s); t = trap_auc(y, s)
        lo, hi, bad = boot_single(spk, y, s, allm, "all")
        cell(f"{tag}_overall", f"{tag} overall zero-shot AUC", v, t, lo, hi, int(len(y)), ck["n_spk"], bad)
        for x in ("conflict", "agreement"):
            m = arm == x
            v = mw_auc(y[m], s[m]); t = trap_auc(y[m], s[m])
            for pool in ("all", "arm"):
                lo, hi, bad = boot_single(spk, y, s, m, pool)
                npool = ck["n_spk"] if pool == "all" else ck["arm_spk"][x]
                cell(f"{tag}_{x}_pool{pool}", f"{tag} {x} arm zero-shot AUC (pool {pool})", v, t, lo, hi,
                     int(m.sum()), npool, bad)
    # paired original minus interviewer-free
    d = mw_auc(y, so) - mw_auc(y, sc)
    lo, hi, bad = boot_paired(spk, y, so, sc, allm, "all")
    cell("paired_overall", "paired original minus interviewer-free, overall", d, None, lo, hi, int(len(y)),
         ck["n_spk"], bad)
    for x in ("conflict", "agreement"):
        m = arm == x
        d = mw_auc(y[m], so[m]) - mw_auc(y[m], sc[m])
        for pool in ("all", "arm"):
            lo, hi, bad = boot_paired(spk, y, so, sc, m, pool)
            npool = ck["n_spk"] if pool == "all" else ck["arm_spk"][x]
            cell(f"paired_{x}_pool{pool}", f"paired original minus interviewer-free, {x} arm (pool {pool})", d, None,
                 lo, hi, int(m.sum()), npool, bad)
    print(json.dumps(ck, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


# ---------------------------------------------------------------------------------------------
# results mode: POD4d_verify.py results RESULT_DIR [FOLD_FILE] [CANONICAL] [OUT_JSON]
# For every <id>.RESULT: load <id>.json and its per-clip <id>.csv, recompute from the csv only,
# and compare to 4 dp.  Pool for the interval = the speakers present in that csv (the compute
# job writes one csv per arm, so for arms this is the arm-only pool).
# ---------------------------------------------------------------------------------------------
PROMPT = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."


def exact_auc_num(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p = s[y == 1]; n = s[y == 0]
    return int(2 * (p[:, None] > n[None, :]).sum() + (p[:, None] == n[None, :]).sum()), 2 * len(p) * len(n)


def r4(x):
    return f"{x:.4f}"


def on_boundary(x):
    t = abs(x) * 1e4
    return abs((t - np.floor(t)) - 0.5) < 1e-7


def boot_cells(spk, y, cols, paired):
    rng = np.random.default_rng(0)
    u = np.unique(spk)
    idx = {p: np.where(spk == p)[0] for p in u}
    vals = []; bad = 0
    for _ in range(NB):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        a = [mw_auc(y[ii], c[ii]) for c in cols]
        if any(np.isnan(x) for x in a):
            bad += 1; continue
        vals.append(a[0] - a[1] if paired else a[0])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), bad, len(vals)


def results_mode(rdir, fold_file, canon_file, out_json):
    import glob, os
    F = {r["clip_id"]: r for r in csv.DictReader(open(fold_file))}
    CAN = {r["clip"]: r for r in csv.DictReader(open(canon_file))}
    RAW = {}
    for nm in ("kimi_pitt_noinv468_scores.csv", "kimi_pitt_orig468_pod_scores.csv"):
        if os.path.exists(os.path.join(rdir, nm)):
            RAW[nm] = list(csv.DictReader(open(os.path.join(rdir, nm))))
    raw_noinv = {r["orig_clip_id"]: r for r in RAW.get("kimi_pitt_noinv468_scores.csv", [])}
    raw_pod = {r["clip"]: r for r in RAW.get("kimi_pitt_orig468_pod_scores.csv", [])}
    report = []
    for rf in sorted(glob.glob(os.path.join(rdir, "*.RESULT"))):
        rid = os.path.basename(rf)[:-7]
        J = json.load(open(os.path.join(rdir, rid + ".json")))
        rec = dict(id=rid, what=J.get("what"), problems=[])
        prob = rec["problems"]
        if J.get("prompt_verbatim") != PROMPT:
            prob.append("prompt_verbatim differs from the Pitt recording-wording prompt")
        if "value" not in J:
            rec["kind"] = "non-cell"
            report.append(rec); continue
        rows = list(csv.DictReader(open(os.path.join(rdir, rid + ".csv"))))
        cols = list(rows[0].keys())
        key = "orig_clip_id" if "orig_clip_id" in cols else "clip"
        keys = [r[key] for r in rows]
        if len(set(keys)) != len(keys):
            prob.append("duplicate clips in per-clip csv")
        spk = np.array([r["speaker"] for r in rows]); y = np.array([int(r["label"]) for r in rows])
        # identity checks against the fold file and the canonical file
        for r, k in zip(rows, keys):
            if F[k]["speaker_id"] != r["speaker"]: prob.append(f"speaker != fold file for {k}"); break
            if F[k]["fold"] != r["fold"]: prob.append(f"fold != fold file for {k}"); break
            if int(CAN[k]["label"]) != int(r["label"]): prob.append(f"label != canonical for {k}"); break
            if r["set"] != k.split("_")[0]: prob.append(f"set column != clip prefix for {k}"); break
        if "p_yes_original" in cols:
            kind = "paired"; a_col, b_col = "p_yes_original", "p_yes_noinv"
        elif "p_yes_canonical" in cols and "p_yes_pod_rerun" in cols:
            kind = "paired"; a_col, b_col = "p_yes_canonical", "p_yes_pod_rerun"
        else:
            kind = "single"; a_col, b_col = "p_yes", None
        rec["kind"] = kind
        A = np.array([float(r[a_col]) for r in rows])
        # per-clip scores must equal the raw score files exactly
        def raw_of(col, k):
            if col == "p_yes_noinv" or (col == "p_yes" and "noinv_clip" not in cols and rid.startswith("kimi_pitt_noinv_")):
                return float(raw_noinv[k]["p_yes"])
            if col == "p_yes_canonical" or (col == "p_yes" and "orig_canonical" in rid):
                return float(CAN[k]["p_yes"])
            if col == "p_yes_pod_rerun" or (col == "p_yes" and "podrerun" in rid):
                return float(raw_pod[k]["p_yes"])
            if col == "p_yes_original":
                return float(CAN[k]["p_yes"]) if rid.startswith("kimi_pitt_orig_minus") else float(raw_pod[k]["p_yes"])
            return None
        for col in [c for c in (a_col, b_col) if c]:
            mm = 0
            for r, k in zip(rows, keys):
                rv = raw_of(col, k)
                if rv is None or rv != float(r[col]): mm += 1
            rec[f"raw_mismatch_{col}"] = mm
            if mm: prob.append(f"{mm} per-clip values in column {col} differ from the raw score file")
        if kind == "single":
            v = mw_auc(y, A); t = trap_auc(y, A); num, den = exact_auc_num(y, A)
            lo, hi, bad, nuse = boot_cells(spk, y, [A], False)
        else:
            B = np.array([float(r[b_col]) for r in rows])
            v = mw_auc(y, A) - mw_auc(y, B); t = trap_auc(y, A) - trap_auc(y, B); num, den = None, None
            lo, hi, bad, nuse = boot_cells(spk, y, [A, B], True)
        rec.update(value_verified=v, trap_verified=t, exact=None if num is None else f"{num}/{den}",
                   lo_verified=lo, hi_verified=hi, degenerate=bad, usable=nuse,
                   n_verified=len(rows), n_spk_verified=int(len(np.unique(spk))),
                   value_computed=J["value"], lo_computed=J["lo"], hi_computed=J["hi"],
                   n_computed=J["n"], n_spk_computed=J["n_speakers"], score_column=J.get("score_column"),
                   per_clip_csv=J.get("per_clip_csv"))
        if abs(v - t) > 1e-12: prob.append(f"MW {v} != trapezoid {t}")
        agree = (r4(v) == r4(J["value"]) and r4(lo) == r4(J["lo"]) and r4(hi) == r4(J["hi"])
                 and len(rows) == J["n"] and int(len(np.unique(spk))) == J["n_speakers"])
        rec["boundary"] = [nm for nm, x in (("value", v), ("lo", lo), ("hi", hi)) if on_boundary(x)]
        rec["agree_4dp"] = bool(agree)
        if J.get("usable_draws") not in (None, nuse): prob.append(f"usable draws {J.get('usable_draws')} vs mine {nuse}")
        arm = "conflict" if rid.endswith("_conflict") else ("agreement" if rid.endswith("_agreement") else "overall")
        exp_n = {"conflict": 146, "agreement": 322, "overall": 468}[arm]
        if len(rows) != exp_n: prob.append(f"n {len(rows)} != expected {exp_n} for {arm}")
        if arm == "overall" and int(len(np.unique(spk))) != 228: prob.append("overall n_spk != 228")
        if arm != "overall" and any(r["set"] != arm for r in rows): prob.append("arm csv contains other-arm clips")
        rec["arm"] = arm
        if kind == "single":
            rec["vs_half"] = "above 0.5" if lo > 0.5 else ("below 0.5" if hi < 0.5 else "interval includes 0.5")
        else:
            rec["vs_half"] = "interval excludes zero" if (lo > 0 or hi < 0) else "interval includes zero"
        flag = "AGREE" if agree and not prob else "DISAGREE"
        print(f"{flag}\t{rid}\tcomputed {r4(J['value'])} [{r4(J['lo'])}, {r4(J['hi'])}] n={J['n']} spk={J['n_speakers']}"
              f"\tverified {v:.6f} [{lo:.6f}, {hi:.6f}] n={len(rows)} spk={len(np.unique(spk))} trap_ok={abs(v-t)<1e-12}"
              f" boundary={rec['boundary']} problems={prob}", flush=True)
        report.append(rec)
    if out_json:
        json.dump(report, open(out_json, "w"), indent=1, default=float)
    return report


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "results":
    a = sys.argv[2:]
    results_mode(a[0],
                 a[1] if len(a) > 1 else "folds/pitt_groupkfold5_pod.csv",
                 a[2] if len(a) > 2 else "<local data dir>/release/overnight2/part3/kimi_pitt_zeroshot_scores.csv",
                 a[3] if len(a) > 3 else None)
    sys.exit(0)


if __name__ == "__main__":
    main()
