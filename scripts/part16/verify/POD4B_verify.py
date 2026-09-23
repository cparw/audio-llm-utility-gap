#!/usr/bin/env python3
"""Independent verification of POD4B. Written from scratch. Does not import
or read the producing scripts. Recomputes every claimed value from the
per-clip csvs on the Mac.
"""
import csv, json, math, os, sys
import numpy as np

BASE = "<local data dir>/Desktop/release/edaic_rerun/part16/POD4B"
OUT  = os.path.join(BASE, "out")

# ---------------------------------------------------------------- AUC
def auc_mwu(y, s):
    """Mann-Whitney U over all concordant pairs, ties counted 0.5.
    Explicit pairwise count, no rank shortcut."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # pairwise comparison matrix
    d = pos[:, None] - neg[None, :]
    wins = float((d > 0).sum())
    ties = float((d == 0).sum())
    return (wins + 0.5 * ties) / (len(pos) * len(neg))

def auc_trapz(y, s):
    """Second, structurally different implementation: trapezoid over the full
    ROC built by sweeping every distinct threshold."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]; ss = s[order]
    tps = [0.0]; fps = [0.0]
    tp = fp = 0.0
    i = 0
    while i < len(ss):
        j = i
        while j < len(ss) and ss[j] == ss[i]:
            if ys[j] == 1: tp += 1
            else: fp += 1
            j += 1
        tps.append(tp); fps.append(fp)
        i = j
    tpr = np.array(tps) / P; fpr = np.array(fps) / N
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))

# ---------------------------------------------------------------- data
def load(corpus):
    p = os.path.join(OUT, f"{corpus}_no_interviewer.csv")
    rows = list(csv.DictReader(open(p)))
    return rows

def col(rows, name, cast=float):
    return np.array([cast(r[name]) for r in rows])

# ---------------------------------------------------------------- bootstrap
def boot_single(y, s, ndraw=2000, mode="integers"):
    n = len(y)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(ndraw):
        idx = rng.integers(0, n, n) if mode == "integers" else rng.choice(n, n, replace=True)
        yy = y[idx]; ss = s[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        vals.append(auc_mwu(yy, ss))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def boot_paired(y, s1, s2, ndraw=2000, mode="integers"):
    """delta = AUC(s2) - AUC(s1), one speaker draw per replicate."""
    n = len(y)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(ndraw):
        idx = rng.integers(0, n, n) if mode == "integers" else rng.choice(n, n, replace=True)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        vals.append(auc_mwu(yy, s2[idx]) - auc_mwu(yy, s1[idx]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def boot_gap(y, szs, senc, ndraw=2000, mode="integers"):
    n = len(y)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(ndraw):
        idx = rng.integers(0, n, n) if mode == "integers" else rng.choice(n, n, replace=True)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        vals.append(auc_mwu(yy, senc[idx]) - auc_mwu(yy, szs[idx]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

# ---------------------------------------------------------------- main
def run(corpus, mode):
    rows = load(corpus)
    y = col(rows, "label", int)
    out = {}
    scores = {
        "A_orig_zs": col(rows, "p_yes_A"),
        "A_orig_enc": col(rows, "p_probe_A"),
        "B_paronly_zs": col(rows, "p_yes_B"),
        "B_paronly_enc": col(rows, "p_probe_B"),
        "C_durmatch_zs": col(rows, "p_yes_C"),
        "C_durmatch_enc": col(rows, "p_probe_C"),
        "published_zs": col(rows, "p_yes_published"),
        "published_enc": col(rows, "p_probe_published"),
    }
    for k, s in scores.items():
        a1 = auc_mwu(y, s); a2 = auc_trapz(y, s)
        lo, hi, nd = boot_single(y, s, mode=mode)
        out[f"4b_{corpus}_{k}"] = dict(value=round(a1, 4), value_trapz=round(a2, 4),
                                       lo=round(lo, 4), hi=round(hi, 4), draws=nd,
                                       impl_agree=abs(a1 - a2) < 1e-9)
    pairs = [("zs", "A_orig", "B_paronly"), ("zs", "A_orig", "C_durmatch"), ("zs", "C_durmatch", "B_paronly"),
             ("enc", "A_orig", "B_paronly"), ("enc", "A_orig", "C_durmatch"), ("enc", "C_durmatch", "B_paronly")]
    shortname = {"A_orig": "A", "B_paronly": "B", "C_durmatch": "C"}
    for kind, a, b in pairs:
        s1 = scores[f"{a}_{kind}"]; s2 = scores[f"{b}_{kind}"]
        d = auc_mwu(y, s2) - auc_mwu(y, s1)
        lo, hi, nd = boot_paired(y, s1, s2, mode=mode)
        out[f"4b_{corpus}_paired_{kind}_{shortname[a]}_to_{shortname[b]}"] = dict(
            value=round(d, 4), lo=round(lo, 4), hi=round(hi, 4), draws=nd)
    for arm in ["A_orig", "B_paronly", "C_durmatch", "published"]:
        szs = scores[f"{arm}_zs"]; senc = scores[f"{arm}_enc"]
        g = auc_mwu(y, senc) - auc_mwu(y, szs)
        lo, hi, nd = boot_gap(y, szs, senc, mode=mode)
        out[f"4b_{corpus}_gap_{arm}"] = dict(value=round(g, 4), lo=round(lo, 4), hi=round(hi, 4), draws=nd)

    # descriptives
    share = col(rows, "inv_share_of_window")
    inv_ms = col(rows, "inv_ms_in_window")
    par_ms = col(rows, "par_ms_in_window")
    dA = col(rows, "dur_A_orig_s"); dB = col(rows, "dur_B_paronly_s"); dC = col(rows, "dur_C_durmatch_s")
    out[f"4b_{corpus}_inv_share_mean"] = dict(value=round(float(share.mean()), 4))
    out[f"4b_{corpus}_inv_share_median"] = dict(value=round(float(np.median(share)), 4))
    out[f"4b_{corpus}_inv_any_count"] = dict(value=int((inv_ms > 0).sum()))
    out[f"4b_{corpus}_inv_gt5s_count"] = dict(value=int((inv_ms > 5000).sum()))
    out[f"4b_{corpus}_dur_removed_mean_s"] = dict(value=round(float((dA - dB).mean()), 4))
    out[f"4b_{corpus}_par_under10s_count"] = dict(value=int((par_ms < 10000).sum()))
    out[f"_{corpus}_meta"] = dict(n=len(rows), n_speakers=len(set(r["speaker"] for r in rows)),
                                 labels={0: int((y == 0).sum()), 1: int((y == 1).sum())},
                                 durA_mean=round(float(dA.mean()), 4),
                                 durB_mean=round(float(dB.mean()), 4),
                                 durC_mean=round(float(dC.mean()), 4),
                                 durBC_maxabsdiff=float(np.abs(dB - dC).max()),
                                 durBC_n_mismatch=int((np.abs(dB - dC) > 1e-6).sum()),
                                 inv_share_recomputed_maxerr=float(np.abs(share - inv_ms / col(rows, "window_span_ms")).max()),
                                 n_distinct_pyesA=len(set(col(rows, "p_yes_A"))),
                                 answer_mass_median_A=round(float(np.median(col(rows, "answer_mass_A"))), 4),
                                 answer_mass_median_B=round(float(np.median(col(rows, "answer_mass_B"))), 4),
                                 answer_mass_median_C=round(float(np.median(col(rows, "answer_mass_C"))), 4),
                                 )
    return out

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "integers"
    res = {}
    for c in ["adress2020", "adresso"]:
        res.update(run(c, mode))
    json.dump(res, open(f"<local data dir>/Desktop/release/edaic_rerun/part16/verify/POD4B_verify_out_{mode}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

# ================= independent CHAT / segmentation parse =================
import re, glob
CHA_ROOT = "<local data dir>/paper work/Hard drive data for paper/DementiaBank/challenges/ADReSS-2020"
SEG_ROOT = os.path.join(BASE, "adresso_seg/ADReSSo21/diagnosis/train/segmentation")

def parse_cha(path):
    """Return (utts, n_untimed). utts = list of (tier, start_ms, end_ms)."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    # join continuation lines (start with tab) onto their header line
    lines = txt.split("\n")
    merged = []
    for ln in lines:
        if ln.startswith("\t") and merged:
            merged[-1] = merged[-1] + " " + ln.lstrip("\t")
        else:
            merged.append(ln)
    utts = []; untimed = 0
    for ln in merged:
        if not ln.startswith("*"):
            continue
        tier = ln[1:ln.index(":")] if ":" in ln else None
        m = re.search(r"\x15(\d+)_(\d+)\x15", ln)
        if m:
            utts.append((tier, int(m.group(1)), int(m.group(2))))
        else:
            untimed += 1
    return utts, untimed

def merge_iv(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if b <= a: continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

def clip_iv(iv, lo, hi):
    out = []
    for a, b in iv:
        a2 = max(a, lo); b2 = min(b, hi)
        if b2 > a2: out.append([a2 - lo, b2 - lo])
    return merge_iv(out)

def parse_seg(path):
    utts = []
    for r in csv.DictReader(open(path)):
        sp = r["speaker"].strip()
        utts.append((sp, int(round(float(r["begin"]))), int(round(float(r["end"])))))
    return utts
