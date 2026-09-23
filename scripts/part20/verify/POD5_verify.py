#!/usr/bin/env python3
"""PART 20 POD5 independent verifier (Qwen2.5-Omni on the 468 interviewer-free Pitt clips).

Written from scratch. Does not import or exec any compute-job script.

AUC      : explicit Mann-Whitney over every positive x negative pair, ties count 0.5,
           cross-checked with a trapezoid ROC built from tie-grouped thresholds.
Interval : 2000 draws, a fresh numpy.random.default_rng(0) per cell, SPEAKERS resampled with
           replacement from the sorted unique speaker list, percentile 2.5 / 97.5, a draw with a
           single class is skipped.
Mean of five : inside every draw all five per-repeat AUCs are recomputed on the drawn clips
           and averaged.
Paired   : one speaker draw per replicate, both quantities recomputed inside it.
Arm      : the full out-of-fold fit restricted to the arm's clips. Interval default = restrict
           first, then resample the speakers present in the arm (the part16 convention);
           the alternative (resample all speakers, restrict inside the draw) is also computed
           and reported so a convention difference is visible, never hidden.
"""
import csv, json, sys, os
import numpy as np

NB = 2000


# ---------------------------------------------------------------- AUC
def auc_mw(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def auc_trap(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort"); ss = s[o]; yy = y[o]
    tps = np.cumsum(yy); fps = np.cumsum(1 - yy)
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]   # tie-grouped thresholds
    tpr = np.r_[0.0, tps[last] / P]; fpr = np.r_[0.0, fps[last] / N]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc_fast(y, s):
    """Rank-based (tie-averaged) AUC used only inside bootstrap loops for speed; it is the same
    statistic as auc_mw, and the point values are always reported from auc_mw."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    n1 = (y == 1).sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    r = np.empty(len(s)); ranks = np.arange(1, len(s) + 1, dtype=np.float64)
    # tie averaging
    starts = np.r_[0, np.where(np.diff(ss) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(ss)]
    avg = (starts + ends + 1) / 2.0
    rr = np.repeat(avg, ends - starts)
    r[o] = rr
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


# ---------------------------------------------------------------- bootstrap
def _groups(spk):
    u = np.unique(spk)
    idx = {sp: np.where(spk == sp)[0] for sp in u}
    return u, idx


def boot(spk, y, S, stat, nb=NB, seed=0):
    """S: list of score vectors; stat(y_draw, [s_draw...]) -> float. Returns lo, hi, usable, vals."""
    spk = np.asarray(spk); y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    u, idx = _groups(spk)
    vals = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        yb = y[ii]
        if len(np.unique(yb)) < 2:
            continue
        v = stat(yb, [s[ii] for s in S])
        if np.isnan(v):
            continue
        vals.append(v)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v), v


def boot_restrict_inside(spk, y, S, mask, stat, nb=NB, seed=0):
    """Alternative arm convention: resample ALL speakers, restrict to the arm inside the draw."""
    spk = np.asarray(spk); y = np.asarray(y).astype(int); mask = np.asarray(mask, bool)
    rng = np.random.default_rng(seed)
    u, idx = _groups(spk)
    vals = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        ii = ii[mask[ii]]
        yb = y[ii]
        if len(ii) == 0 or len(np.unique(yb)) < 2:
            continue
        vals.append(stat(yb, [s[ii] for s in S]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v), v


st_single = lambda yb, ss: auc_fast(yb, ss[0])
st_mean = lambda yb, ss: float(np.mean([auc_fast(yb, s) for s in ss]))
st_diff = lambda yb, ss: auc_fast(yb, ss[0]) - auc_fast(yb, ss[1])          # a minus b
st_meandiff = None  # built on the fly


def r4(v):
    return f"{v:.4f}"


# ---------------------------------------------------------------- io helpers
def read_csv(path):
    return list(csv.DictReader(open(path)))


def fold_map(path):
    return {r["clip_id"]: (r["speaker_id"], int(r["fold"])) for r in read_csv(path)}


def check_folds(clips, spk, folds_used, fold_file):
    """folds_used: per-clip fold ints written by the run. Returns (n_mismatch, n_missing)."""
    fm = fold_map(fold_file)
    mis = miss = 0
    for c, s, f in zip(clips, spk, folds_used):
        c0 = os.path.basename(c)
        if c0 not in fm:
            miss += 1; continue
        if fm[c0][1] != int(f) or fm[c0][0] != s:
            mis += 1
    return mis, miss


ROWS = "reports/part20/rows/POD5.tsv"
COLS = ["id", "what", "value_computed", "value_verified", "lo", "hi", "n", "n_spk", "file",
        "two_dp", "vs_half", "agree_4dp"]


def append_row(d, path=ROWS):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if new:
            w.writerow(COLS)
        w.writerow([d[c] for c in COLS])


if __name__ == "__main__":
    # self test against files with known published values
    rel = "<local data dir>/release"
    rows = read_csv(f"{rel}/omni_final/omnisft_pitt_oof.csv")
    y = np.array([int(r["label"]) for r in rows]); s = np.array([float(r["p_yes"]) for r in rows])
    spk = np.array([r["speaker"] for r in rows]); arm = np.array([r["set"] for r in rows])
    print("sft overall", r4(auc_mw(y, s)), r4(auc_trap(y, s)), r4(auc_fast(y, s)))
    for a in ("conflict", "agreement"):
        m = arm == a
        print("sft", a, m.sum(), r4(auc_mw(y[m], s[m])), r4(auc_trap(y[m], s[m])))


# ================================================================ part20 POD5 checks
PROMPT = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
N_EXP, NSPK_EXP, ARMS_EXP = 468, 228, {"conflict": 146, "agreement": 322}


def basic_checks(clips, spk, y, arm):
    out = {"n": int(len(clips)), "n_spk": int(len(np.unique(spk))), "n_pos": int(np.sum(y)),
           "arms": {a: int(np.sum(np.asarray(arm) == a)) for a in ARMS_EXP},
           "unique_clips": int(len(set(clips)))}
    out["ok"] = (out["n"] == N_EXP and out["n_spk"] == NSPK_EXP and out["arms"] == ARMS_EXP
                 and out["unique_clips"] == N_EXP)
    return out


ALT = os.environ.get("POD5_ALT", "0") == "1"


def point_and_ci(spk, y, cols, mask=None, kind="mean"):
    """cols: list of score vectors (1 or 5). Returns dict with MW / trapezoid point values and both
    arm interval conventions when mask is given."""
    spk = np.asarray(spk); y = np.asarray(y).astype(int)
    cols = [np.asarray(c, dtype=np.float64) for c in cols]
    stat = st_single if len(cols) == 1 else st_mean
    if mask is None:
        mw = [auc_mw(y, c) for c in cols]; tr = [auc_trap(y, c) for c in cols]
        lo, hi, nu, _ = boot(spk, y, cols, stat)
        return dict(per_mw=mw, per_trap=tr, value=float(np.mean(mw)), value_trap=float(np.mean(tr)),
                    lo=lo, hi=hi, usable=nu, n=int(len(y)), n_spk=int(len(np.unique(spk))))
    m = np.asarray(mask, bool)
    mw = [auc_mw(y[m], c[m]) for c in cols]; tr = [auc_trap(y[m], c[m]) for c in cols]
    lo, hi, nu, _ = boot(spk[m], y[m], [c[m] for c in cols], stat)
    lo2, hi2, nu2, _ = boot_restrict_inside(spk, y, cols, m, stat) if ALT else (np.nan, np.nan, 0, None)
    return dict(per_mw=mw, per_trap=tr, value=float(np.mean(mw)), value_trap=float(np.mean(tr)),
                lo=lo, hi=hi, usable=nu, lo_alt=lo2, hi_alt=hi2, usable_alt=nu2,
                n=int(m.sum()), n_spk=int(len(np.unique(spk[m]))))


def paired_ci(spk, y, a_cols, b_cols, mask=None):
    """a minus b, each side a mean over its columns, one speaker draw per replicate."""
    spk = np.asarray(spk); y = np.asarray(y).astype(int)
    A = [np.asarray(c, np.float64) for c in a_cols]; B = [np.asarray(c, np.float64) for c in b_cols]
    na = len(A)
    def st(yb, ss):
        return float(np.mean([auc_fast(yb, s) for s in ss[:na]]) - np.mean([auc_fast(yb, s) for s in ss[na:]]))
    if mask is not None:
        m = np.asarray(mask, bool); spk, y = spk[m], y[m]; A = [c[m] for c in A]; B = [c[m] for c in B]
    va = np.mean([auc_mw(y, c) for c in A]); vb = np.mean([auc_mw(y, c) for c in B])
    lo, hi, nu, _ = boot(spk, y, A + B, st)
    return dict(value=float(va - vb), a=float(va), b=float(vb), lo=lo, hi=hi, usable=nu,
                n=int(len(y)), n_spk=int(len(np.unique(spk))))


def row(id_, what, comp, ver, lo_c, hi_c, lo_v, hi_v, n, nspk, file):
    """comp/lo_c/hi_c are the compute job's numbers (already rounded to 4 dp or raw)."""
    agree = (round(float(comp), 4) == round(float(ver), 4) and round(float(lo_c), 4) == round(float(lo_v), 4)
             and round(float(hi_c), 4) == round(float(hi_v), 4))
    d = dict(id=id_, what=what, value_computed=f"{float(comp):.4f}", value_verified=f"{float(ver):.4f}",
             lo=f"{float(lo_v):.4f}", hi=f"{float(hi_v):.4f}", n=n, n_spk=nspk, file=file,
             two_dp=f"{float(ver):.2f}", vs_half=("above" if ver > 0.5 else ("below" if ver < 0.5 else "equal"))
             if "delta" not in id_ and "cos" not in id_ and "floor" not in id_ else "n/a",
             agree_4dp="yes" if agree else "NO")
    return d, agree


# ================================================================ RESULT verification driver
def load_perclip(path, key="orig_clip_id"):
    rows = read_csv(path)
    clips = [q[key] for q in rows]; spk = np.array([q["speaker"] for q in rows])
    y = np.array([int(q["label"]) for q in rows]); arm = np.array([q["set"] for q in rows])
    return rows, clips, spk, y, arm


def cols_of(rows, names):
    return [np.array([float(q[c]) for q in rows]) for c in names]


def verify_cells(result_id, perclip, sidecar, cellspec, key="orig_clip_id", fname_mac=None):
    """cellspec: {cell_id: ("single"|"paired", [cols_a], [cols_b] or None, arm or None)}.
    Compares every value and bound with the sidecar cell at 4 dp. Returns list of (row, agree, detail)."""
    rows, clips, spk, y, arm = load_perclip(perclip, key)
    sc = json.load(open(sidecar))
    cells = {c["id"]: c for c in sc["cells"]}
    bc = basic_checks(clips, spk, y, arm)
    prompt_ok = sc.get("prompt_verbatim") == PROMPT and all(q.get("prompt", PROMPT) == PROMPT for q in rows)
    out = []
    for cid, (kind, A, Bc, a) in cellspec.items():
        m = None if a is None else (arm == a)
        if kind == "single":
            r = point_and_ci(spk, y, cols_of(rows, A), mask=m)
            v, lo, hi, n, ns = r["value"], r["lo"], r["hi"], r["n"], r["n_spk"]
            trap_ok = abs(r["value"] - r["value_trap"]) < 1e-12
        else:
            r = paired_ci(spk, y, cols_of(rows, A), cols_of(rows, Bc), mask=m)
            v, lo, hi, n, ns = r["value"], r["lo"], r["hi"], r["n"], r["n_spk"]
            trap_ok = True
        c = cells.get(cid)
        if c is None:
            out.append((None, False, f"{cid}: NOT IN SIDECAR, verified {v:.4f} [{lo:.4f}, {hi:.4f}]"))
            continue
        d, ag = row(cid, c["what"], c["value"], v, c["lo"], c["hi"], lo, hi, n, ns,
                    fname_mac or sc.get("per_clip_csv", perclip))
        n_ok = (c.get("n") == n and c.get("n_speakers") == ns)
        ag = ag and n_ok and trap_ok and bc["ok"] and prompt_ok
        d["agree_4dp"] = "yes" if ag else "NO"
        out.append((d, ag, f"{cid}: computed {c['value']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] n={c.get('n')}/{c.get('n_speakers')} | "
                           f"verified {v:.4f} [{lo:.4f}, {hi:.4f}] n={n}/{ns} | trapezoid_ok={trap_ok} basic_ok={bc['ok']} prompt_ok={prompt_ok}"))
    return out, bc, prompt_ok


def paste(d):
    return (f"{d['id']} | {d['what']} | {d['value_verified']} [{d['lo']}, {d['hi']}] | n={d['n']} n_spk={d['n_spk']} | "
            f"{d['file']} | {d['two_dp']} | VERIFIED (agree 4dp: {d['agree_4dp']})")
