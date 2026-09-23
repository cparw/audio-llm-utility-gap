#!/usr/bin/env python3
"""POD4 independent verifier (part 20). Written from scratch; does not import or
exec any compute-job script.

AUC: explicit Mann-Whitney over every positive x negative pair, tie = 0.5.
Cross-check: trapezoid area under the ROC built from every distinct threshold.
Intervals: 2000 draws, numpy.random.default_rng(0) created fresh per cell,
speakers (np.unique order) resampled with replacement via rng.choice,
2.5/97.5 percentiles (numpy linear). A draw needs both classes.
Paired: one speaker draw per replicate, both AUCs inside that draw, diff = a - b.
Arms: clip is in 'conflict' or 'agreement' by its name prefix (or a set/arm column);
per-arm cells subset first, then resample the speakers of that subset.
"""
import csv, json, sys
import numpy as np

NB = 2000

def mw_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # explicit all-pairs comparison (chunked to bound memory)
    tot = 0.0
    for i in range(0, len(pos), 256):
        p = pos[i:i+256][:, None]
        tot += (p > neg[None, :]).sum() + 0.5 * (p == neg[None, :]).sum()
    return float(tot / (len(pos) * len(neg)))

def trap_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    P = (y == 1).sum(); N = (y == 0).sum()
    th = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in th:
        tpr.append(((s >= t) & (y == 1)).sum() / P)
        fpr.append(((s >= t) & (y == 0)).sum() / N)
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))

def fast_auc(y, s):
    # used inside bootstrap only; checked equal to mw_auc on the point estimate
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    _, first, counts = np.unique(ss, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def boot(spk, y, s, nb=NB, mask=None):
    """mask=None: resample the speakers present. mask given (convention B): resample
    ALL speakers of the full cohort, then keep only rows inside mask in that draw."""
    rng = np.random.default_rng(0)
    u = np.unique(spk)
    idx = {sp: np.where(spk == sp)[0] for sp in u}
    v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if mask is not None: ii = ii[mask[ii]]
        a = fast_auc(y[ii], s[ii])
        if not np.isnan(a): v.append(a)
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def boot_paired(spk, y, sa, sb, nb=NB, mask=None):
    """diff = AUC(sa) - AUC(sb)"""
    rng = np.random.default_rng(0)
    u = np.unique(spk)
    idx = {sp: np.where(spk == sp)[0] for sp in u}
    v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if mask is not None: ii = ii[mask[ii]]
        a = fast_auc(y[ii], sa[ii]); b = fast_auc(y[ii], sb[ii])
        if np.isnan(a) or np.isnan(b): continue
        v.append(a - b)
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def arm_of(row):
    for k in ("arm", "set", "subset"):
        if k in row and row[k] in ("conflict", "agreement"):
            return row[k]
    c = row["clip"]
    return "conflict" if c.startswith("conflict_") else ("agreement" if c.startswith("agreement_") else "?")

def load(path, col="p_yes"):
    rows = list(csv.DictReader(open(path)))
    d = {}
    for r in rows:
        d[r["clip"]] = dict(spk=r["speaker"], y=int(float(r["label"])), s=float(r[col]), arm=arm_of(r),
                            prompt=r.get("prompt"))
    return d, rows

def arrays(d, keys):
    return (np.array([d[k]["spk"] for k in keys]), np.array([d[k]["y"] for k in keys]),
            np.array([d[k]["s"] for k in keys]))

def cell(d, arm=None):
    keys = sorted(k for k in d if arm is None or d[k]["arm"] == arm)
    spk, y, s = arrays(d, keys)
    a = mw_auc(y, s); t = trap_auc(y, s); f = fast_auc(y, s)
    assert abs(a - t) < 1e-9 and abs(a - f) < 1e-9, (a, t, f)
    lo, hi, nu = boot(spk, y, s)
    r = dict(value=a, trap=t, lo=lo, hi=hi, n=len(keys), n_spk=int(len(np.unique(spk))),
             n_pos=int(y.sum()), draws=nu)
    if arm is not None:  # convention B: whole-cohort speaker draw, arm subset inside the draw
        allk = sorted(d); S, Y, SS = arrays(d, allk)
        m = np.array([d[k]["arm"] == arm for k in allk])
        r["lo_B"], r["hi_B"], r["draws_B"] = boot(S, Y, SS, mask=m)
        r["n_spk_cohort"] = int(len(np.unique(S)))
    return r

def paired(da, db, arm=None):
    """a minus b over the shared clips"""
    keys = sorted(k for k in set(da) & set(db) if arm is None or da[k]["arm"] == arm)
    for k in keys:
        assert da[k]["y"] == db[k]["y"] and da[k]["spk"] == db[k]["spk"], k
        assert arm is None or db[k]["arm"] == arm
    spk, y, sa = arrays(da, keys); _, _, sb = arrays(db, keys)
    a = mw_auc(y, sa); b = mw_auc(y, sb)
    lo, hi, nu = boot_paired(spk, y, sa, sb)
    r = dict(value=a - b, a=a, b=b, lo=lo, hi=hi, n=len(keys), n_spk=int(len(np.unique(spk))), draws=nu)
    if arm is not None:
        allk = sorted(set(da) & set(db)); S, Y, SA = arrays(da, allk); _, _, SB = arrays(db, allk)
        m = np.array([da[k]["arm"] == arm for k in allk])
        r["lo_B"], r["hi_B"], r["draws_B"] = boot_paired(S, Y, SA, SB, mask=m)
    return r

if __name__ == "__main__" and sys.argv[1] in ("cells", "paired"):
    # usage: POD4_verify.py cells FILE [col]        -> overall/conflict/agreement
    #        POD4_verify.py paired FILE_A FILE_B [colA colB]  -> A minus B per arm
    mode = sys.argv[1]
    out = {}
    if mode == "cells":
        col = sys.argv[3] if len(sys.argv) > 3 else "p_yes"
        d, rows = load(sys.argv[2], col)
        prompts = sorted({r.get("prompt") for r in rows if r.get("prompt") is not None})
        out["prompts"] = prompts
        for arm in (None, "conflict", "agreement"):
            out[arm or "overall"] = cell(d, arm)
    elif mode == "paired":
        ca = sys.argv[4] if len(sys.argv) > 4 else "p_yes"
        cb = sys.argv[5] if len(sys.argv) > 5 else "p_yes"
        da, _ = load(sys.argv[2], ca); db, _ = load(sys.argv[3], cb)
        for arm in (None, "conflict", "agreement"):
            out[arm or "overall"] = paired(da, db, arm)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- result mode
# POD4_verify.py result SIDECAR LOCALDIR OUT_ROWS_JSON
# Recomputes every cell and paired value a sidecar lists, from the per-clip file it
# names. Files under the Mac POD4 folder are read from LOCALDIR (copied off the pod);
# any other path is read where it is.
import os
ARMS = {"conflict": 146, "agreement": 322}
RECORDING = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
VOICE = "Based only on how this person's voice sounds, does this speaker show signs of dementia? Answer with one word, Yes or No."
MAC_POD4 = "scores/part20/POD4"

GDRIVE_LOCAL = {}  # G-Drive path -> local byte-identical copy (sha256 checked before use)

def local(p, localdir):
    p = p.strip()
    if p in GDRIVE_LOCAL: return GDRIVE_LOCAL[p]
    if p.startswith(MAC_POD4) and localdir:
        q = os.path.join(localdir, os.path.basename(p))
        if os.path.exists(q): return q
    return p

def result_mode(sidecar, localdir, outp):
    sc = json.load(open(sidecar))
    rows, notes = [], []
    cache = {}
    def get(path, col):
        k = (path, col)
        if k not in cache: cache[k] = load(local(path, localdir), col)
        return cache[k]
    for c in sc.get("cells", []):
        arm = None if c["arm"] == "overall" else c["arm"]
        d, raw = get(c["file"], c.get("col", "p_yes"))
        v = cell(d, arm)
        # structural checks
        chk = []
        if v["n"] != c["n"]: chk.append(f"n {v['n']} vs {c['n']}")
        if v["n_spk"] != c["n_spk"]: chk.append(f"n_spk {v['n_spk']} vs {c['n_spk']}")
        if "n_pos" in c and v["n_pos"] != c["n_pos"]: chk.append(f"n_pos {v['n_pos']} vs {c['n_pos']}")
        if arm and v["n"] != ARMS[arm]: chk.append(f"arm count {v['n']} != {ARMS[arm]}")
        if not arm and v["n"] != 468: chk.append(f"overall n {v['n']} != 468")
        pr = {r.get("prompt") for r in raw}
        if pr != {None} and len(pr) != 1: chk.append(f"prompts not unique in csv: {pr}")
        rows.append(dict(id=c["id"], what=c["what"], kind="auc",
                         comp=dict(value=c["value"], lo=c["lo"], hi=c["hi"]),
                         ver=dict(value=v["value"], lo=v["lo"], hi=v["hi"]),
                         n=v["n"], n_spk=v["n_spk"], file=c["file"], checks=chk,
                         trap=v["trap"], csv_prompts=sorted(p for p in pr if p)))
    for p in sc.get("paired", []):
        arm = None if p["arm"] == "overall" else p["arm"]
        fa, fb = [x.strip() for x in p["file"].split(" MINUS ")]
        da, _ = get(fa, p.get("col_a", "p_yes")); db, _ = get(fb, p.get("col_b", "p_yes"))
        v = paired(da, db, arm)
        chk = []
        if v["n"] != p["n"]: chk.append(f"n {v['n']} vs {p['n']}")
        if v["n_spk"] != p["n_spk"]: chk.append(f"n_spk {v['n_spk']} vs {p['n_spk']}")
        if "a_auc" in p and f"{v['a']:.4f}" != f"{p['a_auc']:.4f}": chk.append(f"a_auc {v['a']:.4f} vs {p['a_auc']}")
        if "b_auc" in p and f"{v['b']:.4f}" != f"{p['b_auc']:.4f}": chk.append(f"b_auc {v['b']:.4f} vs {p['b_auc']}")
        rows.append(dict(id=p["id"], what=p["what"], kind="diff",
                         comp=dict(value=p["value"], lo=p["lo"], hi=p["hi"]),
                         ver=dict(value=v["value"], lo=v["lo"], hi=v["hi"]),
                         n=v["n"], n_spk=v["n_spk"], file=p["file"], checks=chk,
                         a=v["a"], b=v["b"]))
    json.dump(rows, open(outp, "w"), indent=1)
    for r in rows:
        print(f"{r['id']:70s} comp {r['comp']['value']:+.4f} [{r['comp']['lo']:+.4f},{r['comp']['hi']:+.4f}]  "
              f"ver {r['ver']['value']:+.4f} [{r['ver']['lo']:+.4f},{r['ver']['hi']:+.4f}]  checks={r['checks'] or 'ok'}")

if __name__ == "__main__" and sys.argv[1] == "result":
    import hashlib
    for g, l in (("<local data dir>/paper work/paper1_local_runs/probe2/pitt_zeroshot_scores.csv",
                  os.path.join(sys.argv[3], "paper_q2a_pitt_zeroshot_scores.csv")),):
        if os.path.exists(l) and hashlib.sha256(open(l, "rb").read()).hexdigest() == \
                "9cd9800ed47adee9d9912219bf31db487d4941128364d80babf3d0488fbfff6c":
            GDRIVE_LOCAL[g] = l
    result_mode(sys.argv[2], sys.argv[3], sys.argv[4])
