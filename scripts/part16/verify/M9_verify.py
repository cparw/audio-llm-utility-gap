#!/usr/bin/env python3
"""
INDEPENDENT VERIFIER for PART16 M9 (answer mass).
Written from scratch. Does not import, exec or read the primary job's M9_answer_mass.py.
Goes back to master_lookup.csv and the ORIGINAL per-clip score csv files.

Routes deliberately different from the primary job:
  - median / percentile via numpy (they used statistics.median in their check)
  - AUC via Mann-Whitney U counted on a merge-sort over the sorted union, ties = 0.5,
    cross-checked against a trapezoid integration of the full ROC.
  - bootstrap with numpy default_rng(0), speakers resampled with replacement.
"""
import csv, json, os, sys, math, hashlib
from collections import defaultdict
import numpy as np

REL = "<local data dir>/Desktop/release"
PART16 = os.path.join(REL, "edaic_rerun/part16")
MASTER = os.path.join(REL, "master/master_lookup.csv")

MASS_NAMES = ["mass", "answer_mass", "answer_mass_total", "mass_yes_no", "yes_no_mass"]
SPK_NAMES  = ["speaker", "speaker_id", "spk", "subject", "subject_id", "participant"]
PYES_NAMES = ["p_yes", "prob_yes", "pyes"]
LAB_NAMES  = ["label", "y", "target", "gt"]

# ---------------- AUC: route 1, Mann-Whitney U by counting concordant pairs ----------
def auc_mannwhitney(scores, labels):
    """U / (n_pos*n_neg); ties contribute 0.5. O(n log n) by walking the sorted union."""
    s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=int)
    pos = np.sort(s[y == 1]); neg = np.sort(s[y == 0])
    npos, nneg = len(pos), len(neg)
    if npos == 0 or nneg == 0:
        return float("nan")
    # for every positive, count negatives strictly below + 0.5 * negatives equal
    lo = np.searchsorted(neg, pos, side="left")    # negs strictly < pos
    hi = np.searchsorted(neg, pos, side="right")   # negs <= pos
    U = lo.sum() + 0.5 * (hi - lo).sum()
    return float(U) / (npos * nneg)

# ---------------- AUC: route 2, trapezoid over the full ROC --------------------------
def auc_trapezoid(scores, labels):
    s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=int)
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]; y_sorted = y[order]
    tps = np.cumsum(y_sorted == 1).astype(float)
    fps = np.cumsum(y_sorted == 0).astype(float)
    # keep one point per distinct threshold so ties become a diagonal segment
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(s_sorted) - 1]
    tpr = np.r_[0.0, tps[idx] / npos]
    fpr = np.r_[0.0, fps[idx] / nneg]
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))

# ---------------- csv reading -------------------------------------------------------
def pick(header, names):
    low = {h.strip().lower(): h for h in header}
    for n in names:
        if n in low:
            return low[n]
    return None

def read_perclip(path):
    """Return dict with header, mass list, speaker list, p_yes list, label list."""
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.DictReader(fh)
            header = rd.fieldnames or []
            mcol = pick(header, MASS_NAMES)
            scol = pick(header, SPK_NAMES)
            pcol = pick(header, PYES_NAMES)
            lcol = pick(header, LAB_NAMES)
            mass, spk, pys, lab, nrows = [], [], [], [], 0
            for r in rd:
                nrows += 1
                if mcol is not None:
                    v = r.get(mcol)
                    try: mass.append(float(v))
                    except (TypeError, ValueError): mass.append(float("nan"))
                spk.append((r.get(scol) or "").strip() if scol else "")
                if pcol is not None:
                    try: pys.append(float(r.get(pcol)))
                    except (TypeError, ValueError): pys.append(float("nan"))
                if lcol is not None:
                    try: lab.append(int(float(r.get(lcol))))
                    except (TypeError, ValueError): lab.append(-1)
    except Exception as e:
        return {"error": repr(e), "header": [], "n_rows": 0}
    return {"header": header, "mass_col": mcol, "spk_col": scol, "pyes_col": pcol,
            "label_col": lcol, "mass": mass, "spk": spk, "p_yes": pys,
            "label": lab, "n_rows": nrows, "error": None}

def summarize_mass(mass, spk, seed=0, draws=2000):
    m = np.asarray([x for x in mass], dtype=float)
    ok = ~np.isnan(m)
    m = m[ok]
    sp = [s for s, o in zip(spk, ok) if o]
    n = len(m)
    if n == 0:
        return None
    out = {
        "n": n,
        "n_speakers": len(set(s for s in sp if s != "")) if any(s != "" for s in sp) else n,
        "median": float(np.median(m)),
        "p10": float(np.percentile(m, 10)),
        "min": float(np.min(m)),
        "share_lt_001": float(np.mean(m < 0.01)),
        "share_lt_015": float(np.mean(m < 0.15)),
    }
    # speaker bootstrap of the median
    by = defaultdict(list)
    for s, v in zip(sp, m):
        by[s if s != "" else object()].append(v)
    def _sk(k):
        ks = str(k)
        try:
            return (0, float(ks), "")
        except ValueError:
            return (1, 0.0, ks)
    keys = sorted(by.keys(), key=_sk)  # numeric-aware speaker order (pandas-style type inference)
    arrs = [np.asarray(by[k], dtype=float) for k in keys]
    rng = np.random.default_rng(seed)
    K = len(keys)
    reps = np.empty(draws, dtype=float); usable = 0
    for i in range(draws):
        pickidx = rng.integers(0, K, size=K)
        pool = np.concatenate([arrs[j] for j in pickidx])
        if pool.size:
            reps[usable] = np.median(pool); usable += 1
    reps = reps[:usable]
    out["lo"] = float(np.percentile(reps, 2.5))
    out["hi"] = float(np.percentile(reps, 97.5))
    out["boot_usable"] = usable
    return out

# ---------------- resolve a master_lookup source to a per-clip csv -------------------
def resolve_csv(src):
    if src.lower().endswith(".csv") and os.path.exists(src):
        return src, "direct"
    if src.lower().endswith(".json"):
        stem = src[:-5]
        for suf in (".csv", "_scores.csv", "_perclip.csv", "_clips.csv"):
            cand = stem + suf
            if os.path.exists(cand):
                return cand, "json->sibling" + suf
        # zeroshot json -> *_zeroshot_scores.csv style already covered; try dropping _zeroshot
        if stem.endswith("_zeroshot"):
            for suf in (".csv", "_scores.csv"):
                cand = stem[:-len("_zeroshot")] + suf
                if os.path.exists(cand):
                    return cand, "json->stem-minus-zeroshot" + suf
        # json may name a per-clip file
        try:
            d = json.load(open(src))
            for k in ("per_clip", "per_clip_csv", "csv", "scores_csv", "out_csv", "clip_csv"):
                v = d.get(k) if isinstance(d, dict) else None
                if isinstance(v, str) and os.path.exists(v):
                    return v, "json field " + k
        except Exception:
            pass
    return None, "unresolved"

def main():
    results = {}
    master = list(csv.DictReader(open(MASTER)))
    results["master_lookup_rows"] = len(master)
    ans_cells = [r for r in master if r["stream"] in ("zero shot answer", "transcript only")]
    results["n_answer_stream_cells"] = len(ans_cells)

    cells = []
    for r in ans_cells:
        path, how = resolve_csv(r["source"])
        rec = {"model": r["model"], "dataset": r["dataset"], "stream": r["stream"],
               "master_auc": r["auc"], "master_n": r["n"], "source": r["source"],
               "resolved": path, "how": how}
        if path is None:
            rec["status"] = "UNRESOLVED"
            cells.append(rec); continue
        d = read_perclip(path)
        if d["error"] or d.get("mass_col") is None:
            rec["status"] = "NO_MASS_COL"
            rec["header"] = d.get("header")
            rec["n_rows"] = d.get("n_rows")
            cells.append(rec); continue
        s = summarize_mass(d["mass"], d["spk"])
        rec.update(s or {})
        rec["mass_col"] = d["mass_col"]
        rec["n_rows_file"] = d["n_rows"]
        rec["status"] = "OK"
        # independent AUC from p_yes where we have labels
        if d.get("pyes_col") and d.get("label_col"):
            y = np.asarray(d["label"]); p = np.asarray(d["p_yes"], dtype=float)
            keep = (~np.isnan(p)) & ((y == 0) | (y == 1))
            if keep.sum() and len(set(y[keep].tolist())) == 2:
                rec["auc_mw"] = auc_mannwhitney(p[keep], y[keep])
                rec["auc_trap"] = auc_trapezoid(p[keep], y[keep])
        cells.append(rec)
    results["cells"] = cells
    ok = [c for c in cells if c.get("status") == "OK"]
    results["n_cells_with_mass"] = len(ok)
    if ok:
        worst = min(ok, key=lambda c: c["median"])
        results["min_median_cell"] = {k: worst.get(k) for k in
            ("model","dataset","stream","median","lo","hi","n","n_speakers","resolved","boot_usable")}
        results["cells_median_below_015"] = sum(1 for c in ok if c["median"] < 0.15)
        results["total_clips_in_cells"] = sum(c["n"] for c in ok)
        results["clips_below_015_in_cells"] = sum(int(round(c["share_lt_015"] * c["n"])) for c in ok)
        results["clips_below_001_in_cells"] = sum(int(round(c["share_lt_001"] * c["n"])) for c in ok)
    return results

if __name__ == "__main__":
    out = main()
    json.dump(out, open(os.path.join(PART16, "verify/M9_verify_cells.json"), "w"), indent=1)
    ok = [c for c in out["cells"] if c.get("status") == "OK"]
    print("master rows", out["master_lookup_rows"], "answer-stream cells", out["n_answer_stream_cells"])
    print("cells with mass", out["n_cells_with_mass"])
    print("MIN MEDIAN CELL:", json.dumps(out.get("min_median_cell"), indent=1))
    print("cells with median<0.15:", out["cells_median_below_015"])
    print("clips in cells:", out["total_clips_in_cells"],
          "below .15:", out["clips_below_015_in_cells"],
          "below .01:", out["clips_below_001_in_cells"])
    bad = [c for c in out["cells"] if c.get("status") != "OK"]
    print("non-OK cells:", len(bad))
    for c in bad[:20]:
        print("  ", c["status"], c["model"], c["dataset"], c["stream"], c["source"])

# ============================ DISK-WIDE SWEEP (independent) =========================
def sweep():
    roots = []
    base = "<local data dir>/Desktop/release"
    for d in ("overnight","overnight2","omni_final","omni","edaic_rerun","edaic_conflict",
              "edaic_conflict_new","master","podD2_final","podE_final","part4_text"):
        p = os.path.join(base, d)
        if os.path.isdir(p): roots.append(p)
    gp = "<local data dir>/paper1_local_runs"
    if os.path.isdir(gp): roots.append(gp)
    allcsv = []
    for r in roots:
        for dp, dn, fn in os.walk(r):
            for f in fn:
                if f.lower().endswith(".csv"):
                    allcsv.append(os.path.join(dp, f))
    allcsv = sorted(set(allcsv))
    files_with_mass = []; stubs = []; total_clips = 0
    below015 = 0; below001 = 0
    global_min = (float("inf"), None, None)
    per_file = []
    for p in allcsv:
        try:
            with open(p, newline="", encoding="utf-8", errors="replace") as fh:
                rd = csv.DictReader(fh)
                hdr = rd.fieldnames or []
                mcol = pick(hdr, MASS_NAMES)
                if mcol is None: continue
                scol = pick(hdr, SPK_NAMES)
                vals = []; spk = []
                for row in rd:
                    try: v = float(row[mcol])
                    except (TypeError, ValueError): continue
                    vals.append(v); spk.append((row.get(scol) or "").strip() if scol else "")
        except Exception:
            continue
        files_with_mass.append(p)
        if not vals:
            stubs.append(p); continue
        a = np.asarray(vals)
        per_file.append({"file": p, "n": len(a), "n_speakers": len(set(s for s in spk if s)) or len(a),
                         "median": float(np.median(a)), "min": float(a.min()),
                         "mass_col": mcol})
        total_clips += len(a)
        below015 += int((a < 0.15).sum()); below001 += int((a < 0.01).sum())
        if a.min() < global_min[0]:
            global_min = (float(a.min()), p, mcol)
    worst = min(per_file, key=lambda d: d["median"]) if per_file else None
    return {"n_csv_found": len(allcsv), "n_files_with_mass_col": len(files_with_mass),
            "n_files_with_data": len(per_file), "n_header_only_stubs": len(stubs),
            "stubs": stubs, "total_clips": total_clips,
            "clips_below_015": below015, "clips_below_001": below001,
            "global_min_clip": global_min, "worst_median_file": worst,
            "per_file": per_file, "roots": roots}
