#!/usr/bin/env python3
"""
INDEPENDENT VERIFIER for PART 16 task M4.
Written from scratch. Does not read, import or exec part16/M4_yes_rates.py or
part16/M4_verify.py, and does not consume any of their intermediate files.
Goes back to the ORIGINAL part14 per-clip csvs and the part14 manifest.

Point estimates: pure-python (csv module + fractions/float arithmetic), no pandas.
Bootstrap: numpy.random.default_rng(0), speakers resampled with replacement,
2000 draws, percentile 2.5/97.5.  Several plausible draw conventions are tried so
that an implementation-detail difference can be told apart from a real disagreement.
"""
import csv, os, sys, json, hashlib, statistics
import numpy as np

BASE = "<local data dir>/Desktop/release/edaic_rerun"
MANIFEST = os.path.join(BASE, "part14_manifest_new.csv")
P14 = os.path.join(BASE, "part14")

SOURCES = [
    # name, file, clip column, p_yes column, mass column
    ("Qwen2.5-Omni",       "p14_o25_zeroshot_scores.csv",  "clip",      "p_yes", "mass"),
    ("Qwen2-Audio",        "p14_q2a_zeroshot_scores.csv",  "clip",      "p_yes", "mass"),
    ("Qwen3-Omni-30B-A3B", "p14_q3o_zeroshot_scores.csv",  "clip",      "p_yes", "mass"),
    ("AudioFlamingo2",     "p14_af2.csv",                  "clip_path", "p_yes", "answer_mass"),
    ("AudioFlamingo3",     "p14_af3.csv",                  "clip_path", "p_yes", "answer_mass"),
    ("Kimi-Audio",         "p14_kimi_zeroshot_scores.csv", "clip",      "p_yes", "mass"),
]

def sha_first_mb(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read(1024 * 1024)).hexdigest()

def parse_clip(name):
    """conflict_301_c000010.wav -> ('conflict', '301', 'c000010')"""
    b = os.path.basename(name)
    if b.endswith(".wav"):
        b = b[:-4]
    bits = b.split("_")
    if len(bits) != 3:
        raise ValueError("unexpected clip name %r" % name)
    return bits[0], bits[1], bits[2]

def med(xs):
    """median, pure python, average of the two middle values when n is even"""
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def r4(x):
    return "%.4f" % (x + 0.0)

out = {"checks": [], "problems": []}
def note(msg):
    print(msg)
    out["checks"].append(msg)
def problem(msg):
    print("!! " + msg)
    out["problems"].append(msg)

# ---------------------------------------------------------------- manifest
if not os.path.exists(MANIFEST):
    problem("MANIFEST MISSING: %s" % MANIFEST)
    sys.exit(1)
man = []
with open(MANIFEST, newline="") as f:
    for row in csv.DictReader(f):
        man.append(row)
note("manifest rows=%d sha256_first_1mb=%s" % (len(man), sha_first_mb(MANIFEST)))
man_arm  = [r["set"] for r in man]
man_spk  = [r["speaker_id"] for r in man]
man_lab  = [int(r["label"]) for r in man]
man_uid  = [r["seg_uid"] for r in man]
man_pair = [r["pair_id"] for r in man]
note("manifest: n_conflict=%d n_agreement=%d n_pairs=%d n_speakers=%d"
     % (man_arm.count("conflict"), man_arm.count("agreement"),
        len(set(man_pair)), len(set(man_spk))))
uniq_agree_uid = len({u for u, a in zip(man_uid, man_arm) if a == "agreement"})
uniq_conf_uid  = len({u for u, a in zip(man_uid, man_arm) if a == "conflict"})
note("manifest: unique agreement seg_uid=%d unique conflict seg_uid=%d"
     % (uniq_agree_uid, uniq_conf_uid))

# ---------------------------------------------------------------- load models
data = {}     # model -> list of dicts in file order
for name, fn, clipcol, pcol, mcol in SOURCES:
    path = os.path.join(P14, fn)
    if not os.path.exists(path):
        problem("SOURCE MISSING for %s: %s" % (name, path))
        continue
    rows = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        for i, row in enumerate(rd):
            arm, spk, uid = parse_clip(row[clipcol])
            rows.append({
                "i": i, "arm": arm, "spk": spk, "uid": uid,
                "p": float(row[pcol]), "m": float(row[mcol]),
                "lab": int(row["label"]),
            })
    data[name] = rows
    note("%-18s rows=%d cols=%s sha256_first_1mb=%s"
         % (name, len(rows), ",".join(cols), sha_first_mb(path)))
    if len(rows) != 966:
        problem("%s has %d rows, not 966" % (name, len(rows)))
    # row alignment against manifest (independent path: filename vs manifest)
    bad_arm = sum(1 for r, a in zip(rows, man_arm) if r["arm"] != a)
    bad_spk = sum(1 for r, s in zip(rows, man_spk) if r["spk"] != s)
    bad_uid = sum(1 for r, u in zip(rows, man_uid) if r["uid"] != u)
    bad_lab = sum(1 for r, l in zip(rows, man_lab) if r["lab"] != l)
    note("   align vs manifest: arm_mismatch=%d spk_mismatch=%d uid_mismatch=%d label_mismatch=%d"
         % (bad_arm, bad_spk, bad_uid, bad_lab))
    if bad_arm or bad_spk or bad_uid or bad_lab:
        problem("%s is NOT row-aligned with the manifest" % name)
    # p_yes range sanity
    ps = [r["p"] for r in rows]
    if min(ps) < 0.0 or max(ps) > 1.0:
        problem("%s p_yes outside [0,1]" % name)
    note("   p_yes min=%.6f max=%.6f ; mass min=%.6f max=%.6f"
         % (min(ps), max(ps), min(r["m"] for r in rows), max(r["m"] for r in rows)))

# ---------------------------------------------------------------- duplicates
# their LOW discrepancy: agreement arm has 483 rows but 311 unique seg_uids,
# duplicates carry identical p_yes.
for name, rows in data.items():
    seen = {}
    conflicting = 0
    for r in rows:
        if r["arm"] != "agreement":
            continue
        k = r["uid"]
        if k in seen:
            if abs(seen[k] - r["p"]) > 1e-12:
                conflicting += 1
        else:
            seen[k] = r["p"]
    note("%-18s agreement unique seg_uid=%d duplicate rows with differing p_yes=%d"
         % (name, len(seen), conflicting))

# ---------------------------------------------------------------- point estimates
def cell(rows, arm):
    sub = [r for r in rows if r["arm"] == arm]
    yes = sum(1 for r in sub if r["p"] > 0.5)
    n = len(sub)
    ps = [r["p"] for r in sub]
    return {
        "n": n, "n_spk": len({r["spk"] for r in sub}),
        "yes": yes, "rate": yes / float(n),
        "med_mass": med([r["m"] for r in sub]),
        "sd_sample": statistics.stdev(ps),
        "sd_pop": statistics.pstdev(ps),
    }

pts = {}
print("\n=== POINT ESTIMATES (pure python) ===")
for name, rows in data.items():
    for arm in ("conflict", "agreement"):
        c = cell(rows, arm)
        pts[(name, arm)] = c
        print("%-18s %-9s n=%d n_spk=%d yes=%d rate=%s med_mass=%s sd_samp=%s sd_pop=%s"
              % (name, arm, c["n"], c["n_spk"], c["yes"], r4(c["rate"]),
                 r4(c["med_mass"]), r4(c["sd_sample"]), r4(c["sd_pop"])))
    # overall median mass over all 966 (their AF2 cross-check)
    print("   %-15s overall median mass over 966 = %s" % (name, r4(med([r["m"] for r in rows]))))

# ---------------------------------------------------------------- bootstrap
spk_sorted_int = sorted({int(s) for s in man_spk})
spk_sorted_str = sorted({s for s in man_spk})
spk_firstseen  = list(dict.fromkeys(man_spk))

def build_index(rows, arm, spk_order):
    """map speaker -> numpy array of 0/1 yes flags for that speaker in that arm"""
    d = {s: [] for s in spk_order}
    for r in rows:
        if r["arm"] != arm:
            continue
        d[r["spk"]].append(1 if r["p"] > 0.5 else 0)
    return [np.array(d[s], dtype=np.int64) for s in spk_order]

def boot_cell(rows, arm, spk_order, draw_kind, stat_kind, seed=0, B=2000):
    per = build_index(rows, arm, [str(s) for s in spk_order])
    cnt = np.array([a.sum() for a in per], dtype=np.float64)
    tot = np.array([a.size for a in per], dtype=np.float64)
    ns = len(per)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        if draw_kind == "integers":
            idx = rng.integers(0, ns, ns)
        elif draw_kind == "choice":
            idx = rng.choice(ns, size=ns, replace=True)
        elif draw_kind == "choice_vals":
            picked = rng.choice(np.asarray(spk_order), size=ns, replace=True)
            pos = {s: i for i, s in enumerate(spk_order)}
            idx = np.array([pos[p] for p in picked])
        if stat_kind == "pooled":
            t = tot[idx].sum()
            vals[b] = cnt[idx].sum() / t if t > 0 else np.nan
        else:  # mean of speaker rates
            with np.errstate(invalid="ignore"):
                vals[b] = np.nanmean(np.where(tot[idx] > 0, cnt[idx] / np.maximum(tot[idx], 1), np.nan))
    ok = vals[~np.isnan(vals)]
    return float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)), len(ok)

def boot_diff(rows, spk_order, draw_kind, stat_kind, seed=0, B=2000):
    perc = build_index(rows, "conflict",  [str(s) for s in spk_order])
    pera = build_index(rows, "agreement", [str(s) for s in spk_order])
    cc = np.array([a.sum() for a in perc], float); tc = np.array([a.size for a in perc], float)
    ca = np.array([a.sum() for a in pera], float); ta = np.array([a.size for a in pera], float)
    ns = len(perc)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        if draw_kind == "integers":
            idx = rng.integers(0, ns, ns)
        elif draw_kind == "choice":
            idx = rng.choice(ns, size=ns, replace=True)
        elif draw_kind == "choice_vals":
            picked = rng.choice(np.asarray(spk_order), size=ns, replace=True)
            pos = {s: i for i, s in enumerate(spk_order)}
            idx = np.array([pos[p] for p in picked])
        if stat_kind == "pooled":
            t1, t2 = tc[idx].sum(), ta[idx].sum()
            vals[b] = (cc[idx].sum()/t1) - (ca[idx].sum()/t2) if t1 > 0 and t2 > 0 else np.nan
        else:
            r1 = np.nanmean(np.where(tc[idx] > 0, cc[idx]/np.maximum(tc[idx],1), np.nan))
            r2 = np.nanmean(np.where(ta[idx] > 0, ca[idx]/np.maximum(ta[idx],1), np.nan))
            vals[b] = r1 - r2
    ok = vals[~np.isnan(vals)]
    return float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)), len(ok)

# their reported intervals, read from the csv they wrote (targets only, not inputs)
THEIRS = {}
with open(os.path.join(BASE, "part16", "M4_yes_rates.csv"), newline="") as f:
    for row in csv.DictReader(f):
        THEIRS[(row["model"], row["arm"])] = (float(row["lo"]), float(row["hi"]))
THEIRS_D = {}
with open(os.path.join(BASE, "part16", "M4_yes_rate_diffs.csv"), newline="") as f:
    for row in csv.DictReader(f):
        THEIRS_D[row["model"]] = (float(row["diff"]), float(row["lo"]), float(row["hi"]))

print("\n=== BOOTSTRAP VARIANT SEARCH (Qwen2.5-Omni conflict, target %s) ==="
      % (THEIRS[("Qwen2.5-Omni", "conflict")],))
probe = data["Qwen2.5-Omni"]
best = None
for order_name, order in (("sorted_int", spk_sorted_int),
                          ("sorted_str", spk_sorted_str),
                          ("first_seen", spk_firstseen)):
    for dk in ("integers", "choice", "choice_vals"):
        for sk in ("pooled", "spk_mean"):
            lo, hi, nok = boot_cell(probe, "conflict", order, dk, sk)
            tag = "%s/%s/%s" % (order_name, dk, sk)
            hit = (r4(lo) == r4(THEIRS[("Qwen2.5-Omni","conflict")][0])
                   and r4(hi) == r4(THEIRS[("Qwen2.5-Omni","conflict")][1]))
            print("  %-34s lo=%s hi=%s usable=%d %s" % (tag, r4(lo), r4(hi), nok, "  <== MATCH" if hit else ""))
            if hit and best is None:
                best = (order_name, order, dk, sk)

if best is None:
    print("\n  no variant reproduced their interval on the probe cell; "
          "falling back to sorted_int/integers/pooled as my own convention")
    best = ("sorted_int", spk_sorted_int, "integers", "pooled")
order_name, order, dk, sk = best
print("\nUsing bootstrap convention: %s / %s / %s" % (order_name, dk, sk))

print("\n=== FULL RECOMPUTE ===")
results = []
for name, rows in data.items():
    for arm in ("conflict", "agreement"):
        c = pts[(name, arm)]
        lo, hi, nok = boot_cell(rows, arm, order, dk, sk)
        tl, th = THEIRS[(name, arm)]
        agree = (r4(lo) == r4(tl)) and (r4(hi) == r4(th))
        print("%-18s %-9s rate=%s [%s, %s] usable=%d | theirs [%s, %s] %s"
              % (name, arm, r4(c["rate"]), r4(lo), r4(hi), nok, r4(tl), r4(th),
                 "OK" if agree else "DIFFER"))
        results.append({"model": name, "arm": arm, "rate": r4(c["rate"]),
                        "lo": r4(lo), "hi": r4(hi), "med_mass": r4(c["med_mass"]),
                        "sd_sample": r4(c["sd_sample"]), "sd_pop": r4(c["sd_pop"]),
                        "yes": c["yes"], "n": c["n"], "n_spk": c["n_spk"],
                        "their_lo": r4(tl), "their_hi": r4(th), "ci_agree": agree,
                        "boot_usable": nok})

print()
diffs = []
for name, rows in data.items():
    d = pts[(name, "conflict")]["rate"] - pts[(name, "agreement")]["rate"]
    lo, hi, nok = boot_diff(rows, order, dk, sk)
    td, tl, th = THEIRS_D[name]
    agree = (r4(d) == r4(td)) and (r4(lo) == r4(tl)) and (r4(hi) == r4(th))
    print("%-18s diff=%s [%s, %s] usable=%d | theirs %s [%s, %s] %s"
          % (name, r4(d), r4(lo), r4(hi), nok, r4(td), r4(tl), r4(th),
             "OK" if agree else "DIFFER"))
    diffs.append({"model": name, "diff": r4(d), "lo": r4(lo), "hi": r4(hi),
                  "their_diff": r4(td), "their_lo": r4(tl), "their_hi": r4(th),
                  "agree": agree, "boot_usable": nok})

out["convention"] = "%s/%s/%s" % (order_name, dk, sk)
out["cells"] = results
out["diffs"] = diffs
with open(os.path.join(BASE, "part16", "verify", "M4_verify_out.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nproblems: %d" % len(out["problems"]))
for p in out["problems"]:
    print("  " + p)
