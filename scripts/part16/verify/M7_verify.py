#!/usr/bin/env python3
"""
INDEPENDENT VERIFICATION of PART 16 / M7.
Written from scratch. Does not import, read or reuse the primary job's script
or its intermediate outputs. Goes back to the ORIGINAL source csv in
<local data dir>/Desktop/release/omni_final/.

AUC route: explicit Mann-Whitney U concordant-pair count, ties = 0.5.
NOT a rankdata one-liner.
"""
import csv, json, os, sys, hashlib
import numpy as np

SRC = "<local data dir>/Desktop/release/omni_final"
OUT = "<local data dir>/Desktop/release/edaic_rerun/part16/verify"
DATASETS = ["pitt", "adresso", "adress2020", "pcgita", "neurovoz", "kcl", "edaic"]


def auc_mwu(y, s):
    """AUC as the fraction of concordant (pos,neg) pairs, ties counted 0.5.
    Explicit O(n_pos * n_neg) pair count via broadcasting. No rank formula."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # chunk to bound memory
    total = 0.0
    CH = 4096
    for i in range(0, pos.size, CH):
        blk = pos[i:i + CH][:, None]        # (b,1)
        d = blk - neg[None, :]              # (b, n_neg)
        total += float(np.count_nonzero(d > 0)) + 0.5 * float(np.count_nonzero(d == 0))
    return total / (pos.size * neg.size)


def auc_trapezoid(y, s):
    """Second independent route: trapezoid area under the full ROC curve."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    thr = np.unique(s)
    # sweep thresholds from high to low
    tpr = [0.0]; fpr = [0.0]
    for t in sorted(thr, reverse=True):
        pred = s >= t
        tpr.append(float(((pred) & (y == 1)).sum()) / P)
        fpr.append(float(((pred) & (y == 0)).sum()) / N)
    tpr.append(1.0); fpr.append(1.0)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.trapezoid(tpr, fpr))


def load(path, scorecol):
    clips, spk, lab, sc = [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            clips.append(row["clip"])
            spk.append(row["speaker"])
            lab.append(int(float(row["label"])))
            sc.append(float(row[scorecol]))
    return clips, spk, np.array(lab), np.array(sc, dtype=np.float64)


def sha_first_mb(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()


results = {}
report = []

for ds in DATASETS:
    fp_probe = os.path.join(SRC, f"omni_{ds}_enc_nested_oof.csv")
    fp_zs = os.path.join(SRC, f"omni_{ds}_zeroshot_scores.csv")
    assert os.path.exists(fp_probe), f"MISSING {fp_probe}"
    assert os.path.exists(fp_zs), f"MISSING {fp_zs}"

    c1, s1, y1, p_probe = load(fp_probe, "p_probe")
    c2, s2, y2, p_zs = load(fp_zs, "p_yes")

    # join integrity
    same_order = (c1 == c2)
    md5_1 = hashlib.md5("\n".join(sorted(c1)).encode()).hexdigest()
    md5_2 = hashlib.md5("\n".join(sorted(c2)).encode()).hexdigest()
    label_disagree = int((y1 != y2).sum()) if same_order else None
    spk_disagree = int(sum(a != b for a, b in zip(s1, s2))) if same_order else None

    y = y1
    spk = np.array(s1)
    uspk = np.unique(spk)
    idx_by_spk = {u: np.where(spk == u)[0] for u in uspk}

    a_probe = auc_mwu(y, p_probe)
    a_zs = auc_mwu(y, p_zs)
    a_probe_tz = auc_trapezoid(y, p_probe)
    a_zs_tz = auc_trapezoid(y, p_zs)
    diff = a_probe - a_zs

    # paired speaker bootstrap: ONE speaker draw per replicate, both AUCs inside it
    rng = np.random.default_rng(0)
    bp, bz, bd = [], [], []
    n_bad = 0
    for _ in range(2000):
        pick = rng.integers(0, len(uspk), size=len(uspk))
        idx = np.concatenate([idx_by_spk[uspk[k]] for k in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            n_bad += 1
            continue
        ap = auc_mwu(yy, p_probe[idx])
        az = auc_mwu(yy, p_zs[idx])
        bp.append(ap); bz.append(az); bd.append(ap - az)
    bp = np.array(bp); bz = np.array(bz); bd = np.array(bd)

    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    cp, cz, cd = ci(bp), ci(bz), ci(bd)

    results[ds] = dict(
        n=len(c1), n_spk=int(len(uspk)),
        clips_identical_order=bool(same_order),
        sorted_clip_md5_match=bool(md5_1 == md5_2),
        label_disagreements=label_disagree,
        speaker_disagreements=spk_disagree,
        probe_auc=round(a_probe, 6), probe_auc_trapezoid=round(a_probe_tz, 6),
        zs_auc=round(a_zs, 6), zs_auc_trapezoid=round(a_zs_tz, 6),
        diff=round(diff, 6),
        probe_ci=[round(cp[0], 6), round(cp[1], 6)],
        zs_ci=[round(cz[0], 6), round(cz[1], 6)],
        diff_ci=[round(cd[0], 6), round(cd[1], 6)],
        n_usable_draws=int(len(bp)), n_degenerate=int(n_bad),
        excludes_zero=bool(cd[0] > 0 or cd[1] < 0),
        src_probe=fp_probe, src_zs=fp_zs,
        sha256_first1mb_probe=sha_first_mb(fp_probe),
        sha256_first1mb_zs=sha_first_mb(fp_zs),
    )
    print(f"{ds:11s} n={len(c1):5d} spk={len(uspk):4d} "
          f"probe={a_probe:.4f} [{cp[0]:.4f},{cp[1]:.4f}]  "
          f"zs={a_zs:.4f} [{cz[0]:.4f},{cz[1]:.4f}]  "
          f"diff={diff:+.4f} [{cd[0]:+.4f},{cd[1]:+.4f}]  "
          f"draws={len(bp)}/2000", flush=True)

mean_gap = float(np.mean([results[d]["diff"] for d in DATASETS]))
print(f"\nMEAN GAP across 7 = {mean_gap:.4f}")
results["_mean_gap"] = round(mean_gap, 6)

with open(os.path.join(OUT, "M7_verify_out.json"), "w") as f:
    json.dump(results, f, indent=2)
print("wrote M7_verify_out.json")

# ---------------- PART 2: estimator-discrepancy + pitt npz supplement ----------------
print("\n=== PART 2: estimator discrepancy and pitt rep5 supplement ===")
import csv as _csv

# 2a: does the enc OOF csv reproduce nested.json auc_nested_oof?
for ds in DATASETS:
    j = json.load(open(os.path.join(SRC, f"omni_{ds}_nested.json")))
    print(f"  {ds:11s} my_probe={results[ds]['probe_auc']:.6f}  nested.json enc auc_nested_oof={j['enc']['auc_nested_oof']:.6f}  "
          f"match4dp={round(results[ds]['probe_auc'],4)==round(j['enc']['auc_nested_oof'],4)}  n_json={j['enc']['n']}")

# 2b: master_lookup encoder probe / zero shot rows for Qwen2.5-Omni
ML = "lookup/master_lookup.csv"
ml = {}
with open(ML, newline="") as f:
    for row in _csv.DictReader(f):
        if row["model"] == "Qwen2.5-Omni":
            ml[(row["dataset"], row["stream"])] = row
print("\n  master_lookup comparison (Qwen2.5-Omni):")
for ds in DATASETS:
    zrow = ml.get((ds, "zero shot answer"))
    erow = ml.get((ds, "encoder probe"))
    rep = json.load(open(os.path.join(SRC, f"omni_{ds}_nested_repeats.json")))["enc"]
    print(f"  {ds:11s} zs: mine={results[ds]['zs_auc']:.4f} master={float(zrow['auc']):.4f} match={round(results[ds]['zs_auc'],4)==round(float(zrow['auc']),4)}"
          f" | n: mine={results[ds]['n']} master={zrow['n']} match={str(results[ds]['n'])==zrow['n']}"
          f" | enc: mine={results[ds]['probe_auc']:.4f} master={float(erow['auc']):.4f} repeats_mean={rep['mean']:.4f}"
          f" master==repeats_mean:{round(float(erow['auc']),4)==round(rep['mean'],4)}")

# 2c: pitt 5-repeat npz -> per-repeat AUC, mean, and paired supplement
NPZ = "scores/part10/pitt_enc_nested5_oof.npz"
z = np.load(NPZ, allow_pickle=True)
oof = z["oof"]; lab = z["label"]; spkn = z["spk"].astype(str); nm = z["name"].astype(str)
per_rep = [auc_mwu(lab, oof[i]) for i in range(oof.shape[0])]
print(f"\n  pitt npz per-repeat AUC = {[round(v,4) for v in per_rep]}  mean={np.mean(per_rep):.4f}")

# align npz rows to the pitt zero-shot csv
c2, s2, y2, p_zs_pitt = load(os.path.join(SRC, "omni_pitt_zeroshot_scores.csv"), "p_yes")
zmap = {c: i for i, c in enumerate(c2)}
if set(nm) == set(c2):
    order = np.array([zmap[n] for n in nm])
    zs_al = p_zs_pitt[order]
    lab_ok = bool((np.array(y2)[order] == lab).all())
    mean_oof = oof.mean(axis=0)
    a_p5 = auc_mwu(lab, mean_oof)
    a_z5 = auc_mwu(lab, zs_al)
    uu = np.unique(spkn); ib = {u: np.where(spkn == u)[0] for u in uu}
    rng = np.random.default_rng(0)
    bp5, bd5 = [], []
    for _ in range(2000):
        pick = rng.integers(0, len(uu), size=len(uu))
        idx = np.concatenate([ib[uu[k]] for k in pick])
        yy = lab[idx]
        if yy.min() == yy.max():
            continue
        ap = auc_mwu(yy, mean_oof[idx]); az = auc_mwu(yy, zs_al[idx])
        bp5.append(ap); bd5.append(ap - az)
    bp5 = np.array(bp5); bd5 = np.array(bd5)
    print(f"  npz name set == zeroshot clip set: True ; labels align: {lab_ok}")
    print(f"  SUPPLEMENT pitt rep5mean: probe={a_p5:.4f} [{np.percentile(bp5,2.5):.4f},{np.percentile(bp5,97.5):.4f}]"
          f"  zs={a_z5:.4f}  diff={a_p5-a_z5:+.4f} [{np.percentile(bd5,2.5):+.4f},{np.percentile(bd5,97.5):+.4f}]  draws={len(bp5)}/2000")
else:
    print(f"  npz name set != zeroshot clip set (npz {len(set(nm))} vs csv {len(set(c2))})")
