"""INDEPENDENT verifier for POD3. Written from scratch on the Mac.
Nothing is imported from the pod job's scripts.

AUC implementation 1: explicit pairwise concordance over every (pos,neg) pair,
   counting a tie as 0.5. No ranking, no sorting.
AUC implementation 2: trapezoid integration of the full empirical ROC curve,
   built by sweeping every distinct score as a threshold.
Both are computed for every value; they must agree with each other before the
value is compared to the pod job's number.

Bootstrap: 2000 draws, numpy.random.default_rng(0), speakers resampled with
replacement, percentile 2.5 / 97.5. When several repeat-columns exist, ONE
speaker draw per replicate is shared across the repeats and the replicate
statistic is the mean over repeats (paired).
Two RNG conventions are reported because the spec does not pin the call:
   choice : rng.choice(sorted_unique_speakers, size=k, replace=True)
   integers: sorted_unique_speakers[rng.integers(0, k, size=k)]
"""
import csv, json, os
import numpy as np

P3 = "<local data dir>/Desktop/release/edaic_rerun/part16/POD3"
SHIP = "<local data dir>/Desktop/release/overnight2/q3o_new"


def auc_pairwise(y, s):
    """P(s_pos > s_neg) + 0.5*P(tie), counted pair by pair."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]; neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = 0.0; eq = 0.0
    step = 2048
    for i in range(0, pos.size, step):
        blk = pos[i:i + step][:, None]
        gt += np.count_nonzero(blk > neg[None, :])
        eq += np.count_nonzero(blk == neg[None, :])
    return (gt + 0.5 * eq) / (pos.size * neg.size)


def auc_trapezoid(y, s):
    """Area under the empirical ROC by trapezoid over every distinct threshold."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    thr = np.unique(s)
    tpr = [0.0]; fpr = [0.0]
    for t in thr[::-1]:                      # high score -> predict positive
        pred = s >= t
        tpr.append(float(np.count_nonzero(pred & (y == 1))) / P)
        fpr.append(float(np.count_nonzero(pred & (y == 0))) / N)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def boot_ci(y, spk, score_vectors, seed=0, draws=2000, mode="choice"):
    y = np.asarray(y); spk = np.asarray(spk)
    uniq = np.unique(spk)                      # sorted
    k = uniq.size
    where = {t: np.flatnonzero(spk == t) for t in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        if mode == "choice":
            pick = rng.choice(uniq, size=k, replace=True)
        else:
            pick = uniq[rng.integers(0, k, size=k)]
        idx = np.concatenate([where[t] for t in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        stats.append(float(np.mean([auc_pairwise(yy, sv[idx]) for sv in score_vectors])))
    a = np.asarray(stats)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), a.size


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


ROWS = []


def record(tag, what, claimed, mine, claimed_ci=None, mine_ci=None,
           mine_ci_int=None, n=None, nspk=None, src="", note=""):
    agree = (claimed is not None) and (round(float(claimed), 4) == round(float(mine), 4))
    ROWS.append(dict(id=tag, what=what, claimed=claimed, mine=round(float(mine), 6),
                     agree_4dp=bool(agree), claimed_ci=claimed_ci, mine_ci=mine_ci,
                     mine_ci_integers=mine_ci_int, n=n, n_speakers=nspk,
                     file=src, note=note))
    c = "     n/a" if claimed is None else f"{claimed:8.4f}"
    flag = "OK  " if agree else ("DIFF" if claimed is not None else "NEW ")
    line = f"{flag} {tag:<32} claimed {c}   mine {mine:8.4f}"
    if claimed_ci:
        line += f"  claimCI [{claimed_ci[0]:.4f},{claimed_ci[1]:.4f}]"
    if mine_ci:
        line += f"  myCI [{mine_ci[0]:.4f},{mine_ci[1]:.4f}]"
    print(line, flush=True)


print("=" * 110)
print("PART A  zero-shot answer AUCs, recomputed from the per-clip csv")
print("=" * 110)
ZS = [("3a_pitt_zeroshot_pod", f"{P3}/q3o_pitt_zeroshot_scores.csv", 0.7673),
      ("3a_pitt_zeroshot_shipped", f"{SHIP}/q3o_pitt_zeroshot_scores.csv", 0.7619),
      ("3a_pcgita_zeroshot_pod", f"{P3}/q3o_pcgita_zeroshot_scores.csv", 0.6015),
      ("3a_pcgita_zeroshot_shipped", f"{SHIP}/q3o_pcgita_zeroshot_scores.csv", 0.5967)]
for tag, path, claimed in ZS:
    if not os.path.exists(path):
        print(f"MISSING {tag}: {path}")
        ROWS.append(dict(id=tag, what="zero-shot AUC", claimed=claimed, mine=None,
                         agree_4dp=False, file=path, note="FILE MISSING"))
        continue
    r = load(path)
    y = [int(x["label"]) for x in r]
    s = np.array([float(x["p_yes"]) for x in r])
    spk = [x["speaker"] for x in r]
    a1 = auc_pairwise(y, s); a2 = auc_trapezoid(y, s)
    assert abs(a1 - a2) < 1e-9, (tag, a1, a2)
    lo, hi, nd = boot_ci(y, spk, [s])
    lo2, hi2, _ = boot_ci(y, spk, [s], mode="integers")
    record(tag, "zero-shot answer AUC", claimed, a1, None, (lo, hi), (lo2, hi2),
           len(y), len(set(spk)), path, f"trapezoid={a2:.6f}; usable draws {nd}")

print()
print("=" * 110)
print("PART B  attention-kernel test and determinism reruns")
print("=" * 110)
f = f"{P3}/q3o_pitt_attn_backend.csv"
r = load(f)
y = [int(x["label"]) for x in r]
for col, tag, claimed in (("p_yes_sdpa", "3a_attn_sdpa", 0.7673),
                          ("p_yes_eager", "3a_attn_eager", 0.7679)):
    s = np.array([float(x[col]) for x in r])
    a1 = auc_pairwise(y, s); a2 = auc_trapezoid(y, s)
    assert abs(a1 - a2) < 1e-9
    record(tag, f"zero-shot AUC, {col.split('_')[-1]} kernel", claimed, a1,
           None, None, None, len(r), len(set(x["speaker"] for x in r)), f,
           f"trapezoid={a2:.6f}")
for tag, path, claimed in (("3a_pitt_rescore_runA", f"{P3}/q3o_pitt_rescore_runA.csv", 0.7673),
                           ("3a_pitt_rescore_runB", f"{P3}/q3o_pitt_rescore_runB.csv", 0.7673)):
    r = load(path)
    s = np.array([float(x["p_yes"]) for x in r])
    yy = [int(x["label"]) for x in r]
    record(tag, "zero-shot AUC (rerun)", claimed, auc_pairwise(yy, s), None, None,
           None, len(r), len(set(x["speaker"] for x in r)), path)

print()
print("=" * 110)
print("PART C  nested probe AUCs, recomputed from the per-clip out-of-fold csv")
print("=" * 110)
for ds, nice in (("pitt", "Pitt 468"), ("pcgita", "PC-GITA 1100")):
    summ = json.load(open(f"{P3}/q3o_{ds}_perlayer_summary.json"))["streams"]
    for key, suf in (("enc", "encoder"), ("proj", "proj"), ("llm", "llm"), ("ans", "ans")):
        f = f"{P3}/q3o_{ds}_{suf}_nested_oof.csv"
        if not os.path.exists(f):
            print(f"MISSING {ds} {key}: {f}")
            continue
        r = load(f)
        y = np.array([int(x["label"]) for x in r])
        spk = np.array([x["speaker"] for x in r])
        cols = [c for c in r[0] if c.startswith("p_probe")]
        svs = [np.array([float(x[c]) for x in r]) for c in cols]
        per = [auc_pairwise(y, sv) for sv in svs]
        mine = float(np.mean(per))
        claimed = summ[key]["nested_auc_mean"]
        claimed_per = summ[key].get("nested_per_repeat")
        lo, hi, nd = boot_ci(y, spk, svs)
        lo2, hi2, _ = boot_ci(y, spk, svs, mode="integers")
        pr = " ".join(f"{v:.4f}" for v in per)
        note = f"my per-repeat [{pr}] vs claimed {claimed_per}; draws {nd}"
        record(f"3a_{ds}_{key}_nested", f"{nice} {key} nested probe AUC", claimed,
               mine, tuple(summ[key]["nested_ci"]), (lo, hi), (lo2, hi2),
               len(y), len(set(spk.tolist())), f, note)

print()
print("=" * 110)
print("PART D  ties / precision audit on the PC-GITA projector file")
print("=" * 110)
r = load(f"{P3}/q3o_pcgita_proj_nested_oof.csv")
y = np.array([int(x["label"]) for x in r])
s = np.array([float(x["p_probe_rep0"]) for x in r])
sat = int(np.count_nonzero(s == 1.0))
pos = s[y == 1]; neg = s[y == 0]
ties = int(sum(np.count_nonzero(pos[i:i + 2048][:, None] == neg[None, :])
               for i in range(0, pos.size, 2048)))
dec = max(len(x["p_probe_rep0"].split(".")[-1]) for x in r)
print(f"clips {len(r)}  saturated at exactly 1.0: {sat}  tied pos-neg pairs: {ties}"
      f"  max stored decimals: {dec}")
print(f"AUC pairwise {auc_pairwise(y, s):.6f}   AUC trapezoid {auc_trapezoid(y, s):.6f}")
CLAIMS_D = {"saturated_at_1.0": sat, "tied_pairs": ties, "stored_decimals": dec}

print()
print("=" * 110)
print("PART E  arm composition, recomputed from clip names in the per-clip csv")
print("=" * 110)
r = load(f"{P3}/q3o_pitt_zeroshot_scores.csv")
arms = {}
for x in r:
    a = x["clip"].split("_")[0]
    d = arms.setdefault(a, {"n": 0, "spk": set(), "pos": 0})
    d["n"] += 1; d["spk"].add(x["speaker"]); d["pos"] += int(x["label"])
arms["all"] = {"n": len(r), "spk": set(x["speaker"] for x in r),
               "pos": sum(int(x["label"]) for x in r)}
byarm = {row["arm"]: row for row in load(f"{P3}/readout_direction_q3o_by_arm.csv")}
ARM_OK = {}
for a, d in arms.items():
    c = byarm.get(a)
    ok = c and (int(c["n"]) == d["n"] and int(c["n_speakers"]) == len(d["spk"])
                and int(c["n_positive"]) == d["pos"])
    ARM_OK[a] = bool(ok)
    print(f"{a:<10} mine n={d['n']:4d} spk={len(d['spk']):4d} pos={d['pos']:4d}   "
          f"claimed n={c['n']} spk={c['n_speakers']} pos={c['n_positive']}   "
          f"{'MATCH' if ok else 'MISMATCH'}")

json.dump({"auc": "pairwise concordance with ties at 0.5, cross-checked against trapezoid ROC",
           "bootstrap": "2000 draws, default_rng(0), speakers resampled, pct 2.5/97.5, paired over repeats",
           "rows": ROWS, "pcgita_proj_precision": CLAIMS_D, "arm_composition_match": ARM_OK},
          open("reports/part16/verify/POD3_verify_out.json", "w"),
          indent=1)
n_cmp = sum(1 for o in ROWS if o["claimed"] is not None and o["mine"] is not None)
n_ok = sum(1 for o in ROWS if o["agree_4dp"])
print(f"\nPOINT ESTIMATES: {n_ok}/{n_cmp} agree to 4 dp")


# =====================================================================================
# PART F  3b by-arm probe AUC, rebuilt from the RAW STATE TENSORS.
# Run with:  python3 POD3_verify.py partF
# The pod job's own verifier never recomputed these - it read the aggregate csv cell
# and compared it to itself. This block is the first real recomputation.
# The readout direction d needs q3o_lm_head.npy, which was NOT synced to the Mac, so
# auc_d and auc_probe_without_d stay unverifiable here and are reported as such.
# =====================================================================================
if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "partF":
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    NPZ = "<local data dir>/paper1_local_runs/part16_POD3_states/q3o_pitt_states.npz"
    print("loading", NPZ, flush=True)
    z = np.load(NPZ, allow_pickle=True)
    ans = z["ans"]
    y_all = z["label"].astype(int)
    spk_all = z["spk"].astype(str)
    name_all = z["name"].astype(str)
    print("ans", ans.shape, ans.dtype, "n", len(y_all), flush=True)
    X_all = ans[:, -1, :].astype(np.float64)
    del ans
    arms_all = np.array([n.split("_")[0] for n in name_all])
    print("arm counts", {a: int((arms_all == a).sum()) for a in sorted(set(arms_all))}, flush=True)

    # cross-check the states file against the per-clip zero-shot csv
    zs = {x["clip"]: x for x in load(f"{P3}/q3o_pitt_zeroshot_scores.csv")}
    same_lab = sum(1 for i, n in enumerate(name_all) if int(zs[n]["label"]) == y_all[i])
    if "p_yes" in z.files:
        pz = z["p_yes"].astype(float)
        dmax = max(abs(pz[i] - float(zs[n]["p_yes"])) for i, n in enumerate(name_all))
        print(f"states vs zeroshot csv: labels match {same_lab}/{len(y_all)}, "
              f"max |dp_yes| {dmax:.2e}", flush=True)

    CLAIM = {"all": (0.8307, [0.7833, 0.8741]),
             "conflict": (0.6145, [0.5021, 0.7246]),
             "agreement": (0.9349, [0.9034, 0.9614])}

    for arm in ("all", "conflict", "agreement"):
        m = np.ones(len(y_all), bool) if arm == "all" else (arms_all == arm)
        X = X_all[m]; y = y_all[m]; spk = spk_all[m]
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=spk):
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
        mine = auc_pairwise(y, oof)
        tz = auc_trapezoid(y, oof)
        lo, hi, nd = boot_ci(y, spk, [oof])
        cv, cci = CLAIM[arm]
        record(f"3b_{arm}_auc_probe", f"Pitt {arm} arm, probe AUC (REBUILT from states)",
               cv, mine, tuple(cci), (lo, hi), None, len(y), len(set(spk.tolist())),
               NPZ, f"trapezoid={tz:.6f}; draws {nd}")
        ROWS.append(dict(id=f"3b_{arm}_auc_d", what=f"Pitt {arm} arm, direction-alone AUC",
                         claimed=None, mine=None, agree_4dp=False,
                         note="UNVERIFIABLE ON MAC: q3o_lm_head.npy was never synced"))
        ROWS.append(dict(id=f"3b_{arm}_auc_probe_without_d",
                         what=f"Pitt {arm} arm, probe-minus-direction AUC",
                         claimed=None, mine=None, agree_4dp=False,
                         note="UNVERIFIABLE ON MAC: q3o_lm_head.npy was never synced"))

    json.dump(ROWS, open("reports/part16/verify/POD3_partF_out.json", "w"), indent=1)


# =====================================================================================
# PART G  3b from the PER-CLIP columns in part16/pod_sync/q3o_perp_cols.npz.
# Run with:  python3 POD3_verify.py partG
# This file (written by the primary job on the same pod) stores the per-clip probe OOF
# score, the projection on the readout direction, and the three d-removed variants.
# It is the only per-clip artifact on the Mac that backs any 3b number.
# =====================================================================================
if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "partG":
    NPZ = "scores/part16/pod_sync/q3o_perp_cols.npz"
    g = np.load(NPZ, allow_pickle=True)
    y = g["label"].astype(int); spk = g["spk"].astype(str); nm = g["name"].astype(str)
    arms = np.array([s.split("_")[0] for s in nm])
    print(f"per-clip columns: {list(g.files)}  n={len(y)}  speakers={len(set(spk.tolist()))}")
    POD3 = {("all", "oof"): 0.8307, ("all", "proj_d"): 0.7656, ("all", "perp_trainz"): 0.8305,
            ("conflict", "oof"): 0.6145, ("conflict", "proj_d"): 0.6029, ("conflict", "perp_trainz"): 0.6196,
            ("agreement", "oof"): 0.9349, ("agreement", "proj_d"): 0.8264, ("agreement", "perp_trainz"): 0.9341}
    FIX = {("all", "oof"): 0.8307, ("all", "perp_z"): 0.8276,
           ("conflict", "oof"): 0.6394, ("conflict", "perp_z"): 0.6532,
           ("agreement", "oof"): 0.9032, ("agreement", "perp_z"): 0.8937}
    print(f"\n{'arm':<10} {'column':<12} {'mine':>8} {'POD3':>8} {'3b_fix':>8}  verdict")
    for arm in ("all", "conflict", "agreement"):
        m = np.ones(len(y), bool) if arm == "all" else (arms == arm)
        for col in ("oof", "proj_d", "perp_trainz", "perp_z", "perp_raw"):
            mine = auc_pairwise(y[m], g[col][m])
            tz = auc_trapezoid(y[m], g[col][m])
            assert abs(mine - tz) < 1e-9
            p = POD3.get((arm, col)); fx = FIX.get((arm, col))
            ps = f"{p:8.4f}" if p is not None else "       -"
            fs = f"{fx:8.4f}" if fx is not None else "       -"
            hits = []
            if p is not None and round(mine, 4) == round(p, 4): hits.append("POD3")
            if fx is not None and round(mine, 4) == round(fx, 4): hits.append("3b_fix")
            v = ("matches " + "+".join(hits)) if hits else ("MATCHES NEITHER" if (p or fx) else "")
            print(f"{arm:<10} {col:<12} {mine:8.4f} {ps} {fs}  {v}")
            if p is not None:
                lo, hi, _ = boot_ci(y[m], spk[m], [g[col][m]])
                print(f"{'':<10} {'':<12} my 95% CI [{lo:.4f}, {hi:.4f}]")
