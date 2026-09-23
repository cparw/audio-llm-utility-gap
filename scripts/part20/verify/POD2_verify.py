"""Independent verifier for part20 POD2 (balanced LoRA on Pitt 468).
Written from scratch; imports nothing from the compute job's code.
AUC: explicit Mann-Whitney over every pos x neg pair (ties 0.5), cross-checked by a trapezoid ROC.
Intervals: 2000 draws, fresh numpy default_rng(0) per interval, speakers resampled with replacement
(unique speakers sorted, rng.choice(u, len(u), replace=True)), 2.5 / 97.5 percentiles.
Arm intervals: primary = resample inside the arm's own speakers; also reported: resample all speakers then subset.
Paired differences: one speaker draw per replicate, both score columns scored on the same clips.
usage: POD2_verify.py <balanced_oof.csv> <manifest.csv> <folds.csv> <earlier_oof.csv> [pod1_oof.csv|NONE] [sidecar.json|NONE]
"""
import sys, csv, json, os, hashlib
import numpy as np

PROMPT = ("Based only on this recording, does this speaker show signs of dementia? "
          "Answer with one word, Yes or No.")


def auc_pairs(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    p = s[y == 1]; n = s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return float(gt + 0.5 * eq) / (len(p) * len(n))


def auc_trapezoid(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    P = (y == 1).sum(); N = (y == 0).sum()
    th = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in th:
        tpr.append(((s >= t) & (y == 1)).sum() / P)
        fpr.append(((s >= t) & (y == 0)).sum() / N)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def speaker_draws(spk, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(np.asarray(spk))
    where = {g: np.flatnonzero(np.asarray(spk) == g) for g in u}
    for _ in range(n):
        pick = rng.choice(u, size=len(u), replace=True)
        yield np.concatenate([where[g] for g in pick])


def ci(vals):
    v = np.asarray(vals, float)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))


def boot_one(y, s, spk, sub=None):
    out = []
    for ii in speaker_draws(spk):
        if sub is not None:
            ii = ii[sub[ii]]
        if len(ii) == 0:
            continue
        a = auc_pairs(y[ii], s[ii])
        if a == a:
            out.append(a)
    return ci(out)


def boot_paired(y, s1, s2, spk, sub=None):
    out = []
    for ii in speaker_draws(spk):
        if sub is not None:
            ii = ii[sub[ii]]
        if len(ii) == 0:
            continue
        a = auc_pairs(y[ii], s1[ii]); b = auc_pairs(y[ii], s2[ii])
        if a == a and b == b:
            out.append(a - b)
    return ci(out)


def md5(f):
    return hashlib.md5(open(f, "rb").read()).hexdigest()


def main():
    F, MAN, FOLDS, EARLIER = sys.argv[1:5]
    POD1 = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "NONE" else None
    SIDE = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "NONE" else None
    o = {"file": os.path.abspath(F), "md5_file": md5(F), "folds_file": os.path.abspath(FOLDS),
         "md5_folds": md5(FOLDS)}
    R = list(csv.DictReader(open(F)))
    cols = list(R[0].keys()); o["columns"] = cols
    name_col = "name" if "name" in cols else ("clip_id" if "clip_id" in cols else cols[0])
    spk_col = "spk" if "spk" in cols else ("speaker" if "speaker" in cols else "speaker_id")
    arm_col = "arm" if "arm" in cols else "set"
    score_col = os.environ.get("SCORE_COL", "p_yes")
    names = np.array([os.path.basename(r[name_col]) for r in R])
    y = np.array([int(float(r["label"])) for r in R])
    spk = np.array([r[spk_col] for r in R]); arm = np.array([r[arm_col] for r in R])
    s = np.array([float(r[score_col]) for r in R])
    o.update(n=len(R), n_spk=int(len(np.unique(spk))), n_conflict=int((arm == "conflict").sum()),
             n_agreement=int((arm == "agreement").sum()), n_pos=int(y.sum()),
             n_unique_names=int(len(set(names))), score_col=score_col,
             score_min=float(s.min()), score_max=float(s.max()), n_nan=int(np.isnan(s).sum()))

    # manifest and folds cross-check
    M = list(csv.DictReader(open(MAN)))
    mcols = list(M[0].keys())
    mname = np.array([os.path.basename(r.get("path") or r.get("name") or r.get("clip_id")) for r in M])
    m_y = {n_: int(float(r["label"])) for n_, r in zip(mname, M)}
    m_spk = {n_: (r.get("speaker") or r.get("spk") or r.get("speaker_id")) for n_, r in zip(mname, M)}
    m_arm = {n_: (r.get("set") or r.get("arm")) for n_, r in zip(mname, M)}
    FR = list(csv.DictReader(open(FOLDS)))
    fname = [os.path.basename(r.get("clip_id") or r.get("path") or r.get("name")) for r in FR]
    f_of = {n_: int(r["fold"]) for n_, r in zip(fname, FR)}
    o["manifest_rows"] = len(M); o["fold_rows"] = len(FR)
    o["manifest_matches_oof_labels"] = bool(all(m_y[n_] == y_ for n_, y_ in zip(names, y)))
    o["manifest_matches_oof_speakers"] = bool(all(m_spk[n_] == g for n_, g in zip(names, spk)))
    o["manifest_matches_oof_arms"] = bool(all(m_arm[n_] == a for n_, a in zip(names, arm)))
    if "fold" in cols:
        o["oof_fold_col_matches_fold_file"] = bool(all(int(r["fold"]) == f_of[n_] for r, n_ in zip(R, names)))
    fold = np.array([f_of[n_] for n_ in names])
    o["fold_sizes"] = [int((fold == k).sum()) for k in range(5)]
    # weight table from manifest + folds (independent of the oof file)
    mfold = np.array([f_of[n_] for n_ in mname]); my = np.array([m_y[n_] for n_ in mname])
    marm = np.array([m_arm[n_] for n_ in mname])
    wt = {}
    for k in range(5):
        tr = mfold != k; row = {}
        w = np.zeros(tr.sum()); yt = my[tr]; at = marm[tr]
        for lab in (0, 1):
            Ny = int((yt == lab).sum())
            for a in ("conflict", "agreement"):
                Nya = int(((yt == lab) & (at == a)).sum())
                row[f"y{lab}_{a}"] = {"N_y": Ny, "N_ya": Nya, "w_raw": (Ny / 2) / Nya}
                w[(yt == lab) & (at == a)] = (Ny / 2) / Nya
        mean_raw = float(w.mean()); w2 = w / w.mean()
        for key in row:
            lab = int(key[1]); a = key.split("_", 1)[1]
            row[key]["w_mean1"] = float(w2[(yt == lab) & (at == a)][0])
            row[key]["group_total_weight"] = float(w2[(yt == lab) & (at == a)].sum())
        row["n_train"] = int(tr.sum()); row["mean_raw"] = mean_raw
        wt[str(k)] = row
    o["weight_table"] = wt

    # AUCs and intervals
    cm = arm == "conflict"; am = arm == "agreement"
    res = {}
    for key, m in (("overall", np.ones(len(y), bool)), ("conflict", cm), ("agreement", am)):
        a = auc_pairs(y[m], s[m]); t = auc_trapezoid(y[m], s[m])
        lo, hi, k = boot_one(y[m], s[m], spk[m])
        d = {"auc": a, "auc_trapezoid": t, "lo": lo, "hi": hi, "draws": k,
             "n": int(m.sum()), "n_spk": int(len(np.unique(spk[m])))}
        if key != "overall":
            l2, h2, k2 = boot_one(y, s, spk, sub=m)
            d.update(lo_allspk=l2, hi_allspk=h2, draws_allspk=k2)
        res[key] = d
    o["balanced"] = res

    def paired_vs(other_file, tag, ocol):
        E = list(csv.DictReader(open(other_file)))
        ecols = list(E[0].keys())
        en = "name" if "name" in ecols else ("clip_id" if "clip_id" in ecols else ecols[0])
        es = ocol
        emap = {os.path.basename(r[en]): r for r in E}
        assert set(emap) == set(names), f"{tag}: clip sets differ"
        e = np.array([float(emap[n_][es]) for n_ in names])
        ey = np.array([int(float(emap[n_]["label"])) for n_ in names])
        assert (ey == y).all(), f"{tag}: labels differ"
        out = {"file": os.path.abspath(other_file), "score_col": es}
        for key, m in (("overall", np.ones(len(y), bool)), ("conflict", cm), ("agreement", am)):
            ea = auc_pairs(y[m], e[m]); ba = auc_pairs(y[m], s[m])
            lo, hi, k = boot_one(y[m], e[m], spk[m])
            dlo, dhi, dk = boot_paired(y[m], s[m], e[m], spk[m])
            d = {"other_auc": ea, "other_lo": lo, "other_hi": hi, "diff": ba - ea,
                 "diff_lo": dlo, "diff_hi": dhi, "diff_draws": dk}
            if key == "overall":
                pass
            else:
                l2, h2, k2 = boot_paired(y, s, e, spk, sub=m)
                d.update(diff_lo_allspk=l2, diff_hi_allspk=h2)
            out[key] = d
        return out

    o["vs_earlier_lora"] = paired_vs(EARLIER, "earlier", os.environ.get("EARLIER_COL", "p_yes"))
    if POD1:
        o["vs_pod1"] = paired_vs(POD1, "pod1", os.environ.get("POD1_COL", "p_yes"))
    # within-model gap (conflict minus agreement), one speaker draw over all speakers per replicate
    def gap_boot(sc_list, signs):
        out = []
        for ii in speaker_draws(spk):
            ic = ii[cm[ii]]; ia = ii[am[ii]]
            if len(ic) == 0 or len(ia) == 0:
                continue
            v = 0.0; ok = True
            for sc, sg in zip(sc_list, signs):
                c_ = auc_pairs(y[ic], sc[ic]); a_ = auc_pairs(y[ia], sc[ia])
                if not (c_ == c_ and a_ == a_):
                    ok = False; break
                v += sg * (c_ - a_)
            if ok:
                out.append(v)
        return ci(out)
    gv = auc_pairs(y[cm], s[cm]) - auc_pairs(y[am], s[am])
    glo, ghi, gk = gap_boot([s], [1])
    o["balanced_conflict_minus_agreement"] = {"v": gv, "lo": glo, "hi": ghi, "draws": gk}
    E_ = list(csv.DictReader(open(EARLIER))); ecol = os.environ.get("EARLIER_COL", "p_yes")
    emap_ = {os.path.basename(r["name"]): float(r[ecol]) for r in E_}
    e_ = np.array([emap_[n_] for n_ in names])
    ev = auc_pairs(y[cm], e_[cm]) - auc_pairs(y[am], e_[am])
    clo, chi, ck = gap_boot([s, e_], [1, -1])
    o["gap_change_balanced_minus_earlier"] = {"v": gv - ev, "lo": clo, "hi": chi, "draws": ck,
                                              "earlier_gap": ev}
    FJ = os.environ.get("FOLD_JSONS")
    if FJ:
        chk = {}
        for fj in FJ.split(","):
            d = json.load(open(fj)); k = str(d["fold"]); mine = wt[k]; bad = []
            for t in d["weight_table"]:
                m = mine[f"y{t['label']}_{t['arm']}"]
                for a_, b_ in (("N_y", "N_y"), ("N_ya", "N_ya"), ("w_raw", "w_raw"), ("w_rescaled", "w_mean1"), ("total_weight", "group_total_weight")):
                    if abs(float(t[a_]) - float(m[b_])) > 1e-9: bad.append((t["label"], t["arm"], a_, t[a_], m[b_]))
            if int(d["n_train"]) != mine["n_train"]: bad.append(("n_train", d["n_train"], mine["n_train"]))
            # held-out rows must be exactly fold k of the fold file (manifest row order)
            rows_ = sorted(int(r["row"]) for r in d["results"])
            exp_ = sorted(np.flatnonzero(mfold == int(k)).tolist())
            if rows_ != exp_: bad.append(("heldout_rows_mismatch",))
            # every held-out score must equal the csv
            byname = {n_: i_ for i_, n_ in enumerate(names)}
            for r in d["results"]:
                i_ = byname[mname[int(r["row"])]]
                for c in ("p_yes", "p_yes_2logit"):
                    if c in cols and c in r and abs(float(R[i_][c]) - float(r[c])) > 1e-12: bad.append(("score_mismatch", c, int(r["row"])))
            chk[k] = {"file": fj, "problems": bad[:20], "n_problems": len(bad), "steps": d.get("steps"),
                      "rescale": d.get("rescale")}
        o["fold_checkpoints"] = chk
    if SIDE:
        sd = json.load(open(SIDE))
        flat = json.dumps(sd)
        o["sidecar_prompt_verbatim_present"] = PROMPT in flat
        o["sidecar_mentions_pitt_groupkfold5_pod"] = "pitt_groupkfold5_pod.csv" in flat
        o["sidecar_keys"] = list(sd.keys())
    print(json.dumps(o, indent=1, default=float))


if __name__ == "__main__":
    main()
