"""PART20 POD3 independent verifier (written from scratch, imports nothing from the compute side).
Recomputes every value of the E-DAIC 966-clip projector fine-tune from the per-clip csv:
  AUC   explicit Mann-Whitney over all pos x neg pairs, ties 0.5; cross-checked by trapezoid ROC.
  CI    2000 draws, fresh numpy default_rng(0) per cell, speakers resampled with replacement from the
        sorted unique ids via rng.integers(0, K, K); a drawn speaker contributes all its rows once per draw;
        draws with one class are skipped and counted; np.percentile 2.5 / 97.5.
  paired one speaker draw per replicate, conflict AUC minus agreement AUC inside that draw.
  distinct agreement: first occurrence of seg_uid in row order; repeats must carry identical p_yes.
Also checks n, speakers, arm counts, labels against the reference manifest, the verbatim prompt, and
rebuilds the speaker folds (GroupKFold(5), stable argsort tie order) two ways.
usage: POD3_verify.py --csv C --score COL --ref REF_MANIFEST [--folds F] [--run_manifest M] [--sidecar S] --out J"""
import argparse, csv, json, os, sys, hashlib
import numpy as np

PROMPT = ("Based only on how this person's voice sounds, does this speaker show signs of depression? "
          "Answer with one word, Yes or No.")
DRAWS, SEED = 2000, 0


def auc_mw(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    P = s[y == 1]; N = s[y == 0]
    if len(P) == 0 or len(N) == 0:
        return float("nan")
    gt = (P[:, None] > N[None, :]).sum(); eq = (P[:, None] == N[None, :]).sum()
    return (gt + 0.5 * eq) / (len(P) * len(N))


def auc_trap(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    nP = (y == 1).sum(); nN = (y == 0).sum()
    th = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in th:
        m = s >= t
        tpr.append((m & (y == 1)).sum() / nP); fpr.append((m & (y == 0)).sum() / nN)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def boot_cells(cells, draws=DRAWS, seed=SEED):
    """cells: list of (y, s, spk) arrays. One rng, one speaker draw per replicate shared by all cells
    (single cell -> ordinary bootstrap; two cells -> paired). Speakers = sorted union of cell speakers."""
    allspk = np.unique(np.concatenate([c[2] for c in cells]))
    K = len(allspk)
    idx = [{u: np.where(c[2] == u)[0] for u in allspk} for c in cells]
    rng = np.random.default_rng(seed)
    out = [[] for _ in cells]; usable = 0
    for _ in range(draws):
        pick = allspk[rng.integers(0, K, K)]
        vals = []
        for ci, (y, s, _) in enumerate(cells):
            ii = np.concatenate([idx[ci][u] for u in pick])
            vals.append(auc_mw(y[ii], s[ii]))
        if all(np.isfinite(v) for v in vals):
            usable += 1
            for ci, v in enumerate(vals):
                out[ci].append(v)
    return [np.array(o) for o in out], usable, K


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def stable_gkf(groups, n_splits=5):
    groups = np.asarray(groups)
    ug, gi = np.unique(groups, return_inverse=True)
    cnt = np.bincount(gi)
    order = np.argsort(cnt, kind="stable")[::-1]
    load = np.zeros(n_splits); g2f = np.zeros(len(ug), int)
    for g in order:
        f = int(np.argmin(load)); load[f] += cnt[g]; g2f[g] = f
    return g2f[gi]


def key_of(row):
    for c in ("id", "clip", "clip_id", "path"):
        if c in row and row[c]:
            return os.path.basename(row[c]).replace(".wav", "")
    raise KeyError("no clip id column")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--score", default="p_yes")
    ap.add_argument("--ref", required=True); ap.add_argument("--folds"); ap.add_argument("--run_manifest")
    ap.add_argument("--sidecar"); ap.add_argument("--out", required=True)
    ap.add_argument("--skip_folds_sklearn", action="store_true")
    a = ap.parse_args()
    R = {"csv": os.path.abspath(a.csv), "csv_md5": hashlib.md5(open(a.csv, "rb").read()).hexdigest(), "checks": {}}
    rows = list(csv.DictReader(open(a.csv)))
    ref = list(csv.DictReader(open(a.ref)))
    refkey = [f"{r['set']}_{r['speaker_id']}_{r['seg_uid']}" for r in ref]
    refinfo = {}
    for r, k in zip(ref, refkey):
        refinfo.setdefault(k, set()).add((int(r["label"]), str(r["speaker_id"]), r["set"]))
    keys = [key_of(r) for r in rows]
    arm = np.array([k.split("_")[0] for k in keys]); spk = np.array([k.split("_")[1] for k in keys])
    seg = np.array([k.split("_")[2] for k in keys])
    y = np.array([int(float(r["label"])) for r in rows]); s = np.array([float(r[a.score]) for r in rows])
    # consistency of csv speaker / arm columns with the clip id
    for c in ("speaker", "speaker_id"):
        if c in rows[0]:
            R["checks"][f"csv_{c}_matches_id"] = bool(all(str(r[c]) == sp for r, sp in zip(rows, spk)))
    for c in ("arm", "set"):
        if c in rows[0] and rows[0][c]:
            R["checks"][f"csv_{c}_matches_id"] = bool(all(r[c] == ar for r, ar in zip(rows, arm)))
    # labels against the reference manifest, multiset and positional
    R["checks"]["label_matches_ref"] = bool(all((y[i], spk[i], arm[i]) in refinfo.get(keys[i], set()) for i in range(len(rows))))
    R["checks"]["multiset_equals_ref"] = sorted(keys) == sorted(refkey)
    R["checks"]["positional_order_equals_ref"] = keys == refkey
    R["checks"]["no_nan_scores"] = bool(np.isfinite(s).all())
    R["n"] = len(rows); R["n_speakers"] = int(len(np.unique(spk)))
    R["arm_counts"] = {ar: int((arm == ar).sum()) for ar in np.unique(arm)}
    R["arm_pos"] = {ar: int(y[arm == ar].sum()) for ar in np.unique(arm)}
    R["arm_speakers"] = {ar: int(len(np.unique(spk[arm == ar]))) for ar in np.unique(arm)}
    if "prompt" in rows[0]:
        R["checks"]["csv_prompt_all_verbatim"] = bool(all(r["prompt"] == PROMPT for r in rows))
    if a.sidecar:
        sc = json.load(open(a.sidecar)); R["sidecar"] = os.path.abspath(a.sidecar)
        found = []
        def walk(o, path=""):
            if isinstance(o, dict):
                for kk, vv in o.items(): walk(vv, f"{path}.{kk}")
            elif isinstance(o, list):
                for ii, vv in enumerate(o): walk(vv, f"{path}[{ii}]")
            elif isinstance(o, str) and "depression" in o:
                found.append((path, o == PROMPT))
        walk(sc)
        R["checks"]["sidecar_prompt_fields"] = found
        R["checks"]["sidecar_prompt_verbatim"] = bool(found) and all(v for _, v in found)
    # distinct agreement segments
    mA = arm == "agreement"; mC = arm == "conflict"
    first = {}; spread = 0.0
    for i in np.where(mA)[0]:
        if seg[i] in first:
            spread = max(spread, abs(s[i] - s[first[seg[i]]]))
        else:
            first[seg[i]] = i
    dist = np.array(sorted(first.values()))
    R["distinct_agreement_segments"] = int(len(dist)); R["max_p_yes_spread_within_repeated_segment"] = float(spread)
    cells = {"conflict": np.where(mC)[0], "agreement483": np.where(mA)[0], "agreement311": dist,
             "pooled966": np.arange(len(rows))}
    V = {}
    for nm, ix in cells.items():
        mw = auc_mw(y[ix], s[ix]); tr = auc_trap(y[ix], s[ix])
        (b,), usable, K = boot_cells([(y[ix], s[ix], spk[ix])])
        lo, hi = ci(b)
        V[nm] = {"auc": float(mw), "auc_trap": tr, "mw_minus_trap": float(mw - tr), "lo": lo, "hi": hi,
                 "n": int(len(ix)), "n_spk": int(K), "n_pos": int(y[ix].sum()), "n_neg": int((1 - y[ix]).sum()),
                 "usable": usable}
    for nm, agr in (("paired_conf_minus_agr483", "agreement483"), ("paired_conf_minus_agr311", "agreement311")):
        ic = cells["conflict"]; ia = cells[agr]
        (bc, ba), usable, K = boot_cells([(y[ic], s[ic], spk[ic]), (y[ia], s[ia], spk[ia])])
        d = bc - ba; lo, hi = ci(d)
        V[nm] = {"auc": V["conflict"]["auc"] - V[agr]["auc"], "lo": lo, "hi": hi, "n": int(len(ic) + len(ia)),
                 "n_spk": int(K), "usable": usable, "frac_diff_below0": float((d < 0).mean())}
    R["values"] = V
    # folds
    if a.folds:
        fr = list(csv.DictReader(open(a.folds))); R["folds_file"] = os.path.abspath(a.folds)
        fspk = np.array([str(r["speaker_id"]) for r in fr]); ff = np.array([int(r["fold"]) for r in fr])
        s2f = {}
        bad = 0
        for sp_, f_ in zip(fspk, ff):
            if sp_ in s2f and s2f[sp_] != f_: bad += 1
            s2f.setdefault(sp_, f_)
        R["folds_n_rows"] = len(fr); R["folds_n_speakers"] = len(s2f)
        R["checks"]["folds_speaker_disjoint"] = bad == 0
        R["folds_sizes_rows"] = [int((ff == k).sum()) for k in range(5)]
        mine = stable_gkf(fspk)
        R["checks"]["folds_equal_my_stable_gkf"] = bool((mine == ff).all())
        R["folds_mismatch_vs_my_stable"] = int((mine != ff).sum())
        # the fold file must also cover every row of the csv with the same speaker
        R["checks"]["folds_cover_csv_speakers"] = set(s2f) == set(spk.tolist())
        # every speaker's conflict and agreement rows in one fold by construction
        if a.run_manifest:
            mr = list(csv.DictReader(open(a.run_manifest)))
            mspk = np.array([str(r["speaker"]) for r in mr]); my = np.array([int(r["label"]) for r in mr])
            R["run_manifest"] = os.path.abspath(a.run_manifest); R["run_manifest_n"] = len(mr)
            R["checks"]["run_manifest_speakers_equal_folds_rows"] = bool(len(mr) == len(fr) and (mspk == fspk).all())
            mine2 = stable_gkf(mspk)
            R["checks"]["folds_equal_my_stable_gkf_on_run_manifest"] = bool(len(mr) == len(fr) and (mine2 == ff).all())
            if not a.skip_folds_sklearn:
                from sklearn.model_selection import GroupKFold
                import sklearn
                sk = np.full(len(mr), -1)
                for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(mr)), my, groups=mspk)):
                    sk[te] = k
                R["sklearn_version"] = sklearn.__version__; R["numpy_version"] = np.__version__
                R["checks"]["folds_equal_sklearn_gkf_here"] = bool(len(mr) == len(fr) and (sk == ff).all())
                R["folds_mismatch_vs_sklearn_here"] = int((sk != ff).sum()) if len(mr) == len(fr) else None
    json.dump(R, open(a.out, "w"), indent=1)
    for nm, v in V.items():
        print(f"{nm:28s} {v['auc']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] n={v['n']} spk={v['n_spk']} usable={v['usable']}"
              + (f" trapdiff={v['mw_minus_trap']:.1e}" if 'mw_minus_trap' in v else ""))
    print(json.dumps(R["checks"], indent=0))
    print({k: R[k] for k in R if k in ("n", "n_speakers", "arm_counts", "arm_pos", "arm_speakers",
                                        "distinct_agreement_segments", "max_p_yes_spread_within_repeated_segment",
                                        "folds_sizes_rows", "folds_mismatch_vs_my_stable", "folds_mismatch_vs_sklearn_here")})


if __name__ == "__main__":
    main()
