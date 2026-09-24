"""PART26 table check. Recomputes every cell of the 3 x 7 grid from the files on disk, with its own code.

For each cell it reads the per-clip csv, the zero-shot file named in part25/C/C_build_cells.csv, the saved
release fold files and the saved draws npz. It does not import any project script.
  - per-repeat AUC: own midrank Mann-Whitney (scipy rankdata for average ranks only)
  - answer AUC: same formula on p_yes of the zero-shot file (joined by clip)
  - gap: mean of the five per-repeat AUCs minus the answer AUC
  - speaker bootstrap: 2000 draws, fresh numpy default_rng(0), np.unique speakers of the per-clip file,
    rng.choice(n_spk, n_spk, replace=True), all clips of each drawn speaker; 2.5 and 97.5 percentiles
Read only everywhere except the output json path given on the command line.
"""
import csv, hashlib, json, sys, os
import numpy as np
from scipy.stats import rankdata

P26 = "scores/part26"
FOLDS = "<local data dir>/release/folds"
CB = "<local data dir>/release_from_mac/scores/part25/C/C_build_cells.csv"

DS_KEY = {"PC-GITA": "pcgita", "NeuroVoz": "neurovoz", "MDVR-KCL": "kcl", "E-DAIC": "edaic",
          "Pitt": "pitt", "ADReSSo": "adresso", "ADReSS-2020": "adress2020"}
MODELS = ["Qwen2-Audio", "Qwen3-Omni-30B-A3B", "Kimi-Audio"]
DATASETS = ["PC-GITA", "NeuroVoz", "MDVR-KCL", "E-DAIC", "Pitt", "ADReSSo", "ADReSS-2020"]

CELL = {
 ("Qwen2-Audio", "PC-GITA"): ("Qwen2-Audio_PC-GITA/Qwen2-Audio_PC-GITA_perclip.csv", "Qwen2-Audio_PC-GITA/draws/Qwen2-Audio_PC-GITA_draws.npz", "Qwen2-Audio_PC-GITA/Qwen2-Audio_PC-GITA_perclip.sidecar.json"),
 ("Qwen2-Audio", "NeuroVoz"): ("Qwen2-Audio_NeuroVoz/per_clip/p26_q2a_neurovoz_perclip.csv", "Qwen2-Audio_NeuroVoz/draws/p26_q2a_neurovoz_draws.npz", "Qwen2-Audio_NeuroVoz/per_clip/p26_q2a_neurovoz_perclip.sidecar.json"),
 ("Qwen2-Audio", "MDVR-KCL"): ("Qwen2-Audio_MDVR-KCL/Q2A_kcl_perclip.csv", "Qwen2-Audio_MDVR-KCL/Q2A_kcl_draws.npz", "Qwen2-Audio_MDVR-KCL/Q2A_kcl_perclip.sidecar.json"),
 ("Qwen2-Audio", "E-DAIC"): ("Qwen2-Audio_E-DAIC/Qwen2-Audio_E-DAIC_perclip.csv", "Qwen2-Audio_E-DAIC/Qwen2-Audio_E-DAIC_draws.npz", "Qwen2-Audio_E-DAIC/Qwen2-Audio_E-DAIC_perclip.sidecar.json"),
 ("Qwen2-Audio", "Pitt"): ("Qwen2-Audio_Pitt/per_clip/q2a_pitt_perclip.csv", "Qwen2-Audio_Pitt/draws/q2a_pitt_draws.npz", "Qwen2-Audio_Pitt/per_clip/q2a_pitt_perclip.sidecar.json"),
 ("Qwen2-Audio", "ADReSSo"): ("Qwen2-Audio_ADReSSo/per_clip/Qwen2-Audio_ADReSSo_perclip.csv", "Qwen2-Audio_ADReSSo/draws/Qwen2-Audio_ADReSSo_draws.npz", "Qwen2-Audio_ADReSSo/per_clip/Qwen2-Audio_ADReSSo_perclip.sidecar.json"),
 ("Qwen2-Audio", "ADReSS-2020"): ("Qwen2-Audio_ADReSS-2020/Qwen2-Audio_ADReSS-2020_perclip.csv", "Qwen2-Audio_ADReSS-2020/Qwen2-Audio_ADReSS-2020_draws.npz", "Qwen2-Audio_ADReSS-2020/Qwen2-Audio_ADReSS-2020.sidecar.json"),
 ("Qwen3-Omni-30B-A3B", "PC-GITA"): ("Qwen3-Omni-30B-A3B_PC-GITA/q3o_pcgita_perclip.csv", "Qwen3-Omni-30B-A3B_PC-GITA/q3o_pcgita_draws.npz", "Qwen3-Omni-30B-A3B_PC-GITA/q3o_pcgita_perclip.sidecar.json"),
 ("Qwen3-Omni-30B-A3B", "NeuroVoz"): ("Qwen3-Omni-30B-A3B_NeuroVoz/per_clip/Qwen3-Omni-30B-A3B_NeuroVoz_perclip.csv", "Qwen3-Omni-30B-A3B_NeuroVoz/draws/Qwen3-Omni-30B-A3B_NeuroVoz_draws.npz", "Qwen3-Omni-30B-A3B_NeuroVoz/per_clip/Qwen3-Omni-30B-A3B_NeuroVoz_perclip.sidecar.json"),
 ("Qwen3-Omni-30B-A3B", "MDVR-KCL"): ("Qwen3-Omni-30B-A3B_MDVR-KCL/per_clip/p26_q3o_kcl_perclip.csv", "Qwen3-Omni-30B-A3B_MDVR-KCL/draws/p26_q3o_kcl_draws.npz", "Qwen3-Omni-30B-A3B_MDVR-KCL/per_clip/p26_q3o_kcl_perclip.sidecar.json"),
 ("Qwen3-Omni-30B-A3B", "E-DAIC"): ("Qwen3-Omni-30B-A3B_E-DAIC/Q3O_EDAIC_perclip.csv", "Qwen3-Omni-30B-A3B_E-DAIC/Q3O_EDAIC_draws.npz", "Qwen3-Omni-30B-A3B_E-DAIC/Q3O_EDAIC_perclip.sidecar.json"),
 ("Qwen3-Omni-30B-A3B", "Pitt"): ("Qwen3-Omni-30B-A3B_Pitt/per_clip/Qwen3-Omni-30B-A3B_Pitt_perclip.csv", "Qwen3-Omni-30B-A3B_Pitt/draws/Qwen3-Omni-30B-A3B_Pitt_draws.npz", "Qwen3-Omni-30B-A3B_Pitt/per_clip/Qwen3-Omni-30B-A3B_Pitt_perclip.sidecar.json"),
 ("Qwen3-Omni-30B-A3B", "ADReSSo"): ("Qwen3-Omni-30B-A3B_ADReSSo/Qwen3-Omni-30B-A3B_ADReSSo_perclip.csv", "Qwen3-Omni-30B-A3B_ADReSSo/draws/Qwen3-Omni-30B-A3B_ADReSSo_draws.npz", "Qwen3-Omni-30B-A3B_ADReSSo/Qwen3-Omni-30B-A3B_ADReSSo_perclip.sidecar.json"),
 ("Qwen3-Omni-30B-A3B", "ADReSS-2020"): ("Qwen3-Omni-30B-A3B_ADReSS-2020/per_clip/p26_q3o_adress2020_perclip.csv", "Qwen3-Omni-30B-A3B_ADReSS-2020/draws/p26_q3o_adress2020_draws.npz", "Qwen3-Omni-30B-A3B_ADReSS-2020/per_clip/p26_q3o_adress2020_perclip.sidecar.json"),
 ("Kimi-Audio", "PC-GITA"): ("Kimi-Audio_PC-GITA/Kimi-Audio_PC-GITA_perclip.csv", "Kimi-Audio_PC-GITA/draws/Kimi-Audio_PC-GITA_draws.npz", "Kimi-Audio_PC-GITA/Kimi-Audio_PC-GITA_perclip.sidecar.json"),
 ("Kimi-Audio", "NeuroVoz"): ("Kimi-Audio_NeuroVoz/per_clip/p26_kimi_neurovoz_perclip.csv", "Kimi-Audio_NeuroVoz/draws/p26_kimi_neurovoz_draws.npz", "Kimi-Audio_NeuroVoz/per_clip/p26_kimi_neurovoz_perclip.sidecar.json"),
 ("Kimi-Audio", "MDVR-KCL"): ("Kimi-Audio_MDVR-KCL/per_clip/p26_kimi_kcl_perclip.csv", "Kimi-Audio_MDVR-KCL/draws/p26_kimi_kcl_draws.npz", "Kimi-Audio_MDVR-KCL/per_clip/p26_kimi_kcl_perclip.sidecar.json"),
 ("Kimi-Audio", "E-DAIC"): ("Kimi-Audio_E-DAIC/p26_kimi_edaic_enc_perclip.csv", "Kimi-Audio_E-DAIC/draws/p26_kimi_edaic_enc_draws.npz", "Kimi-Audio_E-DAIC/p26_kimi_edaic_enc_perclip.sidecar.json"),
 ("Kimi-Audio", "Pitt"): ("Kimi-Audio_Pitt/Kimi-Audio_Pitt_perclip.csv", "Kimi-Audio_Pitt/Kimi-Audio_Pitt_draws.npz", "Kimi-Audio_Pitt/Kimi-Audio_Pitt.sidecar.json"),
 ("Kimi-Audio", "ADReSSo"): ("Kimi-Audio_ADReSSo/per_clip/Kimi-Audio_ADReSSo_perclip.csv", "Kimi-Audio_ADReSSo/draws/Kimi-Audio_ADReSSo_draws.npz", "Kimi-Audio_ADReSSo/per_clip/Kimi-Audio_ADReSSo_perclip.sidecar.json"),
 ("Kimi-Audio", "ADReSS-2020"): ("Kimi-Audio_ADReSS-2020/per_clip/p26_kimi_adress2020_perclip.csv", "Kimi-Audio_ADReSS-2020/draws/p26_kimi_adress2020_draws.npz", "Kimi-Audio_ADReSS-2020/per_clip/p26_kimi_adress2020_perclip.sidecar.json"),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)  # average ranks for ties
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def fold_file(model, ds, k):
    key = DS_KEY[ds]
    if model == "Qwen3-Omni-30B-A3B" and ds == "Pitt":
        return f"{FOLDS}/pitt_groupkfold5_pod_Control15ids_seed{k}.csv"
    if ds == "PC-GITA":
        return f"{FOLDS}/pcgita_groupkfold5_pod_seed{k}.csv"
    return f"{FOLDS}/{key}_groupkfold5_pod_seed{k}_UNVERIFIED.csv"


def probe_cols(header):
    a = [f"p_probe_seed{k}" for k in range(5)]
    if all(c in header for c in a):
        return a
    b = [f"p_seed{k}" for k in range(5)]
    if all(c in header for c in b):
        return b
    raise KeyError("no probe columns: " + ",".join(header))


def flatten_numbers(d, out=None):
    if out is None:
        out = []
    if isinstance(d, dict):
        for v in d.values():
            flatten_numbers(v, out)
    elif isinstance(d, list):
        for v in d:
            flatten_numbers(v, out)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out.append(float(d))
    return out


def main(outp):
    cb = {(r["model"], r["dataset"]): r for r in read_csv(CB) if r["probe_stream"].startswith(("encoder", "LM"))}
    res = {}
    for (model, ds), (pc, dr, sc) in CELL.items():
        R = {"model": model, "dataset": ds, "checks": {}, "files": {}}
        C = R["checks"]
        pc, dr, sc = (os.path.join(P26, x) for x in (pc, dr, sc))
        R["files"] = {"perclip": pc, "draws": dr, "sidecar": sc}
        rows = read_csv(pc)
        header = list(rows[0].keys())
        pcols = probe_cols(header)
        clips = [r["clip"] for r in rows]
        y = np.array([int(float(r["label"])) for r in rows])
        spk = np.array([r["speaker"] for r in rows])
        P = np.array([[float(r[c]) for c in pcols] for r in rows])
        n = len(rows)
        C["clips_unique"] = len(set(clips)) == n
        C["probe_scores_finite"] = bool(np.isfinite(P).all())
        # one label per speaker
        lab_by_spk = {}
        ok = True
        for s_, y_ in zip(spk, y):
            if lab_by_spk.setdefault(s_, y_) != y_:
                ok = False
        C["label_constant_within_speaker"] = ok
        # zero-shot file from C_build_cells.csv (the Kimi LM row names the same answer file)
        cbrow = cb[(model.replace("Kimi-Audio", "Kimi-Audio"), ds)]
        zsp = cbrow["zs_file"]
        R["files"]["zeroshot"] = zsp
        zsha = sha(zsp)
        C["zeroshot_sha256_equals_C_build_cells"] = zsha == cbrow["zs_file_sha256"]
        zrows = {r["clip"]: r for r in read_csv(zsp)}
        C["zeroshot_clip_set_equal"] = set(zrows) == set(clips)
        zy = np.array([int(float(zrows[c]["label"])) for c in clips])
        zp = np.array([float(zrows[c]["p_yes"]) for c in clips])
        C["zeroshot_labels_equal"] = bool((zy == y).all())
        if "p_yes_zeroshot" in header:
            pz = np.array([float(r["p_yes_zeroshot"]) for r in rows])
            C["perclip_p_yes_equals_zeroshot_file"] = float(np.max(np.abs(pz - zp))) == 0.0
        # speaker grouping of zero-shot file is a coarsening-free match (bijection), except the known PC-GITA stem case
        zs_spk = np.array([zrows[c]["speaker"] for c in clips])
        pairs = set(zip(spk, zs_spk))
        R["zs_speaker_ids"] = len(set(zs_spk))
        R["zs_speaker_bijection"] = len(pairs) == len(set(spk)) == len(set(zs_spk))
        # folds
        fold_ok = True; disjoint_ok = True; fold_info = {}
        for k in range(5):
            ff = fold_file(model, ds, k)
            fr = {r["clip_id"]: r for r in read_csv(ff)}
            col = np.array([int(float(r[f"fold_seed{k}"])) for r in rows])
            same_set = set(fr) == set(clips)
            eq = same_set and all(int(fr[c]["fold"]) == f for c, f in zip(clips, col))
            # speaker partition of the fold file equals per-clip speaker partition
            part_ok = same_set and len(set(zip(spk, [fr[c]["speaker_id"] for c in clips]))) == len(set(spk))
            dj = all(len(set(col[spk == s_])) == 1 for s_ in np.unique(spk))
            fold_info[f"seed{k}"] = {"file": os.path.basename(ff), "sha256": sha(ff), "equal": bool(eq), "speaker_partition_equal": bool(part_ok), "speaker_disjoint": bool(dj), "sizes": np.bincount(col).tolist()}
            fold_ok &= bool(eq) and bool(part_ok); disjoint_ok &= bool(dj)
        C["fold_columns_equal_release_fold_files"] = fold_ok
        C["outer_folds_speaker_disjoint"] = disjoint_ok
        C["five_repeat_splits_distinct"] = len({tuple(int(float(r[f"fold_seed{k}"])) for r in rows) for k in range(5)}) == 5
        R["folds"] = fold_info
        # point values
        per = np.array([auc(y, P[:, k]) for k in range(5)])
        mean5 = float(per.mean()); ans = float(auc(y, zp)); gap = mean5 - ans
        # bootstrap
        uspk = np.unique(spk)
        idx_of = {s_: np.where(spk == s_)[0] for s_ in uspk}
        rng = np.random.default_rng(0)
        draws = np.full(2000, np.nan); dm = np.full(2000, np.nan); dz = np.full(2000, np.nan)
        allidx = np.empty((2000, len(uspk)), dtype=np.int64)
        for b in range(2000):
            pick = rng.choice(len(uspk), size=len(uspk), replace=True)
            allidx[b] = pick
            ii = np.concatenate([idx_of[uspk[j]] for j in pick])
            yb = y[ii]
            if yb.min() == yb.max():
                continue
            m = np.mean([auc(yb, P[ii, k]) for k in range(5)])
            z = auc(yb, zp[ii])
            dm[b] = m; dz[b] = z; draws[b] = m - z
        use = np.isfinite(draws)
        lo, hi = np.percentile(draws[use], [2.5, 97.5])
        plo, phi = np.percentile(dm[use], [2.5, 97.5]); zlo, zhi = np.percentile(dz[use], [2.5, 97.5])
        R.update({"n": n, "n_pos": int(y.sum()), "n_speakers": int(len(uspk)), "per_repeat": per.tolist(), "probe_mean5": mean5,
                  "answer": ans, "gap": gap, "lo": float(lo), "hi": float(hi), "usable_draws": int(use.sum()),
                  "probe_ci": [float(plo), float(phi)], "answer_ci": [float(zlo), float(zhi)],
                  "share_draws_le0": float(np.mean(draws[use] <= 0)),
                  "interval_above_zero": bool(lo > 0), "interval_below_zero": bool(hi < 0)})
        # compare with saved draws
        Z = np.load(dr, allow_pickle=True)
        sd = Z["diff"]
        C["saved_draws_2000"] = sd.shape == (2000,)
        C["saved_draws_equal_mine_1e-12"] = bool(np.nanmax(np.abs(sd - draws)) < 1e-12) and bool((np.isfinite(sd) == use).all())
        R["saved_draws_maxabs"] = float(np.nanmax(np.abs(sd - draws)))
        if "speaker_idx" in Z.files:
            C["saved_speaker_idx_equal_mine"] = bool((Z["speaker_idx"] == allidx).all())
        spk_key = "speakers" if "speakers" in Z.files else ("speakers_sorted" if "speakers_sorted" in Z.files else None)
        if spk_key:
            C["saved_speaker_list_equal_np_unique"] = [str(s_) for s_ in Z[spk_key]] == [str(s_) for s_ in uspk]
        slo, shi = np.percentile(sd[np.isfinite(sd)], [2.5, 97.5])
        C["saved_interval_equals_mine"] = abs(slo - lo) < 1e-12 and abs(shi - hi) < 1e-12
        if "point_diff" in Z.files:
            C["saved_point_gap_equals_mine"] = abs(float(Z["point_diff"]) - gap) < 1e-12
        # sidecar holds my numbers (exact to 1e-9, or at 4 dp)
        S = json.load(open(sc))
        nums = np.array(flatten_numbers(S))
        def has(v, tol=1e-9):
            return bool(np.any(np.abs(nums - v) < tol))
        def has4(v):
            return has(v) or has(round(v, 4), 5e-9)
        C["sidecar_has_probe_mean5"] = has4(mean5)
        C["sidecar_has_answer"] = has4(ans)
        C["sidecar_has_gap"] = has4(gap)
        C["sidecar_has_lo"] = has4(float(lo))
        C["sidecar_has_hi"] = has4(float(hi))
        C["sidecar_has_all_5_repeats"] = all(has4(float(v)) for v in per)
        R["all_pass"] = all(bool(v) for v in C.values())
        R["failed"] = [k for k, v in C.items() if not v]
        res[f"{model}|{ds}"] = R
        print(f"{model:20s} {ds:12s} probe {mean5:.4f} ans {ans:.4f} gap {gap:+.4f} [{lo:+.4f}, {hi:+.4f}] n={n} spk={len(uspk)} use={int(use.sum())} pass={R['all_pass']} {R['failed']}", flush=True)
    import scipy
    out = {"script": os.path.abspath(__file__), "numpy": np.__version__, "scipy": scipy.__version__, "cells": res}
    json.dump(out, open(outp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main(sys.argv[1])
