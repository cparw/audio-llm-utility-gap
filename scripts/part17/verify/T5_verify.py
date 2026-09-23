#!/usr/local/bin/python3
"""PART 17 task T5, independent verifier.

Written from scratch. Reads only the ORIGINAL source files (zero-shot per clip
scores, per layer encoder curve files, the figure script, master_lookup.csv,
saved hidden states). The main job's outputs are opened only to check that
they exist and to compare its claimed numbers against mine; none of them is
used as an input to any computation here.

AUC: explicit Mann-Whitney over every positive x negative pair, ties 0.5,
cross-checked with a trapezoid over the full empirical ROC.
Interval: 2000 draws, fresh numpy.random.default_rng(0) per cell, speakers
resampled with replacement, percentile 2.5 / 97.5.

usage: /usr/local/bin/python3 T5_verify.py
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P17 = os.path.dirname(HERE)
REL = "<local data dir>/release"
OF = f"{REL}/omni_final"
MIRROR = f"{REL}/overnight/curves/omni"
VAR = f"{REL}/edaic_rerun/variants"
FIGSCRIPT = f"{REL}/edaic_rerun/make_fig1_v4.py"
MASTER = f"{REL}/master/master_lookup.csv"
B = 2000
SEED = 0

# what the main job claimed (typed from its report, for comparison only)
CLAIM = {
    "pitt":          dict(ans=0.6578, lo=0.6033, hi=0.7125, above=31, msd=25, aci=25, mn=0.6517, mnL=2, med=0.7597, mx=0.8083, mxL=27, n=468, spk=228),
    "adresso":       dict(ans=0.6150, lo=0.5432, hi=0.6890, above=32, msd=32, aci=32, mn=0.7413, mnL=0, med=0.8604, mx=0.9030, mxL=27, n=237, spk=237),
    "adress2020":    dict(ans=0.6241, lo=0.5348, hi=0.7148, above=32, msd=26, aci=24, mn=0.6502, mnL=2, med=0.7855, mx=0.8248, mxL=25, n=156, spk=156),
    "pcgita":        dict(ans=0.5635, lo=0.4931, hi=0.6331, above=32, msd=32, aci=32, mn=0.8720, mnL=0, med=0.9065, mx=0.9234, mxL=6, n=1100, spk=100),
    "neurovoz":      dict(ans=0.6951, lo=0.6356, hi=0.7519, above=32, msd=32, aci=32, mn=0.8365, mnL=0, med=0.9218, mx=0.9335, mxL=25, n=1270, spk=107),
    "kcl":           dict(ans=0.7143, lo=0.5210, hi=0.8810, above=32, msd=15, aci=0, mn=0.7470, mnL=0, med=0.8021, mx=0.8482, mxL=19, n=37, spk=37),
    "edaic_full":    dict(ans=0.8285, lo=0.7708, hi=0.8827, above=0, mn=0.5684, med=0.6224, mx=0.6896, mxL=19, n=275, spk=275),
    "edaic_mid300":  dict(ans=0.8122, lo=0.7479, hi=0.8707, above=0, mx=0.6839, mxL=22, n=275, spk=275),
    "edaic_mid30":   dict(ans=0.7205, lo=0.6531, hi=0.7917, above=0, mx=0.6705, mxL=2, n=275, spk=275),
    "edaic_first30": dict(ans=0.6620, lo=0.5831, hi=0.7315, above=2, msd=0, n=275, spk=275),
}

# dataset -> zero-shot file, curve file + column (what is/would be plotted), stats file
CELLS = [
    ("pitt", "AD", "plotted", f"{OF}/omni_pitt_zeroshot_scores.csv", f"{MIRROR}/pitt_encoder.csv", "auc", f"{OF}/omni_pitt_encoder_perlayer.csv", ("Qwen2.5-Omni", "pitt")),
    ("adresso", "AD", "omitted", f"{OF}/omni_adresso_zeroshot_scores.csv", f"{OF}/omni_adresso_encoder_perlayer.csv", "auc_oof", f"{OF}/omni_adresso_encoder_perlayer.csv", ("Qwen2.5-Omni", "adresso")),
    ("adress2020", "AD", "omitted", f"{OF}/omni_adress2020_zeroshot_scores.csv", f"{OF}/omni_adress2020_encoder_perlayer.csv", "auc_oof", f"{OF}/omni_adress2020_encoder_perlayer.csv", ("Qwen2.5-Omni", "adress2020")),
    ("pcgita", "PD", "plotted", f"{OF}/omni_pcgita_zeroshot_scores.csv", f"{MIRROR}/pcgita_encoder.csv", "auc", f"{OF}/omni_pcgita_encoder_perlayer.csv", ("Qwen2.5-Omni", "pcgita")),
    ("neurovoz", "PD", "omitted", f"{OF}/omni_neurovoz_zeroshot_scores.csv", f"{OF}/omni_neurovoz_encoder_perlayer.csv", "auc_oof", f"{OF}/omni_neurovoz_encoder_perlayer.csv", ("Qwen2.5-Omni", "neurovoz")),
    ("kcl", "PD", "omitted", f"{OF}/omni_kcl_zeroshot_scores.csv", f"{OF}/omni_kcl_encoder_perlayer.csv", "auc_oof", f"{OF}/omni_kcl_encoder_perlayer.csv", ("Qwen2.5-Omni", "kcl")),
    ("edaic_full", "MDD", "plotted", f"{VAR}/o25_full_zeroshot_scores.csv", f"{VAR}/o25_full_encoder_perlayer.csv", "auc_oof", f"{VAR}/o25_full_encoder_perlayer.csv", ("Qwen2.5-Omni", "edaic_full")),
    ("edaic_mid300", "MDD", "variant", f"{VAR}/o25_mid300_zeroshot_scores.csv", f"{VAR}/o25_mid300_encoder_perlayer.csv", "auc_oof", f"{VAR}/o25_mid300_encoder_perlayer.csv", ("Qwen2.5-Omni", "edaic_mid300")),
    ("edaic_mid30", "MDD", "variant", f"{VAR}/o25_mid30_zeroshot_scores.csv", f"{VAR}/o25_mid30_encoder_perlayer.csv", "auc_oof", f"{VAR}/o25_mid30_encoder_perlayer.csv", ("Qwen2.5-Omni", "edaic_mid30")),
    ("edaic_first30", "MDD", "reference", f"{OF}/omni_edaic_zeroshot_scores.csv", f"{OF}/omni_edaic_encoder_perlayer.csv", "auc_oof", f"{OF}/omni_edaic_encoder_perlayer.csv", ("Qwen2.5-Omni", "edaic")),
]
# answer values hard-coded in the figure script (read from the script text below, not typed here)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- AUC, two ways
def pair_matrix(s, y):
    pos = s[y == 1]
    neg = s[y == 0]
    return (pos[:, None] > neg[None, :]).astype(np.float64) + 0.5 * (pos[:, None] == neg[None, :])


def auc_mann_whitney(s, y):
    C = pair_matrix(s, y)
    return float(C.sum() / C.size)


def auc_trapezoid(s, y):
    # full empirical ROC: one point per distinct threshold, descending
    thr = np.unique(s)[::-1]
    P = float((y == 1).sum())
    N = float((y == 0).sum())
    tpr = [0.0]
    fpr = [0.0]
    for t in thr:
        tpr.append(((s >= t) & (y == 1)).sum() / P)
        fpr.append(((s >= t) & (y == 0)).sum() / N)
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def speaker_boot(s, y, spk, speaker_order="sorted_str"):
    """Weighted exact pairwise AUC over speaker resamples.
    A speaker drawn k times contributes each of its clips k times, so every clip
    gets weight k; the pairwise AUC of the expanded multiset is
    w_pos^T C w_neg / (sum w_pos * sum w_neg)."""
    if speaker_order == "sorted_str":
        uniq = np.array(sorted(set(spk.tolist())))
    elif speaker_order == "sorted_num":
        uniq = np.array(sorted(set(spk.tolist()), key=lambda v: float(v)))
    elif speaker_order == "appearance":
        seen = []
        for v in spk.tolist():
            if v not in seen:
                seen.append(v)
        uniq = np.array(seen)
    else:
        raise ValueError(speaker_order)
    S = len(uniq)
    index = {v: i for i, v in enumerate(uniq.tolist())}
    clip_spk = np.array([index[v] for v in spk.tolist()])
    C = pair_matrix(s, y)
    pos_spk = clip_spk[y == 1]
    neg_spk = clip_spk[y == 0]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(B):
        draw = rng.integers(0, S, size=S)
        k = np.bincount(draw, minlength=S).astype(np.float64)
        wp = k[pos_spk]
        wn = k[neg_spk]
        if wp.sum() == 0 or wn.sum() == 0:
            continue
        v = float(np.einsum("i,ij,j->", wp, C, wn) / (wp.sum() * wn.sum()))
        if len(vals) < 5:   # non-BLAS cross-check on the first draws
            v2 = float(sum(wp[i] * float((C[i] * wn).sum()) for i in range(len(wp))) / (wp.sum() * wn.sum()))
            assert abs(v - v2) < 1e-12, (v, v2)
        vals.append(v)
    vals = np.array(vals)
    assert np.all(np.isfinite(vals)), "non-finite bootstrap value"
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals), S


# ---------------------------------------------------------------- figure script
def figure_answers():
    txt = open(FIGSCRIPT).read()
    out = {}
    import re
    m = re.search(r'EDAIC_ANSWER\s*=\s*\{([^}]*)\}', txt)
    for k, v in re.findall(r'"(\w+)":\s*([0-9.]+)', m.group(1)):
        out["edaic_" + k] = float(v)
    out["pcgita"] = float(re.search(r'PCGITA_ANSWER\s*=\s*([0-9.]+)', txt).group(1))
    out["pitt"] = float(re.search(r'PITT_ANSWER\s*=\s*([0-9.]+)', txt).group(1))
    curves = re.findall(r'curve\(f"\{(\w+)\}/([\w{}]+\.csv)",\s*"(\w+)"', txt)
    return out, curves, txt


def master_lookup():
    rows = read_rows(MASTER)
    d = {}
    for r in rows:
        if r["stream"] == "zero shot answer":
            d[(r["model"], r["dataset"])] = (float(r["auc"]), int(r["n"]), r["source"])
    return d


def f4(x):
    return "" if x is None else f"{x:.4f}"


# ---------------------------------------------------------------- per layer recompute
def _layer_fit(X, y, spk, l):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    import warnings
    warnings.filterwarnings("ignore")
    oof = np.zeros(len(y))
    fold_auc = []
    for tr, te in GroupKFold(n_splits=5).split(X[:, l], y, groups=spk):
        sc = StandardScaler().fit(X[tr, l])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr, l]), y[tr])
        p = clf.predict_proba(sc.transform(X[te, l]))[:, 1]
        oof[te] = p
        fold_auc.append(auc_mann_whitney(p, y[te]) if len(set(y[te].tolist())) > 1 else np.nan)
    return l, float(np.nanmean(fold_auc)), float(np.nanstd(fold_auc)), auc_mann_whitney(oof, y)


def recompute_from_states(P, ds, npz, zs_path, stat_path, answer):
    from joblib import Parallel, delayed
    z = np.load(npz, allow_pickle=True)
    names = [str(v) for v in z["name"]]
    zs = read_rows(zs_path)
    zs_clip = [r["clip"] for r in zs]
    zs_p = np.array([float(r["p_yes"]) for r in zs])
    same_order = names == zs_clip
    pdiff = float(np.max(np.abs(np.asarray(z["p_yes"], dtype=np.float64) - zs_p))) if same_order else None
    P(f"  [{ds}] states {npz}: enc {z['enc'].shape}; clip order equals zero-shot file: {same_order}; max |p_yes diff| {pdiff}")
    if not same_order or pdiff is None or pdiff > 1e-6:
        P(f"  [{ds}] states do not match the zero-shot run, recompute skipped")
        return dict(dataset=ds, npz=npz, matched=False)
    X = z["enc"].astype(np.float32)
    y = z["label"].astype(int)
    spk = z["spk"].astype(str)
    rows = sorted(Parallel(n_jobs=max(2, (os.cpu_count() or 4) - 2))(delayed(_layer_fit)(X, y, spk, l) for l in range(X.shape[1])))
    saved = read_rows(stat_path)
    s_oof = np.array([float(r["auc_oof"]) for r in saved])
    s_mean = np.array([float(r["auc_mean"]) for r in saved])
    s_sd = np.array([float(r["auc_std"]) for r in saved])
    r_oof = np.array([r[3] for r in rows])
    r_mean = np.array([r[1] for r in rows])
    r_sd = np.array([r[2] for r in rows])
    agree4 = int(sum(f"{a:.4f}" == f"{b:.4f}" for a, b in zip(r_oof, s_oof)))
    P(f"  [{ds}] recomputed auc_oof vs saved: max |diff| {np.max(np.abs(r_oof - s_oof)):.2e}, 4dp agree {agree4}/32; "
      f"mean max |diff| {np.max(np.abs(r_mean - s_mean)):.2e}; std max |diff| {np.max(np.abs(r_sd - s_sd)):.2e}")
    P(f"  [{ds}] layers above answer from MY recomputed curve: {int(np.sum(r_oof > answer))}/32 "
      f"(saved curve {int(np.sum(s_oof > answer))}/32)")
    return dict(dataset=ds, npz=npz, matched=True, max_abs_diff_oof=float(np.max(np.abs(r_oof - s_oof))),
                agree4=agree4, above_recomputed=int(np.sum(r_oof > answer)),
                recomputed_oof=r_oof.tolist())


def _layer_fit_part(X, y, spk, l, seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    import warnings
    warnings.filterwarnings("ignore")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X[:, l], y, groups=spk):
        sc = StandardScaler().fit(X[tr, l])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr, l]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te, l]))[:, 1]
    return seed, l, auc_mann_whitney(oof, y)


def partition_sensitivity(P, ds, npz, answer, n_part=20):
    """How much the per layer auc_oof, and the count of layers above the answer,
    move when only the speaker partition changes (same recipe, same states)."""
    from joblib import Parallel, delayed
    z = np.load(npz, allow_pickle=True)
    X = z["enc"].astype(np.float32)
    y = z["label"].astype(int)
    spk = z["spk"].astype(str)
    jobs = [(l, s) for s in range(n_part) for l in range(X.shape[1])]
    out = Parallel(n_jobs=max(2, (os.cpu_count() or 4) - 2))(delayed(_layer_fit_part)(X, y, spk, l, s) for l, s in jobs)
    M = np.zeros((n_part, X.shape[1]))
    for s, l, a in out:
        M[s, l] = a
    counts = (M > answer).sum(axis=1)
    rng_layer = M.max(axis=0) - M.min(axis=0)
    P(f"  [{ds}] partition sensitivity over {n_part} speaker partitions ({npz}): per-layer auc_oof SD median "
      f"{np.median(M.std(axis=0)):.4f}, max range {rng_layer.max():.4f}; layers above {answer:.4f}: "
      f"min {counts.min()} median {int(np.median(counts))} max {counts.max()} of 32")
    return dict(dataset=ds, npz=npz, n_partitions=n_part, counts=counts.tolist(),
                sd_median=float(np.median(M.std(axis=0))), max_range=float(rng_layer.max()),
                layer_min_over_partitions=M.min(axis=0).tolist())


def extra_checks(P, results):
    P("=" * 90)
    P("EXTRA CHECKS")
    ex = {}
    byds = {r["dataset"]: r for r in results}
    # (a) sidecars
    side_map = {"pitt": "T5_pitt", "adresso": "T5_adresso", "adress2020": "T5_adress2020", "pcgita": "T5_pcgita",
                "neurovoz": "T5_neurovoz", "kcl": "T5_kcl", "edaic_full": "T5_edaic_full",
                "edaic_mid300": "T5_edaic_mid300", "edaic_mid30": "T5_edaic_mid30", "edaic_first30": "T5_edaic_first30"}
    side_ok = {}
    for ds, name in side_map.items():
        p = f"{P17}/sidecars/{name}.json"
        if not os.path.exists(p):
            side_ok[ds] = "MISSING"
            P(f"  sidecar {p}: MISSING")
            continue
        j = json.load(open(p))
        need = ["sources", "n", "n_speakers", "seed", "command"]
        miss = [k for k in need if k not in j]
        hash_bad = [src for src, h in j.get("sources", {}).items() if not os.path.exists(src) or sha(src) != h]
        r = byds[ds]
        srcs = set(j.get("sources", {}).keys())
        has_inputs = r["zeroshot_file"] in srcs and r["curve_file"] in srcs
        ok = (not miss and not hash_bad and j.get("n") == r["n"] and j.get("n_speakers") == r["n_speakers"]
              and j.get("seed") == 0 and has_inputs)
        side_ok[ds] = "OK" if ok else f"PROBLEM miss={miss} hash_bad={hash_bad} n={j.get('n')} spk={j.get('n_speakers')} inputs_listed={has_inputs}"
        P(f"  sidecar {name}: {side_ok[ds]}")
    ex["sidecars"] = side_ok
    # (b) per clip csv written by the main job: exists, n per dataset, values equal the source
    pc = f"{P17}/T5_zeroshot_perclip.csv"
    pcr = read_rows(pc)
    cnt = {}
    for r in pcr:
        cnt[r["dataset"]] = cnt.get(r["dataset"], 0) + 1
    val_bad = {}
    for ds, r in byds.items():
        src = {x["clip"]: float(x["p_yes"]) for x in read_rows(r["zeroshot_file"])}
        mine = [x for x in pcr if x["dataset"] == ds]
        bad = sum(1 for x in mine if x["clip"] not in src or abs(float(x["p_yes"]) - src[x["clip"]]) > 1e-12)
        val_bad[ds] = bad
    P(f"  per-clip csv {pc}: rows {len(pcr)}; per dataset {cnt}; value mismatches vs source {val_bad}")
    ex["perclip_counts"] = cnt
    ex["perclip_value_mismatch"] = val_bad
    cc = read_rows(f"{P17}/T5_omitted_curves.csv")
    ccnt = {}
    for r in cc:
        ccnt[r["dataset"]] = ccnt.get(r["dataset"], 0) + 1
    P(f"  curves csv rows per dataset {ccnt}")
    ex["curves_csv_counts"] = ccnt
    # (c) same clips: nested encoder oof (same states) vs zero-shot clip list
    same = {}
    for ds, tag in (("pitt", "pitt"), ("adresso", "adresso"), ("adress2020", "adress2020"), ("pcgita", "pcgita"),
                    ("neurovoz", "neurovoz"), ("kcl", "kcl"), ("edaic_first30", "edaic")):
        f = f"{OF}/omni_{tag}_enc_nested_oof.csv"
        a = set(x["clip"] for x in read_rows(f))
        b = set(x["clip"] for x in read_rows(byds[ds]["zeroshot_file"]))
        same[ds] = (len(a), len(b), a == b)
    for v in ("full", "mid300", "mid30"):
        j = json.load(open(f"{VAR}/o25_{v}_nested_repeats.json"))
        same["edaic_" + v] = ("nested_repeats n", j.get("n"), j.get("n") == byds["edaic_" + v]["n"])
    P(f"  probe clip set == zero-shot clip set: {same}")
    ex["same_clips"] = {k: list(v) for k, v in same.items()}
    # (d) per layer recompute from saved hidden states where the states match the zero-shot run
    rc = []
    for ds, npz in (("adress2020", f"{REL}/edaic_rerun/part16/POD4B/states/p16_adress2020_orig_states.npz"),
                    ("pitt", f"{REL}/overnight2/part10/o25_pitt_states.npz"),
                    ("edaic_first30", f"{REL}/overnight2/part10/o25_edaic_states.npz")):
        if os.path.exists(npz):
            rc.append(recompute_from_states(P, ds, npz, byds[ds]["zeroshot_file"], byds[ds]["stat_file"],
                                            byds[ds]["answer_mw"]))
    ex["recompute"] = rc
    # (d2) partition sensitivity: pitt and edaic first30 (states match the zero-shot run exactly),
    # adress2020 (Part 16 re-extraction of the same 156 clips; p_yes differs slightly, so indicative only)
    ps = []
    for ds, npz in (("pitt", f"{REL}/overnight2/part10/o25_pitt_states.npz"),
                    ("edaic_first30", f"{REL}/overnight2/part10/o25_edaic_states.npz"),
                    ("adress2020", f"{REL}/edaic_rerun/part16/POD4B/states/p16_adress2020_orig_states.npz")):
        if os.path.exists(npz):
            ps.append(partition_sensitivity(P, ds, npz, byds[ds]["answer_mw"]))
    ex["partition_sensitivity"] = ps
    # (e) zero writes under omni_final since the run began
    newer = [os.path.join(dp, f) for dp, _, fs in os.walk(OF) for f in fs
             if os.path.getmtime(os.path.join(dp, f)) > 1790121600]  # 2026-09-23 00:00 UTC
    P(f"  files under omni_final modified on/after 2026-09-23 UTC: {len(newer)}")
    ex["omni_final_new_files"] = newer
    return ex


def main():
    log = []

    def P(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        log.append(line)

    fig_ans, fig_curves, fig_txt = figure_answers()
    P("figure script", FIGSCRIPT, "sha256", sha(FIGSCRIPT))
    P("figure script curve reads:", fig_curves)
    P("figure script answers:", fig_ans)
    ml = master_lookup()

    results = []
    checks = []
    for (ds, cond, plotted, zs_path, curve_path, curve_col, stat_path, mlkey) in CELLS:
        P("=" * 90)
        P(ds, cond, plotted)
        # ---- zero-shot per clip, fabrication checks
        rows = read_rows(zs_path)
        clips = [r["clip"] for r in rows]
        assert len(clips) == len(set(clips)), f"{ds}: duplicate clips"
        y = np.array([int(r["label"]) for r in rows])
        assert set(y.tolist()) <= {0, 1}
        s = np.array([float(r["p_yes"]) for r in rows], dtype=np.float64)
        assert np.all(np.isfinite(s))
        spk = np.array([str(r["speaker"]) for r in rows])
        n = len(rows)
        nspk = len(set(spk.tolist()))
        npos = int(y.sum())
        # each speaker has one label?
        lab_by_spk = {}
        for sp, lab in zip(spk.tolist(), y.tolist()):
            lab_by_spk.setdefault(sp, set()).add(lab)
        mixed = sum(1 for v in lab_by_spk.values() if len(v) > 1)
        P(f"  zero-shot file {zs_path}  rows {n}  speakers {nspk}  positives {npos}  mixed-label speakers {mixed}")

        a_mw = auc_mann_whitney(s, y)
        a_tr = auc_trapezoid(s, y)
        P(f"  answer AUC  Mann-Whitney {a_mw:.6f}  trapezoid {a_tr:.6f}  diff {abs(a_mw - a_tr):.2e}")
        assert abs(a_mw - a_tr) < 1e-9
        lo, hi, used, S = speaker_boot(s, y, spk, "sorted_str")
        P(f"  interval (sorted str speakers) [{lo:.4f}, {hi:.4f}]  usable {used}/{B}  speakers {S}")
        alt = {}
        for order in ("sorted_num", "appearance"):
            try:
                alt[order] = speaker_boot(s, y, spk, order)[:2]
            except Exception as e:  # non-numeric ids
                alt[order] = None
        P(f"  interval alt orders: {({k: (None if v is None else (round(v[0], 4), round(v[1], 4))) for k, v in alt.items()})}")

        # json and master lookup
        jpath = zs_path.replace("_scores.csv", ".json")
        jauc = None
        if os.path.exists(jpath):
            jj = json.load(open(jpath))
            jauc = jj.get("auc")
            P(f"  run json {jpath}: auc {jauc} n {jj.get('n')} n_positive {jj.get('n_positive')} window {jj.get('window_seconds')}")
        m = ml.get(mlkey)
        P(f"  master_lookup {mlkey}: {m}")

        # ---- curves
        crow = read_rows(curve_path)
        srow = read_rows(stat_path)
        lay_col = "layer"
        layers = [int(r[lay_col]) for r in crow]
        assert layers == list(range(32)), f"{ds}: layers {layers}"
        assert [int(r["layer"]) for r in srow] == list(range(32))
        curve = np.array([float(r[curve_col]) for r in crow])
        oof = np.array([float(r["auc_oof"]) for r in srow])
        mean = np.array([float(r["auc_mean"]) for r in srow])
        sd = np.array([float(r["auc_std"]) for r in srow])
        mirror_diff = float(np.max(np.abs(curve - oof)))
        P(f"  curve file {curve_path} col {curve_col}  rows {len(crow)}  max |curve - stat auc_oof| {mirror_diff:.3e}")
        above = int(np.sum(curve > a_mw))
        above_4dp = int(np.sum(curve > round(a_mw, 4)))
        above_fig = int(np.sum(curve > fig_ans[ds])) if ds in fig_ans else None
        msd = int(np.sum(mean - sd > a_mw))
        aci = int(np.sum(curve > hi))
        below = [(i, round(float(curve[i] - a_mw), 4)) for i in range(32) if curve[i] <= a_mw]
        mn_i = int(np.argmin(curve)); mx_i = int(np.argmax(curve))
        med = float(np.median(curve))
        P(f"  layers above answer {above}/32  (vs 4dp answer {above_4dp}, vs figure value {above_fig})  mean-std above {msd}/32  above CI hi {aci}/32")
        P(f"  probe min {curve[mn_i]:.4f} (L{mn_i})  median {med:.4f}  max {curve[mx_i]:.4f} (L{mx_i})")
        P(f"  layers at or below answer (layer, margin): {below}")
        res = dict(dataset=ds, condition=cond, plotted=plotted, n=n, n_speakers=nspk, n_pos=npos,
                   answer_mw=a_mw, answer_trap=a_tr, lo=lo, hi=hi, usable=used,
                   alt_intervals={k: v for k, v in alt.items()}, json_auc=jauc,
                   master=(None if m is None else m[0]), master_n=(None if m is None else m[1]),
                   figure_answer=fig_ans.get(ds), mirror_diff=mirror_diff,
                   above=above, above_4dp=above_4dp, above_fig=above_fig, mean_minus_std=msd,
                   above_ci_hi=aci, below=below, probe_min=float(curve[mn_i]), probe_min_layer=mn_i,
                   probe_median=med, probe_max=float(curve[mx_i]), probe_max_layer=mx_i,
                   zeroshot_file=zs_path, zeroshot_sha256=sha(zs_path), curve_file=curve_path,
                   curve_col=curve_col, curve_sha256=sha(curve_path), stat_file=stat_path)
        results.append(res)

        # ---- compare with the claim
        c = CLAIM[ds]
        def cmp(what, mine, theirs, kind="f"):
            if theirs is None:
                return
            ok = (f"{mine:.4f}" == f"{theirs:.4f}") if kind == "f" else (int(mine) == int(theirs))
            checks.append(dict(what=f"{ds} {what}", original=theirs, mine=mine, ok=ok, kind=kind))
            P(f"  CHECK {what:<22} claimed {theirs}  mine {f4(mine) if kind == 'f' else mine}  {'AGREE' if ok else 'DISAGREE'}")
        cmp("answer AUC", a_mw, c.get("ans"))
        cmp("interval lo", lo, c.get("lo"))
        cmp("interval hi", hi, c.get("hi"))
        cmp("layers above", above, c.get("above"), "i")
        cmp("mean-std above", msd, c.get("msd"), "i")
        cmp("above CI hi", aci, c.get("aci"), "i")
        cmp("probe min", float(curve[mn_i]), c.get("mn"))
        cmp("probe min layer", mn_i, c.get("mnL"), "i")
        cmp("probe median", med, c.get("med"))
        cmp("probe max", float(curve[mx_i]), c.get("mx"))
        cmp("probe max layer", mx_i, c.get("mxL"), "i")
        cmp("n", n, c.get("n"), "i")
        cmp("n_speakers", nspk, c.get("spk"), "i")

    extra = extra_checks(P, results)
    out = dict(results=results, checks=checks, extra=extra, bootstrap_draws=B, seed=SEED,
               command=f"/usr/local/bin/python3 {os.path.abspath(__file__)}")
    with open(os.path.join(HERE, "T5_verify.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    with open(os.path.join(HERE, "T5_verify.log"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    nd = sum(1 for c in checks if not c["ok"])
    print(f"\nCHECKS {len(checks)}  agree {len(checks) - nd}  disagree {nd}")


if __name__ == "__main__":
    main()
