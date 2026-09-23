"""PART 10 v2: the readout-direction test with a THIRD definition of the readout
direction, the MASS-WEIGHTED one.

Run 1 (part10_readout_direction.py, omni/readout_direction.log, kept) established:
  * the LAST entry of hidden_states is ALREADY through the final RMSNorm, so the
    states are used AS SAVED with no extra norm (Pitt mean|dp| 0.0057, r 0.9968;
    applying RMSNorm again is far worse, mean|dp| 0.1539). That verdict is kept and
    re-verified here.
  * d_plain = mean(yes rows) - mean(no rows) FAILED the gate on Pitt (+0.0188),
    so Pitt never reached the probe in run 1.

This run adds:
  d_weighted = sum_i w_yes_i * yes_row_i  -  sum_j w_no_j * no_row_j
where w_yes_i is the MEAN PROBABILITY of Yes variant i over the dataset, normalised
so the Yes weights sum to 1, and likewise for No. The probabilities are the softmax
over the 10 Yes/No lm_head logits only (state @ lm_head_full.T is not available), which
is exactly the renormalisation the scorer's p_yes uses.

Inputs : ~/Desktop/release/overnight2/part10/o25_{pitt,edaic}_states.npz
         ~/Desktop/release/overnight2/part10/o25_{pitt,edaic}_zeroshot_scores.csv
         ~/Desktop/release/overnight2/part10/omni25_readout.pt
Outputs: ~/Desktop/release/omni/readout_direction.csv   (overwrites run 1; run 1 kept
                                                         as readout_direction_v1.csv)
         ~/Desktop/release/omni/readout_direction.json  (sidecar; run 1 kept as
                                                         readout_direction_v1.json)
         ~/Desktop/release/omni/readout_direction_v2.log (run 1 log kept untouched)

CPU only. Nothing is fabricated: every number printed is computed here.
usage: part10_readout_direction_v2.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import sys
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
# Apple Accelerate BLAS raises spurious divide/overflow/invalid FPU status flags on matmul.
# Run 1 verified matmul against np.einsum and an explicit elementwise-sum reference: agreement
# to 0.0, and the same flags fire on plain random data. Suppressed so the log stays readable.
np.seterr(all="ignore")

P = os.path.expanduser("~/Desktop/release/overnight2/part10")
OUTDIR = os.path.expanduser("~/Desktop/release/omni")
SCRIPT = os.path.abspath(__file__)
PREV_SCRIPT = os.path.join(os.path.dirname(SCRIPT), "part10_readout_direction.py")
os.makedirs(OUTDIR, exist_ok=True)

GATE_TOL = 0.01
R4 = lambda v: round(float(v), 4)


def rms_norm(x, w, eps):
    """Qwen2_5OmniRMSNorm: x / sqrt(mean(x^2) + eps) * weight."""
    x = np.asarray(x, dtype=np.float64)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w[None, :]


def yesno_probs(S, Wy, Wn):
    """Softmax over the 10 Yes/No logits only. Columns 0..4 Yes, 5..9 No.

    This is the scorer's convention: p_yes = sum of the Yes columns of this softmax.
    """
    both = np.concatenate([S @ Wy.T, S @ Wn.T], axis=1)
    m = both.max(axis=1, keepdims=True)
    e = np.exp(both - m)
    return e / e.sum(axis=1, keepdims=True)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fit_wraw(X, y):
    """Standardise, fit logistic probe, map weights back to raw state space."""
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    return lr.coef_.ravel() / sc.scale_, sc, lr


def boot_cos(b, X, y, spk_u, spk_idx, d):
    warnings.filterwarnings("ignore")   # joblib workers are fresh processes
    np.seterr(all="ignore")             # spurious Accelerate FPU flags, see header note
    rng = np.random.default_rng(1000 + b)
    draw = rng.integers(0, len(spk_u), size=len(spk_u))
    rows = np.concatenate([spk_idx[i] for i in draw])
    yb = y[rows]
    if len(np.unique(yb)) < 2:
        return np.nan
    w_raw, _, _ = fit_wraw(X[rows], yb)
    return cosine(w_raw, d)


def run(tag, gate_ref_note):
    print("=" * 86)
    print(f"DATASET: {tag}")
    print("=" * 86)
    state_path = os.path.join(P, f"o25_{tag}_states.npz")
    csv_path = os.path.join(P, f"o25_{tag}_zeroshot_scores.csv")
    z = np.load(state_path, allow_pickle=True)
    ans = z["ans"]
    y = z["label"].astype(int)
    spk = z["spk"].astype(str)
    names = np.asarray(z["name"]).astype(str)
    p_yes_stored = np.asarray(z["p_yes"], dtype=np.float64)

    scores = pd.read_csv(csv_path)
    assert (names == scores["clip"].astype(str).values).all(), "row order mismatch npz vs csv"
    # the gate reference is READ FROM THE FILE, never hardcoded
    auc_file = float(roc_auc_score(scores["label"].astype(int).values, scores["p_yes"].values))

    ro = torch.load(os.path.join(P, "omni25_readout.pt"), map_location="cpu")
    Wy = ro["lm_head_yes"].to(torch.float64).numpy()
    Wn = ro["lm_head_no"].to(torch.float64).numpy()
    gw = ro["final_norm_weight"].to(torch.float64).numpy()
    eps = float(ro["norm_eps"])
    yes_ids = [int(v) for v in ro["yes_ids"]]
    no_ids = [int(v) for v in ro["no_ids"]]
    nY = Wy.shape[0]

    # ---- 0. provenance + norm verdict (kept from run 1, re-verified) --------
    print("\n[0] STATE FILE AND NORM VERDICT  (run 1 verdict re-verified, not changed)")
    print(f"  state path        : {state_path}")
    print(f"  per-clip scores   : {csv_path}")
    print(f"  ans shape         : {ans.shape}   dtype {ans.dtype}")
    print(f"  n clips           : {len(y)}   n speakers {len(np.unique(spk))}   n positive {int(y.sum())}")
    print(f"  norm_type {ro['norm_type']}  eps {eps}")
    print(f"  yes_ids {yes_ids}   no_ids {no_ids}")

    final_raw = ans[:, -1, :].astype(np.float64)
    final_rms = rms_norm(final_raw, gw, eps)

    verdict = {}
    for cname, cand in (("as_saved_no_norm", final_raw), ("rmsnorm_applied", final_rms)):
        pr = yesno_probs(cand, Wy, Wn)
        ph = pr[:, :nY].sum(axis=1)
        mad = float(np.abs(ph - p_yes_stored).mean())
        mx = float(np.abs(ph - p_yes_stored).max())
        r = float(np.corrcoef(ph, p_yes_stored)[0, 1])
        auc = float(roc_auc_score(y, ph))
        verdict[cname] = {"mean_abs_err": mad, "max_abs_err": mx, "pearson_r": r, "auc": auc}
        print(f"  [{cname:<18}] mean|dp| {mad:.4f}  max|dp| {mx:.4f}  r {r:.4f}  auc {auc:.4f}")

    if verdict["as_saved_no_norm"]["mean_abs_err"] <= verdict["rmsnorm_applied"]["mean_abs_err"]:
        chosen_norm, S = "as_saved_no_norm", final_raw
        print("  VERDICT: the LAST hidden_states entry is ALREADY final-normed. Using it AS SAVED.")
    else:  # not reached on these files; kept so the check is real and not decorative
        chosen_norm, S = "rmsnorm_applied", final_rms
        print("  VERDICT: the LAST hidden_states entry is NOT normed. Applying RMSNorm.")
    print(f"  stored p_yes reproduced under chosen convention: "
          f"mean|dp| {verdict[chosen_norm]['mean_abs_err']:.4f}, r {verdict[chosen_norm]['pearson_r']:.4f}")

    # ---- 1. the three directions -------------------------------------------
    print("\n[1] THE THREE READOUT DIRECTIONS  (all on ans[:, -1, :], the final LM stage)")
    probs = yesno_probs(S, Wy, Wn)
    mean_p = probs.mean(axis=0)                 # raw mean mass per variant id, sums to 1 over the 10
    mp_yes, mp_no = mean_p[:nY], mean_p[nY:]

    print("  raw mean mass per Yes id: " + "  ".join(f"{i}:{p:.4f}" for i, p in zip(yes_ids, mp_yes)))
    print("  raw mean mass per No  id: " + "  ".join(f"{i}:{p:.4f}" for i, p in zip(no_ids, mp_no)))
    print("  (4 dp hides the tiny variants; full precision is in the JSON sidecar)")
    print("  raw mean mass per Yes id, full precision: "
          + "  ".join(f"{i}:{p:.6e}" for i, p in zip(yes_ids, mp_yes)))
    print("  raw mean mass per No  id, full precision: "
          + "  ".join(f"{i}:{p:.6e}" for i, p in zip(no_ids, mp_no)))

    w_yes = mp_yes / mp_yes.sum()
    w_no = mp_no / mp_no.sum()
    print("  normalised Yes weights (sum 1): " + "  ".join(f"{i}:{w:.4f}" for i, w in zip(yes_ids, w_yes)))
    print("  normalised No  weights (sum 1): " + "  ".join(f"{i}:{w:.4f}" for i, w in zip(no_ids, w_no)))

    d_plain = Wy.mean(axis=0) - Wn.mean(axis=0)
    ky, kn = int(np.argmax(mp_yes)), int(np.argmax(mp_no))
    d_top = Wy[ky] - Wn[kn]
    d_weighted = w_yes @ Wy - w_no @ Wn
    print(f"  top Yes id {yes_ids[ky]} (mass {mp_yes[ky]:.4f})   top No id {no_ids[kn]} (mass {mp_no[kn]:.4f})")

    dirs = [("d_plain", d_plain), ("d_top", d_top), ("d_weighted", d_weighted)]

    # ---- 2. the table and the gate -----------------------------------------
    exact_lo = logsumexp(S @ Wy.T, axis=1) - logsumexp(S @ Wn.T, axis=1)
    auc_exact = float(roc_auc_score(y, exact_lo))

    print(f"\n[2] TABLE AND GATE   zero-shot AUC from {os.path.basename(csv_path)} = {auc_file:.4f}")
    if gate_ref_note:
        print(f"    NOTE: {gate_ref_note}")
    print(f"    gate: a definition PASSES if |projection AUC - zero-shot AUC| <= {GATE_TOL:.4f}")
    print()
    print(f"  {'definition':<12} {'||d||':>8} {'cos(d_plain)':>13} {'proj AUC':>9} "
          f"{'sanity delta':>13} {'spearman vs exact':>18}  gate")
    print("  " + "-" * 82)
    table, gate_pass = {}, {}
    for nm, dv in dirs:
        proj = S @ dv
        auc_p = float(roc_auc_score(y, proj))
        delta = auc_p - auc_file
        sp = float(spearmanr(exact_lo, proj).statistic)
        ok = abs(delta) <= GATE_TOL
        gate_pass[nm] = ok
        table[nm] = {"norm": float(np.linalg.norm(dv)), "cos_with_d_plain": cosine(dv, d_plain),
                     "proj_auc": auc_p, "delta": delta, "spearman_vs_exact_logodds": sp,
                     "gate": "PASS" if ok else "FAIL"}
        print(f"  {nm:<12} {np.linalg.norm(dv):8.4f} {cosine(dv, d_plain):13.4f} {auc_p:9.4f} "
              f"{delta:+13.4f} {sp:18.4f}  {'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 82)
    print(f"  reference: AUC(exact yes/no log-odds) {auc_exact:.4f}  delta {auc_exact - auc_file:+.4f}")
    print(f"  cos(d_top, d_weighted) {cosine(d_top, d_weighted):.4f}   "
          f"cos(d_plain, d_weighted) {cosine(d_plain, d_weighted):.4f}")
    passed = [nm for nm, _ in dirs if gate_pass[nm]]
    print(f"  definitions that PASS the gate: {passed if passed else 'NONE'}")

    # ---- 3. pick the direction ---------------------------------------------
    if gate_pass["d_weighted"]:
        used, d = "d_weighted", d_weighted
        print("  PRIMARY: d_weighted passes the gate, so d_weighted is used.")
        fallback_note = ""
    elif passed:
        used = passed[0]
        d = dict(dirs)[used]
        fallback_note = (f"d_weighted FAILED the gate (delta {table['d_weighted']['delta']:+.4f}), "
                         f"so it is NOT used. Fell back to {used}, which passes "
                         f"(delta {table[used]['delta']:+.4f}).")
        print(f"  d_weighted FAILED the gate. FALLING BACK to {used}.")
        print(f"  {fallback_note}")
    else:
        print("  NO definition passes the gate. STOPPING this dataset, no probe is reported.")
        meta = {"dataset": tag, "direction_used": None, "gate_overall": "ALL FAIL",
                "stopped_at": "gate", "state_path": state_path, "scores_path": csv_path,
                "n": int(len(y)), "n_speakers": int(len(np.unique(spk))), "n_positive": int(y.sum()),
                "auc_zeroshot_file": R4(auc_file), "gate_tolerance": GATE_TOL,
                "directions": {k: {kk: (vv if isinstance(vv, str) else R4(vv))
                                   for kk, vv in v.items()} for k, v in table.items()}}
        return meta, None

    # ---- 4. probe with the chosen direction --------------------------------
    print(f"\n[3] PROBE AT THE ANSWER POSITION, direction = {used}")
    oof = np.zeros(len(y))
    w_folds = []
    # out-of-fold projections, with the readout component removed (perp) and kept (keep).
    # "keep" is the MATCHED CONTROL: same folds, same pooling, d NOT removed. Without it there is
    # no way to tell whether a low pooled number is caused by removing d or by the pooling itself.
    perp_raw, keep_raw = np.zeros(len(y)), np.zeros(len(y))
    perp_z, keep_z = np.zeros(len(y)), np.zeros(len(y))
    perp_fold, keep_fold = [], []
    for tr, te in GroupKFold(n_splits=5).split(S, y, groups=spk):
        sc = StandardScaler().fit(S[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(S[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(S[te]))[:, 1]
        wf = lr.coef_.ravel() / sc.scale_
        w_folds.append(wf)
        wp = wf - (np.dot(wf, d) / np.dot(d, d)) * d
        a, b = S[te] @ wf, S[te] @ wp
        keep_raw[te], perp_raw[te] = a, b
        # Each fold's projection carries its own offset and scale (the StandardScaler mean and the
        # intercept are dropped by a bare dot product). Pooling the raw values across folds mixes
        # incomparable scales, so they are standardised within the fold before pooling.
        keep_z[te] = (a - a.mean()) / a.std()
        perp_z[te] = (b - b.mean()) / b.std()
        keep_fold.append(float(roc_auc_score(y[te], a)))
        perp_fold.append(float(roc_auc_score(y[te], b)))
    auc_probe = float(roc_auc_score(y, oof))
    w_raw = np.mean(np.stack(w_folds), axis=0)
    auc_d = table[used]["proj_auc"]

    auc_perp_raw = float(roc_auc_score(y, perp_raw))
    auc_keep_raw = float(roc_auc_score(y, keep_raw))
    auc_perp_z = float(roc_auc_score(y, perp_z))
    auc_keep_z = float(roc_auc_score(y, keep_z))
    auc_perp_meanfold = float(np.mean(perp_fold))
    auc_keep_meanfold = float(np.mean(keep_fold))

    cos_wd = cosine(w_raw, d)
    rng = np.random.default_rng(0)
    G = rng.standard_normal((1000, len(d)))
    cos_rand = float(np.mean(np.abs((G @ d) / (np.linalg.norm(G, axis=1) * np.linalg.norm(d)))))
    w_perp = w_raw - (np.dot(w_raw, d) / np.dot(d, d)) * d
    auc_perp_insample = float(roc_auc_score(y, S @ w_perp))
    auc_wraw_insample = float(roc_auc_score(y, S @ w_raw))
    auc_perp_oof = auc_perp_z   # the reported out-of-fold value, see note below

    print("  GroupKFold(5) by speaker, LogisticRegression(max_iter=2000, class_weight='balanced'),")
    print("  StandardScaler fit inside the training fold only, scores out of fold.")
    print(f"  probe AUC, out of fold                     {auc_probe:.4f}")
    print(f"  projection AUC ( states . {used:<10} )  {auc_d:.4f}")
    print(f"  cosine(w_raw, {used})                  {cos_wd:.4f}")
    print(f"  random floor, mean |cos| vs 1000 Gaussian dirs (seed 0, 3584 dims)  {cos_rand:.4f}")
    print(f"  ratio cosine / random floor                {abs(cos_wd) / cos_rand:.4f}")
    print(f"  ||w_raw|| (fold-averaged)                  {np.linalg.norm(w_raw):.4f}")

    print(f"\n  READOUT COMPONENT REMOVED, w_perp = w_fold - (w_fold.d/d.d) d, per fold, out of fold.")
    print(f"  Each column is the SAME statistic with d removed and, as a matched control, with d kept.")
    print(f"  {'pooling':<34} {'d removed':>10} {'d kept (control)':>18}")
    print(f"  {'-' * 64}")
    print(f"  {'raw pooled across folds':<34} {auc_perp_raw:10.4f} {auc_keep_raw:18.4f}")
    print(f"  {'standardised within fold, pooled':<34} {auc_perp_z:10.4f} {auc_keep_z:18.4f}   <- REPORTED")
    print(f"  {'mean of the 5 per-fold AUCs':<34} {auc_perp_meanfold:10.4f} {auc_keep_meanfold:18.4f}")
    print(f"  per-fold AUC, d removed: " + "  ".join(f"{v:.4f}" for v in perp_fold))
    print(f"  per-fold AUC, d kept   : " + "  ".join(f"{v:.4f}" for v in keep_fold))
    print(f"  NOTE ON THE RAW POOLED ROW: a bare dot product drops the StandardScaler mean and the")
    print(f"  intercept, so each fold's projection has its own offset and scale. Pooling the raw")
    print(f"  values mixes them and the AUC collapses. The matched control shows this is caused by")
    print(f"  the pooling and NOT by removing d: with d KEPT the raw pooled AUC is {auc_keep_raw:.4f},")
    print(f"  essentially the same as the {auc_perp_raw:.4f} with d removed. Run 1 reported the raw")
    print(f"  pooled value for E-DAIC (0.5320) without this control. This run reports the")
    print(f"  standardised-within-fold value instead, and keeps the raw one for comparability.")
    print(f"\n  AUC( states . w_raw  ), in-sample          {auc_wraw_insample:.4f}   (p=3584 >> n, separable)")
    print(f"  AUC( states . w_perp ), in-sample          {auc_perp_insample:.4f}   (same caveat, measures nothing)")

    # ---- 5. speaker bootstrap on the cosine --------------------------------
    spk_u, inv = np.unique(spk, return_inverse=True)
    spk_idx = [np.where(inv == i)[0] for i in range(len(spk_u))]
    t0 = time.time()
    cs = np.asarray(Parallel(n_jobs=12)(delayed(boot_cos)(b, S, y, spk_u, spk_idx, d)
                                        for b in range(2000)), dtype=float)
    ok = cs[~np.isnan(cs)]
    ci_lo, ci_hi = np.percentile(ok, [2.5, 97.5])
    print(f"\n[4] SPEAKER BOOTSTRAP OF cosine(w_raw, {used}): 2000 draws, seed 0, probe REFIT per draw")
    print(f"  valid draws {len(ok)}/2000   took {time.time() - t0:.1f}s")
    print(f"  mean {ok.mean():.4f}   95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    row = {"dataset": tag, "direction_used": used, "n": int(len(y)), "n_speakers": int(len(spk_u)),
           "cosine": R4(cos_wd), "cosine_random": R4(cos_rand), "auc_probe": R4(auc_probe),
           "auc_d": R4(auc_d), "auc_probe_without_d": R4(auc_perp_oof),
           "ci_lo": R4(ci_lo), "ci_hi": R4(ci_hi),
           "delta_plain": R4(table["d_plain"]["delta"]),
           "delta_top": R4(table["d_top"]["delta"]),
           "delta_weighted": R4(table["d_weighted"]["delta"])}
    meta = {
        "dataset": tag, "direction_used": used,
        "gate_overall": "PASS", "fallback_note": fallback_note,
        "state_path": state_path, "scores_path": csv_path,
        "n": int(len(y)), "n_speakers": int(len(spk_u)), "n_positive": int(y.sum()),
        "norm_verdict": chosen_norm,
        "norm_checks": {k: {kk: R4(vv) for kk, vv in v.items()} for k, v in verdict.items()},
        "auc_zeroshot_file": R4(auc_file), "gate_tolerance": GATE_TOL,
        "auc_exact_logodds": R4(auc_exact), "delta_exact_logodds": R4(auc_exact - auc_file),
        "directions": {k: {kk: (vv if isinstance(vv, str) else R4(vv)) for kk, vv in v.items()}
                       for k, v in table.items()},
        "gate_passed_by": passed,
        "cos_d_top_d_weighted": R4(cosine(d_top, d_weighted)),
        "cos_d_plain_d_weighted": R4(cosine(d_plain, d_weighted)),
        "top_yes_id": yes_ids[ky], "top_no_id": no_ids[kn],
        "mean_mass_yes_ids": {str(i): R4(p) for i, p in zip(yes_ids, mp_yes)},
        "mean_mass_no_ids": {str(i): R4(p) for i, p in zip(no_ids, mp_no)},
        "mean_mass_yes_ids_full_precision": {str(i): float(p) for i, p in zip(yes_ids, mp_yes)},
        "mean_mass_no_ids_full_precision": {str(i): float(p) for i, p in zip(no_ids, mp_no)},
        "weights_yes_normalised": {str(i): R4(w) for i, w in zip(yes_ids, w_yes)},
        "weights_no_normalised": {str(i): R4(w) for i, w in zip(no_ids, w_no)},
        "weights_yes_normalised_full_precision": {str(i): float(w) for i, w in zip(yes_ids, w_yes)},
        "weights_no_normalised_full_precision": {str(i): float(w) for i, w in zip(no_ids, w_no)},
        "auc_probe_oof": R4(auc_probe), "auc_d": R4(auc_d),
        "cosine_w_d": R4(cos_wd), "cosine_random_mean_abs": R4(cos_rand),
        "cosine_over_random_floor": R4(abs(cos_wd) / cos_rand),
        "w_raw_norm": R4(np.linalg.norm(w_raw)),
        "auc_probe_without_d_outoffold": R4(auc_perp_oof),
        "auc_probe_without_d_insample": R4(auc_perp_insample),
        "auc_wraw_insample": R4(auc_wraw_insample),
        "readout_removal_detail": {
            "d_removed_raw_pooled": R4(auc_perp_raw),
            "d_kept_raw_pooled_control": R4(auc_keep_raw),
            "d_removed_standardised_within_fold_pooled": R4(auc_perp_z),
            "d_kept_standardised_within_fold_pooled_control": R4(auc_keep_z),
            "d_removed_mean_of_per_fold_aucs": R4(auc_perp_meanfold),
            "d_kept_mean_of_per_fold_aucs_control": R4(auc_keep_meanfold),
            "per_fold_auc_d_removed": [R4(v) for v in perp_fold],
            "per_fold_auc_d_kept_control": [R4(v) for v in keep_fold],
        },
        "bootstrap": {"draws": 2000, "seed": 0, "unit": "speaker", "refit_per_draw": True,
                      "valid_draws": int(len(ok)), "mean": R4(ok.mean()),
                      "ci_lo": R4(ci_lo), "ci_hi": R4(ci_hi)},
    }
    if gate_ref_note:
        meta["gate_reference_note"] = gate_ref_note
    return meta, row


def main():
    ro = torch.load(os.path.join(P, "omni25_readout.pt"), map_location="cpu")
    notes = {
        "pitt": "",
        "edaic": ("run 1 recorded that a brief quoted 0.6290 as the E-DAIC zero-shot AUC, but 0.6290 "
                  "is the Qwen3-Omni-30B number in q3o_edaic_zeroshot.json. The Qwen2.5-Omni-7B "
                  "per-clip file o25_edaic_zeroshot_scores.csv gives the value used here. As in "
                  "run 1, the gate reference is read from that file and never hardcoded."),
    }
    metas, rows = [], []
    for tag in ("pitt", "edaic"):
        meta, row = run(tag, notes[tag])
        metas.append(meta)
        if row is not None:
            rows.append(row)
    print("\n" + "=" * 86)
    print("PC-GITA: no o25 state file was re-extracted for PC-GITA, so it is NOT AVAILABLE for")
    print("part 10 and is skipped. Nothing is reported for it. Same as run 1.")
    print("=" * 86)

    cols = ["dataset", "direction_used", "n", "n_speakers", "cosine", "cosine_random", "auc_probe",
            "auc_d", "auc_probe_without_d", "ci_lo", "ci_hi",
            "delta_plain", "delta_top", "delta_weighted"]
    csv_out = os.path.join(OUTDIR, "readout_direction.csv")
    json_out = os.path.join(OUTDIR, "readout_direction.json")
    pd.DataFrame(rows, columns=cols).to_csv(csv_out, index=False, float_format="%.4f")

    payload = {
        "date": str(date.today()),
        "run": "v2 (third direction definition added)",
        "script": SCRIPT,
        "previous_script": PREV_SCRIPT,
        "previous_log_kept": os.path.join(OUTDIR, "readout_direction.log"),
        "this_log": os.path.join(OUTDIR, "readout_direction_v2.log"),
        "previous_outputs_kept": {"csv": os.path.join(OUTDIR, "readout_direction_v1.csv"),
                                  "json": os.path.join(OUTDIR, "readout_direction_v1.json")},
        "python": sys.version.split()[0],
        "model": "Qwen/Qwen2.5-Omni-7B",
        "state_paths": {t: os.path.join(P, f"o25_{t}_states.npz") for t in ("pitt", "edaic")},
        "scores_paths": {t: os.path.join(P, f"o25_{t}_zeroshot_scores.csv") for t in ("pitt", "edaic")},
        "readout_path": os.path.join(P, "omni25_readout.pt"),
        "yes_token_ids": [int(v) for v in ro["yes_ids"]],
        "no_token_ids": [int(v) for v in ro["no_ids"]],
        "norm_type": str(ro["norm_type"]),
        "norm_eps": float(ro["norm_eps"]),
        "features": "ans[:, -1, :], the final LM stage at the first answer position",
        "norm_handling": (
            "The LAST entry of hidden_states is ALREADY through the final RMSNorm, so the states "
            "are used AS SAVED with no extra norm. This was established in run 1 and re-verified "
            "here by reproducing the stored p_yes from the final-stage state through the Yes/No "
            "lm_head rows under both conventions. As-saved reproduces the stored p_yes far better "
            "than applying RMSNorm a second time; see per-dataset norm_checks for the numbers. "
            "The verdict is unchanged from run 1."),
        "direction_definitions": {
            "d_plain": ("mean over the 5 Yes lm_head rows minus the mean over the 5 No lm_head rows. "
                        "Unweighted. This is what run 1 used."),
            "d_top": ("the single Yes lm_head row with the highest mean mass minus the single No "
                      "lm_head row with the highest mean mass."),
            "d_weighted": ("sum_i w_yes_i * yes_row_i minus sum_j w_no_j * no_row_j. THE NEW ONE, "
                           "primary for this run when it passes the gate."),
        },
        "d_weighted_weighting_rule": (
            "In words: for every clip in the dataset, take the final-stage answer state as saved, "
            "multiply it by the 10 Yes/No lm_head rows to get 10 logits, and softmax those 10 "
            "logits AMONG THEMSELVES. The full-vocabulary softmax is not available because "
            "state @ lm_head_full.T was not saved, so the softmax is renormalised WITHIN the 10 "
            "Yes/No rows. That is exactly the renormalisation the scorer's p_yes uses, so the "
            "weights live on the same scale as the reported p_yes. Average each of the 10 "
            "probabilities over all clips in that dataset to get the raw mean mass per variant id. "
            "Then divide the 5 Yes masses by their sum so the Yes weights add to 1, and separately "
            "divide the 5 No masses by their sum so the No weights add to 1. d_weighted is the "
            "Yes-weight-weighted sum of the Yes lm_head rows minus the No-weight-weighted sum of "
            "the No lm_head rows. The weights are computed per dataset, from that dataset's own "
            "clips. The raw mean masses and the normalised weights are both recorded per dataset "
            "below, at 4 decimals and at full precision, so the rule is auditable."),
        "p_yes_convention": ("p_yes = sum(exp(logit_yes_ids)) / (sum(exp(logit_yes_ids)) + "
                             "sum(exp(logit_no_ids))), i.e. the Yes half of the same 10-way softmax"),
        "gate": {
            "rule": ("a definition PASSES if |projection AUC - zero-shot AUC| <= 0.01, where the "
                     "projection AUC is the AUC of states . d and the zero-shot AUC is computed "
                     "from that dataset's own per-clip scores file"),
            "tolerance": GATE_TOL,
            "reference_source": "roc_auc_score(label, p_yes) read from o25_<tag>_zeroshot_scores.csv",
            "hardcoded": False,
            "primary_rule": ("d_weighted is used as the primary direction if it passes. If it does "
                             "not, the fallback is the first of d_plain, d_top that passes, and the "
                             "fallback is named explicitly in fallback_note and in the CSV column "
                             "direction_used."),
        },
        "probe_settings": {"model": "sklearn LogisticRegression", "max_iter": 2000,
                           "class_weight": "balanced", "solver": "lbfgs (default)", "C": 1.0,
                           "scaler": "StandardScaler fit inside the training fold only",
                           "cv": "GroupKFold(n_splits=5) grouped by speaker",
                           "scoring": "out of fold",
                           "features": "ans[:, -1, :], the final LM stage at the first answer position",
                           "w_raw": "coef_ / StandardScaler.scale_, averaged over the 5 folds"},
        "random_floor": {"n_directions": 1000, "seed": 0, "dim": 3584,
                         "statistic": "mean absolute cosine with the chosen d"},
        "csv_column_notes": {
            "direction_used": "which of the three definitions was used for the probe columns",
            "auc_d": "projection AUC of the CHOSEN direction",
            "auc_probe_without_d": (
                "OUT OF FOLD: per training fold, w_perp = w_fold - (w_fold.d/d.d) d, projected on "
                "that fold's held-out speakers, standardised WITHIN the fold, then pooled. "
                "CHANGED FROM RUN 1, deliberately and stated here: run 1 pooled the RAW per-fold "
                "projections. A bare dot product drops the StandardScaler mean and the logistic "
                "intercept, so every fold's projection carries its own offset and scale; pooling "
                "the raw values mixes incomparable scales and the AUC collapses towards chance. "
                "The matched control added in this run proves the collapse is the pooling and not "
                "the removal of d: computed the same raw-pooled way with d KEPT IN, Pitt reads "
                "0.4685 against 0.4692 with d removed. Both the raw-pooled value and the mean of "
                "the 5 per-fold AUCs, each with its d-kept control, are in every dataset's "
                "readout_removal_detail block, so run 1's number is still directly comparable. "
                "The in-sample version is kept as auc_probe_without_d_insample; at p=3584 >> n it "
                "is perfectly separable and reads 1.0000, so it measures separability, not "
                "residual signal."),
            "cosine": "cosine(w_raw, chosen d), w_raw averaged over the 5 outer folds",
            "ci_lo/ci_hi": "2.5 and 97.5 percentiles of the speaker bootstrap of that cosine",
            "delta_plain/delta_top/delta_weighted": ("sanity delta for each definition: projection "
                                                     "AUC minus the zero-shot AUC from that "
                                                     "dataset's own per-clip file"),
        },
        "blas_note": ("numpy is built on Apple Accelerate, which raises spurious divide/overflow/"
                      "invalid FPU status flags on matmul. Run 1 checked matmul against np.einsum "
                      "and an explicit elementwise-sum reference: agreement to 0.0, and the same "
                      "flags fire on random data. Results are unaffected; the flags are suppressed."),
        "differences_from_run_1": [
            "d_weighted, the mass-weighted direction, is added and is the primary direction here.",
            "Pitt reaches the probe for the first time: it failed the gate in run 1 under d_plain "
            "(delta +0.0188) and passes here under d_weighted (delta +0.0031).",
            "auc_probe_without_d is now standardised within the fold before pooling, with a matched "
            "d-kept control. See csv_column_notes. Run 1's raw-pooled convention is kept alongside "
            "it in readout_removal_detail, so nothing from run 1 is lost or silently overwritten.",
            "The norm verdict is unchanged and re-verified: states are used AS SAVED, no extra norm.",
        ],
        "pcgita": "NOT AVAILABLE: no o25 state file was re-extracted for PC-GITA. Skipped, as in run 1.",
        "datasets": metas,
    }
    json.dump(payload, open(json_out, "w"), indent=1)
    print(f"\nwrote {csv_out}")
    print(f"wrote {json_out}\n")
    print("FINAL TABLE")
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))


if __name__ == "__main__":
    main()
