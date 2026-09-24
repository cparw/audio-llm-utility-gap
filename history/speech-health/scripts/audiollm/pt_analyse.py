"""
Analysis for INTERVENTION 1 (prompt-level). Reads the long CSV written by
pt_prompts.py and answers: does any prompt move the model off the words?

Primary metric, comparable across prompts without importing a threshold from
elsewhere: NESTED conflict accuracy. For each of 5 speaker-disjoint folds the
decision threshold is chosen on the AGREEMENT clips of the TRAINING speakers
(maximising balanced accuracy) and applied to the CONFLICT clips of the HELD-OUT
speakers. Nothing about the conflict set touches threshold selection.

Also reported, threshold-free: conflict AUC. Baseline conflict AUC is 0.297,
i.e. reliably ANTI-correlated with the clinical label, which is what following
the words looks like. An intervention that works has to push this above 0.5.

Controls: speaker-level label shuffle through the same nested pipeline, and a
speaker-clustered paired bootstrap of every prompt against P00_baseline.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
try:
    from sklearn.model_selection import StratifiedGroupKFold as SGKF
except Exception:
    SGKF = None
from sklearn.model_selection import GroupKFold

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/final"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "omni"
GRID = np.linspace(0.005, 0.995, 199)

D = pd.read_csv(f"{OUT}/prompt_intervention_{MODEL}_clips.csv")
print("rows", len(D), "prompts", D.prompt_id.nunique(), "clips", D.seg_uid.nunique(),
      "speakers", D.speaker.nunique(), flush=True)


def bal_acc(y, pr):
    a = (pr[y == 1] == 1).mean() if (y == 1).any() else np.nan
    b = (pr[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return 0.5 * (a + b)


def best_t(y, p):
    return float(GRID[int(np.argmax([bal_acc(y, (p >= t).astype(int)) for t in GRID]))])


def folds(y, g, k=5):
    if SGKF is not None:
        try: return list(SGKF(n_splits=k, shuffle=True, random_state=0).split(np.zeros((len(y), 1)), y, g))
        except Exception: pass
    return list(GroupKFold(n_splits=k).split(np.zeros((len(y), 1)), y, g))


def nested_conflict(dp, yvec=None):
    """threshold from agreement clips of training speakers -> applied to conflict
    clips of held-out speakers. returns oof predictions aligned to conflict rows."""
    C = dp[dp.set == "conflict"].reset_index(drop=True)
    A = dp[dp.set == "agreement"].reset_index(drop=True)
    yC = (C.label.values if yvec is None else yvec)
    spk = C.speaker.values
    pred = np.full(len(C), -1); ts = []
    for tr, te in folds(yC, spk):
        trs = set(spk[tr]); tes = set(spk[te])
        assert not (trs & tes), "speaker leak"
        am = A[A.speaker.isin(trs)]
        if am.label.nunique() < 2 or len(am) < 8:
            t = 0.5
        else:
            t = best_t(am.label.values, am.p_dep.values)
        ts.append(t)
        pred[te] = (C.p_dep.values[te] >= t).astype(int)
    return C, pred, float(np.mean(ts)), float(np.std(ts))


def spk_boot(y, a, b, g, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    ug = pd.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
    d = []
    for _ in range(n):
        s = rng.choice(len(ug), len(ug), replace=True)
        ii = np.concatenate([idx[ug[j]] for j in s])
        d.append((b[ii] == y[ii]).mean() - (a[ii] == y[ii]).mean())
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def spk_boot_acc(y, pr, g, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    ug = pd.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
    d = []
    for _ in range(n):
        s = rng.choice(len(ug), len(ug), replace=True)
        ii = np.concatenate([idx[ug[j]] for j in s])
        d.append((pr[ii] == y[ii]).mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


base = None
rows = []
prompt_ids = list(dict.fromkeys(D.prompt_id))
for pid in prompt_ids:
    dp = D[D.prompt_id == pid]
    C, pred, tm, tsd = nested_conflict(dp)
    A = dp[dp.set == "agreement"]
    yC = C.label.values; wC = C.words_say_depressed.values; gC = C.speaker.values
    accC = float((pred == yC).mean())
    folw = float((pred == wC).mean())
    lo, hi = spk_boot_acc(yC, pred, gC)
    # agreement side, same nested machinery but held-out agreement clips
    Ar = A.reset_index(drop=True)
    apred = np.full(len(Ar), -1)
    for tr, te in folds(Ar.label.values, Ar.speaker.values):
        sub = Ar.iloc[tr]
        t = best_t(sub.label.values, sub.p_dep.values) if sub.label.nunique() > 1 else 0.5
        apred[te] = (Ar.p_dep.values[te] >= t).astype(int)
    # speaker-level shuffle control on the conflict side
    sl = C.groupby("speaker").label.first()
    nulls = []
    for r in range(20):
        rr = np.random.RandomState(200 + r)
        m = dict(zip(sl.index, rr.permutation(sl.values)))
        ys = np.array([m[s] for s in C.speaker])
        if len(np.unique(ys)) < 2: continue
        _, pn, _, _ = nested_conflict(dp, yvec=ys)
        nulls.append((pn == ys).mean())
    # raw accuracy on the conflict set is confounded by its base rate (138 of 195
    # clips are label=1), so a prompt that merely shifts the operating point up
    # scores well without discriminating any better. Balanced accuracy is the
    # honest read; AUC is the threshold-free one.
    rows.append(dict(
        prompt_id=pid,
        conflict_balacc_nested=round(float(bal_acc(yC, pred)), 3),
        conflict_majority_baseline=round(float(max(yC.mean(), 1 - yC.mean())), 3),
        mean_p_conflict=round(float(C.p_dep.mean()), 3),
        mean_p_agreement=round(float(A.p_dep.mean()), 3),
        auc_conflict=round(float(roc_auc_score(yC, C.p_dep)), 3),
        auc_agreement=round(float(roc_auc_score(A.label, A.p_dep)), 3),
        thr_mean=round(tm, 3), thr_sd=round(tsd, 3),
        conflict_acc_nested=round(accC, 3),
        conflict_acc_ci_lo=round(lo, 3), conflict_acc_ci_hi=round(hi, 3),
        follows_words=round(folw, 3),
        agreement_acc_nested=round(float((apred == Ar.label.values).mean()), 3),
        pred_dep_rate_conflict=round(float(pred.mean()), 3),
        answer_mass=round(float(dp.mass.mean()), 3),
        shuffle_acc=round(float(np.mean(nulls)), 3) if nulls else np.nan))
    if pid == "P00_baseline":
        base = (yC, pred, gC)
    print("done", pid, flush=True)

T = pd.DataFrame(rows)

# paired bootstrap vs baseline
if base is not None:
    yb, pb, gb = base
    d2 = []
    for pid in prompt_ids:
        dp = D[D.prompt_id == pid]
        C, pred, _, _ = nested_conflict(dp)
        assert (C.label.values == yb).all()
        m, lo, hi = spk_boot(yb, pb, pred, gb)
        d2.append(dict(prompt_id=pid, delta_vs_baseline=round(m, 3),
                       delta_ci_lo=round(lo, 3), delta_ci_hi=round(hi, 3),
                       significant=bool(lo > 0 or hi < 0)))
    T = T.merge(pd.DataFrame(d2), on="prompt_id")

T.to_csv(f"{OUT}/prompt_intervention_{MODEL}_summary.csv", index=False)

print("\n=========== INTERVENTION 1: PROMPTING, model=%s ===========" % MODEL)
h = ("%-26s %7s %7s %7s %7s %8s %9s %9s %8s"
     % ("prompt", "meanPc", "AUCc", "AUCa", "thr", "conf_acc", "conf_bal", "follows_w", "agr_acc"))
print(h); print("-" * len(h))
for _, r in T.iterrows():
    print("%-26s %7.3f %7.3f %7.3f %7.3f %8.3f %9.3f %9.3f %8.3f"
          % (r.prompt_id, r.mean_p_conflict, r.auc_conflict, r.auc_agreement,
             r.thr_mean, r.conflict_acc_nested, r.conflict_balacc_nested,
             r.follows_words, r.agreement_acc_nested))
print("\nconflict-set majority-class accuracy (always predict depressed) = %.3f"
      % T.conflict_majority_baseline.iloc[0])
print("AUC below 0.500 means the score is ANTI-correlated with the clinical label,")
print("i.e. it is tracking the words. conf_acc is base-rate confounded; conf_bal and")
print("AUCc are the discrimination measures.")

print("\n--- conflict accuracy, 95%% CI (speaker-clustered), vs P00_baseline ---")
for _, r in T.iterrows():
    s = "SIG" if r.get("significant", False) else "   "
    print("%-26s %.3f [%.3f, %.3f]   delta %+0.3f [%+0.3f, %+0.3f] %s  shuffle-null %.3f"
          % (r.prompt_id, r.conflict_acc_nested, r.conflict_acc_ci_lo, r.conflict_acc_ci_hi,
             r.get("delta_vs_baseline", np.nan), r.get("delta_ci_lo", np.nan),
             r.get("delta_ci_hi", np.nan), s, r.shuffle_acc))
print("\nwrote", f"{OUT}/prompt_intervention_{MODEL}_summary.csv")
print("JOB_DONE", flush=True)
