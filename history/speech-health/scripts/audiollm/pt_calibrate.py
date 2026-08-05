"""
PART 3, INTERVENTION 2: post-hoc CALIBRATION of the audio-LLM yes/no answers.

The benchmark models are saturated: qwen2-audio answers Yes to almost everything,
Qwen2.5-Omni answers No to almost everything on Parkinson's. That is a broken
DECISION, which is separable from broken DISCRIMINATION. This asks whether a
per-dataset decision rule fitted on held-out speakers turns the existing P(yes)
into a usable decision.

Two recalibrations, both fitted strictly inside training folds:
  (a) threshold  : pick t maximising balanced accuracy on the training speakers
  (b) Platt      : logistic regression on logit P(yes), decide at 0.5
Both are monotone in P(yes), so neither can change AUC. That is the point: the
AUC column is the ceiling this intervention is working under.

Controls: speaker-level label shuffle through the identical nested pipeline
(20 reps), and speaker-clustered bootstrap CIs on the gain.
"""
import os, json, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
try:
    from sklearn.model_selection import StratifiedGroupKFold as SGKF
except Exception:
    SGKF = None
from sklearn.model_selection import GroupKFold

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
SC = "/scratch1/parwatka/scores"
OUT = f"{R}/final"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.RandomState(0)
GRID = np.linspace(0.005, 0.995, 199)

# ---------------------------------------------------------------- sources
SCORES = [  # (file, dataset, task, model, window)
 ("dcaps_interview_af3_v2",     "dcaps",    "interview", "audio-flamingo-3", "30s"),
 ("dcaps_interview_kimi_v2",    "dcaps",    "interview", "kimi-audio",       "30s"),
 ("dcaps_interview300_af3_v2",  "dcaps",    "interview", "audio-flamingo-3", "300s"),
 ("dcaps_interview300_kimi_v2", "dcaps",    "interview", "kimi-audio",       "300s"),
 ("italian_read_af3_v2",        "italian",  "read",      "audio-flamingo-3", "30s"),
 ("italian_read_kimi_v2",       "italian",  "read",      "kimi-audio",       "30s"),
 ("kcl_read_af3_v2",            "kcl",      "read",      "audio-flamingo-3", "30s"),
 ("kcl_read_kimi_v2",           "kcl",      "read",      "kimi-audio",       "30s"),
 ("neurovoz_vowel_af3_v2",      "neurovoz", "vowel",     "audio-flamingo-3", "30s"),
 ("neurovoz_vowel_kimi_v2",     "neurovoz", "vowel",     "kimi-audio",       "30s"),
 ("pcgita_vowel_af3_v2",        "pcgita",   "vowel",     "audio-flamingo-3", "30s"),
 ("pcgita_vowel_kimi_v2",       "pcgita",   "vowel",     "kimi-audio",       "30s"),
 ("pcgita_read_v2check_af3_v2", "pcgita",   "read",      "audio-flamingo-3", "30s"),
 ("pcgita_read_v2check_kimi_v2","pcgita",   "read",      "kimi-audio",       "30s"),
 ("pcgita_read_omni",           "pcgita",   "read",      "qwen2.5-omni",     "full"),
 ("pcgita_read_qwen2audio",     "pcgita",   "read",      "qwen2-audio",      "30s"),
]
ABL = [  # (file, dataset, task, model)
 ("abl_dcaps_omni",       "dcaps",    "interview", "qwen2.5-omni"),
 ("abl_dcaps_qwen",       "dcaps",    "interview", "qwen2-audio"),
 ("abl_italian_read_omni","italian",  "read",      "qwen2.5-omni"),
 ("abl_italian_read_qwen","italian",  "read",      "qwen2-audio"),
 ("abl_italian_vowel_omni","italian", "vowel",     "qwen2.5-omni"),
 ("abl_italian_vowel_qwen","italian", "vowel",     "qwen2-audio"),
 ("abl_kcl_read_omni",    "kcl",      "read",      "qwen2.5-omni"),
 ("abl_kcl_read_qwen",    "kcl",      "read",      "qwen2-audio"),
 ("abl_nv_vowel_omni",    "neurovoz", "vowel",     "qwen2.5-omni"),
 ("abl_nv_vowel_qwen",    "neurovoz", "vowel",     "qwen2-audio"),
 ("abl_pcg_vowel_omni",   "pcgita",   "vowel",     "qwen2.5-omni"),
 ("abl_pcg_vowel_qwen",   "pcgita",   "vowel",     "qwen2-audio"),
]


def load_all():
    out = []
    for f, ds, tk, md, win in SCORES:
        p = f"{SC}/{f}_perclip.csv"
        if not os.path.exists(p): print("MISSING", p); continue
        d = pd.read_csv(p)
        if "p_yes" not in d.columns: print("no p_yes", p); continue
        out.append((f"{ds}|{tk}|{md}|{win}",
                    pd.DataFrame(dict(p=d.p_yes.astype(float),
                                      y=d.label.astype(int),
                                      g=d.speaker_id.astype(str))), ds, tk, md, win))
    for f, ds, tk, md in ABL:
        p = f"{R}/ablation/{f}_clips.csv"
        if not os.path.exists(p): print("MISSING", p); continue
        d = pd.read_csv(p)
        out.append((f"{ds}|{tk}|{md}|abl",
                    pd.DataFrame(dict(p=d.p_orig.astype(float),
                                      y=d.label.astype(int),
                                      g=d.speaker.astype(str))), ds, tk, md, "abl"))
    return out


def bal_acc(y, pred):
    a = (pred[y == 1] == 1).mean() if (y == 1).any() else np.nan
    b = (pred[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return 0.5 * (a + b)


def best_t(y, p):
    return float(GRID[int(np.argmax([bal_acc(y, (p >= t).astype(int)) for t in GRID]))])


def splits(y, g, k=5):
    ug = pd.unique(g)
    kk = min(k, len(ug))
    if kk < 2: return []
    if SGKF is not None:
        try:
            return list(SGKF(n_splits=kk, shuffle=True, random_state=0).split(p_dummy(y), y, g))
        except Exception:
            pass
    return list(GroupKFold(n_splits=kk).split(p_dummy(y), y, g))


def p_dummy(y):
    return np.zeros((len(y), 1))


def nested(y, p, g, seed=0):
    """out-of-fold predictions from (a) threshold search and (b) Platt, both fitted
    on training speakers only."""
    pred_t = np.full(len(y), -1); pred_pl = np.full(len(y), -1); ts = []
    sp = splits(y, g)
    if not sp: return None
    for tr, te in sp:
        assert len(set(g[tr]) & set(g[te])) == 0, "speaker leak"
        if len(np.unique(y[tr])) < 2:
            t = 0.5
        else:
            t = best_t(y[tr], p[tr])
        ts.append(t)
        pred_t[te] = (p[te] >= t).astype(int)
        if len(np.unique(y[tr])) < 2:
            pred_pl[te] = pred_t[te]
        else:
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            lr = LogisticRegression(max_iter=2000).fit(z[tr], y[tr])
            pred_pl[te] = (lr.predict_proba(z[te])[:, 1] >= 0.5).astype(int)
    return pred_t, pred_pl, float(np.mean(ts)), float(np.std(ts))


def boot_delta(y, p, g, pred_a, pred_b, n=2000, seed=0):
    """speaker-clustered paired bootstrap on balanced-accuracy difference b - a"""
    rng = np.random.RandomState(seed)
    ug = pd.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
    d = []
    for _ in range(n):
        s = rng.choice(len(ug), len(ug), replace=True)
        ii = np.concatenate([idx[ug[j]] for j in s])
        if len(np.unique(y[ii])) < 2: continue
        d.append(bal_acc(y[ii], pred_b[ii]) - bal_acc(y[ii], pred_a[ii]))
    if not d: return (np.nan, np.nan, np.nan)
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def shuffle_null(y, p, g, reps=20):
    """permute labels at SPEAKER level, run the identical nested pipeline"""
    ug = pd.unique(g)
    slab = pd.Series(y).groupby(pd.Series(g)).first()
    out = []
    for r in range(reps):
        rng = np.random.RandomState(100 + r)
        perm = rng.permutation(slab.values)
        m = dict(zip(slab.index, perm))
        ys = np.array([m[x] for x in g])
        if len(np.unique(ys)) < 2: continue
        nn = nested(ys, p, g)
        if nn is None: continue
        pt, pl, _, _ = nn
        out.append(bal_acc(ys, pt))
    return float(np.mean(out)) if out else np.nan, float(np.std(out)) if out else np.nan


rows = []
for name, d, ds, tk, md, win in load_all():
    y, p, g = d.y.values, d.p.values, d.g.values
    ok = np.isfinite(p)
    y, p, g = y[ok], p[ok], g[ok]
    if len(np.unique(y)) < 2:
        print("single class, skip", name); continue
    naive = (p >= 0.5).astype(int)
    nn = nested(y, p, g)
    if nn is None:
        print("too few speakers, skip", name); continue
    pt, pl, tmean, tsd = nn
    sm, ss = shuffle_null(y, p, g)
    dm, lo, hi = boot_delta(y, p, g, naive, pt)
    # speaker-level AUC where clips outnumber speakers
    sp = pd.DataFrame(dict(y=y, p=p, g=g)).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    auc_spk = roc_auc_score(sp.y, sp.p) if sp.y.nunique() > 1 else np.nan
    rows.append(dict(
        source=name, dataset=ds, task=tk, model=md, window=win,
        n_clips=len(y), n_spk=len(pd.unique(g)), prevalence=round(float(y.mean()), 3),
        mean_p=round(float(p.mean()), 3),
        yes_rate_at_0p5=round(float(naive.mean()), 3),
        auc_clip=round(float(roc_auc_score(y, p)), 3),
        auc_spk=round(float(auc_spk), 3),
        balacc_naive=round(float(bal_acc(y, naive)), 3),
        acc_naive=round(float((naive == y).mean()), 3),
        f1_naive=round(float(f1_score(y, naive, zero_division=0)), 3),
        balacc_nested_thr=round(float(bal_acc(y, pt)), 3),
        acc_nested_thr=round(float((pt == y).mean()), 3),
        f1_nested_thr=round(float(f1_score(y, pt, zero_division=0)), 3),
        balacc_nested_platt=round(float(bal_acc(y, pl)), 3),
        thr_mean=round(tmean, 3), thr_sd=round(tsd, 3),
        gain_balacc=round(float(bal_acc(y, pt) - bal_acc(y, naive)), 3),
        gain_ci_lo=round(lo, 3), gain_ci_hi=round(hi, 3),
        shuffle_balacc=round(sm, 3), shuffle_sd=round(ss, 3)))
    print("done", name, flush=True)

D = pd.DataFrame(rows).sort_values(["dataset", "task", "model"])
D.to_csv(f"{OUT}/calibration_results.csv", index=False)

print("\n================= CALIBRATION (nested, speaker-disjoint) =================")
hdr = ("%-9s %-10s %-17s %-5s %5s %5s %6s %6s | %6s %6s | %6s %6s %6s | %6s"
       % ("dataset", "task", "model", "win", "n", "spk", "prev", "yes@.5",
          "AUCcl", "AUCsp", "bal.5", "balNST", "balPLT", "shufNST"))
print(hdr); print("-" * len(hdr))
for _, r in D.iterrows():
    print("%-9s %-10s %-17s %-5s %5d %5d %6.2f %6.2f | %6.3f %6.3f | %6.3f %6.3f %6.3f | %6.3f"
          % (r.dataset, r.task, r.model, r.window, r.n_clips, r.n_spk, r.prevalence,
             r.yes_rate_at_0p5, r.auc_clip, r.auc_spk, r.balacc_naive,
             r.balacc_nested_thr, r.balacc_nested_platt, r.shuffle_balacc))

print("\n--- gain from nested threshold over the naive 0.5 rule (speaker-clustered bootstrap 95% CI) ---")
for _, r in D.iterrows():
    sig = "*" if (r.gain_ci_lo > 0 or r.gain_ci_hi < 0) else " "
    print("%-9s %-10s %-17s %-5s  %+0.3f  [%+0.3f, %+0.3f] %s"
          % (r.dataset, r.task, r.model, r.window, r.gain_balacc, r.gain_ci_lo, r.gain_ci_hi, sig))

print("\nnote: threshold and Platt are both monotone in P(yes), so AUC is unchanged by")
print("construction. The AUC columns are the ceiling that recalibration works under.")
print("wrote", f"{OUT}/calibration_results.csv")
print("JOB_DONE", flush=True)
