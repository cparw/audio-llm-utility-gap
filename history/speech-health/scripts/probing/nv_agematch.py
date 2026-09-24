"""
NEUROVOZ AGE MATCHING -- the decisive control.

Neurovoz patients are ~7.1 years older than its controls (71.1 vs 64.0), and age alone
separates the groups. Every neurovoz number therefore has to be rerun on a subset where
age carries no information, or none of them are usable.

What this does
--------------
1. Builds age-matched subsets by OPTIMAL 1:1 assignment (Hungarian) on |age difference|
   with a caliper. Two subsets:
      MATCH_AGE      caliper 3 years
      MATCH_AGE_SEX  caliper 4 years AND identical sex
   Balance is reported: mean age per group, standardized mean difference, Welch t,
   Mann-Whitney U, and the age-only AUC that the matching is supposed to kill.
2. Reruns the three neurovoz headline measures on FULL and on each matched subset with
   IDENTICAL code, so the comparison is apples to apples:
      (a) qwen2 encoder probe, nested layer selection inside training folds only
      (b) silence-only ENV probe (recording artifact probe)
      (c) transcription-error (WER) measure on the read task
3. Every number gets a speaker-clustered bootstrap 95% CI and a speaker-level
   label-shuffle permutation null.

Speaker is the unit of analysis for (a) and (c). For (b) the unit is the clip but folds
are GroupKFold on speaker and disjointness is asserted; a speaker-level AUC is also given.
"""
import os, sys, json, warnings, time, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats
from scipy.stats import rankdata
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/committed"
os.makedirs(OUT, exist_ok=True)
META = "/project2/msoleyma_946/speech_health/spanish_neurovoz/zenodo_upload/metadata"
WERCSV = "/scratch1/parwatka/pd_probing/code_clean/out_neurovoz_read_wer.csv"

ALPHAS = np.logspace(-1, 5, 13)
NFOLD = 5
N_REP = 10
N_REP_PERM = 2
N_PERM_ENC = int(os.environ.get("N_PERM_ENC", 200))
N_PERM_SIMPLE = int(os.environ.get("N_PERM_SIMPLE", 2000))
N_SHUF_SIL = int(os.environ.get("N_SHUF_SIL", 200))
N_BOOT = int(os.environ.get("N_BOOT", 4000))
WORKERS = int(os.environ.get("WORKERS", 8))
K_SIZECTRL = int(os.environ.get("K_SIZECTRL", 25))   # random same-n subsets
N_REP_CTRL = 3


# ---------------------------------------------------------------- metrics
def auc(y, p):
    y = np.asarray(y)
    pos = y == 1
    n1 = int(pos.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    p = np.asarray(p, float)
    if not np.isfinite(p).all():
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def smd(a, b):
    """standardized mean difference, pooled sd."""
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else np.nan


def boot_auc_ci(y, p, rng, nb=N_BOOT):
    """resample SPEAKERS with replacement (y,p are already speaker level)."""
    n = len(y); vals = []
    for _ in range(nb):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v = auc(y[i], p[i])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return (np.nan, np.nan)
    return tuple(float(x) for x in np.percentile(vals, [2.5, 97.5]))


def boot_auc_ci_clustered(y, p, g, rng, nb=N_BOOT):
    """clip-level AUC, speaker-clustered bootstrap: resample SPEAKERS, take their clips."""
    spk = np.unique(g)
    idx = {s: np.flatnonzero(g == s) for s in spk}
    vals = []
    for _ in range(nb):
        pick = rng.integers(0, len(spk), len(spk))
        sel = np.concatenate([idx[spk[k]] for k in pick])
        if len(np.unique(y[sel])) < 2:
            continue
        v = auc(y[sel], p[sel])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return (np.nan, np.nan)
    return tuple(float(x) for x in np.percentile(vals, [2.5, 97.5]))


# ---------------------------------------------------------------- ridge / cv
def kr_paths(Xtr, ytr, Xte, alphas=ALPHAS):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-9] = 1e-9
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    ym = ytr.mean(); yc = ytr - ym
    K = A @ A.T
    Kt = B @ A.T
    s, U = np.linalg.eigh(K)
    s = np.maximum(s, 0.0)
    Uy = U.T @ yc
    out = np.empty((len(alphas), Xte.shape[0]))
    for i, a in enumerate(alphas):
        out[i] = Kt @ (U @ (Uy / (s + a))) + ym
    return out


def folds_of(n, k, rng):
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def cv_oof(Xs, y, n_rep, rng, return_pick=False):
    """Nested (layer-block, alpha) selection inside TRAIN folds only.
    Returns speaker-level OOF prediction averaged over repeats."""
    n = len(y)
    acc = np.zeros(n); cnt = np.zeros(n); picks = []
    for _ in range(n_rep):
        for f in folds_of(n, NFOLD, rng):
            te = np.asarray(f)
            tr = np.setdiff1d(np.arange(n), te)
            assert len(np.intersect1d(tr, te)) == 0, "SPEAKER LEAK"
            ytr = y[tr].astype(float)
            inner = folds_of(len(tr), 3, rng)
            innr = np.zeros((len(Xs), len(ALPHAS), len(tr)))
            for g in inner:
                gpos = np.sort(np.asarray(g))
                ite = tr[gpos]; itr = np.delete(tr, gpos)
                yitr = np.delete(ytr, gpos)
                for bi, X in enumerate(Xs):
                    innr[bi][:, gpos] = np.nan_to_num(
                        kr_paths(X[itr], yitr, X[ite]), nan=0.0, posinf=0.0, neginf=0.0)
            best, bkey = -np.inf, (0, len(ALPHAS) // 2)
            for bi in range(len(Xs)):
                for ai in range(len(ALPHAS)):
                    sc = auc(ytr, innr[bi, ai])
                    if np.isfinite(sc) and sc > best:
                        best, bkey = sc, (bi, ai)
            bi, ai = bkey
            picks.append(bi)
            pred = np.nan_to_num(kr_paths(Xs[bi][tr], ytr, Xs[bi][te])[ai],
                                 nan=0.0, posinf=0.0, neginf=0.0)
            acc[te] += pred; cnt[te] += 1
    p = acc / np.maximum(cnt, 1)
    return (p, picks) if return_pick else p


# ---------------------------------------------------------------- matching
def load_meta():
    p = pd.read_csv(f"{META}/data_pd.csv").drop_duplicates("ID")
    h = pd.read_csv(f"{META}/data_hc.csv").drop_duplicates("ID")
    p["is_pd"] = 1; h["is_pd"] = 0
    x = pd.concat([p, h], ignore_index=True)
    x["speaker"] = x["ID"].astype(int).astype(str)
    x["age"] = pd.to_numeric(x["Age"], errors="coerce")
    x["sex"] = pd.to_numeric(x["Sex"], errors="coerce")
    return x[["speaker", "is_pd", "age", "sex"]]


def match(meta, pool, caliper, sex_match):
    """Optimal 1:1 assignment on |age diff| restricted to `pool` speakers."""
    m = meta[meta.speaker.isin(pool) & meta.age.notna()]
    if sex_match:
        m = m[m.sex.notna()]
    P = m[m.is_pd == 1].reset_index(drop=True)
    H = m[m.is_pd == 0].reset_index(drop=True)
    C = np.abs(P.age.values[:, None] - H.age.values[None, :])
    bad = C > caliper
    if sex_match:
        bad = bad | (P.sex.values[:, None] != H.sex.values[None, :])
    C2 = np.where(bad, 1e6, C)
    r, c = linear_sum_assignment(C2)
    ok = C2[r, c] < 1e5
    return list(P.speaker.values[r[ok]]), list(H.speaker.values[c[ok]])


def balance_report(meta, spk, tag):
    m = meta[meta.speaker.isin(spk)]
    a1 = m.age[m.is_pd == 1].dropna().values
    a0 = m.age[m.is_pd == 0].dropna().values
    y = m.is_pd.values.astype(int)
    ageu = m.age.values.astype(float)
    ok = np.isfinite(ageu)
    d = dict(subset=tag, n=int(len(m)), n_pd=int((m.is_pd == 1).sum()),
             n_hc=int((m.is_pd == 0).sum()),
             age_pd_mean=round(float(a1.mean()), 2), age_pd_sd=round(float(a1.std(ddof=1)), 2),
             age_hc_mean=round(float(a0.mean()), 2), age_hc_sd=round(float(a0.std(ddof=1)), 2),
             age_diff=round(float(a1.mean() - a0.mean()), 2),
             age_smd=round(smd(a1, a0), 3),
             welch_t_p=round(float(stats.ttest_ind(a1, a0, equal_var=False)[1]), 4),
             mannwhitney_p=round(float(stats.mannwhitneyu(a1, a0)[1]), 4),
             age_only_auc=round(auc(y[ok], ageu[ok]), 3))
    frac_m = m.groupby("is_pd").sex.mean()
    d["sex_frac_pd"] = round(float(frac_m.get(1, np.nan)), 3)
    d["sex_frac_hc"] = round(float(frac_m.get(0, np.nan)), 3)
    return d


# ---------------------------------------------------------------- (a) encoder
def enc_job(args):
    tag, task, speakers, labels, blocks_flat, shape, seed = args
    n_l = shape[0]
    Xs = [blocks_flat[i] for i in range(n_l)]
    y = np.asarray(labels, float)
    rng = np.random.default_rng(seed)
    p, picks = cv_oof(Xs, y, N_REP, rng, return_pick=True)
    obs = auc(y, p)
    lo, hi = boot_auc_ci(y, p, rng)
    null = []
    for _ in range(N_PERM_ENC):
        yp = rng.permutation(y)
        pp = cv_oof(Xs, yp, N_REP_PERM, rng)
        v = auc(yp, pp)
        if np.isfinite(v):
            null.append(v)
    null = np.array(null)
    pval = (1 + (null >= obs).sum()) / (1 + len(null))
    # single-layer curve, no selection, for reference
    curve = []
    for li in range(n_l):
        pl = cv_oof([Xs[li]], y, 3, np.random.default_rng(seed + 1000 + li))
        curve.append(round(auc(y, pl), 4))
    return dict(measure="encoder", subset=tag, task=task, n=int(len(y)),
                n_pd=int((y == 1).sum()), n_hc=int((y == 0).sum()),
                auc=round(obs, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                perm_null_mean=round(float(null.mean()), 4) if len(null) else np.nan,
                perm_null_p95=round(float(np.percentile(null, 95)), 4) if len(null) else np.nan,
                perm_p=round(float(pval), 5), n_perm=len(null),
                layer_mode=int(np.bincount(picks).argmax()),
                layer_spread=round(float(np.std(picks)), 2),
                best_single_layer=int(np.argmax(curve)),
                best_single_layer_auc=float(np.max(curve)),
                layer_curve=";".join(str(c) for c in curve))


def sizectrl_enc_job(args):
    """Same n as the matched subset but speakers drawn AT RANDOM from the full pool,
    ignoring age. Separates 'age matching killed the signal' from 'smaller n killed it'."""
    task, keep, labels, Xs, n_pd, n_hc, K, seed = args
    y_all = np.asarray(labels, float)
    ipd = np.flatnonzero(y_all == 1); ihc = np.flatnonzero(y_all == 0)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(K):
        sel = np.concatenate([rng.choice(ipd, n_pd, replace=False),
                              rng.choice(ihc, n_hc, replace=False)])
        Xsub = [X[sel] for X in Xs]
        ysub = y_all[sel]
        p = cv_oof(Xsub, ysub, N_REP_CTRL, rng)
        v = auc(ysub, p)
        if np.isfinite(v):
            vals.append(v)
    return task, vals


# ---------------------------------------------------------------- (b) silence
def silence_run(X, y, g, tag, block, rng):
    if len(np.unique(g)) < 8 or len(np.unique(y)) < 2:
        return None

    def oof(Xa, ya, ga):
        ns = min(5, len(np.unique(ga)))
        o = np.full(len(ya), np.nan)
        for tr, te in GroupKFold(ns).split(Xa, ya, ga):
            assert not (set(ga[tr]) & set(ga[te])), "SPEAKER LEAK"
            if len(np.unique(ya[tr])) < 2:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, class_weight="balanced"))
            clf.fit(Xa[tr], ya[tr])
            o[te] = clf.predict_proba(Xa[te])[:, 1]
        return o

    p = oof(X, y, g)
    m = ~np.isnan(p)
    clip_auc = roc_auc_score(y[m], p[m])
    # speaker level
    spk = {}
    for gi, yi, pi in zip(g[m], y[m], p[m]):
        spk.setdefault(gi, [yi, []])[1].append(pi)
    ys = np.array([v[0] for v in spk.values()])
    ps = np.array([np.mean(v[1]) for v in spk.values()])
    spk_auc = roc_auc_score(ys, ps)
    lo, hi = boot_auc_ci_clustered(y[m], p[m], g[m], rng)      # CI on the CLIP-level AUC
    slo, shi = boot_auc_ci(ys, ps, rng)                        # CI on the SPEAKER-level AUC
    # speaker-level label shuffle, whole CV rerun
    null = []
    us = np.unique(g)
    lab = np.array([y[g == s][0] for s in us])
    for _ in range(N_SHUF_SIL):
        perm = rng.permutation(lab)
        mp = dict(zip(us, perm))
        ysh = np.array([mp[s] for s in g])
        psh = oof(X, ysh, g)
        mm = ~np.isnan(psh)
        if len(np.unique(ysh[mm])) < 2:
            continue
        null.append(roc_auc_score(ysh[mm], psh[mm]))
    null = np.array(null)
    return dict(measure="silence_env", subset=tag, task=block, n_clips=int(m.sum()),
                n=int(len(ys)), n_pd=int((ys == 1).sum()), n_hc=int((ys == 0).sum()),
                auc=round(float(clip_auc), 4), speaker_auc=round(float(spk_auc), 4),
                ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                speaker_ci_lo=round(slo, 4), speaker_ci_hi=round(shi, 4),
                perm_null_mean=round(float(null.mean()), 4) if len(null) else np.nan,
                perm_null_p95=round(float(np.percentile(null, 95)), 4) if len(null) else np.nan,
                perm_p=round(float((1 + (null >= clip_auc).sum()) / (1 + len(null))), 5),
                n_perm=len(null))


def silence_run_quick(X, y, g):
    """clip-level OOF AUC only, no CI / no null. Used for the size control."""
    if len(np.unique(g)) < 8 or len(np.unique(y)) < 2:
        return None
    ns = min(5, len(np.unique(g)))
    o = np.full(len(y), np.nan)
    for tr, te in GroupKFold(ns).split(X, y, g):
        assert not (set(g[tr]) & set(g[te])), "SPEAKER LEAK"
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        o[te] = clf.predict_proba(X[te])[:, 1]
    m = ~np.isnan(o)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m], o[m]))


# ---------------------------------------------------------------- (c) WER
def wer_run(sp, tag, rng):
    y = sp.label.values.astype(int)
    x = sp.wer.values.astype(float)
    obs = auc(y, x)
    lo, hi = boot_auc_ci(y, x, rng)
    null = []
    for _ in range(N_PERM_SIMPLE):
        null.append(auc(rng.permutation(y), x))
    null = np.array(null)
    return dict(measure="wer_read", subset=tag, task="read", n=int(len(y)),
                n_pd=int((y == 1).sum()), n_hc=int((y == 0).sum()),
                auc=round(obs, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                mean_wer_pd=round(float(x[y == 1].mean()), 4),
                mean_wer_hc=round(float(x[y == 0].mean()), 4),
                mannwhitney_p=round(float(stats.mannwhitneyu(x[y == 1], x[y == 0])[1]), 5),
                perm_null_mean=round(float(null.mean()), 4),
                perm_null_p95=round(float(np.percentile(null, 95)), 4),
                perm_p=round(float((1 + (null >= obs).sum()) / (1 + len(null))), 5),
                n_perm=len(null))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    meta = load_meta()
    enc = np.load(f"{R}/features/qwen2/neurovoz/encoder_features.npz", allow_pickle=True)
    enc_spk = np.array([str(int(s)) for s in enc["speaker"]])
    enc_task = enc["task"].astype(str)
    n_layers = int(enc["n_layers"])
    sil = np.load(f"{R}/soleymani_checks/silence_feats_neurovoz.npz", allow_pickle=True)
    sil_g = np.array([str(int(s)) for s in sil["g"]])
    wer = pd.read_csv(WERCSV)
    wer["speaker"] = wer.speaker.astype(int).astype(str)
    wspk = wer.groupby("speaker").agg(label=("label", "first"),
                                      wer=("wer", "mean"), nclip=("wer", "size")).reset_index()

    pool = sorted(set(meta.speaker) & set(enc_spk))
    print(f"neurovoz speakers with metadata AND encoder features: {len(pool)}", flush=True)
    mp = meta[meta.speaker.isin(pool)]
    print(f"  PD={int((mp.is_pd==1).sum())} HC={int((mp.is_pd==0).sum())} "
          f"age missing={int(mp.age.isna().sum())}", flush=True)

    subsets = {}
    subsets["FULL"] = sorted(pool)
    for tag, cal, sx in [("MATCH_AGE", 3, False), ("MATCH_AGE_SEX", 4, True)]:
        a, b = match(meta, pool, cal, sx)
        subsets[tag] = sorted(a + b)
        print(f"{tag}: caliper={cal}y sex_match={sx} -> {len(a)} matched pairs "
              f"({len(a)+len(b)} speakers)", flush=True)

    bal = [balance_report(meta, s, t) for t, s in subsets.items()]
    bdf = pd.DataFrame(bal)
    bdf.to_csv(f"{OUT}/neurovoz_agematch_balance.csv", index=False)
    print("\n=== AGE BALANCE ===", flush=True)
    print(bdf.to_string(index=False), flush=True)

    # roster of who is in each subset
    rows = []
    for t, s in subsets.items():
        mm = meta[meta.speaker.isin(s)]
        for _, r in mm.iterrows():
            rows.append(dict(subset=t, speaker=r.speaker, is_pd=int(r.is_pd),
                             age=r.age, sex=r.sex))
    pd.DataFrame(rows).to_csv(f"{OUT}/neurovoz_agematch_roster.csv", index=False)

    # ---- (a) encoder ----
    tasks = ["read", "vowel", "ddk", "all"]
    lab = meta.set_index("speaker").is_pd.to_dict()
    jobs = []
    full_blocks = {}
    for tag, spks in subsets.items():
        spks = [s for s in spks if s in set(enc_spk)]
        for task in tasks:
            m = np.ones(len(enc_spk), bool) if task == "all" else (enc_task == task)
            idx = {s: np.flatnonzero(m & (enc_spk == s)) for s in spks}
            keep = [s for s in spks if len(idx[s])]
            if len(keep) < 20:
                print(f"SKIP encoder {tag}/{task}: only {len(keep)} speakers", flush=True)
                continue
            Xs = []
            for l in range(n_layers):
                X = enc[f"layer_{l:02d}"]
                Xs.append(np.stack([X[idx[s]].mean(0) for s in keep]).astype(np.float64))
            y = [lab[s] for s in keep]
            if tag == "FULL":
                full_blocks[task] = (keep, y, Xs)
            jobs.append((tag, task, keep, y, Xs, (n_layers,),
                         abs(hash((tag, task))) % (2 ** 31)))
    print(f"\nencoder jobs: {len(jobs)}", flush=True)
    with Pool(min(WORKERS, len(jobs))) as pl:
        enc_rows = pl.map(enc_job, jobs)
    for r in enc_rows:
        print(f"  ENC {r['subset']:<14} {r['task']:<6} n={r['n']:<4} "
              f"AUC={r['auc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
              f"null={r['perm_null_mean']:.3f} p={r['perm_p']:.4f}", flush=True)

    # ---- size control: random same-n subsets, age ignored ----
    n_pd_m = int(bdf.set_index("subset").loc["MATCH_AGE", "n_pd"])
    n_hc_m = int(bdf.set_index("subset").loc["MATCH_AGE", "n_hc"])
    cjobs = [(t, *full_blocks[t], n_pd_m, n_hc_m, K_SIZECTRL,
              abs(hash(("ctrl", t))) % (2 ** 31)) for t in tasks if t in full_blocks]
    with Pool(min(WORKERS, max(1, len(cjobs)))) as pl:
        cres = pl.map(sizectrl_enc_job, cjobs)
    ctrl_rows = []
    matched_auc = {r["task"]: r["auc"] for r in enc_rows if r["subset"] == "MATCH_AGE"}
    for task, vals in cres:
        v = np.array(vals)
        ma = matched_auc.get(task, np.nan)
        row = dict(measure="encoder", subset="SIZE_CTRL_random", task=task,
                   n=n_pd_m + n_hc_m, n_pd=n_pd_m, n_hc=n_hc_m,
                   auc=round(float(v.mean()), 4),
                   ci_lo=round(float(np.percentile(v, 2.5)), 4),
                   ci_hi=round(float(np.percentile(v, 97.5)), 4),
                   n_draws=len(v), matched_auc=ma,
                   frac_random_below_matched=round(float((v <= ma).mean()), 3))
        ctrl_rows.append(row)
        print(f"  CTRL encoder   {task:<6} random same-n AUC={v.mean():.3f} "
              f"[{np.percentile(v,2.5):.3f},{np.percentile(v,97.5):.3f}] "
              f"vs MATCH_AGE {ma:.3f} "
              f"(frac random <= matched: {(v<=ma).mean():.2f})", flush=True)

    # ---- (b) silence ----
    rngs = np.random.default_rng(11)
    sil_rows = []
    Xenv = np.asarray(sil["X"], float)
    ysil = np.asarray(sil["y"], int)
    tsil = sil["task"].astype(str)
    for tag, spks in subsets.items():
        S = set(spks)
        for block in ["ALL", "read", "ddk"]:
            m = np.array([g in S for g in sil_g])
            if block != "ALL":
                m &= (tsil == block)
            if m.sum() < 30:
                continue
            r = silence_run(Xenv[m], ysil[m], sil_g[m], tag, block, rngs)
            if r:
                sil_rows.append(r)
                print(f"  SIL {tag:<14} {block:<6} clips={r['n_clips']:<5} "
                      f"spk={r['n']:<4} clipAUC={r['auc']:.3f} "
                      f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] spkAUC={r['speaker_auc']:.3f} "
                      f"null={r['perm_null_mean']:.3f} p={r['perm_p']:.4f}", flush=True)

    # size control for the silence probe: random same-n speaker subsets
    m_all = np.array([g in set(subsets["FULL"]) for g in sil_g])
    pool_spk = np.unique(sil_g[m_all])
    spk_lab = {s: int(ysil[sil_g == s][0]) for s in pool_spk}
    ipd = [s for s in pool_spk if spk_lab[s] == 1]
    ihc = [s for s in pool_spk if spk_lab[s] == 0]
    matched_sil = {r["task"]: r["auc"] for r in sil_rows if r["subset"] == "MATCH_AGE"}
    rr = np.random.default_rng(31)
    for block in ["ALL", "read"]:
        vals = []
        for _ in range(K_SIZECTRL):
            S = set(list(rr.choice(ipd, min(n_pd_m, len(ipd)), replace=False)) +
                    list(rr.choice(ihc, min(n_hc_m, len(ihc)), replace=False)))
            mm = np.array([g in S for g in sil_g])
            if block != "ALL":
                mm &= (tsil == block)
            r = silence_run_quick(Xenv[mm], ysil[mm], sil_g[mm])
            if r is not None and np.isfinite(r):
                vals.append(r)
        if not vals:
            continue
        v = np.array(vals); ma = matched_sil.get(block, np.nan)
        ctrl_rows.append(dict(measure="silence_env", subset="SIZE_CTRL_random", task=block,
                              n=n_pd_m + n_hc_m, n_pd=n_pd_m, n_hc=n_hc_m,
                              auc=round(float(v.mean()), 4),
                              ci_lo=round(float(np.percentile(v, 2.5)), 4),
                              ci_hi=round(float(np.percentile(v, 97.5)), 4),
                              n_draws=len(v), matched_auc=ma,
                              frac_random_below_matched=round(float((v <= ma).mean()), 3)))
        print(f"  CTRL silence   {block:<6} random same-n clipAUC={v.mean():.3f} "
              f"[{np.percentile(v,2.5):.3f},{np.percentile(v,97.5):.3f}] "
              f"vs MATCH_AGE {ma:.3f} "
              f"(frac random <= matched: {(v<=ma).mean():.2f})", flush=True)

    # ---- (c) WER ----
    wer_rows = []
    for tag, spks in subsets.items():
        sp = wspk[wspk.speaker.isin(spks)]
        if len(sp) < 20 or sp.label.nunique() < 2:
            print(f"SKIP wer {tag}: n={len(sp)}", flush=True)
            continue
        r = wer_run(sp, tag, np.random.default_rng(23))
        wer_rows.append(r)
        print(f"  WER {tag:<14} n={r['n']:<4} AUC={r['auc']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
              f"PD={r['mean_wer_pd']:.3f} HC={r['mean_wer_hc']:.3f} "
              f"p={r['perm_p']:.4f}", flush=True)

    # size control for WER
    matched_wer = next((r["auc"] for r in wer_rows if r["subset"] == "MATCH_AGE"), np.nan)
    wpool = wspk[wspk.speaker.isin(subsets["FULL"])]
    wpd = wpool.index[wpool.label == 1].values; whc = wpool.index[wpool.label == 0].values
    rr2 = np.random.default_rng(41); vals = []
    for _ in range(200):
        sel = np.concatenate([rr2.choice(wpd, min(n_pd_m, len(wpd)), replace=False),
                              rr2.choice(whc, min(n_hc_m, len(whc)), replace=False)])
        s = wpool.loc[sel]
        vals.append(auc(s.label.values.astype(int), s.wer.values.astype(float)))
    v = np.array([x for x in vals if np.isfinite(x)])
    ctrl_rows.append(dict(measure="wer_read", subset="SIZE_CTRL_random", task="read",
                          n=int(min(n_pd_m, len(wpd)) + min(n_hc_m, len(whc))),
                          n_pd=int(min(n_pd_m, len(wpd))), n_hc=int(min(n_hc_m, len(whc))),
                          auc=round(float(v.mean()), 4),
                          ci_lo=round(float(np.percentile(v, 2.5)), 4),
                          ci_hi=round(float(np.percentile(v, 97.5)), 4),
                          n_draws=len(v), matched_auc=matched_wer,
                          frac_random_below_matched=round(float((v <= matched_wer).mean()), 3)))
    print(f"  CTRL wer       read   random same-n AUC={v.mean():.3f} "
          f"[{np.percentile(v,2.5):.3f},{np.percentile(v,97.5):.3f}] "
          f"vs MATCH_AGE {matched_wer:.3f} "
          f"(frac random <= matched: {(v<=matched_wer).mean():.2f})", flush=True)

    allr = pd.DataFrame(enc_rows + sil_rows + wer_rows + ctrl_rows)
    allr.to_csv(f"{OUT}/neurovoz_agematch_results.csv", index=False)

    print("\n================ NEUROVOZ AGE MATCHING SUMMARY ================", flush=True)
    print(bdf[["subset", "n", "n_pd", "n_hc", "age_pd_mean", "age_hc_mean", "age_diff",
               "age_smd", "welch_t_p", "age_only_auc"]].to_string(index=False), flush=True)
    print("\nmeasure                 FULL            MATCH_AGE       MATCH_AGE_SEX   "
          "SIZE_CTRL_random", flush=True)
    d = allr.set_index(["measure", "task", "subset"])
    for meas, task in [("encoder", "read"), ("encoder", "vowel"), ("encoder", "ddk"),
                       ("encoder", "all"), ("silence_env", "ALL"), ("silence_env", "read"),
                       ("wer_read", "read")]:
        cells = []
        for s in ["FULL", "MATCH_AGE", "MATCH_AGE_SEX", "SIZE_CTRL_random"]:
            try:
                r = d.loc[(meas, task, s)]
                cells.append(f"{r['auc']:.3f}(n={int(r['n'])})")
            except KeyError:
                cells.append("--")
        print(f"{meas + '/' + task:<22} " + "  ".join(f"{c:<14}" for c in cells), flush=True)
    print(f"\nwrote {OUT}/neurovoz_agematch_balance.csv", flush=True)
    print(f"wrote {OUT}/neurovoz_agematch_results.csv", flush=True)
    print(f"wrote {OUT}/neurovoz_agematch_roster.csv", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
