"""POD3 3a: per-layer probe AUCs + nested-with-selection AUCs (with speaker bootstrap CIs)
from an already-extracted states npz.

Per-layer: a single layer carries no layer choice, so the inner GroupKFold(4) selection is
degenerate; the value is the outer GroupKFold(5) out-of-fold AUC for THAT layer, averaged over
R speaker-permutation repeats (identical outer machinery to nested_repeats_all.py, which itself
sets best=0 when n_layers==1).
Nested: outer GroupKFold(5) by speaker, inner GroupKFold(4) on the training speakers picks the
layer, refit, out-of-fold AUC. Averaged over the same R repeats.

usage: perlayer_nested.py STATES.npz OUTPREFIX [R]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, csv, time, datetime, warnings
warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

NPZ, OUTP = sys.argv[1:3]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 5
t0 = time.time()
z = np.load(NPZ, allow_pickle=True)
y = z["label"].astype(int)
spk = z["spk"].astype(str)
NJ = max(2, (os.cpu_count() or 8) - 8)
print(f"loaded {NPZ} n={len(y)} pos={int(y.sum())} n_spk={len(set(spk))} njobs={NJ}", flush=True)


def rank_auc(yv, s):
    """AUC by the rank formula (ties averaged). Independent of sklearn."""
    yv = np.asarray(yv); s = np.asarray(s, dtype=np.float64)
    n1 = int((yv == 1).sum()); n0 = int((yv == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[yv == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]


def groups_for(seed):
    rng = np.random.default_rng(seed)
    u = np.unique(spk)
    perm = {s: i for i, s in enumerate(rng.permutation(u))}
    return np.array([perm[s] for s in spk])


def oof_single(X, l, g):
    """Outer 5-fold OOF for one fixed layer l."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        oof[te] = fitpred(X[tr][:, l], y[tr], X[te][:, l])
    return oof


def inner(X, tr, l, g):
    s = 0.0
    for itr, ite in GroupKFold(n_splits=4).split(X[tr], y[tr], groups=g[tr]):
        p = fitpred(X[tr][itr, l], y[tr][itr], X[tr][ite, l])
        s += roc_auc_score(y[tr][ite], p) if len(set(y[tr][ite])) > 1 else 0.5
    return s / 4


def oof_nested(X, g, nl):
    oof = np.zeros(len(y)); chosen = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        if nl == 1:
            best = 0
        else:
            best = int(np.argmax(Parallel(n_jobs=min(nl, NJ))(
                delayed(inner)(X, tr, l, g) for l in range(nl))))
        chosen.append(best)
        oof[te] = fitpred(X[tr][:, best], y[tr], X[te][:, best])
    return oof, chosen


def boot_ci(oofs, draws=2000, seed=0):
    """Speaker bootstrap on the mean-over-repeats AUC. Resamples SPEAKERS, not clips."""
    rng = np.random.default_rng(seed)
    u = np.unique(spk)
    idx_by = {s: np.where(spk == s)[0] for s in u}
    vals = []
    for _ in range(draws):
        pick = rng.choice(u, size=len(u), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        yy = y[idx]
        if len(set(yy)) < 2:
            continue
        vals.append(float(np.mean([rank_auc(yy, o[idx]) for o in oofs])))
    v = np.array(vals)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


STAGE_NAME = {
    "enc": lambda i, nl: f"audio_encoder_layer_{i}",
    "proj": lambda i, nl: "projected_audio_input",
    "llm": lambda i, nl: ("thinker_embeddings" if i == 0 else f"thinker_layer_{i}"),
    "ans": lambda i, nl: ("thinker_embeddings" if i == 0 else f"thinker_layer_{i}"),
}
FILE_SUFFIX = {"enc": "encoder", "proj": "proj", "llm": "llm", "ans": "ans"}
POOLING = {
    "enc": "mean over audio encoder frames (forward hook on the encoder layer output)",
    "proj": "mean over projector output positions (forward hook on the projector)",
    "llm": "mean over the AUDIO TOKEN positions of the thinker hidden state",
    "ans": "the first answer position (last prompt position) of the thinker hidden state",
}

summary = {}
GS = [groups_for(s) for s in range(R)]

for key in [k for k in ("enc", "proj", "llm", "ans") if k in z.files]:
    X = z[key].astype(np.float32)
    nl = X.shape[1]
    reps = 1 if nl == 1 else R
    print(f"\n=== {key}  n_points={nl}  repeats={reps}  {time.time()-t0:.0f}s ===", flush=True)

    # ---- per layer ----
    per_layer = {}
    for l in range(nl):
        oofs = [oof_single(X, l, GS[s]) for s in range(reps)]
        aucs = [rank_auc(y, o) for o in oofs]
        lo, hi, nd = boot_ci(oofs)
        per_layer[l] = {"oofs": oofs, "aucs": aucs, "mean": float(np.mean(aucs)),
                        "std": float(np.std(aucs)), "lo": lo, "hi": hi, "draws": nd}
        print(f"  {key} stage {l:>2} auc_mean {np.mean(aucs):.4f} "
              f"[{lo:.4f},{hi:.4f}] per_repeat {[round(a,4) for a in aucs]}", flush=True)

    out_csv = f"{OUTP}_{FILE_SUFFIX[key]}_perlayer.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "stage_name", "auc_mean", "auc_std", "ci_lo", "ci_hi",
                    "boot_draws", "per_repeat", "n", "n_speakers"])
        for l in range(nl):
            d = per_layer[l]
            w.writerow([l, STAGE_NAME[key](l, nl), round(d["mean"], 6), round(d["std"], 6),
                        round(d["lo"], 6), round(d["hi"], 6), d["draws"],
                        "|".join(f"{a:.6f}" for a in d["aucs"]), len(y), len(set(spk))])
    print(f"  wrote {out_csv}", flush=True)

    best = max(range(nl), key=lambda l: per_layer[l]["mean"])
    bd = per_layer[best]

    # ---- nested (layer chosen inside the training folds) ----
    n_oofs, chosen_all = [], []
    for s in range(reps):
        o, ch = oof_nested(X, GS[s], nl)
        n_oofs.append(o); chosen_all.append(ch)
        print(f"  nested repeat {s} auc {rank_auc(y,o):.4f} chosen {ch}  {time.time()-t0:.0f}s", flush=True)
    n_aucs = [rank_auc(y, o) for o in n_oofs]
    nlo, nhi, ndr = boot_ci(n_oofs)
    with open(f"{OUTP}_{FILE_SUFFIX[key]}_nested_oof.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "speaker", "label"] + [f"p_probe_rep{s}" for s in range(reps)])
        names = z["name"].astype(str)
        for i in range(len(y)):
            w.writerow([names[i], spk[i], int(y[i])] + [f"{o[i]:.8f}" for o in n_oofs])

    summary[key] = {
        "n_points": nl, "repeats": reps, "n": int(len(y)), "n_speakers": int(len(set(spk))),
        "pooling": POOLING[key],
        "best_stage": int(best), "best_stage_name": STAGE_NAME[key](best, nl),
        "best_stage_auc_mean": round(bd["mean"], 4),
        "best_stage_ci": [round(bd["lo"], 4), round(bd["hi"], 4)],
        "best_stage_per_repeat": [round(a, 4) for a in bd["aucs"]],
        "nested_auc_mean": round(float(np.mean(n_aucs)), 4),
        "nested_ci": [round(nlo, 4), round(nhi, 4)],
        "nested_boot_draws": ndr,
        "nested_per_repeat": [round(a, 4) for a in n_aucs],
        "nested_chosen_layers": chosen_all,
    }
    print(f"  >>> {key}: NESTED {np.mean(n_aucs):.4f} [{nlo:.4f},{nhi:.4f}] | "
          f"BEST STAGE {best} ({STAGE_NAME[key](best,nl)}) {bd['mean']:.4f} "
          f"[{bd['lo']:.4f},{bd['hi']:.4f}]", flush=True)

json.dump({
    "states": os.path.abspath(NPZ), "repeats": R, "bootstrap_draws": 2000,
    "bootstrap_seed": 0, "bootstrap_unit": "speaker",
    "folds": "GroupKFold(5) by speaker (outer), GroupKFold(4) by speaker (inner layer choice); "
             "speaker order permuted per repeat with numpy.random.default_rng(seed)",
    "fold_file": "NONE FOUND - no saved fold assignment exists for these clips; "
                 "no fold column in any *_oof.csv for them. GroupKFold by speaker used instead.",
    "auc": "rank formula, ties averaged, recomputed from the per-clip out-of-fold scores",
    "date": datetime.datetime.now().isoformat(timespec="seconds"),
    "streams": summary,
}, open(f"{OUTP}_perlayer_summary.json", "w"), indent=1)
print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
