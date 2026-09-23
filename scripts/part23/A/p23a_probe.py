"""PART 23 A, item 3 probes on the E-DAIC WHOLE-window states (pod, CPU).
modes:
  probe <enc|proj|llm|ans>  nested probe exactly as part16/EDAICFULL/probe_boot.py run_stage(), except that the OUTER
                            GroupKFold(5, shuffle=True, random_state=rep) split is READ from the saved fold files
                            folds/edaic_full_groupkfold5_shuffle_rs{rep}.csv (never recomputed). Inner layer selection:
                            GroupKFold(4, shuffle=True, random_state=rep) on the outer-train rows, Parallel(n_jobs=16),
                            StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0), float32 features.
                            One npz per repeat is written as soon as that repeat is done (resume-safe).
  perlayer                  per-layer curves, method of part16/EDAICFULL/perlayer.py layer_curve() verbatim
                            (GroupKFold(n_splits=5) by speaker, no shuffle), written per stream as each finishes.
  fixed <enc|llm|ans>       SUPPLEMENT (not the paper convention): every single stage probed on the same five saved outer
                            splits with no inner selection; per-repeat OOF saved per stage.
"""
import os, sys, csv, json, time, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

D = "/workspace/p23a"; SH = f"{D}/shards"; PR = f"{D}/probe"; SUF = ".npz"
if os.environ.get("CTL300") == "1":   # same-GPU 300 s control: shards written by the unmodified extract_full.py (default processor call)
    SH = f"{D}/ctl300/shards"; PR = f"{D}/probe_ctl300"; SUF = ".wav.npz"
os.makedirs(PR, exist_ok=True)
REPEATS = 5

def auc_rank(y, s):  # identical to probe_boot.py
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); sv = s[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j+1] == sv[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))

# ---------- load (clip order = windows_full.csv order = part16 order) ----------
WIN = [r["pid"] for r in csv.DictReader(open(f"{D}/edaicfull/windows_full.csv"))]
LAB = {r["pid"]: int(r["label"]) for r in csv.DictReader(open(f"{D}/ref/edaic_lifted_perclip.csv"))}
clips = [f"{p}.wav" for p in WIN]
y = np.array([LAB[p] for p in WIN])
spk = np.array(WIN)            # E-DAIC: speaker id == pid, one clip per speaker (as in part16 full_zeroshot_scores.csv)
FOLD = {}
for rep in range(REPEATS):
    fr = {r["clip_id"]: r for r in csv.DictReader(open(f"{D}/folds/edaic_full_groupkfold5_shuffle_rs{rep}.csv"))}
    assert all(fr[c]["speaker_id"] == p for c, p in zip(clips, WIN))
    FOLD[rep] = np.array([int(fr[c]["fold"]) for c in clips])

def load_stream(key):
    return np.stack([np.load(f"{SH}/{p}{SUF}")[key] for p in WIN])   # float32 as stored

def probe(Xl, yl, tr, te):  # identical to probe_boot.py
    ss = StandardScaler().fit(Xl[tr])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(ss.transform(Xl[tr]), yl[tr])
    return lr.predict_proba(ss.transform(Xl[te]))[:, 1]

mode = sys.argv[1]
t0 = time.time()
if mode == "probe":
    key = sys.argv[2]
    X = load_stream(key); nL = X.shape[1]
    print(f"{key}: {X.shape} {X.dtype}  n {len(y)} n_spk {len(set(spk))} pos {int(y.sum())}", flush=True)
    for rep in range(REPEATS):
        fn = f"{PR}/{key}_rep{rep}.npz"
        if os.path.exists(fn):
            print(f"  {key} repeat {rep}: exists, skipped (resume)", flush=True); continue
        oof = np.zeros(len(y)); picks = []; inner_scores = []
        for k in range(5):                                   # saved outer folds
            te = np.where(FOLD[rep] == k)[0]; tr = np.where(FOLD[rep] != k)[0]
            if nL == 1:
                best = 0; sc_l = [float("nan")]
            else:
                inner = GroupKFold(n_splits=4, shuffle=True, random_state=rep)
                isp = list(inner.split(X[tr], y[tr], groups=spk[tr]))
                ytr = y[tr]; Xtr = X[tr]
                def score_layer(L):
                    o = np.zeros(len(tr))
                    for itr, ite in isp:
                        o[ite] = probe(Xtr[:, L, :], ytr, itr, ite)
                    return auc_rank(ytr, o)
                sc_l = Parallel(n_jobs=16, prefer="processes")(delayed(score_layer)(L) for L in range(nL))
                best = int(np.argmax(sc_l))
            picks.append(best); inner_scores.append(sc_l)
            oof[te] = probe(X[:, best, :], y, tr, te)
        a = auc_rank(y, oof)
        np.savez(fn, oof=oof, picks=np.array(picks), inner_scores=np.array(inner_scores, float), auc=a, fold=FOLD[rep])
        print(f"  {key} repeat {rep}: AUC {a:.4f}  layers {picks}  {time.time()-t0:.0f}s", flush=True)
    per = [float(np.load(f"{PR}/{key}_rep{r}.npz")["auc"]) for r in range(REPEATS)]
    print(f"{key} DONE per_repeat {[round(a,4) for a in per]} mean-of-5 {np.mean(per):.4f}", flush=True)
    open(f"{PR}/{key}_nested.RESULT", "w").write(json.dumps(dict(per_repeat=per, mean=float(np.mean(per)))) + "\n")

elif mode == "perlayer":
    def layer_curve(X, L):   # perlayer.py verbatim
        o = np.zeros(len(y)); fold = []
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=spk):
            ss = StandardScaler().fit(X[tr, L, :])
            lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(ss.transform(X[tr, L, :]), y[tr])
            p = lr.predict_proba(ss.transform(X[te, L, :]))[:, 1]; o[te] = p; fold.append(auc_rank(y[te], p))
        return float(np.mean(fold)), float(np.std(fold)), auc_rank(y, o)
    for key, name, first in [("enc", "o25_whole_encoder_perlayer.csv", "layer"), ("llm", "o25_whole_llm_perlayer.csv", "stage"),
                             ("ans", "o25_whole_ans_perlayer.csv", "stage"), ("proj", "o25_whole_proj_perlayer.csv", "stage")]:
        X = load_stream(key); nL = X.shape[1]
        res = Parallel(n_jobs=24, prefer="processes")(delayed(layer_curve)(X, L) for L in range(nL))
        with open(f"{PR}/{name}", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow([first, "auc_mean", "auc_std", "auc_oof"])
            for L in range(nL): w.writerow([L, res[L][0], res[L][1], res[L][2]])
        oo = np.array([r[2] for r in res])
        print(f"perlayer {key}: n_layers {nL} peak auc_oof {oo.max():.4f} @ {int(oo.argmax())}  {time.time()-t0:.0f}s", flush=True)
        open(f"{PR}/{name}.RESULT", "w").write("done\n")

elif mode == "fixed":
    key = sys.argv[2]
    X = load_stream(key); nL = X.shape[1]
    def one_stage(L):
        oo = []
        for rep in range(REPEATS):
            oof = np.zeros(len(y))
            for k in range(5):
                te = np.where(FOLD[rep] == k)[0]; tr = np.where(FOLD[rep] != k)[0]
                oof[te] = probe(X[:, L, :], y, tr, te)
            oo.append(oof)
        return np.array(oo)
    res = Parallel(n_jobs=16, prefer="processes")(delayed(one_stage)(L) for L in range(nL))
    OOF = np.stack(res)  # (nL, 5, n)
    per = np.array([[auc_rank(y, OOF[L, r]) for r in range(REPEATS)] for L in range(nL)])
    np.savez(f"{PR}/fixed_{key}.npz", oof=OOF, per_repeat=per)
    m = per.mean(1)
    print(f"fixed {key}: best stage {int(m.argmax())} mean-of-5 {m.max():.4f}  {time.time()-t0:.0f}s", flush=True)
    open(f"{PR}/fixed_{key}.RESULT", "w").write(json.dumps(dict(best=int(m.argmax()), meanof5=float(m.max()))) + "\n")
