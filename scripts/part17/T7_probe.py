"""T7 step 1: rerun the E-DAIC 300 s window nested probe exactly as
part16/EDAICFULL/probe_boot.py did, this time SAVING the per-repeat per-clip OOF vectors.

Same data order (full_zeroshot_scores.csv row order), same shards, same folds
(outer GroupKFold(5, shuffle=True, random_state=rep), inner GroupKFold(4, shuffle=True,
random_state=rep), rep 0..4), same probe (StandardScaler train only + LogisticRegression
max_iter=2000, class_weight='balanced', C=1.0), same layer pick (argmax of inner rank AUC).

Only change vs probe_boot.py: the inner per-layer scorer is a module-level function that takes
the outer-train block as an argument (joblib memmaps it) instead of a closure that pickles it
once per layer. The arithmetic inside is line for line the same.

Writes T7_scratch/T7_oof_repeats.npz and T7_scratch/T7_gate.json. Does not bootstrap.
"""
import os, csv, json, sys, time, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

P16 = "<local data dir>/release/edaic_rerun/part16/EDAICFULL"
P17 = "<local data dir>/release/edaic_rerun/part17"
SCR = f"{P17}/T7_scratch"
SHARDS = f"{SCR}/shards"
REPEATS = 5
NJOBS = int(os.environ.get("T7_NJOBS", "16"))
STAGES = sys.argv[1:] or ["enc", "proj", "llm", "ans"]
TAG = os.environ.get("T7_TAG", "A")  # variant label: which interpreter/BLAS produced the vectors


def auc_rank(y, s):
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


def probe(Xl, yl, tr, te):
    ss = StandardScaler().fit(Xl[tr])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(ss.transform(Xl[tr]), yl[tr])
    return lr.predict_proba(ss.transform(Xl[te]))[:, 1]


def score_layer(Xtr, ytr, isp, L):
    o = np.zeros(len(ytr))
    for itr, ite in isp:
        o[ite] = probe(Xtr[:, L, :], ytr, itr, ite)
    return auc_rank(ytr, o)


def main():
    sc = list(csv.DictReader(open(f"{P16}/full_zeroshot_scores.csv")))
    clips = [r["clip"] for r in sc]
    y = np.array([int(r["label"]) for r in sc])
    spk = np.array([r["speaker"] for r in sc])
    print(f"n clips {len(clips)}  n speakers {len(set(spk))}  positives {int(y.sum())}", flush=True)
    assert len(set(spk)) == len(clips)

    S = {k: [] for k in ("enc", "proj", "llm", "ans")}
    for c in clips:
        z = np.load(f"{SHARDS}/{c}.npz")
        for k in S: S[k].append(z[k])
    for k in S:
        S[k] = np.stack(S[k]); print(f"  {k}: {S[k].shape} {S[k].dtype}", flush=True)

    out_npz = f"{SCR}/T7_oof_repeats_{TAG}.npz"
    saved = dict(np.load(out_npz, allow_pickle=False)) if os.path.exists(out_npz) else {}
    fold_log = {}
    res = {}
    for key in STAGES:
        t0 = time.time()
        X = S[key]; nL = X.shape[1]
        per_repeat, oof_all, picks, folds = [], [], [], []
        print(f"--- stage {key} ---", flush=True)
        for rep in range(REPEATS):
            oof = np.zeros(len(y)); fold_id = np.full(len(y), -1)
            outer = GroupKFold(n_splits=5, shuffle=True, random_state=rep)
            for f, (tr, te) in enumerate(outer.split(X, y, groups=spk)):
                fold_id[te] = f
                if nL == 1:
                    best = 0
                else:
                    inner = GroupKFold(n_splits=4, shuffle=True, random_state=rep)
                    isp = list(inner.split(X[tr], y[tr], groups=spk[tr]))
                    ytr = y[tr]; Xtr = X[tr]
                    sc_l = Parallel(n_jobs=NJOBS, prefer="processes")(
                        delayed(score_layer)(Xtr, ytr, isp, L) for L in range(nL))
                    best = int(np.argmax(sc_l))
                picks.append(best)
                oof[te] = probe(X[:, best, :], y, tr, te)
            a = auc_rank(y, oof)
            per_repeat.append(a); oof_all.append(oof); folds.append(fold_id)
            print(f"  {key} repeat {rep}: AUC {a:.4f}  layers {picks[-5:]}  ({time.time()-t0:.0f}s)", flush=True)
        res[key] = dict(per_repeat=per_repeat, picks=picks)
        saved[f"{key}_oof_repeats"] = np.array(oof_all)
        saved[f"{key}_layers_picked"] = np.array(picks)
        saved[f"{key}_per_repeat_auc"] = np.array(per_repeat)
        saved[f"{key}_outer_fold"] = np.array(folds)
        saved["label"] = y; saved["speaker"] = spk; saved["clip"] = np.array(clips)
        np.savez_compressed(out_npz, **saved)
        print(f"  saved {key} -> {out_npz}", flush=True)

    # ---------- reproduction gate ----------
    pub = json.load(open(f"{P16}/edaic_full_nested_repeats.json"))
    perclip = list(csv.DictReader(open(f"{P16}/edaic_full_perclip.csv")))
    assert [r["pid"] for r in perclip] == [c.replace(".wav", "") for c in clips]
    gate_path = f"{SCR}/T7_gate_{TAG}.json"
    gate = json.load(open(gate_path)) if os.path.exists(gate_path) else {}
    for key in res:
        mine4 = [round(a, 4) for a in res[key]["per_repeat"]]
        oof_mean = saved[f"{key}_oof_repeats"].mean(axis=0)
        pubcol = np.array([float(r[f"oof_{key}"]) for r in perclip])
        g = dict(
            per_repeat_mine=mine4, per_repeat_published=pub[key]["per_repeat"],
            per_repeat_match=mine4 == pub[key]["per_repeat"],
            layers_mine=[int(x) for x in res[key]["picks"]], layers_published=pub[key]["layers_picked"],
            layers_match=[int(x) for x in res[key]["picks"]] == pub[key]["layers_picked"],
            mean_of_repeat_aucs_mine_fullprec=float(np.mean(res[key]["per_repeat"])),
            mean_of_repeat_aucs_mine_of_rounded=round(float(np.mean(mine4)), 4),
            mean_of_repeat_aucs_published=pub[key]["mean_of_repeat_aucs"],
            auc_of_mean_oof_mine=auc_rank(y, oof_mean), auc_of_mean_oof_published=pub[key]["auc_of_mean_oof"],
            max_abs_diff_mean_oof_vs_published_perclip=float(np.max(np.abs(oof_mean - pubcol))),
        )
        g["reproduced"] = bool(g["per_repeat_match"] and g["layers_match"])
        gate[key] = g
        print(f"GATE {key}: per_repeat match {g['per_repeat_match']}  layers match {g['layers_match']}  "
              f"max|mean OOF - published| {g['max_abs_diff_mean_oof_vs_published_perclip']:.2e}", flush=True)
        print(f"     mine {mine4}\n     pub  {pub[key]['per_repeat']}", flush=True)
    json.dump(gate, open(gate_path, "w"), indent=1)


if __name__ == "__main__":
    main()
