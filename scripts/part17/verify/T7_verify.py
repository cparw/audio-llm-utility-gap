#!/usr/local/bin/python3
"""PART17 T7 independent verifier. Written from scratch; imports nothing from the main job.

Sources (originals only):
  shards  : extracted by THIS script's operator from
            <local data dir>/paper work/paper1_local_runs/part16_EDAICFULL_states/edaicfull_shards.tgz
            into part17/verify/T7_verify_scratch/shards
  labels/order/p_yes : part16/EDAICFULL/full_zeroshot_scores.csv   (pod output)
  pod results        : part16/EDAICFULL/edaic_full_nested_repeats.json, probe2.log, edaic_full_perclip.csv

Modes:
  probe <stream>...   own nested probe (outer group 5-fold, inner group 4-fold picks the layer, 5 repeats)
  forced <stream>...  same outer folds, layer forced to the pod's recorded pick
  boot                AUCs, intervals, cross checks, comparison with the main job's numbers
"""
import os, sys, csv, json, time, hashlib
import numpy as np

BASE = "<local data dir>/release/edaic_rerun"
SRC = f"{BASE}/part16/EDAICFULL"
VD = f"{BASE}/part17/verify"
SCR = f"{VD}/T7_verify_scratch"
SHARDS = f"{SCR}/shards"
TGZ = "<local data dir>/paper work/paper1_local_runs/part16_EDAICFULL_states/edaicfull_shards.tgz"
REPEATS = 5
TAG = os.environ.get("T7V_TAG", "")  # suffix for outputs of a labelled alternate-interpreter run
NB = 2000


# ---------------------------------------------------------------- AUC, two independent ways
def auc_mw(y, s):
    """Mann-Whitney over every (positive, negative) pair, ties count 0.5."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    P = s[y == 1]; N = s[y == 0]
    if len(P) == 0 or len(N) == 0:
        return float("nan")
    d = P[:, None] - N[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(P) * len(N)))


def auc_trap(y, s):
    """Area under the empirical ROC by the trapezoid rule over distinct thresholds."""
    y = np.asarray(y).astype(np.int64); s = np.asarray(s, dtype=np.float64)
    o = np.argsort(-s, kind="mergesort")
    ss = s[o]; yy = y[o]
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    tp = np.cumsum(yy)[last].astype(float); fp = (last + 1) - tp
    if tp[-1] == 0 or fp[-1] == 0:
        return float("nan")
    tpr = np.r_[0.0, tp / tp[-1]]; fpr = np.r_[0.0, fp / fp[-1]]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


# ---------------------------------------------------------------- data
def load_scores():
    rows = list(csv.DictReader(open(f"{SRC}/full_zeroshot_scores.csv")))
    clips = [r["clip"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    spk = np.array([r["speaker"] for r in rows])
    p = np.array([float(r["p_yes"]) for r in rows])
    return clips, y, spk, p


def load_stream(clips, key):
    return np.stack([np.load(f"{SHARDS}/{c}.npz")[key] for c in clips])  # keep float32 like the pod


# ---------------------------------------------------------------- folds (own implementation)
def group_folds(groups, k, seed):
    """Speaker folds: sorted unique groups, permuted by RandomState(seed), split into k near-equal parts.
    Train and test indices are returned in ascending order."""
    ug = np.unique(groups)
    perm = np.random.RandomState(seed).permutation(ug)
    allidx = np.arange(len(groups))
    out = []
    for part in np.array_split(perm, k):
        m = np.isin(groups, part)
        out.append((allidx[~m], allidx[m]))
    return out


def fit_predict(Xa, ya, Xb):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(Xa)
    m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    m.fit(sc.transform(Xa), ya)
    return m.predict_proba(sc.transform(Xb))[:, 1]


def inner_score(XtrL, ytr, isplits):
    o = np.zeros(len(ytr))
    for a, b in isplits:
        o[b] = fit_predict(XtrL[a], ytr[a], XtrL[b])
    return auc_mw(ytr, o)


def pod_picks():
    j = json.load(open(f"{SRC}/edaic_full_nested_repeats.json"))
    return {k: j[k]["layers_picked"] for k in j}


def run_stream(key, mode, clips, y, spk):
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold
    X = load_stream(clips, key)
    nL = X.shape[1]
    pp = pod_picks()[key]
    oofs, picks, inner_all, fold_te = [], [], [], []
    t0 = time.time()
    for rep in range(REPEATS):
        oof = np.full(len(y), np.nan)
        outer = group_folds(spk, 5, rep)
        # cross check own folds against sklearn's GroupKFold(shuffle=True)
        sk = list(GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(X[:, 0, :], y, groups=spk))
        assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(outer, sk)), "fold mismatch"
        for f, (tr, te) in enumerate(outer):
            fold_te.append(te.tolist())
            if mode == "forced" or nL == 1:
                best = int(pp[rep * 5 + f]) if mode == "forced" else 0
                sc_l = []
            else:
                Xtr = X[tr]; ytr = y[tr]; str_ = spk[tr]
                isp = group_folds(str_, 4, rep)
                sk_i = list(GroupKFold(n_splits=4, shuffle=True, random_state=rep).split(Xtr[:, 0, :], ytr, groups=str_))
                assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(isp, sk_i)), "inner fold mismatch"
                sc_l = Parallel(n_jobs=16, prefer="processes")(
                    delayed(inner_score)(Xtr[:, L, :], ytr, isp) for L in range(nL))
                best = int(np.argmax(sc_l))
            picks.append(best); inner_all.append([float(v) for v in sc_l])
            XL = X[:, best, :]
            oof[te] = fit_predict(XL[tr], y[tr], XL[te])
        assert not np.isnan(oof).any()
        oofs.append(oof)
        print(f"  {mode} {key} rep {rep}: AUC_mw {auc_mw(y, oof):.6f} trap {auc_trap(y, oof):.6f} "
              f"picks {picks[-5:]} pod {pp[rep*5:rep*5+5]}  t={time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(f"{SCR}/V{TAG}_{mode}_{key}.npz", oof=np.array(oofs), picks=np.array(picks),
                        inner=np.array(inner_all, dtype=object), y=y, spk=spk, clip=np.array(clips))
    json.dump(dict(key=key, mode=mode, picks=picks, pod_picks=pp, inner_auc_per_layer=inner_all,
                   per_repeat_mw=[auc_mw(y, o) for o in oofs], per_repeat_trap=[auc_trap(y, o) for o in oofs],
                   fold_test_idx=fold_te),
              open(f"{SCR}/V{TAG}_{mode}_{key}.json", "w"))


# ---------------------------------------------------------------- bootstrap
def draws(n_spk_sorted_to_idx):
    rng = np.random.default_rng(0)
    m = len(n_spk_sorted_to_idx)
    return [n_spk_sorted_to_idx[rng.choice(m, size=m, replace=True)] for _ in range(NB)]


def boot_cell(D, y, arm_vecs, ref=None, aucf=auc_mw):
    """arm_vecs: list of score vectors; statistic = mean over them of AUC (one vector -> plain AUC).
    ref: answer score vector for the paired difference; the SAME draw is used for both."""
    st, df = [], []
    for ii in D:
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        a = float(np.mean([aucf(yy, v[ii]) for v in arm_vecs]))
        st.append(a)
        if ref is not None:
            df.append(a - aucf(yy, ref[ii]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out = dict(usable=len(st), lo=q(st)[0], hi=q(st)[1])
    if ref is not None:
        out.update(dlo=q(df)[0], dhi=q(df)[1])
    return out


def sha(p, first=None):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read(first) if first else fh.read())
    return h.hexdigest()


def main_boot():
    clips, y, spk, p = load_scores()
    res = dict(env={}, checks={}, cells={})
    import sklearn, scipy
    res["env"] = dict(python=sys.executable, numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__)
    # ---- input consistency with the per-clip file
    pc = list(csv.DictReader(open(f"{SRC}/edaic_full_perclip.csv")))
    assert [r["pid"] for r in pc] == [c.replace(".wav", "") for c in clips]
    assert all(int(r["label"]) == v for r, v in zip(pc, y))
    pa = np.array([float(r["p_yes_answer"]) for r in pc])
    res["checks"]["p_yes_maxabs_scores_vs_perclip"] = float(np.max(np.abs(pa - p)))
    res["checks"]["n"] = int(len(y)); res["checks"]["n_spk"] = int(len(set(spk))); res["checks"]["n_pos"] = int(y.sum())
    pub = {k: np.array([float(r[f"oof_{k}"]) for r in pc]) for k in ("enc", "proj", "llm", "ans")}

    uspk = np.array(sorted(set(spk)))
    pos = {s: i for i, s in enumerate(spk)}
    D = draws(np.array([pos[s] for s in uspk]))

    def cell(name, vecs, ref=None):
        pt_mw = float(np.mean([auc_mw(y, v) for v in vecs])); pt_tr = float(np.mean([auc_trap(y, v) for v in vecs]))
        b = boot_cell(D, y, vecs, ref, auc_mw); bt = boot_cell(D, y, vecs, ref, auc_trap)
        c = dict(point_mw=pt_mw, point_trap=pt_tr, lo=b["lo"], hi=b["hi"], lo_trap=bt["lo"], hi_trap=bt["hi"],
                 usable=b["usable"], per_vec_mw=[auc_mw(y, v) for v in vecs])
        if ref is not None:
            aref = auc_mw(y, ref)
            c.update(diff_point=pt_mw - aref, dlo=b["dlo"], dhi=b["dhi"], dlo_trap=bt["dlo"], dhi_trap=bt["dhi"],
                     excl0="yes" if (b["dlo"] > 0 or b["dhi"] < 0) else "no")
        res["cells"][name] = c
        s = f"{name:34s} {pt_mw:.4f} (trap {pt_tr:.4f}) [{b['lo']:.4f}, {b['hi']:.4f}] trap[{bt['lo']:.4f}, {bt['hi']:.4f}]"
        if ref is not None:
            s += f"  diff {c['diff_point']:.4f} [{b['dlo']:.4f}, {b['dhi']:.4f}] trap[{bt['dlo']:.4f}, {bt['dhi']:.4f}] excl0 {c['excl0']}"
        print(s, flush=True)

    print("=== answer ===")
    cell("answer_p_yes", [p])
    print("=== published per-clip (AUC of mean OOF, NOT USED rows) ===")
    for k in ("llm", "enc", "ans", "proj"):
        cell(f"pub_{k}_aucofmean", [pub[k]], p)

    pod = json.load(open(f"{SRC}/edaic_full_nested_repeats.json"))
    for mode in ("probe", "forced"):
        for k in ("llm", "enc", "ans", "proj"):
            f = f"{SCR}/V{TAG}_{mode}_{k}.npz"
            if not os.path.exists(f):
                print(f"missing {f}"); continue
            z = np.load(f, allow_pickle=True)
            oofs = list(z["oof"]); picks = z["picks"].tolist()
            pr = [auc_mw(y, o) for o in oofs]
            podpr = pod[k]["per_repeat"]
            res["checks"][f"{mode}_{k}"] = dict(
                per_repeat_mine=[round(v, 6) for v in pr], per_repeat_pod=podpr,
                per_repeat_match_4dp=[round(a, 4) == b for a, b in zip(pr, podpr)],
                max_abs_diff_vs_pod=float(np.max(np.abs(np.round(pr, 4) - np.array(podpr)))),
                mean_mine=float(np.mean(pr)), mean_pod=pod[k]["mean_of_repeat_aucs"],
                picks_match=int(sum(a == b for a, b in zip(picks, pod[k]["layers_picked"]))),
                picks_mismatch=[(i // 5, i % 5, a, b) for i, (a, b) in enumerate(zip(picks, pod[k]["layers_picked"])) if a != b],
                auc_of_mean_oof_mine=auc_mw(y, np.mean(oofs, axis=0)), auc_of_mean_oof_pod=pod[k]["auc_of_mean_oof"],
                r_meanoof_vs_published=float(np.corrcoef(np.mean(oofs, axis=0), pub[k])[0, 1]),
                maxabs_meanoof_vs_published=float(np.max(np.abs(np.mean(oofs, axis=0) - pub[k]))))
            print(f"--- {mode} {k}: mine {[round(v,4) for v in pr]} pod {podpr} "
                  f"picks match {res['checks'][f'{mode}_{k}']['picks_match']}/25 "
                  f"mismatch {res['checks'][f'{mode}_{k}']['picks_mismatch']}")
            cell(f"{mode}_{k}_meanof5", oofs, p)
            # inner-AUC margins where my layer pick differs from the pod's
            if mode == "probe" and z["inner"].size and len(z["inner"][0]):
                marg = []
                for (r_, f_, mine, podl) in res["checks"][f"{mode}_{k}"]["picks_mismatch"]:
                    row = list(z["inner"][r_ * 5 + f_])
                    marg.append(dict(rep=r_, fold=f_, mine_layer=mine, mine_inner=row[mine], pod_layer=podl,
                                     pod_layer_inner_on_mac=row[podl], gap=row[mine] - row[podl]))
                res["checks"][f"{mode}_{k}"]["flip_margins"] = marg
                print(f"    flip margins {k}: {marg}")
            # compare against the main job's per-clip vectors (check only, never used as input)
            if TAG == "":
                tf = {"forced": f"{BASE}/part17/T7_edaic300_meanof5_perclip.csv",
                      "probe": f"{BASE}/part17/T7_edaic300_meanof5_perclip_macselected.csv"}[mode]
                tr_ = list(csv.DictReader(open(tf)))
                assert [r["pid"] for r in tr_] == [c.replace(".wav", "") for c in clips]
                mx = max(float(np.max(np.abs(np.array([float(r[f"oof_{k}_r{i}"]) for r in tr_]) - oofs[i])))
                         for i in range(REPEATS))
                res["checks"][f"{mode}_{k}"]["maxabs_vs_task_perclip"] = mx
                print(f"    max |mine - task per-clip| {k} {mode}: {mx:.3e}")
    for k in ("llm", "enc", "ans", "proj"):
        res["checks"][f"pod_{k}_mean_of_rounded_per_repeat"] = float(np.mean(pod[k]["per_repeat"]))
    json.dump(res, open(f"{VD}/T7_verify{TAG}.json", "w"), indent=1)
    print("wrote", f"{VD}/T7_verify{TAG}.json")


if __name__ == "__main__":
    if sys.argv[1] in ("probe", "forced"):
        clips, y, spk, p = load_scores()
        for k in sys.argv[2:]:
            print(f"=== {sys.argv[1]} {k} ===", flush=True)
            run_stream(k, sys.argv[1], clips, y, spk)
    elif sys.argv[1] == "boot":
        main_boot()
