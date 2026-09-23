"""PART17 T2 independent verifier: refit the saved probes with the released fold files as written.

Written from scratch. Reads only original run outputs (states, saved OOF, saved layer choices)
and the fold CSVs in release/folds. Writes only into part17/verify/ and a scratch folder.

usage: T2_verify.py SECTION   (mac_single | pod_single | pcgita_seeds | pitt_mac_seeds |
                                pitt15_seeds | edaicfull | omni_pitt_repeats | derive | files)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json, csv, time, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

REL = "<local data dir>/release"
GD = "<local data dir>/paper work/paper1_local_runs"
FOLDS = f"{REL}/folds"
OUTD = f"{REL}/edaic_rerun/part17/verify"
SCR = "<local data dir>/scratch"
NJ = 14
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- AUC, two independent ways
def auc_mw(y, s):
    """Mann-Whitney over all positive/negative pairs, ties count 0.5."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))


def auc_trap(y, s):
    """Trapezoid area under the empirical ROC, one point per distinct threshold."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    P = int(y.sum()); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort"); ss = s[o]; yy = y[o]
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    tp = np.cumsum(yy)[last]; fp = (last + 1) - tp
    tpr = np.r_[0.0, tp / P]; fpr = np.r_[0.0, fp / N]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc2(y, s):
    a, b = auc_mw(y, s), auc_trap(y, s)
    return {"mw": round(a, 6), "trap": round(b, 6), "agree_4dp": bool(round(a, 4) == round(b, 4))}


# ---------------------------------------------------------------- probe
def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def my_gkf(groups, n_splits, tie):
    """GroupKFold without shuffle, my own code. tie='mac': default numpy argsort on this machine;
    tie='stable': argsort(kind='stable'). Largest group first to the lightest fold."""
    ug, gi = np.unique(np.asarray(groups), return_inverse=True)
    cnt = np.bincount(gi)
    if tie == "mac":
        order = np.argsort(cnt)[::-1]
    elif tie == "stable":
        order = np.argsort(cnt, kind="stable")[::-1]
    else:
        raise ValueError(tie)
    load = np.zeros(n_splits); g2f = np.zeros(len(ug), dtype=int)
    for j in order:
        f = int(np.argmin(load)); load[f] += cnt[j]; g2f[j] = f
    return g2f[gi]


def sk_gkf(groups, n_splits, shuffle=False, rs=None):
    fold = np.full(len(groups), -1, dtype=int)
    kw = {"shuffle": True, "random_state": rs} if shuffle else {}
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits, **kw).split(np.zeros(len(groups)), groups=groups)):
        fold[te] = k
    return fold


def renumber(spk, seed):
    rng = np.random.default_rng(seed); u = np.unique(spk)
    perm = {s: i for i, s in enumerate(rng.permutation(u))}
    return np.array([perm[s] for s in spk])


# ---------------------------------------------------------------- fold files
def read_folds(fname, names, spk_expected=None):
    p = f"{FOLDS}/{fname}"
    df = pd.read_csv(p, dtype={"clip_id": str, "speaker_id": str, "fold": int})
    chk = {"file": fname, "rows": int(len(df)), "cols": list(df.columns)}
    assert list(df.columns) == ["clip_id", "speaker_id", "fold"], df.columns
    assert df.clip_id.is_unique, f"{fname} duplicate clip ids"
    m = dict(zip(df.clip_id, zip(df.speaker_id, df.fold)))
    names = [str(n) for n in names]
    miss = [n for n in names if n not in m]; extra = set(m) - set(names)
    assert not miss and not extra, f"{fname}: missing {len(miss)} extra {len(extra)}"
    fold = np.array([m[n][1] for n in names]); fspk = np.array([m[n][0] for n in names])
    chk["fold_sizes"] = np.bincount(fold, minlength=5).tolist()
    # speaker disjoint within the file's own labels
    bad = [s for s in set(fspk) if len(set(fold[fspk == s])) > 1]
    chk["speakers_across_folds"] = len(bad)
    if spk_expected is not None:
        chk["speaker_ids_match_run"] = bool((fspk == np.asarray(spk_expected).astype(str)).all())
    chk["sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return fold, fspk, chk


def refit_layers(X, y, fold, layers):
    oof = np.full(len(y), np.nan)
    for k in range(5):
        tr = fold != k; te = fold == k
        L = layers[k]
        oof[te] = fitpred(X[tr][:, L], y[tr], X[te][:, L])
    return oof


def refit_layers_par(X, y, fold, layers):
    def one(k):
        tr = fold != k; te = fold == k
        return k, fitpred(X[tr][:, layers[k]], y[tr], X[te][:, layers[k]])
    oof = np.full(len(y), np.nan)
    for k, p in Parallel(n_jobs=5)(delayed(one)(k) for k in range(5)):
        oof[fold == k] = p
    return oof


INNER_AUC = os.environ.get("T2V_INNER", "mw")   # "sk": sklearn roc_auc_score in the inner loop, as the runs did


def inner_score(Xl_tr, y_tr, ifold):
    """Inner layer score. The runs used sklearn roc_auc_score; its float rounding decides exact ties
    between layers, so T2V_INNER=sk replicates their tie-breaking. Default is my own Mann-Whitney."""
    if INNER_AUC == "sk":
        from sklearn.metrics import roc_auc_score as f
    else:
        f = auc_mw
    s = 0.0
    for j in range(4):
        te = ifold == j; tr = ~te
        p = fitpred(Xl_tr[tr], y_tr[tr], Xl_tr[te])
        s += f(y_tr[te], p) if len(set(y_tr[te])) > 1 else 0.5
    return s / 4


def derive_layers(X, y, fold, groups, tie):
    """Re-run the inner GroupKFold(4) layer choice for each outer fold of the given fold vector."""
    nl = X.shape[1]; chosen = []
    for k in range(5):
        tr = np.where(fold != k)[0]
        ifold = my_gkf(groups[tr], 4, tie)
        Xtr = X[tr]; ytr = y[tr]
        sc = Parallel(n_jobs=min(NJ, nl))(delayed(inner_score)(Xtr[:, l], ytr, ifold) for l in range(nl))
        chosen.append(int(np.argmax(sc)))
    return chosen


def mad(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return z


def save(section, obj):
    fn = f"{OUTD}/T2_verify_{section}{'_skinner' if INNER_AUC == 'sk' else ''}.json"
    json.dump(obj, open(fn, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("wrote", fn)


DS7 = ["pitt", "adresso", "adress2020", "pcgita", "neurovoz", "kcl", "edaic"]


# ================================================================ sections
def sec_mac_single():
    out = {}
    for ds in DS7:
        z = load_npz(f"{GD}/probe2/{ds}_states.npz")
        names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
        fold, fspk, chk = read_folds(f"{ds}_groupkfold5_mac.csv", names, spk)
        fpod, _, _ = read_folds(f"{ds}_groupkfold5_pod.csv", names, spk)
        r = {"file_check": chk,
             "equals_sklearn_here": bool((sk_gkf(spk, 5) == fold).all()),
             "equals_my_mac_rule": bool((my_gkf(spk, 5, "mac") == fold).all()),
             "pod_file_equals_my_stable_rule": bool((my_gkf(spk, 5, "stable") == fpod).all()),
             "clips_in_different_fold_mac_vs_pod": int((fold != fpod).sum()),
             "streams": {}}
        nj = json.load(open(f"{GD}/probe2/{ds}_nested.json"))
        for key in [k for k in ("enc", "llm", "ans") if k in z.files]:
            X = z[key].astype(np.float32)
            sv = pd.read_csv(f"{GD}/probe2/{ds}_{key}_nested_oof.csv", dtype={"clip": str, "speaker": str})
            assert (sv["clip"].values == names).all() and (sv.label.values == y).all()
            layers = nj[key]["chosen_layers"]
            oof = refit_layers_par(X, y, fold, layers)
            alt = refit_layers_par(X, y, fpod, layers)
            der = derive_layers(X, y, fold, spk, "mac")
            r["streams"][key] = {
                "saved_layers": layers, "rederived_layers": der, "layers_match": der == layers,
                "maxabs_refit_vs_saved": mad(oof, sv.p_probe), "maxabs_other_tie_file": mad(alt, sv.p_probe),
                "auc_saved_json": nj[key]["auc_nested_oof"], "auc_saved_oof": auc2(y, sv.p_probe.values),
                "auc_refit": auc2(y, oof)}
            log(ds, key, r["streams"][key]["maxabs_refit_vs_saved"], r["streams"][key]["maxabs_other_tie_file"], der == layers)
        # eGeMAPS
        e = load_npz(f"{REL}/overnight/egemaps_{ds}.npz")
        ecl = e["clips"].astype(str); eg = e["groups"].astype(str); ey = e["y"].astype(int)
        es = pd.read_csv(f"{REL}/overnight/perclip/egemaps_{ds}_oof.csv", dtype={"clip": str, "speaker": str})
        assert (es["clip"].values == ecl).all()
        try:
            efold, _, echk = read_folds(f"{ds}_groupkfold5_mac.csv", ecl, eg)
        except AssertionError as ex:
            r["egemaps"] = {"error": str(ex)}; out[ds] = r; continue
        efpod, _, _ = read_folds(f"{ds}_groupkfold5_pod.csv", ecl, eg)
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        EX = np.asarray(e["X"], dtype=np.float64).copy(); EX[~np.isfinite(EX)] = np.nan

        def eg_oof(fv):
            o = np.full(len(ey), np.nan)
            for k in range(5):
                tr = fv != k; te = fv == k
                pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                 ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(EX[tr], ey[tr])
                o[te] = pipe.predict_proba(EX[te])[:, 1]
            return o
        eo = eg_oof(efold); ea = eg_oof(efpod)
        r["egemaps"] = {"n": int(len(ey)), "file_check_speakers": echk.get("speaker_ids_match_run"),
                        "maxabs_refit_vs_saved": mad(eo, es.p_egemaps), "maxabs_other_tie_file": mad(ea, es.p_egemaps),
                        "auc_saved": auc2(ey, es.p_egemaps.values), "auc_refit": auc2(ey, eo)}
        log(ds, "egemaps", r["egemaps"]["maxabs_refit_vs_saved"], r["egemaps"]["maxabs_other_tie_file"])
        out[ds] = r
    save("mac_single", out)


def pod_single_one(tag, states, oofpat, nestedjson, fold_name, alt_name, streams, derive=True):
    z = load_npz(states)
    names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
    fold, fspk, chk = read_folds(fold_name, names, spk)
    falt, _, _ = read_folds(alt_name, names, spk)
    nj = json.load(open(nestedjson))
    r = {"states": states, "file_check": chk,
         "equals_my_stable_rule": bool((my_gkf(spk, 5, "stable") == fold).all()),
         "equals_sklearn_here": bool((sk_gkf(spk, 5) == fold).all()), "streams": {}}
    for key in streams:
        if key not in z.files:
            continue
        X = z[key].astype(np.float32)
        sv = pd.read_csv(oofpat.format(key=key), dtype={"clip": str, "speaker": str})
        assert (sv["clip"].values == names).all(), (tag, key, "clip order")
        assert (sv.label.values == y).all()
        layers = nj[key]["chosen_layers"]
        oof = refit_layers_par(X, y, fold, layers)
        alt = refit_layers_par(X, y, falt, layers)
        d = np.abs(oof - sv.p_probe.values)
        rr = {"saved_layers": layers,
              "maxabs_refit_vs_saved": float(d.max()), "median_abs": float(np.median(d)),
              "share_clips_within_0.01": float((d < 0.01).mean()),
              "maxabs_other_tie_file": mad(alt, sv.p_probe),
              "median_abs_other_tie": float(np.median(np.abs(alt - sv.p_probe.values))),
              "auc_saved_json": nj[key]["auc_nested_oof"], "auc_saved_oof": auc2(y, sv.p_probe.values),
              "auc_refit": auc2(y, oof), "auc_other_tie_refit": auc2(y, alt)}
        if derive and X.shape[1] > 1:
            der = derive_layers(X, y, fold, spk, "stable")
            rr["rederived_layers"] = der; rr["layers_match"] = der == layers
        r["streams"][key] = rr
        log(tag, key, rr["maxabs_refit_vs_saved"], rr["median_abs"], rr["maxabs_other_tie_file"], rr.get("layers_match"))
    return r


def sec_pod_single():
    out = {}
    of = f"{REL}/omni_final"
    out["omni_final_pitt"] = pod_single_one("omni_pitt", f"{REL}/overnight2/part10/o25_pitt_states.npz",
                                            of + "/omni_pitt_{key}_nested_oof.csv", of + "/omni_pitt_nested.json",
                                            "pitt_groupkfold5_pod.csv", "pitt_groupkfold5_mac.csv", ["enc", "proj", "llm", "ans"])
    out["omni_final_edaic"] = pod_single_one("omni_edaic", f"{REL}/overnight2/part10/o25_edaic_states.npz",
                                             of + "/omni_edaic_{key}_nested_oof.csv", of + "/omni_edaic_nested.json",
                                             "edaic_groupkfold5_pod.csv", "edaic_groupkfold5_mac.csv", ["enc", "proj", "llm", "ans"])
    q = f"{REL}/overnight2/q3o_new"
    for ds in ["adresso", "adress2020", "pcgita", "neurovoz", "kcl"]:
        out[f"q3o_new_{ds}"] = pod_single_one(f"q3o_{ds}", f"{REL}/overnight2/q3o/q3o_{ds}_states.npz",
                                              q + f"/q3o_{ds}_" + "{key}_nested_oof.csv", q + f"/q3o_{ds}_nested.json",
                                              f"{ds}_groupkfold5_pod.csv", f"{ds}_groupkfold5_mac.csv", ["enc", "proj", "llm", "ans"])
    out["q2a_kcl_local"] = pod_single_one("q2a_kcl_local", f"{GD}/kcl_probe_recut/kcl_states.npz",
                                          f"{GD}/probe2/kcl_local_" + "{key}_nested_oof.csv", f"{GD}/probe2/kcl_local_nested.json",
                                          "kcl_groupkfold5_pod.csv", "kcl_groupkfold5_mac.csv", ["enc", "llm"])
    # pod-written fold files
    fp = pd.read_csv(f"{FOLDS}/pitt_groupkfold5_pod.csv", dtype=str)
    pw = pd.read_csv(f"{REL}/edaic_rerun/part16/POD2/pitt468_folds.csv", dtype=str)
    pw["clip"] = pw.path.map(os.path.basename)
    m = dict(zip(fp.clip_id, fp.fold)); mm = dict(zip(fp.clip_id, fp.speaker_id))
    out["pod_written_pitt468"] = {"n": int(len(pw)), "fold_agree": int(sum(m.get(c) == f for c, f in zip(pw["clip"], pw.fold))),
                                  "speaker_agree": int(sum(mm.get(c) == s for c, s in zip(pw["clip"], pw.speaker)))}
    fm = pd.read_csv(f"{FOLDS}/pitt_groupkfold5_mac.csv", dtype=str); m2 = dict(zip(fm.clip_id, fm.fold))
    out["pod_written_pitt468"]["mac_file_agree"] = int(sum(m2.get(c) == f for c, f in zip(pw["clip"], pw.fold)))
    sj = json.load(open(f"{REL}/edaic_rerun/part16/POD1/sft1c_folds.json"))
    fe = pd.read_csv(f"{FOLDS}/edaic_groupkfold5_pod.csv", dtype=str); me = dict(zip(fe.clip_id, fe.fold))
    fem = pd.read_csv(f"{FOLDS}/edaic_groupkfold5_mac.csv", dtype=str); mem = dict(zip(fem.clip_id, fem.fold))
    agree = agree_m = tot = 0; sample = {k: (v[:3] if isinstance(v, list) else v) for k, v in list(sj.items())[:2]}
    for k, v in sj.items():
        items = v if isinstance(v, list) else v.get("test", v.get("te", []))
        for it in items:
            c = os.path.basename(str(it))
            if not c.endswith(".wav"):
                c = c + ".wav"
            tot += 1; agree += int(me.get(c) == str(k)); agree_m += int(mem.get(c) == str(k))
    out["pod_written_sft1c"] = {"structure_sample": sample, "n": tot, "fold_agree_pod_file": agree, "fold_agree_mac_file": agree_m}
    log("pod-written", out["pod_written_pitt468"], out["pod_written_sft1c"]["n"], agree, agree_m)
    save("pod_single", out)


def seeds_one(tag, states, oofpat, summary, stem, streams, tie_file, tie_other, spk_override=None):
    z = load_npz(states)
    names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
    sm = json.load(open(summary))["streams"]
    r = {"states": states, "seeds": {}}
    for s in range(5):
        fold, fspk, chk = read_folds(f"{stem}{s}.csv", names, spk)
        g = renumber(spk, s)
        other = my_gkf(g, 5, tie_other)
        rs = {"file_check": chk, f"equals_my_{tie_file}_rule_renumbered": bool((my_gkf(g, 5, tie_file) == fold).all()),
              "streams": {}}
        for key in streams:
            if key not in z.files:
                continue
            reps = sm[key]["repeats"]
            if s >= reps:
                continue
            X = z[key].astype(np.float32)
            sv = pd.read_csv(oofpat.format(key={"enc": "encoder"}.get(key, key)), dtype={"clip": str, "speaker": str})
            assert (sv["clip"].values == names).all()
            col = f"p_probe_rep{s}"
            layers = sm[key]["nested_chosen_layers"][s]
            oof = refit_layers_par(X, y, fold, layers)
            alt = refit_layers_par(X, y, other, layers)
            d = np.abs(oof - sv[col].values)
            rr = {"saved_layers": layers, "maxabs_refit_vs_saved": float(d.max()), "median_abs": float(np.median(d)),
                  "maxabs_other_tie": mad(alt, sv[col]),
                  "auc_saved_summary": sm[key]["nested_per_repeat"][s], "auc_saved_oof": auc2(y, sv[col].values),
                  "auc_refit": auc2(y, oof)}
            if X.shape[1] > 1:
                der = derive_layers(X, y, fold, g, tie_file)
                rr["rederived_layers"] = der; rr["layers_match"] = der == layers
            rs["streams"][key] = rr
            log(tag, s, key, rr["maxabs_refit_vs_saved"], rr["median_abs"], rr["maxabs_other_tie"], rr.get("layers_match"))
        r["seeds"][s] = rs
    return r


def sec_pcgita_seeds():
    P = f"{REL}/edaic_rerun/part16/POD3"
    out = seeds_one("pod3_pcgita", f"{GD}/part16_POD3_states/q3o_pcgita_states.npz",
                    P + "/q3o_pcgita_{key}_nested_oof.csv", P + "/q3o_pcgita_perlayer_summary.json",
                    "pcgita_groupkfold5_pod_seed", ["enc", "proj", "llm", "ans"], "stable", "mac")
    save("pcgita_seeds", out)


def sec_pitt15_seeds():
    P = f"{REL}/edaic_rerun/part16/POD3"
    out = seeds_one("pod3_pitt15", f"{GD}/part16_POD3_states/q3o_pitt_states.npz",
                    P + "/q3o_pitt_{key}_nested_oof.csv", P + "/q3o_pitt_perlayer_summary.json",
                    "pitt_groupkfold5_pod_Control15ids_seed", ["enc", "proj", "llm", "ans"], "stable", "mac")
    save("pitt15_seeds", out)


def sec_pitt_mac_seeds():
    z = load_npz(f"{REL}/overnight2/part10/o25_pitt_states.npz")
    names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
    n5 = load_npz(f"{REL}/overnight2/part10/pitt_enc_nested5_oof.npz")
    assert (n5["name"].astype(str) == names).all() and (n5["label"] == y).all() and (n5["spk"].astype(str) == spk).all()
    X = z["enc"].astype(np.float32); nl = X.shape[1]
    out = {"seeds": {}}
    aucs = []
    for s in range(5):
        fold, fspk, chk = read_folds(f"pitt_groupkfold5_mac_seed{s}.csv", names, spk)
        g = renumber(spk, s)
        saved = n5["oof"][s]
        rs = {"file_check": chk, "equals_sklearn_renumbered": bool((sk_gkf(g, 5) == fold).all()),
              "equals_my_mac_rule_renumbered": bool((my_gkf(g, 5, "mac") == fold).all()), "folds": {}}
        best_layers = []; pred = np.full(len(y), np.nan)
        for k in range(5):
            tr = fold != k; te = fold == k
            P = Parallel(n_jobs=NJ)(delayed(fitpred)(X[tr][:, l], y[tr], X[te][:, l]) for l in range(nl))
            diffs = [float(np.max(np.abs(p - saved[te]))) for p in P]
            b = int(np.argmin(diffs)); best_layers.append(b); pred[te] = P[b]
            srt = sorted(diffs)
            rs["folds"][k] = {"best_match_layer": b, "min_maxabs": srt[0], "second_best_maxabs": srt[1]}
        der = derive_layers(X, y, fold, g, "mac")
        # other tie order: refit at the re-derived layers for that split
        oth = my_gkf(g, 5, "stable")
        der_o = derive_layers(X, y, oth, g, "stable")
        po = refit_layers_par(X, y, oth, der_o)
        rs.update({"best_match_layers": best_layers, "rederived_layers_mac": der, "rederived_match": der == best_layers,
                   "maxabs_refit_vs_saved": mad(pred, saved), "maxabs_other_tie": mad(po, saved),
                   "auc_saved": auc2(y, saved), "auc_refit": auc2(y, pred)})
        aucs.append(auc_mw(y, saved))
        out["seeds"][s] = rs
        log("pitt_mac_seed", s, rs["maxabs_refit_vs_saved"], rs["maxabs_other_tie"], best_layers, der)
    out["mean_auc_saved_mw"] = float(np.mean(aucs))
    save("pitt_mac_seeds", out)


def sec_edaicfull():
    E = f"{REL}/edaic_rerun/part16/EDAICFULL"
    sc = pd.read_csv(f"{E}/full_zeroshot_scores.csv", dtype={"clip": str, "speaker": str})
    clips = sc["clip"].values; spk = sc.speaker.values; y = sc.label.values.astype(int)
    pc = pd.read_csv(f"{E}/edaic_full_perclip.csv", dtype={"pid": str})
    assert (pc.pid.values == np.array([c.replace(".wav", "") for c in clips])).all() and (pc.label.values == y).all()
    js = json.load(open(f"{E}/edaic_full_nested_repeats.json"))
    S = {k: [] for k in ("enc", "proj", "llm", "ans")}
    for c in clips:
        zz = np.load(f"{SCR}/t2v_edaicfull/shards/{c}.npz")
        for k in S:
            S[k].append(zz[k])
    S = {k: np.stack(v) for k, v in S.items()}
    folds = []
    out = {"files": {}}
    for r in range(5):
        fold, fspk, chk = read_folds(f"edaic_full_groupkfold5_shuffle_rs{r}.csv", clips, spk)
        chk["equals_sklearn_shuffle_rs"] = bool((sk_gkf(spk, 5, True, r) == fold).all())
        out["files"][r] = chk; folds.append(fold)
    out["streams"] = {}
    for k in S:
        X = S[k]; picks = js[k]["layers_picked"]; reps = []
        for r in range(5):
            reps.append(refit_layers_par(X, y, folds[r], picks[r * 5:(r + 1) * 5]))
        mean = np.mean(reps, axis=0)
        # contrast: wrong fold assignment (unshuffled GroupKFold) at the same picks
        wrong = np.mean([refit_layers_par(X, y, sk_gkf(spk, 5), picks[r * 5:(r + 1) * 5]) for r in range(5)], axis=0)
        d = np.abs(mean - pc[f"oof_{k}"].values)
        out["streams"][k] = {"maxabs_mean_refit_vs_saved": float(d.max()), "median_abs": float(np.median(d)),
                             "maxabs_unshuffled_contrast": mad(wrong, pc[f"oof_{k}"]),
                             "per_repeat_auc_refit": [round(auc_mw(y, o), 4) for o in reps],
                             "per_repeat_auc_saved": js[k]["per_repeat"],
                             "auc_of_mean_refit": auc2(y, mean), "auc_of_mean_saved_json": js[k]["auc_of_mean_oof"],
                             "auc_of_mean_saved_csv": auc2(y, pc[f"oof_{k}"].values)}
        log("edaicfull", k, out["streams"][k]["maxabs_mean_refit_vs_saved"], out["streams"][k]["median_abs"],
            out["streams"][k]["maxabs_unshuffled_contrast"], out["streams"][k]["per_repeat_auc_refit"], js[k]["per_repeat"])
    save("edaicfull", out)


def sec_omni_pitt_repeats():
    """HIGH finding: Omni Pitt encoder five repeats on the pod seed files vs saved 0.7761 per-repeat."""
    z = load_npz(f"{REL}/overnight2/part10/o25_pitt_states.npz")
    names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
    X = z["enc"].astype(np.float32)
    saved = json.load(open(f"{REL}/omni_final/omni_pitt_nested_repeats.json"))["enc"]
    n5 = load_npz(f"{REL}/overnight2/part10/pitt_enc_nested5_oof.npz")
    out = {"saved_pod_json": saved, "seeds": {}}
    pod_oofs = []
    for s in range(5):
        fold, _, chk = read_folds(f"pitt_groupkfold5_pod_seed{s}_UNVERIFIED.csv", names, spk)
        g = renumber(spk, s)
        der = derive_layers(X, y, fold, g, "stable")
        o = refit_layers_par(X, y, fold, der)
        pod_oofs.append(o)
        out["seeds"][s] = {"file_equals_my_stable_rule": bool((my_gkf(g, 5, "stable") == fold).all()),
                           "layers": der, "auc_refit_pod_split": auc2(y, o), "auc_saved_pod": saved["per_repeat"][s],
                           "auc_mac_split_saved_npz": auc2(y, n5["oof"][s])}
        log("omni_pitt_rep", s, out["seeds"][s]["auc_refit_pod_split"], saved["per_repeat"][s])
    out["mean_refit_pod"] = float(np.mean([auc_mw(y, o) for o in pod_oofs]))
    out["mean_mac_npz"] = float(np.mean([auc_mw(y, o) for o in n5["oof"]]))
    # paired speaker bootstrap of mean-of-5 AUC: pod-split refit minus Mac-split saved
    rng = np.random.default_rng(0); u = np.unique(spk); idx_by = {s_: np.where(spk == s_)[0] for s_ in u}
    va, vb, vd = [], [], []
    for _ in range(2000):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx_by[s_] for s_ in pick]); yy = y[ii]
        if yy.min() == yy.max():
            continue
        a = np.mean([auc_mw(yy, o[ii]) for o in pod_oofs]); b = np.mean([auc_mw(yy, o[ii]) for o in n5["oof"]])
        va.append(a); vb.append(b); vd.append(a - b)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    out["bootstrap"] = {"draws": 2000, "usable": len(vd), "pod_mean_ci": q(va), "mac_mean_ci": q(vb),
                        "diff_point": round(out["mean_refit_pod"] - out["mean_mac_npz"], 4), "diff_ci": q(vd)}
    log("omni_pitt_repeats", out["mean_refit_pod"], out["mean_mac_npz"], out["bootstrap"])
    save("omni_pitt_repeats", out)


def sec_derive():
    """Every fold file re-derived from its speaker labels with my own GroupKFold code."""
    out = {}
    src = {"pitt": f"{GD}/probe2/pitt_states.npz", "adresso": f"{GD}/probe2/adresso_states.npz",
           "adress2020": f"{GD}/probe2/adress2020_states.npz", "pcgita": f"{GD}/probe2/pcgita_states.npz",
           "neurovoz": f"{GD}/probe2/neurovoz_states.npz", "kcl": f"{GD}/probe2/kcl_states.npz",
           "edaic": f"{GD}/probe2/edaic_states.npz"}
    for ds, p in src.items():
        z = load_npz(p); names = z["name"].astype(str); spk = z["spk"].astype(str)
        for s in range(5):
            f, _, chk = read_folds(f"{ds}_groupkfold5_pod_seed{s}_UNVERIFIED.csv" if ds != "pcgita" else f"pcgita_groupkfold5_pod_seed{s}.csv", names, spk)
            g = renumber(spk, s)
            out[f"{ds}_pod_seed{s}"] = {"equals_my_stable": bool((my_gkf(g, 5, "stable") == f).all()),
                                        "equals_my_mac": bool((my_gkf(g, 5, "mac") == f).all()),
                                        "fold_sizes": chk["fold_sizes"], "speakers_across_folds": chk["speakers_across_folds"],
                                        "speaker_ids_match_run": chk["speaker_ids_match_run"]}
    # participant 172 in every Pitt file
    p172 = {}
    for fn in sorted(os.listdir(FOLDS)):
        if fn.startswith("pitt") and fn.endswith(".csv"):
            df = pd.read_csv(f"{FOLDS}/{fn}", dtype=str)
            p172[fn] = {s: sorted(set(df.fold[df.speaker_id == s])) for s in ("Control172", "Dementia172")}
    out["participant172"] = p172
    # generic checks over every csv
    allf = {}
    for fn in sorted(os.listdir(FOLDS)):
        if not fn.endswith(".csv"):
            continue
        df = pd.read_csv(f"{FOLDS}/{fn}", dtype={"clip_id": str, "speaker_id": str, "fold": int})
        bad = int(sum(df[df.speaker_id == s].fold.nunique() > 1 for s in df.speaker_id.unique()))
        allf[fn] = {"rows": int(len(df)), "unique_clips": bool(df.clip_id.is_unique), "n_spk": int(df.speaker_id.nunique()),
                    "fold_sizes": np.bincount(df.fold, minlength=5).tolist(), "speakers_across_folds": bad,
                    "folds_range_ok": bool(df.fold.between(0, 4).all())}
    out["all_files"] = allf
    save("derive", out)


def sec_unverified_spot():
    """Spot check of *_UNVERIFIED seed files: refit five-repeat nested probes (inner layer choice re-run,
    stable tie on renumbered ids) and compare per-repeat AUC with the saved per-repeat AUC."""
    jobs = [
        ("kimi_kcl_llm", f"{REL}/overnight2/kimi_new/kimi_kcl_states.npz", "llm", "kcl",
         json.load(open(f"{REL}/overnight2/kimi_new/kimi_kcl_nested_repeats.json"))["llm"]["per_repeat"]),
        ("q2a_kcl_enc", f"{GD}/probe2/kcl_states.npz", "enc", "kcl",
         json.load(open(f"{REL}/omni_final/q2a_kcl_nested_repeats.json"))["enc"]["per_repeat"]),
        ("omni_edaic_enc", f"{REL}/overnight2/part10/o25_edaic_states.npz", "enc", "edaic",
         json.load(open(f"{REL}/omni_final/omni_edaic_nested_repeats.json"))["enc"]["per_repeat"]),
        ("kimi_adresso_llm", f"{REL}/overnight2/kimi_new/kimi_adresso_states.npz", "llm", "adresso",
         json.load(open(f"{REL}/overnight2/kimi_new/kimi_adresso_nested_repeats.json"))["llm"]["per_repeat"]),
    ]
    out = {}
    for tag, st, key, ds, saved in jobs:
        z = load_npz(st); names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
        X = z[key].astype(np.float32)
        rr = {"states": st, "saved_per_repeat": saved, "refit_pod": [], "refit_mac_seed0": None}
        for s in range(5):
            fold, _, chk = read_folds(f"{ds}_groupkfold5_pod_seed{s}_UNVERIFIED.csv", names, spk)
            g = renumber(spk, s)
            der = derive_layers(X, y, fold, g, "stable")
            o = refit_layers_par(X, y, fold, der)
            rr["refit_pod"].append(round(auc_mw(y, o), 4))
            if s == 0:
                fm = my_gkf(g, 5, "mac"); dm = derive_layers(X, y, fm, g, "mac")
                rr["refit_mac_seed0"] = round(auc_mw(y, refit_layers_par(X, y, fm, dm)), 4)
        rr["maxabs_dauc_pod"] = float(np.max(np.abs(np.array(rr["refit_pod"]) - np.array(saved))))
        rr["dauc_mac_seed0"] = abs(rr["refit_mac_seed0"] - saved[0])
        out[tag] = rr
        log(tag, rr["refit_pod"], saved, rr["refit_mac_seed0"])
    save("unverified_spot", out)


def sec_curves():
    """AUC-only support: pod per-layer curves (auc_oof per fixed layer, single GroupKFold(5)) vs pod and Mac files."""
    jobs = [("omni_pitt_encoder", f"{REL}/overnight2/part10/o25_pitt_states.npz", "enc", "pitt", f"{REL}/omni_final/omni_pitt_encoder_perlayer.csv"),
            ("omni_edaic_encoder", f"{REL}/overnight2/part10/o25_edaic_states.npz", "enc", "edaic", f"{REL}/omni_final/omni_edaic_encoder_perlayer.csv"),
            ("q2a_kcl_encoder", f"{GD}/probe2/kcl_states.npz", "enc", "kcl", f"{REL}/omni_final/q2a_kcl_encoder_perlayer.csv"),
            ("q2a_adresso_encoder", f"{GD}/probe2/adresso_states.npz", "enc", "adresso", f"{REL}/omni_final/q2a_adresso_encoder_perlayer.csv")]
    out = {}
    for tag, st, key, ds, cf in jobs:
        z = load_npz(st); names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
        X = z[key].astype(np.float32); cur = pd.read_csv(cf)
        res = {}
        for which in ("pod", "mac"):
            fold, _, _ = read_folds(f"{ds}_groupkfold5_{which}.csv", names, spk)
            aucs = Parallel(n_jobs=NJ)(delayed(lambda l: auc_mw(y, refit_layers(X, y, fold, [l] * 5)))(l) for l in range(X.shape[1]))
            res[which] = float(np.max(np.abs(np.array(aucs) - cur.auc_oof.values[:len(aucs)])))
        out[tag] = {"n_layers_curve": int(len(cur)), "n_layers_states": int(X.shape[1]), "max_dauc_pod": res["pod"], "max_dauc_mac": res["mac"]}
        log(tag, out[tag])
    save("curves", out)


if __name__ == "__main__":
    sec = sys.argv[1]
    {"mac_single": sec_mac_single, "pod_single": sec_pod_single, "pcgita_seeds": sec_pcgita_seeds,
     "pitt_mac_seeds": sec_pitt_mac_seeds, "pitt15_seeds": sec_pitt15_seeds, "edaicfull": sec_edaicfull,
     "omni_pitt_repeats": sec_omni_pitt_repeats, "derive": sec_derive,
     "unverified_spot": sec_unverified_spot, "curves": sec_curves}[sec]()
    log("DONE", sec)
