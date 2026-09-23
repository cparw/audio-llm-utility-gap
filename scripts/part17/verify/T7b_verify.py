"""PART17 T7b independent verifier. Runs on the T7b pod (Linux x86_64, sklearn 1.9.1).

Own code. Does not import or exec any task script.

  python3 T7b_verify.py stack                 read shards in full_zeroshot_scores.csv order, write stacked_<key>.npy
  python3 T7b_verify.py probe llm enc [..]    nested five-repeat probe, writes V_oof_<key>.npz
  python3 T7b_verify.py check <task_npz>      AUCs (Mann-Whitney + trapezoid) and intervals from saved vectors

Probe spec being checked (part16/EDAICFULL/probe_boot.py):
  outer GroupKFold(5, shuffle=True, random_state=rep) by speaker, rep 0..4
  inner GroupKFold(4, shuffle=True, random_state=rep) on the outer-train block picks the layer
  (argmax of inner out-of-fold AUC, first max wins)
  probe = StandardScaler fit on train rows only + LogisticRegression(max_iter=2000,
  class_weight='balanced', C=1.0), fit on the float32 states as stored (no cast).
Bootstrap spec: 2000 draws, fresh numpy default_rng(0) per cell, speakers (sorted unique ids)
resampled with replacement via rng.choice, percentiles 2.5 / 97.5 (numpy linear). Each draw
recomputes all five per-repeat AUCs and averages them; paired cells use the same draw for the
answer AUC.
"""
import os
# Threading (matters for float32 lbfgs at 4 dp, see task diag_omp1): inner layer scoring runs in
# loky workers limited to 1 BLAS thread (what joblib gave the original 16 workers on a 14-CPU quota);
# the final outer-fold fits run in THIS process at VT_FINAL_THREADS BLAS threads
# ("default" = no limit, i.e. what the original main process had). Nothing is set globally.
import sys, csv, json, time, hashlib, platform
import numpy as np

W = os.environ.get("VT_DIR", "/workspace/verify_t7b")
NJOBS = int(os.environ.get("VT_NJOBS", "16"))
REP = 5
NB = 2000
STAGES_ALL = ("enc", "proj", "llm", "ans")


# ---------------------------------------------------------------- AUC, two independent ways
def auc_mw(y, s):
    """Mann-Whitney U / (n1 n0), explicit pairwise counts, ties count 0.5. Exact integer counts."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]; neg = s[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    gt = int(np.count_nonzero(pos[:, None] > neg[None, :]))
    eq = int(np.count_nonzero(pos[:, None] == neg[None, :]))
    return (2 * gt + eq) / (2.0 * n1 * n0)


def auc_trap(y, s):
    """Area under the empirical ROC by the trapezoid rule over distinct thresholds."""
    y = np.asarray(y).astype(np.int64); s = np.asarray(s, dtype=np.float64)
    P = int(y.sum()); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    ss, yy = s[o], y[o]
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]   # last index of each tied block
    tp = np.cumsum(yy)[last]
    fp = (last + 1) - tp
    tpr = np.r_[0, tp] / P
    fpr = np.r_[0, fp] / N
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


# ---------------------------------------------------------------- data
def load_meta():
    rows = list(csv.DictReader(open(f"{W}/in/full_zeroshot_scores.csv")))
    clips = [r["clip"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    spk = np.array([r["speaker"] for r in rows])
    pz = np.array([float(r["p_yes"]) for r in rows])
    return clips, y, spk, pz


def env_info():
    import sklearn, scipy
    info = dict(python=sys.version.split()[0], machine=platform.machine(), system=platform.system(),
                sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__)
    try:
        from threadpoolctl import threadpool_info
        info["blas"] = [(d.get("internal_api"), d.get("version"), d.get("num_threads")) for d in threadpool_info()]
    except Exception as e:
        info["blas"] = repr(e)
    return info


def cmd_stack():
    clips, y, spk, pz = load_meta()
    for key in STAGES_ALL:
        arr = np.stack([np.load(f"{W}/shards/{c}.npz")[key] for c in clips])
        np.save(f"{W}/stacked_{key}.npy", arr)
        print(f"stacked {key} {arr.shape} {arr.dtype} C={arr.flags['C_CONTIGUOUS']}", flush=True)


# ---------------------------------------------------------------- probe
_CACHE = {}


def _X(key):
    if key not in _CACHE:
        _CACHE[key] = np.load(f"{W}/stacked_{key}.npy", mmap_mode="r")
    return _CACHE[key]


def _fit_predict(Xa, ya, Xb):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(Xa)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(sc.transform(Xa), ya)
    return clf.predict_proba(sc.transform(Xb))[:, 1]


def _inner_task(key, rep, fold, L, tr, isp, y):
    """inner out-of-fold AUC of layer L on outer-train rows tr"""
    from threadpoolctl import threadpool_limits
    X = _X(key)
    ytr = y[tr]
    oof = np.zeros(len(tr))
    with threadpool_limits(1):
        for a, b in isp:
            Xa = np.ascontiguousarray(X[tr[a], L, :])
            Xb = np.ascontiguousarray(X[tr[b], L, :])
            oof[b] = _fit_predict(Xa, ytr[a], Xb)
    return key, rep, fold, L, auc_mw(ytr, oof)


def _final_task(key, rep, fold, L, tr, te, y):
    """runs in the main process"""
    from threadpoolctl import threadpool_limits
    X = _X(key)
    ft = os.environ.get("VT_FINAL_THREADS", "default")
    Xa = np.ascontiguousarray(X[tr, L, :]); Xb = np.ascontiguousarray(X[te, L, :])
    if ft == "default":
        p = _fit_predict(Xa, y[tr], Xb)
    else:
        with threadpool_limits(int(ft)):
            p = _fit_predict(Xa, y[tr], Xb)
    return key, rep, fold, L, te, p


def cmd_probe(stages):
    from sklearn.model_selection import GroupKFold
    from joblib import Parallel, delayed
    clips, y, spk, pz = load_meta()
    t0 = time.time()
    splits = {}
    inner_jobs = []
    for key in stages:
        nL = _X(key).shape[1]
        for rep in range(REP):
            outer = GroupKFold(n_splits=5, shuffle=True, random_state=rep)
            for f, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y, groups=spk)):
                inner = GroupKFold(n_splits=4, shuffle=True, random_state=rep)
                isp = list(inner.split(np.zeros(len(tr)), y[tr], groups=spk[tr]))
                splits[(key, rep, f)] = (tr, te, isp)
                if nL > 1:
                    for L in range(nL):
                        inner_jobs.append((key, rep, f, L, tr, isp))
    TAG = os.environ.get("VT_TAG", "main")
    print(f"probe {stages} tag {TAG}: {len(inner_jobs)} inner layer jobs, n_jobs={NJOBS}, inner threads 1, "
          f"final threads {os.environ.get('VT_FINAL_THREADS', 'default')}", flush=True)
    res = Parallel(n_jobs=NJOBS, backend="loky", verbose=5)(
        delayed(_inner_task)(k, r, f, L, tr, isp, y) for (k, r, f, L, tr, isp) in inner_jobs)
    inner_auc = {}
    for k, r, f, L, a in res:
        inner_auc.setdefault((k, r, f), {})[L] = a
    picks = {}
    for (k, r, f), (tr, te, isp) in splits.items():
        nL = _X(k).shape[1]
        if nL == 1:
            picks[(k, r, f)] = 0
        else:
            v = [inner_auc[(k, r, f)][L] for L in range(nL)]
            picks[(k, r, f)] = int(np.argmax(v))
    print(f"inner done {time.time()-t0:.0f}s", flush=True)
    from threadpoolctl import threadpool_info
    final_blas = [(d.get("internal_api"), d.get("num_threads")) for d in threadpool_info()]
    fres = [_final_task(k, r, f, picks[(k, r, f)], splits[(k, r, f)][0], splits[(k, r, f)][1], y)
            for (k, r, f) in splits]
    oof = {k: np.full((REP, len(y)), np.nan) for k in stages}
    fold_of = {k: np.full((REP, len(y)), -1) for k in stages}
    for k, r, f, L, te, p in fres:
        oof[k][r, te] = p
        fold_of[k][r, te] = f
    info = env_info()
    for k in stages:
        assert not np.isnan(oof[k]).any()
        lay = np.array([picks[(k, r, f)] for r in range(REP) for f in range(5)])
        inner_tab = np.array([[[inner_auc.get((k, r, f), {}).get(L, np.nan) for L in range(_X(k).shape[1])]
                               for f in range(5)] for r in range(REP)])
        np.savez_compressed(f"{W}/V_oof_{k}{'' if TAG == 'main' else '_' + TAG}.npz", oof_repeats=oof[k], layers_picked=lay, outer_fold=fold_of[k],
                            inner_auc=inner_tab, label=y, speaker=spk, clip=np.array(clips))
        pr = [auc_mw(y, oof[k][r]) for r in range(REP)]
        print(f"{k}: per_repeat {[round(a, 6) for a in pr]}  layers {lay.tolist()}", flush=True)
    json.dump(dict(env=info, seconds=time.time() - t0, stages=list(stages), tag=TAG, inner_threads=1,
                   final_threads=os.environ.get("VT_FINAL_THREADS", "default"), final_blas_main_process=final_blas,
                   njobs=NJOBS),
              open(f"{W}/V_probe_env_{TAG}_{'_'.join(stages)}.json", "w"), indent=1)
    print(f"probe done {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- bootstrap
def boot_cell(y, spk, vecs, ref=None, auc=auc_mw):
    """vecs: list of score vectors, statistic = mean of their AUCs. ref: paired answer vector."""
    uspk = np.array(sorted(set(spk.tolist())))
    rows_of = {s: np.nonzero(spk == s)[0] for s in uspk}
    rng = np.random.default_rng(0)
    va, vd, skipped = [], [], 0
    for _ in range(NB):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.concatenate([rows_of[uspk[p]] for p in pick])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            skipped += 1
            continue
        a = float(np.mean([auc(yy, v[ii]) for v in vecs]))
        va.append(a)
        if ref is not None:
            vd.append(a - auc(yy, ref[ii]))
    out = dict(usable=len(va), skipped=skipped,
               lo=float(np.percentile(va, 2.5)), hi=float(np.percentile(va, 97.5)))
    if ref is not None:
        out.update(dlo=float(np.percentile(vd, 2.5)), dhi=float(np.percentile(vd, 97.5)))
    return out


def cmd_check(task_npz, task_json=None, task_perclip=None, task_tsv=None):
    clips, y, spk, pz = load_meta()
    pub = json.load(open(f"{W}/in/edaic_full_nested_repeats.json"))
    perclip = list(csv.DictReader(open(f"{W}/in/edaic_full_perclip.csv")))
    assert [r["pid"] for r in perclip] == [c.replace(".wav", "") for c in clips]
    y_pc = np.array([int(r["label"]) for r in perclip]); assert np.array_equal(y_pc, y)
    pz_pc = np.array([float(r["p_yes_answer"]) for r in perclip])
    out = dict(env=env_info(), n=len(y), n_spk=len(set(spk.tolist())), n_pos=int(y.sum()),
               p_yes_maxabs_scores_vs_perclip=float(np.max(np.abs(pz - pz_pc))))
    T = np.load(task_npz, allow_pickle=False)
    out["task_npz"] = task_npz
    out["task_npz_keys"] = sorted(T.files)
    out["task_npz_sha256"] = hashlib.sha256(open(task_npz, "rb").read()).hexdigest()
    # task npz alignment
    if "label" in T.files:
        out["task_label_equal"] = bool(np.array_equal(np.asarray(T["label"]).astype(int), y))
    if "clip" in T.files:
        out["task_clip_equal"] = [str(c) for c in T["clip"]] == clips
    TJ = json.load(open(task_json)) if task_json else None
    TP = list(csv.DictReader(open(task_perclip))) if task_perclip else None
    if TP is not None:
        out["task_perclip_pid_order_equal"] = [r["pid"] for r in TP] == [c.replace(".wav", "") for c in clips]
        out["task_perclip_label_equal"] = [int(r["label"]) for r in TP] == y.tolist()
        out["task_perclip_pyes_maxabs"] = float(np.max(np.abs(np.array([float(r["p_yes_answer"]) for r in TP]) - pz)))
    stages = [k for k in STAGES_ALL if os.path.exists(f"{W}/V_oof_{k}.npz")]
    out["stages_rerun_by_verifier"] = stages
    vec = {}
    for k in STAGES_ALL:
        d = {}
        pubk = pub[k]
        tk = f"{k}_oof_repeats"
        tv = np.asarray(T[tk], dtype=np.float64) if tk in T.files else None
        mine = np.load(f"{W}/V_oof_{k}.npz") if k in stages else None
        if tv is not None:
            d["task_per_repeat_mw"] = [auc_mw(y, tv[r]) for r in range(REP)]
            d["task_per_repeat_trap"] = [auc_trap(y, tv[r]) for r in range(REP)]
            d["task_per_repeat_4dp_match_published"] = [round(a, 4) for a in d["task_per_repeat_mw"]] == pubk["per_repeat"]
            for lk in (f"{k}_layers_picked", f"{k}_layers"):
                if lk in T.files:
                    d["task_layers"] = [int(v) for v in T[lk]]
            if "task_layers" not in d and TJ is not None:
                d["task_layers"] = [int(v) for v in TJ[k]["layers_picked"]]
                d["task_layers_source"] = task_json
                d["task_json_per_repeat"] = TJ[k]["per_repeat"]
                d["task_json_per_repeat_equals_npz_4dp"] = TJ[k]["per_repeat"] == [round(a, 4) for a in d["task_per_repeat_mw"]]
            if "task_layers" in d:
                d["task_layers_match_published"] = d["task_layers"] == pubk["layers_picked"]
            if TP is not None:
                pc = np.array([[float(r[f"oof_{k}_r{rr}"]) for r in TP] for rr in range(REP)])
                d["task_perclip_csv_vs_npz_maxabs"] = float(np.max(np.abs(pc - tv)))
            d["task_auc_of_mean_oof"] = auc_mw(y, tv.mean(axis=0))
            pubcol = np.array([float(r[f"oof_{k}"]) for r in perclip])
            d["task_meanoof_maxabs_vs_published_perclip"] = float(np.max(np.abs(tv.mean(axis=0) - pubcol)))
        if mine is not None:
            mv = mine["oof_repeats"]
            d["mine_per_repeat_mw"] = [auc_mw(y, mv[r]) for r in range(REP)]
            d["mine_per_repeat_trap"] = [auc_trap(y, mv[r]) for r in range(REP)]
            d["mine_per_repeat_4dp"] = [round(a, 4) for a in d["mine_per_repeat_mw"]]
            d["mine_per_repeat_4dp_match_published"] = d["mine_per_repeat_4dp"] == pubk["per_repeat"]
            d["mine_layers"] = [int(v) for v in mine["layers_picked"]]
            d["mine_layers_match_published"] = d["mine_layers"] == pubk["layers_picked"]
            d["mine_auc_of_mean_oof"] = auc_mw(y, mv.mean(axis=0))
            pubcol = np.array([float(r[f"oof_{k}"]) for r in perclip])
            d["mine_meanoof_maxabs_vs_published_perclip"] = float(np.max(np.abs(mv.mean(axis=0) - pubcol)))
            if tv is not None:
                d["mine_vs_task_maxabs_perclip"] = float(np.max(np.abs(mv - tv)))
                d["mine_vs_task_bitwise_equal"] = bool(np.array_equal(mv, tv))
                if "task_layers" in d:
                    d["mine_vs_task_layers_equal"] = d["mine_layers"] == d["task_layers"]
        for tag in ("final1",):
            fn = f"{W}/V_oof_{k}_{tag}.npz"
            if os.path.exists(fn):
                zz = np.load(fn)
                d[f"variant_{tag}_per_repeat_4dp"] = [round(auc_mw(y, zz["oof_repeats"][r]), 4) for r in range(REP)]
                d[f"variant_{tag}_layers_match_published"] = [int(v) for v in zz["layers_picked"]] == pubk["layers_picked"]
                d[f"variant_{tag}_mean_of_repeat_aucs"] = float(np.mean([auc_mw(y, zz["oof_repeats"][r]) for r in range(REP)]))
        d["published_per_repeat"] = pubk["per_repeat"]
        d["published_layers"] = pubk["layers_picked"]
        d["published_mean_of_repeat_aucs"] = pubk["mean_of_repeat_aucs"]
        d["published_auc_of_mean_oof"] = pubk["auc_of_mean_oof"]
        out[f"stage_{k}"] = d
        vec[k] = dict(task=tv, mine=(mine["oof_repeats"] if mine is not None else None))
        print(f"[{k}] pub {pubk['per_repeat']}", flush=True)
        if tv is not None:
            print(f"     task mw {[round(a,6) for a in d['task_per_repeat_mw']]} layers-match {d.get('task_layers_match_published')}", flush=True)
        if mine is not None:
            print(f"     mine mw {[round(a,6) for a in d['mine_per_repeat_mw']]} layers-match {d['mine_layers_match_published']} "
                  f"mine-vs-task maxabs {d.get('mine_vs_task_maxabs_perclip')}", flush=True)

    # ---------------- cells
    cells = {}

    def cell(name, vecs, ref=None):
        r = {}
        r["point_mw"] = float(np.mean([auc_mw(y, v) for v in vecs]))
        r["point_trap"] = float(np.mean([auc_trap(y, v) for v in vecs]))
        b_mw = boot_cell(y, spk, vecs, ref, auc_mw)
        b_tr = boot_cell(y, spk, vecs, ref, auc_trap)
        r.update({f"{kk}_mw": vv for kk, vv in b_mw.items()})
        r.update({f"{kk}_trap": vv for kk, vv in b_tr.items()})
        if ref is not None:
            r["diff_point_mw"] = r["point_mw"] - auc_mw(y, ref)
            r["diff_point_trap"] = r["point_trap"] - auc_trap(y, ref)
        cells[name] = r
        s = f"{name:44s} {r['point_mw']:.4f} [{r['lo_mw']:.4f}, {r['hi_mw']:.4f}]"
        if ref is not None:
            s += f"   diff {r['diff_point_mw']:.4f} [{r['dlo_mw']:.4f}, {r['dhi_mw']:.4f}]"
        print(s, flush=True)

    t0 = time.time()
    cell("answer", [pz])
    # machinery check against probe_boot.py's published intervals (averaged-OOF convention, published per-clip file)
    for k in ("llm", "enc"):
        pubcol = np.array([float(r[f"oof_{k}"]) for r in perclip])
        cell(f"published_perclip_{k}_avgprob_vs_answer", [pubcol], pz)
    for src in ("task", "mine"):
        for k in STAGES_ALL:
            v = vec[k][src]
            if v is None:
                continue
            cell(f"{src}_{k}_meanof5_vs_answer", [v[r] for r in range(REP)], pz)
            cell(f"{src}_{k}_avgprob_vs_answer_NOTUSED", [v.mean(axis=0)], pz)
    out["cells"] = cells
    out["boot_seconds"] = time.time() - t0

    # ---------------- compare the task's rows at 4 dp
    if task_tsv:
        cmp = []
        want = {"T7b_zeroshot_answer": ("answer", "")}
        for k in STAGES_ALL:
            want[f"T7b_{k}_meanof5"] = (f"{{src}}_{k}_meanof5_vs_answer", "")
            want[f"T7b_{k}_meanof5_minus_answer"] = (f"{{src}}_{k}_meanof5_vs_answer", "diff")
            want[f"T7b_notused_{k}_aucofmeanoof"] = (f"{{src}}_{k}_avgprob_vs_answer_NOTUSED", "")
            want[f"T7b_notused_{k}_aucofmeanoof_minus_answer"] = (f"{{src}}_{k}_avgprob_vs_answer_NOTUSED", "diff")
        for r in csv.DictReader(open(task_tsv), delimiter="\t"):
            if r["id"] not in want:
                cmp.append(dict(id=r["id"], status="NOT CHECKED (no matching verifier cell)"))
                continue
            cname, kind = want[r["id"]]
            tv = tuple(round(float(r[c]), 4) for c in ("value", "lo", "hi"))
            recs = {}
            for src in (["task", "mine"] if "{src}" in cname else [""]):
                c = cells.get(cname.format(src=src))
                if c is None:
                    continue
                for m in ("mw", "trap"):
                    if kind == "diff":
                        v = (c[f"diff_point_{m}"], c[f"dlo_{m}"], c[f"dhi_{m}"])
                    else:
                        v = (c[f"point_{m}"], c[f"lo_{m}"], c[f"hi_{m}"])
                    recs[f"{src or 'answer'}_{m}"] = [round(x, 4) for x in v]
            agree = all(tuple(v) == tv for v in recs.values()) and len(recs) > 0
            cmp.append(dict(id=r["id"], task=list(tv), verifier=recs, n=r["n"], n_spk=r["n_spk"],
                            n_ok=(r["n"] == str(len(y)) and r["n_spk"] == str(len(set(spk.tolist())))),
                            agree_4dp=bool(agree)))
            print(f"{r['id']:48s} task {tv}  {'AGREE' if agree else 'DISAGREE'}  {recs}", flush=True)
        out["tsv_comparison"] = cmp
    json.dump(out, open(f"{W}/T7b_verify.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("CHECK DONE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stack":
        cmd_stack()
    elif cmd == "probe":
        cmd_probe(sys.argv[2:])
    elif cmd == "check":
        cmd_check(*sys.argv[2:6])
    else:
        raise SystemExit("unknown command")
