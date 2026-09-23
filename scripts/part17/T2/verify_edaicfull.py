"""PART17 T2: E-DAIC full-window nested probe (part16/EDAICFULL/probe_boot.py, POD2) uses
GroupKFold(5, shuffle=True, random_state=rep) outer and GroupKFold(4, shuffle=True, random_state=rep) inner.
The shuffle branch permutes the unique groups with RandomState(rep) and array_splits them: no argsort tie
order is involved. Refit at the saved layers, average the 5 repeats, compare with the saved per-clip oof_mean;
then re-run the inner layer choice and compare with the saved layers_picked."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v,"1")
import json, csv, time, hashlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from eng import rank_auc
E = "<local data dir>/release/edaic_rerun/part16/EDAICFULL"
SH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edaicfull/shards")
TGZ = "<local data dir>/paper work/paper1_local_runs/part16_EDAICFULL_states/edaicfull_shards.tgz"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edaicfull_results.json")
def sha1mb(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read(1 << 20)); return h.hexdigest()
sc = list(csv.DictReader(open(f"{E}/full_zeroshot_scores.csv")))
clips = [r["clip"] for r in sc]; y = np.array([int(r["label"]) for r in sc]); spk = np.array([r["speaker"] for r in sc])
S = {k: [] for k in ("enc", "proj", "llm", "ans")}
for c in clips:
    z = np.load(f"{SH}/{c}.npz")
    for k in S: S[k].append(z[k])
for k in S: S[k] = np.stack(S[k])
J = json.load(open(f"{E}/edaic_full_nested_repeats.json")); P = pd.read_csv(f"{E}/edaic_full_perclip.csv", dtype={"pid": str})
assert (P["pid"].values == np.array([c.replace(".wav", "") for c in clips])).all()
def probe(Xl, yl, tr, te):
    ss = StandardScaler().fit(Xl[tr]); lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(ss.transform(Xl[tr]), yl[tr])
    return lr.predict_proba(ss.transform(Xl[te]))[:, 1]
res = dict(states_tgz=TGZ, sha_tgz=sha1mb(TGZ), perclip=f"{E}/edaic_full_perclip.csv", sha_perclip=sha1mb(f"{E}/edaic_full_perclip.csv"),
           json=f"{E}/edaic_full_nested_repeats.json", sha_json=sha1mb(f"{E}/edaic_full_nested_repeats.json"), script=f"{E}/probe_boot.py",
           n=int(len(y)), n_spk=int(len(set(spk))), streams={}, folds={})
for rep in range(5):
    f = np.full(len(y), -1)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, groups=spk)): f[te] = k
    res["folds"][rep] = f.tolist()
res["clip"] = clips; res["spk"] = spk.tolist()
for key in ("enc", "proj", "llm", "ans"):
    t0 = time.time(); X = S[key]; nL = X.shape[1]; picks = J[key]["layers_picked"]; oofs = []; rederived = []
    for rep in range(5):
        oof = np.zeros(len(y)); splits = list(GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(X, y, groups=spk))
        for i, (tr, te) in enumerate(splits):
            oof[te] = probe(X[:, picks[rep*5+i], :], y, tr, te)
            if nL == 1: rederived.append(0); continue
            isp = list(GroupKFold(n_splits=4, shuffle=True, random_state=rep).split(X[tr], y[tr], groups=spk[tr])); ytr = y[tr]; Xtr = X[tr]
            def score_layer(L):
                o = np.zeros(len(tr))
                for itr, ite in isp: o[ite] = probe(Xtr[:, L, :], ytr, itr, ite)
                return rank_auc(ytr, o)
            sl = Parallel(n_jobs=14)(delayed(score_layer)(L) for L in range(nL)); rederived.append(int(np.argmax(sl)))
        oofs.append(oof)
    om = np.mean(oofs, axis=0); col = f"oof_{key}"; d = np.abs(om - P[col].values)
    per = [round(rank_auc(y, o), 4) for o in oofs]
    res["streams"][key] = dict(maxabs_oof_mean=float(d.max()), medabs=float(np.median(d)), per_repeat_mine=per, per_repeat_saved=J[key]["per_repeat"],
                               per_repeat_match=[a == b for a, b in zip(per, J[key]["per_repeat"])], layers_saved=picks, layers_rederived=rederived,
                               layers_match=(picks == rederived), auc_mean_oof_mine=round(rank_auc(y, om), 4), auc_mean_oof_saved=J[key]["auc_of_mean_oof"])
    print(key, {k: v for k, v in res["streams"][key].items() if k not in ("layers_saved", "layers_rederived")}, f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(OUT, "w"))
print("EDAICFULL DONE")
