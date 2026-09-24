import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import csv, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
D="scores/part26/Qwen3-Omni-30B-A3B_Pitt"
z=np.load(D+"/stage/q3o_pitt_encstates.npz",allow_pickle=True)
X=z["enc"].astype(np.float32); y=z["label"].astype(int); names=z["name"].astype(str); spk=z["spk"].astype(str)
J=json.load(open(D+"/pull/p26-q3o-pitt/out/p26_q3o_pitt_enc_nested5.json"))
oof={r["clip"]:r for r in csv.DictReader(open(D+"/pull/p26-q3o-pitt/out/p26_q3o_pitt_enc_nested5_oof.csv"))}
assert set(oof)==set(names)
res={}
for s in range(5):
    t=J["tags"][f"seed{s}"]; layers=t["layers"]; inner=np.array(t["inner"])
    argmax=[int(np.argmax(r)) for r in inner]
    fold=np.array([int(oof[n][f"fold_seed{s}"]) for n in names])
    ff={r["clip_id"]:int(r["fold"]) for r in csv.DictReader(open(f"<local data dir>/release/folds/pitt_groupkfold5_pod_Control15ids_seed{s}.csv"))}
    assert all(ff[n]==f for n,f in zip(names,fold))
    podp=np.array([float(oof[n][f"p_seed{s}"]) for n in names]); myp=np.full(len(y),np.nan)
    for k in range(5):
        tr,te=np.where(fold!=k)[0],np.where(fold==k)[0]; L=layers[k]
        sc=StandardScaler().fit(X[tr][:,L])
        myp[te]=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr][:,L]),y[tr]).predict_proba(sc.transform(X[te][:,L]))[:,1]
    # wrong-fold control: use seed (s+1)%5 folds at the same layers
    res[s]={"layers":layers,"layers_eq_argmax_inner":layers==argmax,
            "runnerup_gap_min":float(min(np.sort(r)[-1]-np.sort(r)[-2] for r in inner)),
            "max_abs_diff_refit_vs_pod":float(np.nanmax(np.abs(myp-podp))),
            "median_abs_diff":float(np.median(np.abs(myp-podp)))}
print(json.dumps(res,indent=1))
print("labels/spk in states equal oof:", all(int(oof[n]["label"])==l and oof[n]["speaker"]==s_ for n,l,s_ in zip(names,y,spk)))
