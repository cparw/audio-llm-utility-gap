# Own outer refit at the pod's saved layers: checks the saved OOF scores are out of fold on the saved folds.
import json, csv, numpy as np, time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
CELL="scores/part26/Qwen2-Audio_E-DAIC"
S=np.load("<local data dir>/paper1_local_runs/probe2/edaic_states.npz",allow_pickle=False)
names=[str(x) for x in S["name"]]; enc=S["enc"]; lab=S["label"].astype(int); sspk=[str(x) for x in S["spk"]]
P=list(csv.DictReader(open(f"{CELL}/Qwen2-Audio_E-DAIC_perclip.csv")))
clips=[r["clip"] for r in P]
# map clip -> state row (names may lack .wav)
key={n if n.endswith('.wav') else n+'.wav':i for i,n in enumerate(names)}
ix=np.array([key[c] for c in clips]); X=enc[ix]; y=np.array([int(r["label"]) for r in P])
out={"states_names_cover":len(set(key))==275,"labels_eq_states":bool((lab[ix]==y).all()),
     "spk_eq_states":bool((np.array(sspk)[ix]==np.array([r['speaker'] for r in P])).all())}
J=json.load(open(f"{CELL}/pull/p26-q2a-edaic/out/p26_q2a_edaic_enc_nested5.json"))
t=time.time(); res={}
for k in range(5):
    f=np.array([int(r[f"fold_seed{k}"]) for r in P]); p=np.array([float(r[f"p_probe_seed{k}"]) for r in P])
    L=J["tags"][f"seed{k}"]["layers"]; q=np.full(275,np.nan); ins=[]
    for o in range(5):
        te=f==o; tr=~te
        sc=StandardScaler().fit(X[tr][:,L[o]])
        m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr][:,L[o]]),y[tr])
        q[te]=m.predict_proba(sc.transform(X[te][:,L[o]]))[:,1]
        ins.append(roc_auc_score(y[tr],m.predict_proba(sc.transform(X[tr][:,L[o]]))[:,1]))
    res[f"seed{k}"]=dict(layers=L,refit_auc=float(roc_auc_score(y,q)),saved_auc=float(roc_auc_score(y,p)),
        max_abs_pred_diff=float(np.abs(q-p).max()),median_abs_pred_diff=float(np.median(np.abs(q-p))),
        spearman=float(np.corrcoef(np.argsort(np.argsort(q)),np.argsort(np.argsort(p)))[0,1]),
        train_insample_auc_min=float(min(ins)))
out["refit"]=res; out["seconds"]=round(time.time()-t,1)
print(json.dumps(out,indent=1))
