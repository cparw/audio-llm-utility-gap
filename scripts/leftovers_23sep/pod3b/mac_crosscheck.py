"""Mac diagnostic (not a pod value): Mac sklearn 1.7.2 / numpy 2.2.6 refit of the 3b probe with
(a) the Mac's own GroupKFold folds (this is the audit's partF 0.7864) and (b) the Linux pod folds read from
pull/lo-pod3b-3/out/pod3b_gap_perclip.csv. If (b) lands on 0.8307, the fit is not the cause; the folds are."""
import numpy as np, csv, json, sys, sklearn, platform
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
P="scores/leftovers_23sep/pod3b/lo-pod3b-3/pod3b_gap_perclip.csv"
z=np.load("<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_pitt_states.npz",allow_pickle=True)
X_all=z["ans"][:,-1,:].astype(np.float64); y_all=z["label"].astype(int); spk_all=z["spk"].astype(str); nm=z["name"].astype(str)
g=np.load("<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_perp_cols.npz",allow_pickle=True)
arms=np.array([n.split("_")[0] for n in nm])
R=list(csv.DictReader(open(P)))
def auc(y,s):
    y=np.asarray(y); s=np.asarray(s); pos=s[y==1]; neg=s[y==0]
    return float(((pos[:,None]>neg[None,:]).sum()+0.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))
out={"env":{"sklearn":sklearn.__version__,"numpy":np.__version__,"machine":platform.machine()}}
for arm in ("all","conflict","agreement"):
    m=np.ones(len(y_all),bool) if arm=="all" else arms==arm
    X=X_all[m]; y=y_all[m]; spk=spk_all[m]; names=nm[m]
    fl={r["name"]:int(r["fold_linux"]) for r in R if r["arm"]==arm}; FL=np.array([fl[n] for n in names])
    FM=np.full(len(y),-1)
    for k,(tr,te) in enumerate(GroupKFold(5).split(X,y,groups=spk)): FM[te]=k
    res={}
    for tag,F in (("mac_folds",FM),("linux_folds",FL)):
        oof=np.zeros(len(y))
        for k in range(5):
            tr=F!=k; te=F==k
            sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr]),y[tr])
            oof[te]=lr.predict_proba(sc.transform(X[te]))[:,1]
        res[tag]=auc(y,oof)
        if arm=="all" and tag=="linux_folds": res["linux_folds_oof_maxabs_vs_pod"]=float(np.abs(oof-g["oof"]).max())
    out[arm]=res; print(arm,res,flush=True)
json.dump(out,open("mac_crosscheck.json","w"),indent=1)
