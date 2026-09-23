# per-layer curve recompute with the pod's library versions (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1)
import numpy as np, pandas as pd, sys, warnings
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
B="scores/part23"
order=sys.argv[1]
if order=="pid": D=pd.read_csv(f"{B}/A/A2_edaic_whole_zeroshot_perclip.csv").sort_values("pid")
elif order=="win": D=pd.read_csv("manifests/edaic_windows_full.csv")
else: D=pd.read_csv(f"{B}/A/A_extract_perclip.csv")
pids=D.pid.astype(int).values
L=pd.read_csv(f"{B}/A/A2_edaic_whole_zeroshot_perclip.csv").set_index("pid").label
y=L.loc[pids].values; g=np.array([str(p) for p in pids])
Z=[np.load(f"<local data dir>/shards/{p}.npz") for p in pids]
out=[]
for stream,fn,n in [("encoder","enc",32),("llm","llm",29)]:
    ref=pd.read_csv(f"{B}/o25_whole_{stream}_perlayer.csv")
    X=np.stack([z[fn] for z in Z])
    stages=range(n) if len(sys.argv)<3 else [0]
    for s in stages:
        oof=np.zeros(len(y))
        for tr,te in GroupKFold(n_splits=5).split(X[:,s],y,g):
            sc=StandardScaler().fit(X[tr,s,:]); m=LogisticRegression(max_iter=2000,class_weight="balanced",C=1.0).fit(sc.transform(X[tr,s,:]),y[tr])
            oof[te]=m.predict_proba(sc.transform(X[te,s,:]))[:,1]
        a=roc_auc_score(y,oof); r=ref.auc_oof[s]; out.append((stream,s,a,r,abs(a-r)))
o=pd.DataFrame(out,columns=["stream","stage","recomputed","pod","absdiff"])
print(order, "n",len(o),"max absdiff %.6f"%o.absdiff.max(), "n match 4dp", int((o.recomputed.round(4)==o.pod.round(4)).sum()))
if len(sys.argv)<3: o.to_csv(f"scores/part23/verify/curves_191_{order}.csv",index=False)
