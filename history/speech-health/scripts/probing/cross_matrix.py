import os, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R="/project2/msoleyma_946/speech_health/results_chaitanya"
CAND={"kcl":f"{R}/features/qwen2/kcl/encoder_features.npz",
      "italian":f"{R}/features/qwen2/italian/encoder_features.npz",
      "neurovoz":f"{R}/features/qwen2/neurovoz/encoder_features.npz",
      "pcgita":f"{R}/features/qwen2/pcgita/encoder_features.npz"}
def load(p):
    d=np.load(p,allow_pickle=True); idx=np.where(d["task"]=="read")[0]
    return d, idx, d["label"][idx].astype(int), d["speaker"][idx], int(d["n_layers"])
data={n:load(p) for n,p in CAND.items() if os.path.exists(p)}
names=[n for n in ["kcl","italian","neurovoz","pcgita"] if n in data]
print("datasets with read features:", names, flush=True)

def cv_perlayer(d,idx,y,spk,L):
    out=[]
    ns=min(5,len(set(spk)))
    for li in range(L):
        X=d[f"layer_{li:02d}"][idx]; fold=[]
        for tr,te in GroupKFold(ns).split(X,y,spk):
            if len(set(y[tr]))<2 or len(set(y[te]))<2: continue
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
            clf.fit(X[tr],y[tr])
            try: fold.append(roc_auc_score(y[te],clf.predict_proba(X[te])[:,1]))
            except Exception: pass
        out.append(np.mean(fold) if fold else np.nan)
    return out

best_layer={}; within={}
for n in names:
    d,idx,y,spk,L=data[n]
    a=cv_perlayer(d,idx,y,spk,L); bl=int(np.nanargmax(a))
    best_layer[n]=bl; within[n]=a[bl]
    print(f"{n}: {len(idx)} read clips, {len(set(spk))} spk, best read layer {bl}, within-CV AUC {a[bl]:.3f}", flush=True)

M=np.full((len(names),len(names)),np.nan)
for i,tr in enumerate(names):
    d,idx,y,spk,L=data[tr]; bl=best_layer[tr]
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
    clf.fit(d[f"layer_{bl:02d}"][idx],y)
    for j,te in enumerate(names):
        if te==tr: M[i,j]=within[tr]; continue
        d2,idx2,y2,spk2,L2=data[te]
        try: M[i,j]=roc_auc_score(y2, clf.predict_proba(d2[f"layer_{bl:02d}"][idx2])[:,1])
        except Exception: pass

with open(f"{R}/crossdataset/cross_matrix_read.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["train\\test"]+names)
    for i,tr in enumerate(names): w.writerow([tr]+[("%.3f"%M[i,j] if not np.isnan(M[i,j]) else "") for j in range(len(names))])
fig,ax=plt.subplots(figsize=(6.5,5.5)); im=ax.imshow(M,vmin=0.5,vmax=1.0,cmap="RdYlGn")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=30,ha="right")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlabel("test on"); ax.set_ylabel("train on")
for i in range(len(names)):
    for j in range(len(names)):
        if not np.isnan(M[i,j]):
            ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=10,color="black")
ax.set_title("Parkinson's (read): cross-dataset AUC\nQwen2-Audio encoder, train-best layer, speaker-disjoint")
fig.colorbar(im,fraction=0.046); fig.tight_layout(); fig.savefig(f"{R}/plots/cross_matrix_read.png",dpi=150)
print("SAVED cross_matrix_read.csv + png", flush=True)
print("diagonal=within-dataset (inflated where confounded), off-diagonal=honest generalization", flush=True)
