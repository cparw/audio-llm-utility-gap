# p12real: independent recompute of the line 149 cosine (probe weight vs Yes minus No readout direction), Pitt, Qwen2.5-Omni. Read only.
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
P='<local data dir>/release/overnight2/part10/'
z=np.load(P+'o25_pitt_states.npz',allow_pickle=True); X=z['ans'][:,-1,:].astype(np.float64); y=z['label']; g=z['spk']
r=torch.load(P+'omni25_readout.pt',map_location='cpu',weights_only=False)
Y=r['lm_head_yes'].double().numpy(); N=r['lm_head_no'].double().numpy()
L=X@np.vstack([Y,N]).T; L-=L.max(1,keepdims=True); p=np.exp(L); p/=p.sum(1,keepdims=True)
m=p.mean(0); wy=m[:5]/m[:5].sum(); wn=m[5:]/m[5:].sum()
d=wy@Y-wn@N
d_plain=Y.mean(0)-N.mean(0)
print('n',len(y),'speakers',len(np.unique(g)),'pos',int(y.sum()))
print('weights yes',np.round(wy,4),'no',np.round(wn,4))
print('projection AUC on d_weighted', round(roc_auc_score(y,X@d),4), ' zero-shot p_yes AUC in npz', round(roc_auc_score(y,z['p_yes']),4))
W=[]; oof=np.zeros(len(y))
for tr,te in GroupKFold(n_splits=5).split(X,y,g):
    sc=StandardScaler().fit(X[tr]); clf=LogisticRegression(max_iter=2000,class_weight='balanced',C=1.0).fit(sc.transform(X[tr]),y[tr])
    W.append(clf.coef_[0]/sc.scale_); oof[te]=clf.decision_function(sc.transform(X[te]))
w=np.mean(W,0)
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
print('probe OOF AUC', round(roc_auc_score(y,oof),4))
print('cosine(w_raw, d_weighted) =', round(cos(w,d),4), ' full', cos(w,d))
print('cosine(w_raw, d_plain)    =', round(cos(w,d_plain),4))
print('per-fold cosines', [round(cos(a,d),4) for a in W])
R=np.random.default_rng(0).standard_normal((1000,X.shape[1]))
print('random floor mean |cos| with d (1000 dirs, rng 0) =', round(float(np.mean([abs(cos(v,d)) for v in R])),4))
