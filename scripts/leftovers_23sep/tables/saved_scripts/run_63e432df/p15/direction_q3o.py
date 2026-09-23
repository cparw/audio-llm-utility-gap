"""Part 15 E: readout-direction test on Qwen3-Omni Pitt answer states."""
import numpy as np, json, torch, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
d=np.load("/workspace/out/q3o_pitt_states.npz",allow_pickle=True)
ans=d["ans"]; y=d["label"].astype(int); spk=d["spk"]
X=ans[:,-1,:].astype(np.float64)          # final LM stage at the first answer position
print("states",ans.shape,"final-stage X",X.shape,flush=True)
MID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
from transformers import Qwen3OmniMoeProcessor as Proc, Qwen3OmniMoeForConditionalGeneration as Model
proc=Proc.from_pretrained(MID); tok=proc.tokenizer
full=Model.from_pretrained(MID,dtype=torch.bfloat16)
net=full.thinker
W=net.lm_head.weight.detach().float().cpu().numpy()
def ids(ws):
    out=[]
    for w in ws:
        for f in (w," "+w):
            t=tok.encode(f,add_special_tokens=False)
            if len(t)==1: out.append(t[0])
    return sorted(set(out))
YES=ids(["Yes","yes","YES"]); NO=ids(["No","no","NO"])
print("yes ids",YES,"no ids",NO,flush=True)
# mass-weighted readout direction, as in the Omni run
import torch.nn.functional as F
sc=torch.softmax(torch.tensor(X@W.T),dim=-1).numpy()
wy=sc[:,YES].mean(0); wn=sc[:,NO].mean(0)
wy=wy/max(wy.sum(),1e-12); wn=wn/max(wn.sum(),1e-12)
dvec=(W[YES].T@wy)-(W[NO].T@wn)
dvec=dvec/np.linalg.norm(dvec)
gkf=GroupKFold(n_splits=5); oof=np.zeros(len(y)); ws=[]
for tr,te in gkf.split(X,y,groups=spk):
    ss=StandardScaler().fit(X[tr])
    lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(ss.transform(X[tr]),y[tr])
    oof[te]=lr.predict_proba(ss.transform(X[te]))[:,1]
    ws.append(lr.coef_[0]/ss.scale_)
w_raw=np.mean(ws,axis=0); w_raw=w_raw/np.linalg.norm(w_raw)
cos=float(w_raw@dvec)
rng=np.random.default_rng(0); rnd=[]
for _ in range(2000):
    r=rng.normal(size=len(dvec)); r/=np.linalg.norm(r); rnd.append(abs(float(w_raw@r)))
rnd=np.array(rnd)
auc_probe=roc_auc_score(y,oof); auc_d=roc_auc_score(y,X@dvec)
oof2=np.zeros(len(y))
for tr,te in gkf.split(X,y,groups=spk):
    ss=StandardScaler().fit(X[tr]); Xt=ss.transform(X[tr]); Xe=ss.transform(X[te])
    lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xt,y[tr])
    wf=lr.coef_[0]; wp=wf-(wf@dvec)/(dvec@dvec)*dvec
    oof2[te]=(Xe@wp-(Xt@wp).mean())/((Xt@wp).std()+1e-12)
auc_wo=roc_auc_score(y,oof2)
print("DIR n=%d cos=%.4f rand_mean_abs=%.4f rand_ci=[%.4f,%.4f] auc_probe=%.4f auc_d=%.4f auc_without_d=%.4f"%(
    len(y),cos,rnd.mean(),np.percentile(rnd,2.5),np.percentile(rnd,97.5),auc_probe,auc_d,auc_wo),flush=True)
json.dump({"model":MID,"n":int(len(y)),"n_speakers":int(len(set(spk))),
 "cosine_w_d":round(cos,4),"cosine_random_mean_abs":round(float(rnd.mean()),4),
 "cosine_random_ci":[round(float(np.percentile(rnd,2.5)),4),round(float(np.percentile(rnd,97.5)),4)],
 "auc_probe_oof":round(float(auc_probe),4),"auc_d":round(float(auc_d),4),
 "auc_probe_without_d":round(float(auc_wo),4),"seed":0,"draws":2000,
 "states":"/workspace/out/q3o_pitt_states.npz","direction":"mass-weighted Yes minus No lm_head rows",
 "folds":"GroupKFold(5) by speaker, scaler fit inside the training fold"},
 open("/workspace/out/readout_direction_q3o.json","w"),indent=1)
import csv
with open("/workspace/out/readout_direction_q3o.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["model","n","n_speakers","cosine","cosine_random","auc_probe","auc_d","auc_probe_without_d"])
    w.writerow([MID,len(y),len(set(spk)),round(cos,4),round(float(rnd.mean()),4),
                round(float(auc_probe),4),round(float(auc_d),4),round(float(auc_wo),4)])
print("wrote readout_direction_q3o.csv/.json",flush=True)
