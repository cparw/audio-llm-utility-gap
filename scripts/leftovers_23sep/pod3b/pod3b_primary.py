"""pod3b primary (lo-pod3b-1). Extended logging around the unmodified direction_by_arm.py run.
Same data, same direction, same folds and fits as direction_by_arm.py run(); adds per-clip checks:
 (1) p_yes from softmax(X @ W.T) vs the p_yes saved in q3o_pitt_states.npz  -> lm_head is the one the pod used
 (2) X @ d vs proj_d saved by the pod in q3o_perp_cols.npz                    -> d is the pod's d
 (3) probe OOF vs oof saved in q3o_perp_cols.npz                              -> the refit is the pod's fit
 (4) the Linux GroupKFold fold ids, for the gap test.
usage: python3 pod3b_primary.py <lm_head npy> <tag>"""
import os, sys, json, csv, hashlib, datetime, platform
os.environ.setdefault("OMP_NUM_THREADS","8")
import numpy as np, sklearn, scipy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
LMH=sys.argv[1]; TAG=sys.argv[2]
NPZ="/workspace/q3o_pitt_states.npz"; PERP="/workspace/q3o_perp_cols.npz"; MAN="/workspace/mf_pitt468.csv"
YES=[7414,9454,9693,9834,14004]; NO=[902,2152,2308,2753,8996]
d=np.load(NPZ,allow_pickle=True)
ans=d["ans"]; y_all=d["label"].astype(int); spk_all=d["spk"].astype(str); name_all=d["name"].astype(str)
X_all=ans[:,-1,:].astype(np.float64); p_yes_saved=d["p_yes"].astype(float); del ans
W=np.load(LMH).astype(np.float32)
g=np.load(PERP,allow_pickle=True); assert (g["name"].astype(str)==name_all).all()
arm_of={os.path.basename(r["path"]):r["set"] for r in csv.DictReader(open(MAN))}
arms=np.array([arm_of[n] for n in name_all]); pref=np.array([n.split("_")[0] for n in name_all])
res={"tag":TAG,"lm_head":LMH,"lm_head_dtype_on_disk":str(np.load(LMH,mmap_mode="r").dtype),
     "manifest_set_equals_name_prefix":int((arms==pref).sum()),"n":int(len(y_all))}
def rank_auc(yv,s):  # copied from direction_by_arm.py
    yv=np.asarray(yv); s=np.asarray(s,float)
    n1=int((yv==1).sum()); n0=int((yv==0).sum())
    if n1==0 or n0==0: return float("nan")
    o=np.argsort(s,kind="mergesort"); ss=s[o]; rk=np.empty(len(s))
    i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        rk[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    return float((rk[yv==1].sum()-n1*(n1+1)/2.0)/(n1*n0))
# (1) p_yes check, full-vocab softmax, all 468 clips
logits=X_all@W.T; m=logits.max(axis=-1,keepdims=True); e=np.exp(logits-m); Z=e.sum(axis=-1,keepdims=True); sc=e/Z
lse=(np.log(Z)+m)[:,0]
py=sc[:,YES].sum(1)/(sc[:,YES].sum(1)+sc[:,NO].sum(1)); mass=sc[:,YES].sum(1)+sc[:,NO].sum(1)
res["p_yes_check"]={"max_abs_diff":float(np.abs(py-p_yes_saved).max()),"median_abs_diff":float(np.median(np.abs(py-p_yes_saved))),
    "pearson":float(np.corrcoef(py,p_yes_saved)[0,1]),"auc_recomputed":rank_auc(y_all,py),"auc_saved":rank_auc(y_all,p_yes_saved),
    "answer_mass_min":float(mass.min()),"answer_mass_median":float(np.median(mass))}
rows={}
def run(X,y,spk,tag,idx):
    lg=X@W.T; mm=lg.max(axis=-1,keepdims=True); ee=np.exp(lg-mm); s=ee/ee.sum(axis=-1,keepdims=True)
    wy=s[:,YES].mean(0); wn=s[:,NO].mean(0)
    wy=wy/max(wy.sum(),1e-12); wn=wn/max(wn.sum(),1e-12)
    dvec=(W[YES].T.astype(np.float64)@wy)-(W[NO].T.astype(np.float64)@wn)
    dvec=dvec/np.linalg.norm(dvec)
    gkf=GroupKFold(n_splits=5); oof=np.zeros(len(y)); ws=[]; fold=np.full(len(y),-1); iters=[]
    for k,(tr,te) in enumerate(gkf.split(X,y,groups=spk)):
        fold[te]=k
        ss=StandardScaler().fit(X[tr])
        lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(ss.transform(X[tr]),y[tr])
        oof[te]=lr.predict_proba(ss.transform(X[te]))[:,1]; iters.append(int(lr.n_iter_[0]))
        ws.append(lr.coef_[0]/ss.scale_)
    w_raw=np.mean(ws,axis=0); w_raw=w_raw/np.linalg.norm(w_raw)
    cos=float(w_raw@dvec)
    s_d=X@dvec
    oof2=np.zeros(len(y)); pz=np.zeros(len(y))
    for tr,te in gkf.split(X,y,groups=spk):
        ss=StandardScaler().fit(X[tr]); Xt=ss.transform(X[tr]); Xe=ss.transform(X[te])
        lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xt,y[tr])
        wf=lr.coef_[0]; wp=wf-(wf@dvec)/(dvec@dvec)*dvec
        oof2[te]=(Xe@wp-(Xt@wp).mean())/((Xt@wp).std()+1e-12)
        b=Xe@wp; pz[te]=(b-b.mean())/b.std()          # canonical held-out-fold z (Q3O_direction_FIXED.json)
    r={"arm":tag,"n":int(len(y)),"n_speakers":int(len(set(spk))),"cosine_w_d":cos,"auc_probe_oof":rank_auc(y,oof),
       "auc_d":rank_auc(y,s_d),"auc_probe_without_d_trainz":rank_auc(y,oof2),"auc_probe_without_d_foldz":rank_auc(y,pz),
       "lbfgs_iters":iters,"fold_sizes":np.bincount(fold).tolist()}
    if tag=="all":
        r["oof_vs_pod_perp_cols"]={"max_abs_diff":float(np.abs(oof-g["oof"]).max()),"pearson":float(np.corrcoef(oof,g["oof"])[0,1])}
        r["proj_d_vs_pod_perp_cols"]={"max_abs_diff":float(np.abs(s_d-g["proj_d"]).max()),"max_rel":float(np.abs(s_d-g["proj_d"]).max()/np.abs(g["proj_d"]).max())}
        r["perp_trainz_vs_pod"]=float(np.abs(oof2-g["perp_trainz"]).max()); r["perp_z_vs_pod"]=float(np.abs(pz-g["perp_z"]).max())
    rows[tag]=dict(idx=idx,fold=fold,oof=oof,s_d=s_d,oof2=oof2,pz=pz,dvec=dvec,w_raw=w_raw)
    print(TAG,json.dumps(r),flush=True); return r
res["arms"]=[run(X_all,y_all,spk_all,"all",np.arange(len(y_all)))]
for a in ("conflict","agreement"):
    mk=arms==a; res["arms"].append(run(X_all[mk],y_all[mk],spk_all[mk],a,np.where(mk)[0]))
os.makedirs("/workspace/out",exist_ok=True)
with open(f"/workspace/out/pod3b_primary_{TAG}_perclip.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["arm","name","speaker","label","fold_linux","oof","proj_d","perp_trainz","perp_foldz"])
    for a,rr in rows.items():
        for j,i in enumerate(rr["idx"]):
            w.writerow([a,name_all[i],spk_all[i],y_all[i],rr["fold"][j],repr(float(rr["oof"][j])),repr(float(rr["s_d"][j])),repr(float(rr["oof2"][j])),repr(float(rr["pz"][j]))])
np.savez(f"/workspace/out/pod3b_primary_{TAG}_vectors.npz",**{f"{a}_dvec":rr["dvec"] for a,rr in rows.items()},**{f"{a}_w_raw":rr["w_raw"] for a,rr in rows.items()},
         lse=lse,p_yes_recomputed=py,answer_mass=mass,W_yes=W[YES],W_no=W[NO],yes_ids=np.array(YES),no_ids=np.array(NO))
res["env"]={"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"machine":platform.machine(),
            "date":datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(res,open(f"/workspace/out/pod3b_primary_{TAG}.json","w"),indent=1)
print("DONE",TAG,flush=True)
