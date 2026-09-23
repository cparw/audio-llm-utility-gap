"""pod3b INDEPENDENT VERIFIER (lo-pod3b-2). Written separately from pod3b_primary.py / pod3b_gap.py.
Own lm_head route (full shard via huggingface_hub + safetensors/torch, not HTTP range + manual bf16 parse),
own softmax (scipy logsumexp), own AUC (scipy mannwhitneyu), own fold builder (checked against sklearn),
own bootstrap per the leftovers rule: fresh default_rng(0) per cell, idx = rng.choice(N, N, replace=True)
over the sorted unique speaker ids, 2000 draws, 2.5/97.5 percentiles, same idx for every quantity (paired).
The per-fold estimator is the defined one: StandardScaler on the training fold + LogisticRegression(max_iter=2000, class_weight='balanced')."""
import os, sys, json, csv, hashlib, datetime, platform
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np, scipy, sklearn, torch
from scipy.special import logsumexp
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
from huggingface_hub import hf_hub_download
from safetensors import safe_open
REPO="Qwen/Qwen3-Omni-30B-A3B-Instruct"; REV="26291f793822fb6be9555850f06dfe95f2d7e695"
YES=[7414,9454,9693,9834,14004]; NO=[902,2152,2308,2753,8996]
V={}
def put(k,v): V[k]=v; print(k,v,flush=True)
# ---------- lm_head, own route ----------
p=hf_hub_download(REPO,"model-00013-of-00015.safetensors",revision=REV,local_dir="/workspace/hf")
with safe_open(p,framework="pt") as f: T=f.get_tensor("thinker.lm_head.weight")
put("lm_head_shape",list(T.shape)); put("lm_head_dtype",str(T.dtype))
put("lm_head_sha256_bf16",hashlib.sha256(T.view(torch.int16).numpy().tobytes()).hexdigest())
W32=T.float().numpy()                              # bf16 exact
Wr=np.load("/workspace/lmh/q3o_lm_head.npy")      # the primary's HTTP-range copy on this pod
put("range_copy_bitwise_equal",bool(np.array_equal(W32.view(np.uint32),Wr.view(np.uint32)))); del Wr
W16=W32.astype(np.float16).astype(np.float32)      # POD3 dump: rescore_probe.py line 22 W.astype(np.float16)
put("fp16_dump_rows_changed",int((W16!=W32).any(1).sum())); put("fp16_dump_entries_changed",int((W16!=W32).sum()))
# ---------- data ----------
z=np.load("/workspace/q3o_pitt_states.npz",allow_pickle=True)
X_all=np.asarray(z["ans"][:,-1,:],dtype=np.float64); y_all=np.asarray(z["label"]).astype(int)
spk_all=np.asarray(z["spk"]).astype(str); nm=np.asarray(z["name"]).astype(str); psaved=np.asarray(z["p_yes"],float)
arm_all=np.char.partition(nm,"_")[:,0]
g=np.load("/workspace/q3o_perp_cols.npz",allow_pickle=True)
mac=np.load("/workspace/mac_folds.npz",allow_pickle=True)
def auc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); U=mannwhitneyu(s[y==1],s[y==0],alternative="two-sided",method="asymptotic").statistic
    return float(U/((y==1).sum()*(y==0).sum()))
def boot(y,spk,scores,draws=2000):
    us=np.unique(spk); N=len(us); pos={s:np.flatnonzero(spk==s) for s in us}; out=[]
    for sc in scores:
        rng=np.random.default_rng(0); v=[]
        for _ in range(draws):
            idx=rng.choice(N,size=N,replace=True); ii=np.concatenate([pos[us[k]] for k in idx])
            if len(np.unique(y[ii]))<2: continue
            v.append(auc(y[ii],sc[ii]))
        out.append((float(np.percentile(v,2.5)),float(np.percentile(v,97.5)),len(v)))
    return out
def my_groupkfold(spk,k=5,kind="stable"):
    # sklearn 1.9.1 sorts the group sizes with kind="stable"; sklearn 1.7.2 (the Mac) uses the default kind.
    u,inv=np.unique(spk,return_inverse=True); c=np.bincount(inv); order=np.argsort(c,kind=kind)[::-1]
    load=np.zeros(k); gf=np.zeros(len(u),int)
    for gi in order: j=int(np.argmin(load)); load[j]+=c[gi]; gf[gi]=j
    return gf[inv]
def perm_folds(spk,seed,k=5):
    u,inv=np.unique(spk,return_inverse=True); c=np.bincount(inv); pr=np.random.default_rng(seed).permutation(len(u))
    order=pr[np.argsort(-c[pr],kind="stable")]; load=np.zeros(k); gf=np.zeros(len(u),int)
    for gi in order: j=int(np.argmin(load)); load[j]+=c[gi]; gf[gi]=j
    return gf[inv]
def fits(X,y,fold,dvec):
    oof=np.zeros(len(y)); pt=np.zeros(len(y)); pz=np.zeros(len(y)); ws=[]
    for k in range(5):
        tr=np.flatnonzero(fold!=k); te=np.flatnonzero(fold==k)
        sc=StandardScaler().fit(X[tr]); A=sc.transform(X[tr]); B=sc.transform(X[te])
        m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(A,y[tr])
        oof[te]=m.predict_proba(B)[:,1]; w=m.coef_[0]; ws.append(w/sc.scale_)
        wp=w-np.dot(w,dvec)/np.dot(dvec,dvec)*dvec; a=A@wp; b=B@wp
        pt[te]=(b-a.mean())/(a.std()+1e-12); pz[te]=(b-b.mean())/b.std()
    wr=np.mean(ws,0); wr=wr/np.linalg.norm(wr)
    return oof,pt,pz,wr
def readout(X,W):
    L=X@W.T; P=np.exp(L-logsumexp(L,axis=1,keepdims=True))
    a=P[:,YES].mean(0); b=P[:,NO].mean(0); a=a/a.sum(); b=b/b.sum()
    dv=W[YES].astype(np.float64).T@a-W[NO].astype(np.float64).T@b
    return dv/np.linalg.norm(dv), P
put("env",{"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"machine":platform.machine()})
for wtag,W in (("fp16",W16),("bf16exact",W32)):
    for arm in ("all","conflict","agreement"):
        m=np.ones(len(y_all),bool) if arm=="all" else arm_all==arm
        X=X_all[m]; y=y_all[m]; spk=spk_all[m]
        dvec,P=readout(X,W)
        if arm=="all":
            py=P[:,YES].sum(1)/(P[:,YES].sum(1)+P[:,NO].sum(1))
            put(f"{wtag}_pyes_maxabs_vs_saved",float(np.abs(py-psaved).max()))
            put(f"{wtag}_projd_maxabs_vs_pod",float(np.abs(X@dvec-g["proj_d"]).max()))
        fl=my_groupkfold(spk,kind="stable"); fleg=my_groupkfold(spk,kind=None)
        sk=np.full(len(y),-1)
        for k,(tr,te) in enumerate(GroupKFold(5).split(X,y,groups=spk)): sk[te]=k
        put(f"{wtag}_{arm}_own_folds_equal_sklearn",bool((fl==sk).all()))
        oof,pt,pz,wr=fits(X,y,fl,dvec)
        cos=float(wr@dvec); sd=X@dvec
        rng=np.random.default_rng(0); rr=np.array([abs(float(wr@(v/np.linalg.norm(v)))) for v in (rng.normal(size=len(dvec)) for _ in range(2000))])
        a_p,a_d,a_t,a_z=auc(y,oof),auc(y,sd),auc(y,pt),auc(y,pz)
        (pl,ph,nd),(dl,dh,_),(tl,th,_),(zl,zh,_)=boot(y,spk,[oof,sd,pt,pz])
        put(f"{wtag}_{arm}",{"n":int(len(y)),"n_speakers":int(len(np.unique(spk))),"cosine":cos,"rand_mean":float(rr.mean()),
            "rand_lo":float(np.percentile(rr,2.5)),"rand_hi":float(np.percentile(rr,97.5)),
            "auc_probe":a_p,"auc_probe_ci":[pl,ph],"auc_d":a_d,"auc_d_ci":[dl,dh],"auc_wo_trainz":a_t,"auc_wo_trainz_ci":[tl,th],
            "auc_wo_foldz":a_z,"auc_wo_foldz_ci":[zl,zh],"draws":nd,
            **({"oof_maxabs_vs_pod":float(np.abs(oof-g["oof"]).max())} if arm=="all" else {})})
        if wtag=="fp16":
            oL,_,_,wL=fits(X,y,fleg,dvec)
            put(f"legacyrule_linux_{arm}",{"auc_probe":auc(y,oL),"cosine":float(wL@dvec),"note":"sklearn 1.7.2 fold rule (default-kind argsort) run on Linux numpy 2.1.2"})
        if wtag=="fp16":
            fm=mac[arm].astype(int); oM,_,_,wM=fits(X,y,fm,dvec); (ml,mh,_),=boot(y,spk,[oM])
            put(f"macfolds_{arm}",{"auc_probe":auc(y,oM),"auc_probe_ci":[ml,mh],"cosine":float(wM@dvec),
                 "speakers_with_different_foldmates":int(sum(set(spk[fl==fl[spk==s][0]])!=set(spk[fm==fm[spk==s][0]]) for s in np.unique(spk)))})
            if os.environ.get("SKIP_SPREAD")=="1": continue
            def one(seed):
                f=perm_folds(spk,seed); o,_,_,w=fits(X,y,f,dvec); return auc(y,o),float(w@dvec)
            C=np.array(Parallel(n_jobs=32)(delayed(one)(s) for s in range(200)))
            put(f"spread_{arm}",{"auc_mean":float(C[:,0].mean()),"auc_sd":float(C[:,0].std(ddof=1)),"auc_min":float(C[:,0].min()),"auc_max":float(C[:,0].max()),
                 "auc_p2_5":float(np.percentile(C[:,0],2.5)),"auc_p97_5":float(np.percentile(C[:,0],97.5)),
                 "cos_mean":float(C[:,1].mean()),"cos_min":float(C[:,1].min()),"cos_max":float(C[:,1].max())})
V["date"]=datetime.datetime.now().isoformat(timespec="seconds")
json.dump(V,open(os.environ.get("VOUT","/workspace/out/pod3b_verify.json"),"w"),indent=1); print("DONE",flush=True)
