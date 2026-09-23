"""pod3b gap test (lo-pod3b-3). Why does the Mac refit give 0.7864 where the pod gave 0.8307?
Hypothesis: GroupKFold assigns speakers to folds after np.argsort of the group sizes (default kind, not stable).
228 speakers share only 8 distinct sizes (112 have 1 clip), so the tie order decides the folds, and the tie
order differs between numpy on Linux x86 (SIMD sort) and numpy on the arm64 Mac.
Test: same Linux sklearn 1.9.1 fit, (A) Linux GroupKFold folds, (B) the Mac's GroupKFold folds (mac_folds.npz,
made on the Mac with sklearn 1.7.2 / numpy 2.2.6). Then (C) the spread of the same probe over 200 other valid
speaker-grouped greedy folds that differ only in the tie order (seeded permutations), with cosine too.
Recipe per fold is direction_by_arm.py's: StandardScaler on train, LogisticRegression(max_iter=2000, class_weight='balanced')."""
import os, sys, json, csv, datetime, platform
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import numpy as np, sklearn, scipy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
NPZ="/workspace/q3o_pitt_states.npz"; PERP="/workspace/q3o_perp_cols.npz"
LMH=sys.argv[1]
YES=[7414,9454,9693,9834,14004]; NO=[902,2152,2308,2753,8996]
d=np.load(NPZ,allow_pickle=True)
X_all=d["ans"][:,-1,:].astype(np.float64); y_all=d["label"].astype(int); spk_all=d["spk"].astype(str); name_all=d["name"].astype(str)
arms_all=np.array([n.split("_")[0] for n in name_all])
mac=np.load("/workspace/mac_folds.npz",allow_pickle=True); assert (mac["name"].astype(str)==name_all).all()
g=np.load(PERP,allow_pickle=True)
W=np.load(LMH).astype(np.float32)
def rank_auc(yv,s):
    yv=np.asarray(yv); s=np.asarray(s,float); n1=int((yv==1).sum()); n0=int((yv==0).sum())
    o=np.argsort(s,kind="mergesort"); ss=s[o]; rk=np.empty(len(s)); i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        rk[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    return float((rk[yv==1].sum()-n1*(n1+1)/2.0)/(n1*n0))
def direction(X):
    lg=X@W.T; mm=lg.max(axis=-1,keepdims=True); ee=np.exp(lg-mm); s=ee/ee.sum(axis=-1,keepdims=True)
    wy=s[:,YES].mean(0); wn=s[:,NO].mean(0); wy=wy/max(wy.sum(),1e-12); wn=wn/max(wn.sum(),1e-12)
    dv=(W[YES].T.astype(np.float64)@wy)-(W[NO].T.astype(np.float64)@wn); return dv/np.linalg.norm(dv)
def fit_fold(X,y,tr,te):
    ss=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(ss.transform(X[tr]),y[tr])
    return te, lr.predict_proba(ss.transform(X[te]))[:,1], lr.coef_[0]/ss.scale_
def probe(X,y,fold,dvec,par=True):
    jobs=[(np.where(fold!=k)[0],np.where(fold==k)[0]) for k in range(5)]
    outs=Parallel(n_jobs=5)(delayed(fit_fold)(X,y,tr,te) for tr,te in jobs) if par else [fit_fold(X,y,tr,te) for tr,te in jobs]
    oof=np.zeros(len(y)); ws=[]
    for te,p,w in outs: oof[te]=p; ws.append(w)
    w_raw=np.mean(ws,axis=0); w_raw/=np.linalg.norm(w_raw)
    return oof, float(w_raw@dvec)
def linux_folds(spk):
    f=np.full(len(spk),-1)
    for k,(tr,te) in enumerate(GroupKFold(n_splits=5).split(np.zeros((len(spk),1)),groups=spk)): f[te]=k
    return f
def greedy_folds(spk,perm_seed):
    """sklearn's greedy rule (largest group first into the lightest fold), ties broken by a seeded permutation."""
    u,gi=np.unique(spk,return_inverse=True); cnt=np.bincount(gi)
    pr=np.random.default_rng(perm_seed).permutation(len(u))
    order=pr[np.argsort(-cnt[pr],kind="stable")]
    load=np.zeros(5); g2f=np.zeros(len(u),int)
    for gidx in order:
        k=int(np.argmin(load)); load[k]+=cnt[gidx]; g2f[gidx]=k
    return g2f[gi]
res={"env":{"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__,"scipy":scipy.__version__,
            "machine":platform.machine(),"lm_head":LMH,"date":datetime.datetime.now().isoformat(timespec="seconds")}}
u,gi=np.unique(spk_all,return_inverse=True); cnt=np.bincount(gi)
res["argsort_linux_all"]=np.argsort(cnt).tolist()
mj=json.load(open("/workspace/mac_argsort.json")); res["argsort_mac_all"]=mj["argsort_all"]
res["argsort_identical"]=bool(res["argsort_linux_all"]==mj["argsort_all"])
res["argsort_positions_differing"]=int(sum(a!=b for a,b in zip(res["argsort_linux_all"],mj["argsort_all"])))
rows=[]
for arm in ("all","conflict","agreement"):
    m=np.ones(len(y_all),bool) if arm=="all" else arms_all==arm
    X=X_all[m]; y=y_all[m]; spk=spk_all[m]; dvec=direction(X)
    fl=linux_folds(spk); fm=mac[arm].astype(int)
    # same partition up to fold relabelling?  compare as sets of speakers per fold
    part=lambda f: sorted(tuple(sorted(set(spk[f==k]))) for k in range(5))
    same_partition=part(fl)==part(fm)
    moved=int(sum(1 for s in set(spk) if len({int(v) for v in fl[spk==s]})==1 and True) )
    # clips whose held-out companions differ: count speakers whose fold-mates set differs
    mates_l={s:frozenset(spk[fl==fl[spk==s][0]]) for s in set(spk)}; mates_m={s:frozenset(spk[fm==fm[spk==s][0]]) for s in set(spk)}
    diff_spk=sum(mates_l[s]!=mates_m[s] for s in set(spk))
    oofL,cosL=probe(X,y,fl,dvec); oofM,cosM=probe(X,y,fm,dvec)
    r={"arm":arm,"n":int(len(y)),"n_speakers":int(len(set(spk))),"same_partition_linux_vs_mac":bool(same_partition),
       "speakers_with_different_foldmates":int(diff_spk),
       "auc_linux_folds":rank_auc(y,oofL),"cos_linux_folds":cosL,"auc_mac_folds":rank_auc(y,oofM),"cos_mac_folds":cosM}
    if arm=="all":
        r["linux_oof_vs_pod_oof_maxabs"]=float(np.abs(oofL-g["oof"]).max())
    print(json.dumps(r),flush=True); res.setdefault("AB",[]).append(r)
    for i,nm in enumerate(name_all[m]):
        rows.append([arm,nm,spk[i],y[i],fl[i],fm[i],repr(float(oofL[i])),repr(float(oofM[i]))])
    # (C) spread over 200 tie-order permutations (fits in parallel across seeds, each seed serial)
    def one(seed):
        f=greedy_folds(spk,seed); o,c=probe(X,y,f,dvec,par=False); return seed,rank_auc(y,o),c
    C=Parallel(n_jobs=32)(delayed(one)(s) for s in range(200))
    aucs=np.array([c[1] for c in C]); coss=np.array([c[2] for c in C])
    sp={"arm":arm,"n_perm":200,"auc_mean":float(aucs.mean()),"auc_sd":float(aucs.std(ddof=1)),"auc_min":float(aucs.min()),"auc_max":float(aucs.max()),
        "auc_p2_5":float(np.percentile(aucs,2.5)),"auc_p97_5":float(np.percentile(aucs,97.5)),
        "frac_ge_linux":float((aucs>=r["auc_linux_folds"]-1e-12).mean()),"frac_le_mac":float((aucs<=r["auc_mac_folds"]+1e-12).mean()),
        "cos_mean":float(coss.mean()),"cos_min":float(coss.min()),"cos_max":float(coss.max()),"abs_cos_max":float(np.abs(coss).max())}
    print(json.dumps(sp),flush=True); res.setdefault("spread",[]).append(sp)
    with open(f"/workspace/out/pod3b_gap_spread_{arm}.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["perm_seed","auc_probe","cosine"]); [w.writerow([s,repr(a),repr(c)]) for s,a,c in C]
with open("/workspace/out/pod3b_gap_perclip.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["arm","name","speaker","label","fold_linux","fold_mac","oof_linux_folds","oof_mac_folds"]); w.writerows(rows)
json.dump(res,open("/workspace/out/pod3b_gap.json","w"),indent=1); print("DONE",flush=True)
