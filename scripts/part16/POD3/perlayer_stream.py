"""POD3: same computation as perlayer_nested.py but restricted to ONE stream and parallelised
over layers, with a vectorised tie-aware rank AUC. Numerically identical by construction:
same folds (same rng seeds, same GroupKFold), same estimator, same bootstrap seed and draws.
usage: perlayer_stream.py STATES.npz OUTPREFIX STREAM [R]"""
import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import sys, json, csv, time, datetime, warnings; warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

NPZ, OUTP, KEY = sys.argv[1:4]
R = int(sys.argv[4]) if len(sys.argv) > 4 else 5
t0=time.time()
z=np.load(NPZ, allow_pickle=True)
y=z["label"].astype(int); spk=z["spk"].astype(str)
NJ=max(2,(os.cpu_count() or 8)-8)
print(f"loaded {NPZ} stream={KEY} n={len(y)} pos={int(y.sum())} n_spk={len(set(spk))} njobs={NJ}",flush=True)

def rank_auc(yv,s):
    """Tie-aware rank AUC, fully vectorised. Same value as the scalar version."""
    yv=np.asarray(yv); s=np.asarray(s,dtype=np.float64); n=len(s)
    n1=int((yv==1).sum()); n0=n-n1
    if n1==0 or n0==0: return float("nan")
    order=np.argsort(s,kind="mergesort"); ss=s[order]
    _,inv,cnt=np.unique(ss,return_inverse=True,return_counts=True)
    csum=np.cumsum(cnt); start=csum-cnt
    ranks_sorted=(start+(cnt+1)/2.0)[inv]
    ranks=np.empty(n,dtype=np.float64); ranks[order]=ranks_sorted
    return float((ranks[yv==1].sum()-n1*(n1+1)/2.0)/(n1*n0))

def fitpred(Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000,class_weight="balanced").fit(
        sc.transform(Xtr),ytr).predict_proba(sc.transform(Xte))[:,1]

def groups_for(seed):
    rng=np.random.default_rng(seed); u=np.unique(spk)
    perm={s:i for i,s in enumerate(rng.permutation(u))}
    return np.array([perm[s] for s in spk])

X=z[KEY].astype(np.float32); nl=X.shape[1]; reps=1 if nl==1 else R
GS=[groups_for(s) for s in range(reps)]

def oof_single(l,g):
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        oof[te]=fitpred(X[tr][:,l],y[tr],X[te][:,l])
    return oof

def layer_job(l):
    oofs=[oof_single(l,GS[s]) for s in range(reps)]
    return l,oofs

print(f"per-layer: {nl} stages x {reps} repeats, parallel over layers",flush=True)
res=dict(Parallel(n_jobs=min(nl,NJ))(delayed(layer_job)(l) for l in range(nl)))
print(f"per-layer fits done {time.time()-t0:.0f}s",flush=True)

u=np.unique(spk); idx_by={s:np.where(spk==s)[0] for s in u}
def boot_ci(oofs,draws=2000,seed=0):
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(draws):
        pick=rng.choice(u,size=len(u),replace=True)
        idx=np.concatenate([idx_by[s] for s in pick]); yy=y[idx]
        if len(set(yy))<2: continue
        vals.append(float(np.mean([rank_auc(yy,o[idx]) for o in oofs])))
    v=np.array(vals)
    if len(v)==0: return float("nan"),float("nan"),0
    return float(np.percentile(v,2.5)),float(np.percentile(v,97.5)),len(v)

cis=Parallel(n_jobs=min(nl,NJ))(delayed(boot_ci)(res[l]) for l in range(nl))
print(f"per-layer bootstrap done {time.time()-t0:.0f}s",flush=True)

NAME={"enc":lambda i:f"audio_encoder_layer_{i}","proj":lambda i:"projected_audio_input",
      "llm":lambda i:("thinker_embeddings" if i==0 else f"thinker_layer_{i}"),
      "ans":lambda i:("thinker_embeddings" if i==0 else f"thinker_layer_{i}")}[KEY]
SUF={"enc":"encoder","proj":"proj","llm":"llm","ans":"ans"}[KEY]
POOL={"enc":"mean over audio encoder frames (forward hook on the encoder layer output)",
 "proj":"mean over projector output positions (forward hook on the projector)",
 "llm":"mean over the AUDIO TOKEN positions of the thinker hidden state",
 "ans":"the first answer position (last prompt position) of the thinker hidden state"}[KEY]

per={}
for l in range(nl):
    aucs=[rank_auc(y,o) for o in res[l]]
    lo,hi,nd=cis[l]
    per[l]={"aucs":aucs,"mean":float(np.mean(aucs)),"std":float(np.std(aucs)),"lo":lo,"hi":hi,"draws":nd}
    print(f"  {KEY} stage {l:>2} auc_mean {per[l]['mean']:.4f} [{lo:.4f},{hi:.4f}] "
          f"per_repeat {[round(a,4) for a in aucs]}",flush=True)
with open(f"{OUTP}_{SUF}_perlayer.csv","w",newline="") as fh:
    w=csv.writer(fh)
    w.writerow(["stage","stage_name","auc_mean","auc_std","ci_lo","ci_hi","boot_draws","per_repeat","n","n_speakers"])
    for l in range(nl):
        d=per[l]
        w.writerow([l,NAME(l),round(d["mean"],6),round(d["std"],6),round(d["lo"],6),round(d["hi"],6),
                    d["draws"],"|".join(f"{a:.6f}" for a in d["aucs"]),len(y),len(set(spk))])
best=max(range(nl),key=lambda l:per[l]["mean"]); bd=per[best]

def inner(tr,l,g):
    s=0.0
    for itr,ite in GroupKFold(n_splits=4).split(X[tr],y[tr],groups=g[tr]):
        p=fitpred(X[tr][itr,l],y[tr][itr],X[tr][ite,l])
        s+=roc_auc_score(y[tr][ite],p) if len(set(y[tr][ite]))>1 else 0.5
    return s/4
n_oofs=[];chosen_all=[]
for s in range(reps):
    g=GS[s]; oof=np.zeros(len(y)); ch=[]
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        b=0 if nl==1 else int(np.argmax(Parallel(n_jobs=min(nl,NJ))(delayed(inner)(tr,l,g) for l in range(nl))))
        ch.append(b); oof[te]=fitpred(X[tr][:,b],y[tr],X[te][:,b])
    n_oofs.append(oof); chosen_all.append(ch)
    print(f"  nested repeat {s} auc {rank_auc(y,oof):.4f} chosen {ch}  {time.time()-t0:.0f}s",flush=True)
n_aucs=[rank_auc(y,o) for o in n_oofs]; nlo,nhi,ndr=boot_ci(n_oofs)
with open(f"{OUTP}_{SUF}_nested_oof.csv","w",newline="") as fh:
    w=csv.writer(fh); names=z["name"].astype(str)
    w.writerow(["clip","speaker","label"]+[f"p_probe_rep{s}" for s in range(reps)])
    for i in range(len(y)):
        w.writerow([names[i],spk[i],int(y[i])]+[f"{o[i]:.8f}" for o in n_oofs])
summary={KEY:{"n_points":nl,"repeats":reps,"n":int(len(y)),"n_speakers":int(len(set(spk))),
 "pooling":POOL,"best_stage":int(best),"best_stage_name":NAME(best),
 "best_stage_auc_mean":round(bd["mean"],4),"best_stage_ci":[round(bd["lo"],4),round(bd["hi"],4)],
 "best_stage_per_repeat":[round(a,4) for a in bd["aucs"]],
 "nested_auc_mean":round(float(np.mean(n_aucs)),4),"nested_ci":[round(nlo,4),round(nhi,4)],
 "nested_boot_draws":ndr,"nested_per_repeat":[round(a,4) for a in n_aucs],
 "nested_chosen_layers":chosen_all}}
json.dump({"states":os.path.abspath(NPZ),"repeats":R,"bootstrap_draws":2000,"bootstrap_seed":0,
 "bootstrap_unit":"speaker",
 "folds":"GroupKFold(5) by speaker (outer), GroupKFold(4) by speaker (inner layer choice); "
         "speaker order permuted per repeat with numpy.random.default_rng(seed)",
 "fold_file":"NONE FOUND - no saved fold assignment exists for these clips.",
 "auc":"rank formula, ties averaged, recomputed from the per-clip out-of-fold scores",
 "date":datetime.datetime.now().isoformat(timespec="seconds"),"streams":summary},
 open(f"{OUTP}_{SUF}_perlayer_summary.json","w"),indent=1)
print(f"  >>> {KEY}: NESTED {np.mean(n_aucs):.4f} [{nlo:.4f},{nhi:.4f}] | "
      f"BEST STAGE {best} ({NAME(best)}) {bd['mean']:.4f} [{bd['lo']:.4f},{bd['hi']:.4f}]",flush=True)
print(f"DONE {time.time()-t0:.0f}s",flush=True)
