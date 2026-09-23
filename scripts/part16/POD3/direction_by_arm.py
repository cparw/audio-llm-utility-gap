"""POD3 3b-extension: readout-direction test SPLIT BY ARM (conflict 146 / agreement 322).
Same method as direction_q3o.py: final thinker stage at the first answer position, mass-weighted
Yes-minus-No lm_head direction, GroupKFold(5) by speaker, scaler fit inside the training fold.
The lm_head matrix is read from a dumped npy so the 30B model need not be reloaded."""
import os
os.environ.setdefault("OMP_NUM_THREADS","8")
import numpy as np, json, csv, sys, datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

NPZ="/workspace/out/q3o_pitt_states.npz"
MANIFEST="/workspace/mf_pitt468.csv"
LMH="/workspace/out/q3o_lm_head.npy"
MID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
YES=[7414,9454,9693,9834,14004]      # verified identical in the Sept-19 log and today's rerun
NO=[902,2152,2308,2753,8996]

d=np.load(NPZ,allow_pickle=True)
ans=d["ans"]; y_all=d["label"].astype(int); spk_all=d["spk"].astype(str); name_all=d["name"].astype(str)
X_all=ans[:,-1,:].astype(np.float64)
W=np.load(LMH).astype(np.float32)
print("states",ans.shape,"X",X_all.shape,"lm_head",W.shape,flush=True)

arm_of={}
for r in csv.DictReader(open(MANIFEST)):
    arm_of[os.path.basename(r["path"])]=r["set"]
arms=np.array([arm_of[n] for n in name_all])
print("arm counts",{a:int((arms==a).sum()) for a in sorted(set(arms))},flush=True)

def rank_auc(yv,s):
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

def run(X,y,spk,tag):
    n_spk=len(set(spk))
    # mass-weighted readout direction, computed within this arm
    logits=X@W.T
    m=logits.max(axis=-1,keepdims=True)
    e=np.exp(logits-m); sc=e/e.sum(axis=-1,keepdims=True)
    wy=sc[:,YES].mean(0); wn=sc[:,NO].mean(0)
    wy=wy/max(wy.sum(),1e-12); wn=wn/max(wn.sum(),1e-12)
    dvec=(W[YES].T.astype(np.float64)@wy)-(W[NO].T.astype(np.float64)@wn)
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
    s_d=X@dvec
    oof2=np.zeros(len(y))
    for tr,te in gkf.split(X,y,groups=spk):
        ss=StandardScaler().fit(X[tr]); Xt=ss.transform(X[tr]); Xe=ss.transform(X[te])
        lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xt,y[tr])
        wf=lr.coef_[0]; wp=wf-(wf@dvec)/(dvec@dvec)*dvec
        oof2[te]=(Xe@wp-(Xt@wp).mean())/((Xt@wp).std()+1e-12)
    a_p,a_d,a_wo=rank_auc(y,oof),rank_auc(y,s_d),rank_auc(y,oof2)
    # paired speaker bootstrap: one speaker draw per replicate, all three quantities inside it
    u=np.unique(spk); idx_by={s:np.where(spk==s)[0] for s in u}
    rb=np.random.default_rng(0); bp=[];bd=[];bw=[]
    for _ in range(2000):
        pick=rb.choice(u,size=len(u),replace=True)
        idx=np.concatenate([idx_by[s] for s in pick])
        yy=y[idx]
        if len(set(yy))<2: continue
        bp.append(rank_auc(yy,oof[idx])); bd.append(rank_auc(yy,s_d[idx])); bw.append(rank_auc(yy,oof2[idx]))
    ci=lambda v:[round(float(np.percentile(v,2.5)),4),round(float(np.percentile(v,97.5)),4)]
    res={"arm":tag,"n":int(len(y)),"n_speakers":int(n_spk),"n_positive":int(y.sum()),
      "cosine_w_d":round(cos,4),
      "cosine_random_mean_abs":round(float(rnd.mean()),4),
      "cosine_random_ci":[round(float(np.percentile(rnd,2.5)),4),round(float(np.percentile(rnd,97.5)),4)],
      "auc_probe_oof":round(a_p,4),"auc_probe_ci":ci(bp),
      "auc_d":round(a_d,4),"auc_d_ci":ci(bd),
      "auc_probe_without_d":round(a_wo,4),"auc_probe_without_d_ci":ci(bw),
      "boot_usable_draws":len(bp)}
    print(f"[{tag}] n={len(y)} spk={n_spk} cos={cos:.4f} rand_floor={rnd.mean():.4f} "
          f"[{np.percentile(rnd,2.5):.4f},{np.percentile(rnd,97.5):.4f}] "
          f"auc_probe={a_p:.4f}{ci(bp)} auc_d={a_d:.4f}{ci(bd)} auc_without_d={a_wo:.4f}{ci(bw)}",flush=True)
    return res

out=[run(X_all,y_all,spk_all,"all")]
for a in ("conflict","agreement"):
    m=arms==a
    out.append(run(X_all[m],y_all[m],spk_all[m],a))

json.dump({"model":MID,"states":NPZ,"manifest":MANIFEST,"lm_head":LMH,
 "direction":"mass-weighted Yes minus No lm_head rows, recomputed within each arm",
 "folds":"GroupKFold(5) by speaker, scaler fit inside the training fold",
 "bootstrap":{"draws":2000,"seed":0,"unit":"speaker","paired":True},
 "yes_ids":YES,"no_ids":NO,"auc":"rank formula, ties averaged",
 "date":datetime.datetime.now().isoformat(timespec="seconds"),"arms":out},
 open("/workspace/out/readout_direction_q3o_by_arm.json","w"),indent=1)
with open("/workspace/out/readout_direction_q3o_by_arm.csv","w",newline="") as fh:
    w=csv.writer(fh)
    w.writerow(["model","arm","n","n_speakers","n_positive","cosine","cosine_random","cos_rand_lo","cos_rand_hi",
                "auc_probe","auc_probe_lo","auc_probe_hi","auc_d","auc_d_lo","auc_d_hi",
                "auc_probe_without_d","auc_wo_lo","auc_wo_hi","boot_draws"])
    for r in out:
        w.writerow([MID,r["arm"],r["n"],r["n_speakers"],r["n_positive"],r["cosine_w_d"],
                    r["cosine_random_mean_abs"],r["cosine_random_ci"][0],r["cosine_random_ci"][1],
                    r["auc_probe_oof"],r["auc_probe_ci"][0],r["auc_probe_ci"][1],
                    r["auc_d"],r["auc_d_ci"][0],r["auc_d_ci"][1],
                    r["auc_probe_without_d"],r["auc_probe_without_d_ci"][0],r["auc_probe_without_d_ci"][1],
                    r["boot_usable_draws"]])
print("wrote readout_direction_q3o_by_arm.csv/.json",flush=True)
