# Seed-jitter study: how much do the 2000-draw percentile bounds move with the
# RNG stream alone? If their bounds sit inside this spread, the CI gap is Monte
# Carlo noise rather than a methodological disagreement.
import csv, hashlib
import numpy as np

SRC = {
 "pitt":"<local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv",
 "pcgita":"<local data dir>/Desktop/release/omni_final/omni_pcgita_zeroshot_scores.csv",
 "neurovoz":"<local data dir>/Desktop/release/omni_final/omni_neurovoz_zeroshot_scores.csv"}

THEIRS = {
 "pitt":     dict(clip=(0.6033,0.7125), spk=(0.6096,0.7494), diff=(-0.0565, 0.0172)),
 "pcgita":   dict(clip=(0.4931,0.6331), spk=(0.4778,0.7089), diff=(-0.0820, 0.0179)),
 "neurovoz": dict(clip=(0.6356,0.7519), spk=(0.7045,0.8733), diff=(-0.1325,-0.0613))}

def auc_pairs(y,s):
    y=np.asarray(y); s=np.asarray(s,float)
    pos=s[y==1]; neg=s[y==0]
    if pos.size==0 or neg.size==0: return float('nan')
    w=0.0; t=0.0
    step=max(1,int(2e7//max(1,neg.size)))
    for i in range(0,pos.size,step):
        d=pos[i:i+step][:,None]-neg[None,:]
        w+=float((d>0).sum()); t+=float((d==0).sum())
    return (w+0.5*t)/(pos.size*neg.size)

def load(p):
    sp=[];lb=[];pv=[]
    for r in csv.DictReader(open(p,newline="")):
        sp.append(r["speaker"]); lb.append(int(r["label"])); pv.append(float(r["p_yes"]))
    return np.array(sp),np.array(lb),np.array(pv,float)

for ds,path in SRC.items():
    spk,lab,p=load(path)
    order=[];seen=set()
    for s in spk:
        if s not in seen: seen.add(s); order.append(s)
    idx={s:i for i,s in enumerate(order)}
    groups=[[] for _ in order]
    for i,s in enumerate(spk): groups[idx[s]].append(i)
    s_lab=np.array([lab[g][0] for g in groups])
    s_mean=np.array([p[g].mean() for g in groups])
    K=len(order)

    rows={"clip":[],"spk":[],"diff":[]}
    for seed in range(12):
        rng=np.random.default_rng(seed)
        cl=[];sp_=[];df=[]
        for _ in range(2000):
            pick=rng.integers(0,K,size=K)
            bl=s_lab[pick]
            if bl.min()==bl.max(): continue
            ci=np.concatenate([groups[k] for k in pick])
            a=auc_pairs(lab[ci],p[ci]); b=auc_pairs(bl,s_mean[pick])
            cl.append(a); sp_.append(b); df.append(a-b)
        for key,v in (("clip",cl),("spk",sp_),("diff",df)):
            rows[key].append((np.percentile(v,2.5),np.percentile(v,97.5)))

    print(f"=== {ds} ===")
    for key in ("clip","spk","diff"):
        arr=np.array(rows[key])
        lo,hi=arr[:,0],arr[:,1]
        tlo,thi=THEIRS[ds][key]
        mine=rows[key][0]
        print(f"  {key:5s} mine(seed0)=[{mine[0]:.4f},{mine[1]:.4f}]  theirs=[{tlo:.4f},{thi:.4f}]  "
              f"|d_lo|={abs(mine[0]-tlo):.4f} |d_hi|={abs(mine[1]-thi):.4f}")
        print(f"        12-seed spread of lo=[{lo.min():.4f},{lo.max():.4f}] (sd {lo.std():.4f}), "
              f"hi=[{hi.min():.4f},{hi.max():.4f}] (sd {hi.std():.4f})  "
              f"theirs_inside_lo={lo.min()<=tlo<=lo.max()} theirs_inside_hi={hi.min()<=thi<=hi.max()}")
