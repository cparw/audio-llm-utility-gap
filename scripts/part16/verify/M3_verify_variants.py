# Which plausible bootstrap variant, if any, reproduces their bounds?
import csv, itertools
import numpy as np

SRC={"pitt":"<local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv",
     "pcgita":"<local data dir>/Desktop/release/omni_final/omni_pcgita_zeroshot_scores.csv",
     "neurovoz":"<local data dir>/Desktop/release/omni_final/omni_neurovoz_zeroshot_scores.csv"}
THEIRS={"pitt":dict(clip=(0.6033,0.7125),spk=(0.6096,0.7494),diff=(-0.0565,0.0172)),
        "pcgita":dict(clip=(0.4931,0.6331),spk=(0.4778,0.7089),diff=(-0.0820,0.0179)),
        "neurovoz":dict(clip=(0.6356,0.7519),spk=(0.7045,0.8733),diff=(-0.1325,-0.0613))}

def auc(y,s):
    y=np.asarray(y); s=np.asarray(s,float)
    pos=s[y==1]; neg=s[y==0]
    if pos.size==0 or neg.size==0: return float('nan')
    w=0.0;t=0.0; step=max(1,int(2e7//max(1,neg.size)))
    for i in range(0,pos.size,step):
        d=pos[i:i+step][:,None]-neg[None,:]
        w+=float((d>0).sum()); t+=float((d==0).sum())
    return (w+0.5*t)/(pos.size*neg.size)

D={}
for ds,path in SRC.items():
    sp=[];lb=[];pv=[]
    for r in csv.DictReader(open(path,newline="")):
        sp.append(r["speaker"]);lb.append(int(r["label"]));pv.append(float(r["p_yes"]))
    sp=np.array(sp);lb=np.array(lb);pv=np.array(pv,float)
    order=[];seen=set()
    for s in sp:
        if s not in seen: seen.add(s);order.append(s)
    idx={s:i for i,s in enumerate(order)}
    g=[[] for _ in order]
    for i,s in enumerate(sp): g[idx[s]].append(i)
    D[ds]=dict(lab=lb,p=pv,groups=[np.array(x) for x in g],
               s_lab=np.array([lb[x][0] for x in g]),
               s_mean=np.array([pv[x].mean() for x in g]),K=len(order))

def run(ds,rng,method="linear",stratify=False):
    d=D[ds]; K=d["K"]; cl=[];sk=[];df=[]
    pos_i=np.where(d["s_lab"]==1)[0]; neg_i=np.where(d["s_lab"]==0)[0]
    for _ in range(2000):
        if stratify:
            pick=np.concatenate([pos_i[rng.integers(0,pos_i.size,pos_i.size)],
                                 neg_i[rng.integers(0,neg_i.size,neg_i.size)]])
        else:
            pick=rng.integers(0,K,size=K)
        bl=d["s_lab"][pick]
        if bl.min()==bl.max(): continue
        ci=np.concatenate([d["groups"][k] for k in pick])
        a=auc(d["lab"][ci],d["p"][ci]); b=auc(bl,d["s_mean"][pick])
        cl.append(a);sk.append(b);df.append(a-b)
    q=lambda v:(float(np.percentile(v,2.5,method=method)),float(np.percentile(v,97.5,method=method)))
    return dict(clip=q(cl),spk=q(sk),diff=q(df))

def score(ds,res):
    hits=0
    for k in ("clip","spk","diff"):
        for a,b in zip(res[k],THEIRS[ds][k]):
            if round(a,4)==round(b,4): hits+=1
    return hits

print("variant sweep -- how many of the 6 bounds per dataset land on theirs (max 6)")
variants=[]
for method in ("linear","nearest","lower","higher","midpoint"):
    variants.append((f"fresh rng(0), pct={method}", lambda ds,m=method: run(ds,np.random.default_rng(0),m)))
variants.append(("fresh rng(0), stratified by label", lambda ds: run(ds,np.random.default_rng(0),stratify=True)))

for name,fn in variants:
    tot=0; line=[]
    for ds in ("pitt","pcgita","neurovoz"):
        r=fn(ds); h=score(ds,r); tot+=h; line.append(f"{ds}={h}/6")
    print(f"  {name:38s} {' '.join(line)}  total={tot}/18")

# shared rng across datasets, every processing order
print("\nshared single rng(0) consumed across all three datasets, all 6 orders:")
for perm in itertools.permutations(("pitt","pcgita","neurovoz")):
    rng=np.random.default_rng(0); tot=0; line=[]
    for ds in perm:
        r=run(ds,rng); h=score(ds,r); tot+=h; line.append(f"{ds}={h}/6")
    print(f"  order={'>'.join(perm):28s} {' '.join(line)}  total={tot}/18")
