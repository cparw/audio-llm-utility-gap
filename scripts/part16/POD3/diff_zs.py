import csv, numpy as np, json, os
SH="<local data dir>/Desktop/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"
PD="<local data dir>/scratch/pod_q3o_pitt_zs.csv"
def load(p):
    return list(csv.DictReader(open(p)))
a=load(SH); b=load(PD)
print("n shipped",len(a),"n pod",len(b))
print("clip order identical:", [r["clip"] for r in a]==[r["clip"] for r in b])
print("labels identical:", [r["label"] for r in a]==[r["label"] for r in b])
print("prompt identical:", a[0]["prompt"]==b[0]["prompt"])
print("shipped prompt:",repr(a[0]["prompt"]))
print("pod     prompt:",repr(b[0]["prompt"]))
sa=set(r["speaker"] for r in a); sb=set(r["speaker"] for r in b)
print("n_spk shipped",len(sa),"n_spk pod",len(sb))
print("speaker label examples shipped",sorted(sa)[:4],"pod",sorted(sb)[:4])
pa=np.array([float(r["p_yes"]) for r in a]); pb=np.array([float(r["p_yes"]) for r in b])
ma=np.array([float(r["mass"]) for r in a]); mb=np.array([float(r["mass"]) for r in b])
y =np.array([int(r["label"]) for r in a])
d=np.abs(pa-pb)
print("\n--- p_yes abs diff ---")
for t in [1e-6,1e-5,1e-4,1e-3,1e-2,5e-2,1e-1,2e-1]:
    print(f"  > {t:g}: {(d>t).sum()} clips ({100*(d>t).mean():.1f}%)")
print("  max",d.max(),"mean",d.mean(),"median",np.median(d))
print("  mass max abs diff",np.abs(ma-mb).max())
def rank_auc(yv,s):
    yv=np.asarray(yv); s=np.asarray(s,float)
    n1=int((yv==1).sum()); n0=int((yv==0).sum())
    o=np.argsort(s,kind="mergesort"); ss=s[o]; r=np.empty(len(s))
    i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        r[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    return float((r[yv==1].sum()-n1*(n1+1)/2.0)/(n1*n0))
print("\n--- AUC (rank formula) ---")
print("  shipped %.6f   pod %.6f   delta %.6f"%(rank_auc(y,pa),rank_auc(y,pb),rank_auc(y,pb)-rank_auc(y,pa)))
# duration relation
mf="manifests/pitt_conflict_manifest_468.csv"
dur={}
arm={}
if os.path.exists(mf):
    for r in csv.DictReader(open(mf)):
        k=os.path.basename(r["segment_path"])
        dur[k]=float(r["dur"]); arm[k]=r["set"]
dd=np.array([dur.get(r["clip"],np.nan) for r in a])
big=d>1e-3
print("\n--- does the difference track clip duration? ---")
print("  clips with dur available:",int(np.isfinite(dd).sum()))
for lo,hi in [(0,20),(20,30),(30,1e9)]:
    m=np.isfinite(dd)&(dd>=lo)&(dd<hi)
    if m.sum(): print(f"  dur [{lo},{hi}) n={m.sum():4d}  frac diff>1e-3: {big[m].mean():.3f}  mean|d| {d[m].mean():.5f}  max|d| {d[m].max():.5f}")
print("\n--- top 10 largest differences ---")
for i in np.argsort(-d)[:10]:
    print(f"  {a[i]['clip']:<34} dur {dur.get(a[i]['clip'],float('nan')):6.1f}s  shipped {pa[i]:.6f}  pod {pb[i]:.6f}  |d| {d[i]:.6f}")
print("\n--- correlation ---")
print("  pearson r(p_yes) %.6f"%np.corrcoef(pa,pb)[0,1])
print("  spearman r %.6f"%np.corrcoef(np.argsort(np.argsort(pa)),np.argsort(np.argsort(pb)))[0,1])
