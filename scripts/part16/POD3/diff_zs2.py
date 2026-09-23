import csv, numpy as np
SH="<local data dir>/Desktop/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"
PD="<local data dir>/scratch/pod_q3o_pitt_zs.csv"
a=list(csv.DictReader(open(SH))); b=list(csv.DictReader(open(PD)))
pa=np.array([float(r["p_yes"]) for r in a]); pb=np.array([float(r["p_yes"]) for r in b])
y=np.array([int(r["label"]) for r in a])
print("distinct p_yes values: shipped %d / %d   pod %d / %d"%(len(set(pa.round(6))),len(pa),len(set(pb.round(6))),len(pb)))
ua,ca=np.unique(pa.round(6),return_counts=True); ub,cb=np.unique(pb.round(6),return_counts=True)
print("\nshipped top levels (value, count):")
for v,c in sorted(zip(ua,ca),key=lambda t:-t[1])[:15]: print("   %.6f  x%d"%(v,c))
print("pod top levels (value, count):")
for v,c in sorted(zip(ub,cb),key=lambda t:-t[1])[:15]: print("   %.6f  x%d"%(v,c))
common=set(ua)&set(ub)
print("\nlevels shared between the two files: %d (shipped %d, pod %d)"%(len(common),len(ua),len(ub)))
print("share of shipped clips sitting on a shared level: %.3f"%np.isin(pa.round(6),list(common)).mean())
d=pb-pa
print("\nsigned diff: mean %.6f  median %.6f  frac pod>shipped %.3f"%(d.mean(),np.median(d),(d>0).mean()))
print("std of signed diff %.6f"%d.std())
# Is the difference just moving between adjacent levels of a shared ladder?
allv=np.array(sorted(set(ua)|set(ub)))
def lvl(x): return np.searchsorted(allv,x)
steps=np.array([lvl(x)-lvl(z) for x,z in zip(pb.round(6),pa.round(6))])
print("\nladder-step displacement (on the union of %d levels):"%len(allv))
u,c=np.unique(steps,return_counts=True)
for v,n in sorted(zip(u,c),key=lambda t:-t[1])[:12]: print("   step %+d : %d clips"%(v,n))
print("   frac |step|<=2: %.3f"%(np.abs(steps)<=2).mean())
print("\nlogit-gap view: log(p/(1-p))")
la=np.log(pa/(1-pa)); lb=np.log(pb/(1-pb))
print("  shipped logit mean %.4f sd %.4f | pod logit mean %.4f sd %.4f"%(la.mean(),la.std(),lb.mean(),lb.std()))
print("  mean |dlogit| %.4f"%np.abs(lb-la).mean())
print("\nyes-rate (p_yes>0.5): shipped %.4f  pod %.4f"%((pa>0.5).mean(),(pb>0.5).mean()))
print("mean p_yes: shipped %.4f  pod %.4f"%(pa.mean(),pb.mean()))
for lab in (0,1):
    m=y==lab
    print("  label %d: shipped mean %.4f  pod mean %.4f"%(lab,pa[m].mean(),pb[m].mean()))
