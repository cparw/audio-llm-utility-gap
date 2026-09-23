# independent recompute of PART 23 values from per-clip files (separate check code, not the pod report code)
import numpy as np, pandas as pd, glob, os
from sklearn.metrics import roc_auc_score
B="scores/part23"
def f(x): return float(str(x).replace('np.float64(','').replace(')',''))
def boot(y,s,s2=None,n=2000):
    rng=np.random.default_rng(0); y=np.asarray(y); N=len(y); out=[]
    for _ in range(n):
        i=rng.choice(N,size=N,replace=True)
        if len(set(y[i]))<2: continue
        a=roc_auc_score(y[i],s[i]); out.append(a-roc_auc_score(y[i],s2[i]) if s2 is not None else a)
    return np.percentile(out,[2.5,97.5])
def row(name,y,s,s2=None):
    v=roc_auc_score(y,s)-(roc_auc_score(y,s2) if s2 is not None else 0); lo,hi=boot(y,s,s2); print(f"{name}\t{v:.4f}\t{lo:.4f}\t{hi:.4f}")
A=pd.read_csv(f"{B}/A/A3_edaic_whole_meanof5_perclip.csv").sort_values('pid')
y=A.label.values; pw=A.p_yes_whole.map(f).values
Z=pd.read_csv(f"{B}/A/A2_edaic_whole_zeroshot_perclip.csv").sort_values('pid')
assert (Z.pid.values==A.pid.values).all()
p300=Z.p_yes_300s_T7b.map(f).values
row("zs_whole",y,pw); row("zs_300_T7b",y,p300); row("zs_whole_minus_300",y,pw,p300)
for st in ['enc','proj','llm','ans']:
    aucs=[roc_auc_score(y,A[f'oof_{st}_r{r}'].values) for r in range(5)]
    # mean-of-five convention: mean of per-repeat AUCs; interval = bootstrap of the mean of per-repeat AUCs
    rng=np.random.default_rng(0); M=np.stack([A[f'oof_{st}_r{r}'].values for r in range(5)]); bs=[]; bd=[]
    for _ in range(2000):
        i=rng.choice(len(y),size=len(y),replace=True)
        m=np.mean([roc_auc_score(y[i],M[r,i]) for r in range(5)]); bs.append(m); bd.append(m-roc_auc_score(y[i],pw[i]))
    print(f"{st}_meanof5\t{np.mean(aucs):.4f}\t{np.percentile(bs,2.5):.4f}\t{np.percentile(bs,97.5):.4f}\tper-rep {[round(a,4) for a in aucs]}")
    print(f"{st}_minus_zs_whole\t{np.mean(aucs)-roc_auc_score(y,pw):.4f}\t{np.percentile(bd,2.5):.4f}\t{np.percentile(bd,97.5):.4f}")
# fine-tune: pooled out-of-fold AUC over the 5 folds, whole and 300 s control
for mode in ['whole','ctrl300']:
    F=pd.concat([pd.read_csv(g) for g in sorted(glob.glob(f"{B}/FT/FT_{mode}_fold*_oof.csv"))]).sort_values('pid')
    print(f"FT_{mode}_n\t{len(F)}\tfolds {sorted(F.fold.unique())}\tper-fold {[round(roc_auc_score(g.label,g.p_yes),4) for _,g in F.groupby('fold')]}")
    row(f"FT_{mode}_pooled",F.label.values,F.p_yes.values)
    if mode=='whole': Fw=F
    M2=F.merge(Z[['pid']].assign(pw=pw),on='pid'); assert len(M2)==275
    row(f"FT_{mode}_minus_zs_whole",M2.label.values,M2.p_yes.values,M2.pw.values)
Fc=pd.concat([pd.read_csv(g) for g in sorted(glob.glob(f"{B}/FT/FT_ctrl300_fold*_oof.csv"))]).sort_values('pid')
M3=Fw.merge(Fc[['pid','p_yes']],on='pid',suffixes=('','_c')); row("FT_whole_minus_ctrl300",M3.label.values,M3.p_yes.values,M3.p_yes_c.values)
print("FT_whole_mass_min\t%.4f\tfold3_mass_median %.4f"%(Fw.answer_mass.min(),Fw[Fw.fold==3].answer_mass.median()))
