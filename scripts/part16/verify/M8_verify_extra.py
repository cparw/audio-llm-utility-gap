"""Supplementary independent checks for M8: third AUC route (scipy/sklearn),
duplicate-clip sensitivity in the Set A agreement arm, and full-set AUC
cross-check against master_lookup.csv."""
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

REL="<local data dir>/Desktop/release"
a=pd.read_csv(f"{REL}/edaic_rerun/part14_manifest_new.csv")
akey=(a['set'].astype(str)+'_'+a['speaker_id'].astype(str)+'_'+a['seg_uid'].astype(str)+'.wav')

def auc_u(y,s):
    y=np.asarray(y); s=np.asarray(s,float)
    p=s[y==1]; n=s[y==0]
    ns=np.sort(n)
    lo=np.searchsorted(ns,p,'left'); hi=np.searchsorted(ns,p,'right')
    return (lo.sum()+0.5*(hi-lo).sum())/(p.size*n.size)

SRC_A={'Qwen2.5-Omni':'edaic_rerun/part14/p14_o25_zeroshot_scores.csv',
 'Qwen2-Audio':'edaic_rerun/part14/p14_q2a_zeroshot_scores.csv',
 'Qwen3-Omni-30B-A3B':'edaic_rerun/part14/p14_q3o_zeroshot_scores.csv',
 'Audio Flamingo 2':'edaic_rerun/part14/p14_af2.csv',
 'Audio Flamingo 3':'edaic_rerun/part14/p14_af3.csv',
 'Kimi-Audio':'edaic_rerun/part14/p14_kimi_zeroshot_scores.csv'}

print("THIRD ROUTE (scipy mannwhitneyu + sklearn roc_auc_score) vs my U-count, Set A")
print(f"{'model':<20}{'arm':<12}{'ucount':>9}{'scipy':>9}{'sklearn':>9}")
for m,f in SRC_A.items():
    s=pd.read_csv(f"{REL}/{f}")
    py=s['p_yes'].values
    for arm in ('conflict','agreement'):
        msk=(a['set'].values==arm)
        y=a['label'].values[msk]; sc=py[msk]
        u=mannwhitneyu(sc[y==1],sc[y==0],alternative='two-sided').statistic/((y==1).sum()*(y==0).sum())
        print(f"{m:<20}{arm:<12}{auc_u(y,sc):>9.4f}{u:>9.4f}{roc_auc_score(y,sc):>9.4f}")

print()
print("DUPLICATE-CLIP SENSITIVITY, Set A")
dup=akey.duplicated(keep=False)
print("rows whose clip key repeats:", int(dup.sum()),
      " by arm:", a.loc[dup.values,'set'].value_counts().to_dict())
for arm in ('conflict','agreement'):
    msk=(a['set'].values==arm)
    print(f"  {arm}: rows {msk.sum()}, unique clips {akey[msk].nunique()}")
print()
print(f"{'model':<20}{'agr_all483':>12}{'agr_dedup':>12}{'n_dedup':>9}{'conf_all':>10}{'diff_all':>10}{'diff_dedup':>12}")
for m,f in SRC_A.items():
    s=pd.read_csv(f"{REL}/{f}"); py=s['p_yes'].values
    mc=(a['set'].values=='conflict'); ma=(a['set'].values=='agreement')
    conf=auc_u(a['label'].values[mc],py[mc])
    agr=auc_u(a['label'].values[ma],py[ma])
    d=pd.DataFrame({'k':akey[ma].values,'y':a['label'].values[ma],'s':py[ma]}).drop_duplicates('k')
    agrd=auc_u(d['y'].values,d['s'].values)
    print(f"{m:<20}{agr:>12.4f}{agrd:>12.4f}{len(d):>9}{conf:>10.4f}{conf-agr:>10.4f}{conf-agrd:>12.4f}")

print()
print("FULL-468 AUC cross-check vs master_lookup.csv (AF2 pitt = 0.5724)")
b=pd.read_csv("manifests/pitt_conflict_manifest_468.csv",dtype={'spk':str})
af2=pd.read_csv("<local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv")
print("  af2_pitt468 full-468 AUC  = %.4f"%auc_u(b['label'].values,af2['p_yes'].values))
af2b=pd.read_csv("<local data dir>/paper1_local_runs/af2_pitt_audio.csv")
print("  af2_pitt_audio full-468   = %.4f"%auc_u(b['label'].values,af2b['p_yes'].values))
