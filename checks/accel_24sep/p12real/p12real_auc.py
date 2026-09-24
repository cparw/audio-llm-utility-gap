# p12real: independent recompute of the WRONG COPY rows (lines 125, 171, 172). Read only.
import re, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
H=os.path.expanduser('<local data dir>/release'); G='<local data dir>/release_from_mac/scores'
def f(x):
    if isinstance(x,str):
        m=re.match(r'np\.float64\((.*)\)',x); return float(m.group(1)) if m else float(x)
    return float(x)
def boot(y,s,spk):
    y=np.asarray(y); s=np.asarray(s,float); spk=np.asarray(spk).astype(str)
    u=np.unique(spk); idx={k:np.where(spk==k)[0] for k in u}
    rng=np.random.default_rng(0); out=[]; skipped=0
    for _ in range(2000):
        d=rng.choice(u,size=len(u),replace=True); ii=np.concatenate([idx[k] for k in d])
        if len(np.unique(y[ii]))<2: skipped+=1; continue
        out.append(roc_auc_score(y[ii],s[ii]))
    return np.percentile(out,2.5),np.percentile(out,97.5),skipped
def cell(name,df,ycol,scol,spkcol,ci=True):
    y=df[ycol].astype(int).values; s=df[scol].map(f).values; spk=df[spkcol].astype(str).values
    a=roc_auc_score(y,s); r=dict(cell=name,n=len(df),n_spk=len(np.unique(spk)),n_pos=int(y.sum()),auc=round(a,4))
    if ci: lo,hi,sk=boot(y,s,spk); r.update(ci_lo=round(lo,4),ci_hi=round(hi,4),skipped_draws=sk)
    return r
rows=[]
# L125 zero-shot audio (0.84)
A2=pd.read_csv(f'{G}/part23/A/A2_edaic_whole_zeroshot_perclip.csv')
rows.append(dict(line=125,tex=0.84,role='right',file=f'{G}/part23/A/A2_edaic_whole_zeroshot_perclip.csv [p_yes_whole]',**cell('edaic_zs_whole',A2,'label','p_yes_whole','pid')))
M=pd.read_csv(f'{H}/edaic_rerun/variants/o25_full_zeroshot_scores.csv')
rows.append(dict(line=125,tex=0.84,role='matcher',file=f'{H}/edaic_rerun/variants/o25_full_zeroshot_scores.csv (json: o25_full_zeroshot.json auc 0.8285)',**cell('edaic_zs_300s',M,'label','p_yes','speaker')))
# L125 fine-tuned audio (0.86)
FT=pd.read_csv(f'{G}/part24/FT/FT24_whole_seed0_perclip.csv')
rows.append(dict(line=125,tex=0.86,role='right',file=f'{G}/part24/FT/FT24_whole_seed0_perclip.csv [p_yes]',**cell('edaic_ft_whole_s0',FT,'label','p_yes','pid')))
S=pd.read_csv(f'{H}/edaic_rerun/variants/sft_full_oof.csv')
rows.append(dict(line=125,tex=0.86,role='matcher',file=f'{H}/edaic_rerun/variants/sft_full_oof.csv (json: sft_full_sft.json auc_oof 0.6421)',**cell('edaic_sft_30s',S,'label','p_yes','speaker')))
# L171 AF2 Alzheimer's conflict 0.58, agree 0.53
P4=f'{G}/part20/POD4c'
m2=pd.read_csv('<local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv')
m2['arm']=m2.clip_path.str.extract(r'/(conflict|agreement)_')[0]
for arm,tex in [('conflict',0.58),('agreement',0.53)]:
    r=pd.read_csv(f'{P4}/af2_orig30_{arm}.csv')
    rows.append(dict(line=171,tex=tex,role='right',file=f'{P4}/af2_orig30_{arm}.csv [p_yes]',**cell(f'af2_{arm}_orig30',r,'label','p_yes','speaker_id')))
    rows.append(dict(line=171,tex=tex,role='matcher',file='<local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv [p_yes, arm from clip_path]',**cell(f'af2_{arm}_pitt468',m2[m2.arm==arm],'label','p_yes','speaker_id')))
# L172 AF3 Alzheimer's conflict 0.61, agree 0.59
p10=pd.read_csv(f'{H}/overnight2/part10/af3_pitt.csv'); p3=pd.read_csv(f'{H}/overnight2/part3/af3_pitt.csv')
for d in (p10,p3): d['arm']=d.clip_path.str.extract(r'/(conflict|agreement)_')[0]
for arm,tex in [('conflict',0.61),('agreement',0.59)]:
    rows.append(dict(line=172,tex=tex,role='right',file=f'{H}/overnight2/part10/af3_pitt.csv [p_yes, arm from clip_path]',**cell(f'af3_{arm}_part10',p10[p10.arm==arm],'label','p_yes','speaker_id')))
    rows.append(dict(line=172,tex=tex,role='matcher',file=f'{H}/overnight2/part3/af3_pitt.csv [p_yes, arm from clip_path]',**cell(f'af3_{arm}_part3',p3[p3.arm==arm],'label','p_yes','speaker_id')))
out=pd.DataFrame(rows); out['rounds_to_tex']=(out.auc.round(2)==out.tex)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print(out.drop(columns=['file']).to_string(index=False))
out.to_csv('checks/accel_24sep/p12real/p12real_auc.csv',index=False)
# clip-set checks
print('A2 vs FT pid sets equal:', set(A2.pid)==set(FT.pid), 'labels agree:', (A2.set_index('pid').label.sort_index()==FT.set_index('pid').label.sort_index()).all())
for arm in ('conflict','agreement'):
    r=pd.read_csv(f'{P4}/af2_orig30_{arm}.csv'); a=set(r.orig_clip_id); b=set(m2[m2.arm==arm].clip_path.str.split('/').str[-1]); c=set(p10[p10.arm==arm].clip_path.str.split('/').str[-1])
    print(arm,'orig30 clips == af2_pitt468 clips:',a==b,'== af3 part10 clips:',a==c, len(a),len(b),len(c))
print('arm NaN counts', m2.arm.isna().sum(), p10.arm.isna().sum(), p3.arm.isna().sum())
