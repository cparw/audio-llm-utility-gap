import numpy as np, pandas as pd
F='/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun/confounds/clip_features.csv'
OUT='/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun/confounds/confound_auc.csv'
FEATS=['noise_floor_db','snr_db','quiet_centroid_hz','quiet_rolloff85_hz','quiet_tilt_db_per_khz','dur_s','loudness_db','sample_rate']
def auc(s,y):
    s=np.asarray(s,dtype=float); y=np.asarray(y,dtype=int)
    ok=~np.isnan(s); s=s[ok]; y=y[ok]
    n1=(y==1).sum(); n0=(y==0).sum()
    if n1==0 or n0==0 or len(np.unique(s))<2: return np.nan,n1,n0
    r=pd.Series(s).rank().values
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0), n1, n0
d=pd.read_csv(F)
d=d[d.err.fillna('')=='']
rows=[]
for ds,g in d.groupby('dataset'):
    sp=g.groupby(['speaker','label'])[FEATS].mean().reset_index()
    for f in FEATS:
        a,n1,n0=auc(sp[f],sp['label'])
        nun=sp[f].nunique()
        rows.append(dict(dataset=ds,feature=f,auc=a,auc_rev=(np.nan if np.isnan(a) else 1-a),
                         strength=(np.nan if np.isnan(a) else max(a,1-a)),
                         n_speakers=len(sp),n_pos=n1,n_neg=n0,n_clips=len(g),n_unique=nun,
                         mean_pos=sp.loc[sp.label==1,f].mean(),mean_neg=sp.loc[sp.label==0,f].mean()))
r=pd.DataFrame(rows)
r['FLAG']=np.where(r.strength.isna(),'NA',np.where(r.strength>0.65,'FLAG',''))
r=r.round(4)
r.to_csv(OUT,index=False)
pd.set_option('display.width',250)
for ds,g in r.groupby('dataset'):
    print('==',ds)
    print(g[['feature','auc','auc_rev','strength','n_speakers','n_pos','n_neg','n_clips','FLAG']].to_string(index=False))
print('\n== RANKED FLAGS')
print(r[r.FLAG=='FLAG'].sort_values('strength',ascending=False)[['dataset','feature','strength','auc','n_speakers','mean_pos','mean_neg']].to_string(index=False))
