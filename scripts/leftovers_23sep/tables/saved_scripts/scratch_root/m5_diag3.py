import os,json,csv
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np
np.seterr(all="ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
exec(open('<local data dir>/scratch/m5_diag.py').read().split('their={r["clip"]')[0])
their={r["clip"]:r for r in csv.DictReader(open("<local data dir>/release/edaic_rerun/part16/M5_readout_direction_q2a.csv"))}
tz=np.array([float(their[x]["z_probe_oof_without_d_all"]) for x in name])
def auc(yv,s):
    a=s[yv==1]; b=s[yv==0]
    return float((np.count_nonzero(a[:,None]>b[None,:])+0.5*np.count_nonzero(a[:,None]==b[None,:]))/(a.size*b.size))
S32=S.astype(np.float32); d32=d.astype(np.float32)
res={k:np.zeros(len(y)) for k in ["f32_train_z","f32_test_z","trainscaler_obj"]}
for tr,te in GroupKFold(5).split(S,y,groups=spk):
    sc=StandardScaler().fit(S[tr]); lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(S[tr]),y[tr])
    wf=lr.coef_.ravel()/sc.scale_
    wp32=(wf.astype(np.float32)-np.float32(np.dot(wf.astype(np.float32),d32))*d32)
    b32=S32[te]@wp32; btr32=S32[tr]@wp32
    res["f32_train_z"][te]=(b32-btr32.mean())/btr32.std()
    res["f32_test_z"][te]=(b32-b32.mean())/b32.std()
    wp=wf-np.dot(wf,d)*d
    s2=StandardScaler().fit((S[tr]@wp).reshape(-1,1))
    res["trainscaler_obj"][te]=s2.transform((S[te]@wp).reshape(-1,1)).ravel()
for k,v in res.items():
    print(f"  {k:<18} auc={auc(y,v):.10f}  max|diff vs their col|={np.abs(v-tz).max():.3e}")
