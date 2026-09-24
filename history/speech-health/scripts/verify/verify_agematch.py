import csv, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R="/project2/msoleyma_946/speech_health/results_chaitanya"
print("=== C3: SPEAKER-DISJOINT ASSERTION (all datasets) ===")
for ds in ["kcl","italian","neurovoz","pcgita","dcaps"]:
    d=np.load(f"{R}/features/qwen2/{ds}/encoder_features.npz",allow_pickle=True)
    spk=d["speaker"]; y=d["label"].astype(int); ns=min(5,len(set(spk)))
    overlap=0
    for tr,te in GroupKFold(ns).split(np.zeros((len(y),1)),y,spk):
        overlap+=len(set(spk[tr]) & set(spk[te]))
    assert overlap==0, f"{ds} SPEAKER LEAK"
    print(f"  {ds}: PASS, 0 speaker overlap across {ns} folds, {len(set(spk))} speakers")
print()
print("=== C5: AGE-MATCHED NEUROVOZ (read) ===")
man={r["speaker_id"]:r for r in csv.DictReader(open(f"{R}/manifests/neurovoz.csv"))}
d=np.load(f"{R}/features/qwen2/neurovoz/encoder_features.npz",allow_pickle=True)
task=d["task"]; spk=d["speaker"]; y=d["label"].astype(int)
def ageof(s):
    try: return float(man.get(s,{}).get("age",""))
    except Exception: return np.nan
ages=np.array([ageof(s) for s in spk]); rd=(task=="read")
pd_a=ages[rd&(y==1)]; lo,hi=np.nanpercentile(pd_a[~np.isnan(pd_a)],[5,95])
keep=rd & ((y==1)|((y==0)&(ages>=lo)&(ages<=hi)))
idx=np.where(keep & ~np.isnan(ages))[0]
print(f"  PD age range {np.nanmin(pd_a):.0f}-{np.nanmax(pd_a):.0f}; HC restricted to [{lo:.0f},{hi:.0f}]; n={len(idx)} clips, {len(set(spk[idx]))} speakers, PD={int(y[idx].sum())} HC={int((y[idx]==0).sum())}")
best=(-1,-1.0); rows=[]
for li in range(int(d["n_layers"])):
    X=d[f"layer_{li:02d}"][idx]; yy=y[idx]; gg=spk[idx]; aucs=[]
    for tr,te in GroupKFold(min(5,len(set(gg)))).split(X,yy,gg):
        if len(set(yy[tr]))<2 or len(set(yy[te]))<2: continue
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
        clf.fit(X[tr],yy[tr])
        try: aucs.append(roc_auc_score(yy[te],clf.predict_proba(X[te])[:,1]))
        except Exception: pass
    a=float(np.mean(aucs)) if aucs else float("nan"); rows.append((li,a))
    if a==a and a>best[1]: best=(li,a)
with open(f"{R}/probing/neurovoz_read_agematched.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["layer","auc"]); w.writerows([(l,round(a,3)) for l,a in rows])
print(f"  neurovoz age-matched read: peak layer {best[0]} AUC {best[1]:.3f} (vs all-HC 0.948) -> neurovoz_read_agematched.csv")
