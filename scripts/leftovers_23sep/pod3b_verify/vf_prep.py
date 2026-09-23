# Verifier prep (own code). Builds the small input bundle for the pods from the original POD3 files.
import numpy as np, csv, os, json, hashlib, sklearn, platform
from sklearn.model_selection import GroupKFold
S="<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_pitt_states.npz"
P="<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_perp_cols.npz"
M="<local data dir>/release_from_mac/session_scratch/p16/mf_pitt468_pod.csv"
z=np.load(S,allow_pickle=True)
X=np.ascontiguousarray(z["ans"][:,-1,:]).astype(np.float32)
y=z["label"].astype(np.int64); spk=z["spk"].astype(str); name=z["name"].astype(str); pyes=z["p_yes"].astype(np.float64)
setof={}
for r in csv.DictReader(open(M)): setof[os.path.basename(r["path"])]=r["set"]
arm=np.array([setof[n+".wav"] if n+".wav" in setof else setof[n] for n in name])
pref=np.array([n.split("_")[0] for n in name])
g=np.load(P,allow_pickle=True)
assert (g["name"].astype(str)==name).all() and (g["spk"].astype(str)==spk).all() and (g["label"]==y).all()
out={}
# Mac folds: real sklearn GroupKFold on this Mac, per arm
folds={}
for a in ("all","conflict","agreement"):
    m=np.ones(len(y),bool) if a=="all" else arm==a
    f=np.full(m.sum(),-1)
    for k,(tr,te) in enumerate(GroupKFold(n_splits=5).split(np.zeros((m.sum(),1)),y[m],groups=spk[m])): f[te]=k
    folds[a]=f
# compare with task's mac_folds.npz as a sanity check only
T="scores/leftovers_23sep/pod3b/mac/mac_folds.npz"
t=np.load(T,allow_pickle=True)
cmp={a:bool((t[a].astype(int)==folds[a]).all()) for a in folds}
u,cnt=np.unique(spk,return_counts=True)
np.savez("mac/vf_inputs.npz",X=X,y=y,spk=spk,name=name,arm=arm,p_yes=pyes,
  pod3_oof=g["oof"],pod3_proj_d=g["proj_d"],pod3_perp_z=g["perp_z"],pod3_perp_trainz=g["perp_trainz"],
  macfold_all=folds["all"],macfold_conflict=folds["conflict"],macfold_agreement=folds["agreement"])
info={"n":int(len(y)),"n_spk":int(len(u)),"distinct_clip_counts":sorted(set(cnt.tolist())),"n_distinct_counts":len(set(cnt.tolist())),
 "n_spk_with_1clip":int((cnt==1).sum()),"arm_counts":{a:int((arm==a).sum()) for a in ("conflict","agreement")},
 "arm_eq_prefix":int((arm==pref).sum()),"n_pos":int(y.sum()),
 "mac_folds_equal_task_mac_folds":cmp,"sklearn":sklearn.__version__,"numpy":np.__version__,"machine":platform.machine(),
 "X_sha256":hashlib.sha256(X.tobytes()).hexdigest(),
 "argsort_default_mac_all":np.argsort(cnt).tolist()}
json.dump(info,open("mac/vf_prep.json","w"),indent=1)
print({k:v for k,v in info.items() if k!="argsort_default_mac_all"})
