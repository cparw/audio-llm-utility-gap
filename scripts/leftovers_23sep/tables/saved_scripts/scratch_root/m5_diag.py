import os, json, csv
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np
np.seterr(all="ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

NPZ="<local data dir>/paper1_local_runs/probe2/pitt_states.npz"
SNAP="<local data dir>/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a"
SHARD=os.path.join(SNAP,"model-00005-of-00005.safetensors")
z=np.load(NPZ,allow_pickle=True)
name=z["name"].astype(str); y=z["label"].astype(int); spk=z["spk"].astype(str)
S=z["ans"][:,-1,:].astype(np.float64)

tok=json.load(open(os.path.join(SNAP,"tokenizer.json"))); V=tok["model"]["vocab"]
ids=lambda ws: sorted({V[f] for w in ws for f in (w,"Ġ"+w) if f in V})
yes_ids=ids(["Yes","yes","YES"]); no_ids=ids(["No","no","NO"])
def rows(rs):
    with open(SHARD,"rb") as fh:
        hl=int.from_bytes(fh.read(8),"little"); hdr=json.loads(fh.read(hl)); base=8+hl
        info=hdr["language_model.lm_head.weight"]; nr,nc=info["shape"]; st=base+info["data_offsets"][0]
        out=np.empty((len(rs),nc))
        for j,i in enumerate(rs):
            fh.seek(st+i*nc*2)
            raw=np.frombuffer(fh.read(nc*2),dtype="<u2").astype(np.uint32)
            out[j]=((raw<<np.uint32(16)).astype(np.uint32)).view(np.float32).astype(np.float64)
    return out
Wy=rows(yes_ids); Wn=rows(no_ids)
both=np.concatenate([S@Wy.T,S@Wn.T],1); m=both.max(1,keepdims=True); e=np.exp(both-m); P=e/e.sum(1,keepdims=True)
mp=P.mean(0); ny=len(yes_ids)
wy=mp[:ny]/mp[:ny].sum(); wn=mp[ny:]/mp[ny:].sum()
d=wy@Wy-wn@Wn; d=d/np.linalg.norm(d)

their={r["clip"]:r for r in csv.DictReader(open("<local data dir>/release/edaic_rerun/part16/M5_readout_direction_q2a.csv"))}
tz=np.array([float(their[x]["z_probe_oof_without_d_all"]) for x in name])

def auc(yv,s):
    a=s[yv==1]; b=s[yv==0]
    return float((np.count_nonzero(a[:,None]>b[None,:])+0.5*np.count_nonzero(a[:,None]==b[None,:]))/(a.size*b.size))

variants={k:np.zeros(len(y)) for k in
  ["A_raw_perp_z","B_ddof1","C_stdspace","D_std_coef_raw_d","E_train_z","F_raw_nopool","G_perp_proba","H_keep_z"]}
for tr,te in GroupKFold(5).split(S,y,groups=spk):
    sc=StandardScaler().fit(S[tr])
    lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(S[tr]),y[tr])
    wf=lr.coef_.ravel()/sc.scale_
    cz=lr.coef_.ravel()
    zt=sc.transform(S[te])
    # A
    wp=wf-np.dot(wf,d)*d
    b=S[te]@wp; variants["A_raw_perp_z"][te]=(b-b.mean())/b.std()
    variants["B_ddof1"][te]=(b-b.mean())/b.std(ddof=1)
    variants["F_raw_nopool"][te]=b
    # C: project in standardized space, d mapped as d*scale
    dz=d*sc.scale_; dz=dz/np.linalg.norm(dz)
    cp=cz-np.dot(cz,dz)*dz
    bc=zt@cp; variants["C_stdspace"][te]=(bc-bc.mean())/bc.std()
    # D: standardized coef minus raw-d projection (dimensional mismatch bug)
    cp2=cz-np.dot(cz,d)*d
    bd=zt@cp2; variants["D_std_coef_raw_d"][te]=(bd-bd.mean())/bd.std()
    # E: z using train fold stats
    btr=S[tr]@wp
    variants["E_train_z"][te]=(b-btr.mean())/btr.std()
    # G: sigmoid of perp score + intercept
    variants["G_perp_proba"][te]=1/(1+np.exp(-(b+lr.intercept_[0])))
    # H: keep d (no removal), z per fold  -> control
    a=S[te]@wf; variants["H_keep_z"][te]=(a-a.mean())/a.std()

print("their auc_without_d (summary) = 0.8326811313444401")
for k,v in variants.items():
    md=np.abs(v-tz).max()
    # rank-correlation with their column
    from scipy.stats import spearmanr
    sp=spearmanr(v,tz).statistic
    print(f"  {k:<18} auc={auc(y,v):.10f}  max|diff vs their z col|={md:.3e}  spearman={sp:.6f}")

print("\n--- per fold: is their z an exact affine map of my raw residual b? ---")
folds=np.full(len(y),-1); braw=np.zeros(len(y)); traw=[]
k=0
for tr,te in GroupKFold(5).split(S,y,groups=spk):
    sc=StandardScaler().fit(S[tr])
    lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(S[tr]),y[tr])
    wf=lr.coef_.ravel()/sc.scale_
    wp=wf-np.dot(wf,d)*d
    braw[te]=S[te]@wp; folds[te]=k
    btr=S[tr]@wp
    traw.append((btr, S[te]@wp, te))
    k+=1
for k in range(5):
    m=folds==k
    b=braw[m]; t=tz[m]
    A=np.vstack([b,np.ones(b.size)]).T
    coef,res,rank,sv=np.linalg.lstsq(A,t,rcond=None)
    pred=A@coef
    btr,bte,te=traw[k]
    print(f" fold{k} n={m.sum()} affine fit slope={coef[0]:.8f} intercept={coef[1]:.8f} maxresid={np.abs(pred-t).max():.3e}")
    print(f"        implied center={-coef[1]/coef[0]:.6f} scale={1/coef[0]:.6f} | test mean={bte.mean():.6f} std={bte.std():.6f} std1={bte.std(ddof=1):.6f}")
    print(f"        train mean={btr.mean():.6f} std={btr.std():.6f} std1={btr.std(ddof=1):.6f}")
