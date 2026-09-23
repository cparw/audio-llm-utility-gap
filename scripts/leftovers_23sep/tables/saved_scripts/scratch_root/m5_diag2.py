import os, json, csv, time
os.environ.setdefault("OMP_NUM_THREADS","4")
import numpy as np
np.seterr(all="ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
exec(open('<local data dir>/scratch/m5_diag.py').read().split('their={r["clip"]')[0])   # reuse loaders only (my own code)

their={r["clip"]:r for r in csv.DictReader(open("<local data dir>/release/edaic_rerun/part16/M5_readout_direction_q2a.csv"))}
tz=np.array([float(their[x]["z_probe_oof_without_d_all"]) for x in name])
def auc(yv,s):
    a=s[yv==1]; b=s[yv==0]
    return float((np.count_nonzero(a[:,None]>b[None,:])+0.5*np.count_nonzero(a[:,None]==b[None,:]))/(a.size*b.size))

def run_with_d(dv,label):
    dv=dv/np.linalg.norm(dv)
    out={k:np.zeros(len(y)) for k in ["test_z","train_z0","train_z1"]}
    for tr,te in GroupKFold(5).split(S,y,groups=spk):
        sc=StandardScaler().fit(S[tr])
        lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(S[tr]),y[tr])
        wf=lr.coef_.ravel()/sc.scale_
        wp=wf-np.dot(wf,dv)*dv
        b=S[te]@wp; btr=S[tr]@wp
        out["test_z"][te]=(b-b.mean())/b.std()
        out["train_z0"][te]=(b-btr.mean())/btr.std()
        out["train_z1"][te]=(b-btr.mean())/btr.std(ddof=1)
    for k,v in out.items():
        print(f"  {label:<16} {k:<9} auc={auc(y,v):.10f}  max|diff vs their col|={np.abs(v-tz).max():.3e}")
    return out

print("THEIR auc_without_d = 0.8326811313444401 ; THEIR col is the reference")
run_with_d(d,"d_12logit")

# full-vocab mass weights, streamed
t=time.time()
with open(SHARD,"rb") as fh:
    hl=int.from_bytes(fh.read(8),"little"); hdr=json.loads(fh.read(hl)); base=8+hl
info=hdr["language_model.lm_head.weight"]; nr,nc=info["shape"]; st=base+info["data_offsets"][0]
mx=np.full(len(S),-np.inf); 
CH=8192
with open(SHARD,"rb") as fh:
    for s0 in range(0,nr,CH):
        s1=min(nr,s0+CH); fh.seek(st+s0*nc*2)
        raw=np.frombuffer(fh.read((s1-s0)*nc*2),dtype="<u2").astype(np.uint32)
        W=((raw<<np.uint32(16)).astype(np.uint32)).view(np.float32).reshape(s1-s0,nc).astype(np.float32)
        L=(S.astype(np.float32)@W.T)
        mx=np.maximum(mx,L.max(1))
acc=np.zeros(len(S))
with open(SHARD,"rb") as fh:
    for s0 in range(0,nr,CH):
        s1=min(nr,s0+CH); fh.seek(st+s0*nc*2)
        raw=np.frombuffer(fh.read((s1-s0)*nc*2),dtype="<u2").astype(np.uint32)
        W=((raw<<np.uint32(16)).astype(np.uint32)).view(np.float32).reshape(s1-s0,nc).astype(np.float32)
        L=(S.astype(np.float32)@W.T).astype(np.float64)
        acc+=np.exp(L-mx[:,None]).sum(1)
logZ=mx+np.log(acc)
Ly=S@Wy.T; Ln=S@Wn.T
py=np.exp(Ly-logZ[:,None]); pn=np.exp(Ln-logZ[:,None])
mpy=py.mean(0); mpn=pn.mean(0)
wyf=mpy/mpy.sum(); wnf=mpn/mpn.sum()
d_full=wyf@Wy-wnf@Wn
print(f"\nfull-vocab weights yes {np.round(wyf,8).tolist()}")
print(f"full-vocab weights no  {np.round(wnf,8).tolist()}")
print(f"cos(d_12logit, d_fullvocab) = {float(np.dot(d,d_full)/(np.linalg.norm(d)*np.linalg.norm(d_full))):.12f}  ({time.time()-t:.1f}s)")
run_with_d(d_full,"d_fullvocab")

print("\n--- random-direction floor: seed / shape sensitivity (dim 4096) ---")
for seed in [0,1,2,3,4]:
    rng=np.random.default_rng(seed); G=rng.standard_normal((2000,4096))
    c=np.abs((G@d)/(np.linalg.norm(G,axis=1)*np.linalg.norm(d)))
    print(f"  seed={seed} draws=2000  mean={c.mean():.6f}  p2.5={np.percentile(c,2.5):.6f}  p97.5={np.percentile(c,97.5):.6f}")
rng=np.random.default_rng(0); G=rng.standard_normal((1000,4096))
c=np.abs((G@d)/(np.linalg.norm(G,axis=1)*np.linalg.norm(d)))
print(f"  seed=0 draws=1000 (part10 convention)  mean={c.mean():.6f}  p2.5={np.percentile(c,2.5):.6f}  p97.5={np.percentile(c,97.5):.6f}")
print(f"  theory: mean={np.sqrt(2/np.pi)/64:.6f}  p2.5={0.031334/64:.6f}  p97.5={2.241403/64:.6f}")
