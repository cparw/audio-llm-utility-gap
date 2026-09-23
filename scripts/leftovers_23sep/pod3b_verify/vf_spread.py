# Verifier spread (own code): 200 speaker-grouped greedy partitions differing only in tie order.
# usage: python3 vf_spread.py <arm> <seed_lo> <seed_hi>
import os, sys, json, csv, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
sys.path.insert(0,"/workspace")
from vf_common import *
import vf_lmh
from joblib import Parallel, delayed
a=sys.argv[1]; lo=int(sys.argv[2]); hi=int(sys.argv[3]); t0=time.time()
Z=np.load("/workspace/vf_inputs.npz",allow_pickle=True)
X=Z["X"].astype(np.float64); y=Z["y"]; spk=Z["spk"].astype(str); arm=Z["arm"].astype(str)
m=np.ones(len(y),bool) if a=="all" else arm==a
Xa=X[m]; ya=y[m]; sa=spk[m]
Wb,meta=vf_lmh.load(); W16=np.asarray(Wb,dtype=np.float32).astype(np.float16).astype(np.float32)
d,_=readout_dir(Xa,W16); del W16
print("d ready",time.time()-t0,flush=True)
u=np.unique(sa)
def one(seed):
    pr=np.random.default_rng(seed).permutation(len(u)); rank=np.empty(len(u),np.int64); rank[pr]=np.arange(len(u))
    f=greedy_groupkfold(sa,tie_rank=rank)
    oof,w,wt,wf=fit_folds(Xa,ya,f,d)
    return seed,auc(ya,oof),float(w@d),auc(ya,wt),auc(ya,wf)
res=Parallel(n_jobs=32)(delayed(one)(s) for s in range(lo,hi))
os.makedirs("/workspace/out",exist_ok=True)
with open(f"/workspace/out/vf_spread_{a}_{lo}_{hi}.csv","w",newline="") as fh:
    wr=csv.writer(fh); wr.writerow(["seed","auc_probe","cosine","auc_wo_trainz","auc_wo_foldz"])
    for r in res: wr.writerow([r[0]]+[repr(v) for v in r[1:]])
json.dump({"arm":a,"lo":lo,"hi":hi,"lmh_sha":meta["sha256_bf16_bytes"],"secs":time.time()-t0},open(f"/workspace/out/vf_spread_{a}_{lo}_{hi}.json","w"))
print("DONE",a,time.time()-t0,flush=True)
