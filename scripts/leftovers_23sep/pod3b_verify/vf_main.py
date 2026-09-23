# Verifier main (own code), run on a Linux x86 CPU pod with sklearn 1.9.1 / numpy 2.1.2 / scipy 1.18.1.
import os, sys, json, subprocess, platform, time
os.environ.setdefault("OMP_NUM_THREADS","16")
import numpy as np, sklearn, scipy
from sklearn.model_selection import GroupKFold
sys.path.insert(0,"/workspace")
from vf_common import *
import vf_lmh
t0=time.time()
Z=np.load("/workspace/vf_inputs.npz",allow_pickle=True)
X=Z["X"].astype(np.float64); y=Z["y"]; spk=Z["spk"].astype(str); arm=Z["arm"].astype(str)
Wb,meta=vf_lmh.load(); Wb=np.asarray(Wb,dtype=np.float32)
W16=Wb.astype(np.float16).astype(np.float32)
R={"env":{"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"machine":platform.machine()},
   "lmh":meta,"fp16_rows_changed":int((W16!=Wb).any(axis=1).sum()),"fp16_entries_changed":int((W16!=Wb).sum())}
print("lmh",meta["sha256_bf16_bytes"],time.time()-t0,flush=True)
# sklearn 1.7.2 GroupKFold on this Linux box, via a side install
sk172={}
try:
    subprocess.run([sys.executable,"-m","pip","install","--break-system-packages","--no-deps","-q","--target","/opt/sk172","scikit-learn==1.7.2"],check=True)
    code=r'''
import sys,json,numpy as np,sklearn
from sklearn.model_selection import GroupKFold
Z=np.load("/workspace/vf_inputs.npz",allow_pickle=True); spk=Z["spk"].astype(str); arm=Z["arm"].astype(str)
o={"sklearn":sklearn.__version__,"file":sklearn.__file__,"numpy":np.__version__}
u,gi=np.unique(spk,return_inverse=True); o["argsort_default_all"]=np.argsort(np.bincount(gi)).tolist()
for a in ("all","conflict","agreement"):
    m=np.ones(len(spk),bool) if a=="all" else arm==a
    f=np.full(m.sum(),-1)
    for k,(tr,te) in enumerate(GroupKFold(n_splits=5).split(np.zeros((m.sum(),1)),groups=spk[m])): f[te]=k
    o[a]=f.tolist()
json.dump(o,open("/workspace/out/vf_sk172_linux_folds.json","w"))
'''
    subprocess.run([sys.executable,"-c",code],check=True,env=dict(os.environ,PYTHONPATH="/opt/sk172"))
    sk172=json.load(open("/workspace/out/vf_sk172_linux_folds.json"))
    R["sk172_env"]={"sklearn":sk172["sklearn"],"file":sk172["file"],"numpy":sk172["numpy"]}
except Exception as e:
    R["sk172_error"]=repr(e)
# source check of the GroupKFold sort line
def grep_src(path):
    try:
        t=open(path).read(); i=t.find("class GroupKFold"); j=t.find("class ",i+10); seg=t[i:j]
        return [l.strip() for l in seg.splitlines() if "argsort" in l]
    except Exception as e: return repr(e)
R["src_argsort_1_9_1"]=grep_src(os.path.join(os.path.dirname(sklearn.__file__),"model_selection","_split.py"))
R["src_argsort_1_7_2"]=grep_src("/opt/sk172/sklearn/model_selection/_split.py")
try:
    subprocess.run([sys.executable,"-m","pip","install","--break-system-packages","--no-deps","-q","--target","/opt/sk180","scikit-learn==1.8.0"],check=True)
    R["src_argsort_1_8_0"]=grep_src("/opt/sk180/sklearn/model_selection/_split.py")
except Exception as e: R["src_argsort_1_8_0"]=repr(e)
try:
    R["cpu_flags_avx512f"]="avx512f" in open("/proc/cpuinfo").read()
    R["cpu_model"]=[l for l in open("/proc/cpuinfo").read().splitlines() if l.startswith("model name")][0]
except Exception as e: R["cpu"]=repr(e)
rows=[]
R["arms"]={}
for a in ("all","conflict","agreement"):
    m=np.ones(len(y),bool) if a=="all" else arm==a
    Xa=X[m]; ya=y[m]; sa=spk[m]; ix=np.where(m)[0]
    d16,pyes16=readout_dir(Xa,W16); dbf,_=readout_dir(Xa,Wb)
    fl=np.full(m.sum(),-1)
    for k,(tr,te) in enumerate(GroupKFold(n_splits=5).split(Xa,ya,groups=sa)): fl[te]=k
    mine=greedy_groupkfold(sa)
    A={"n":int(m.sum()),"n_spk":int(len(np.unique(sa))),"own_folds_equal_sklearn191":bool((mine==fl).all())}
    oof,w,wt,wf=fit_folds(Xa,ya,fl,d16)
    A["cos_fp16"]=float(w@d16); A["cos_bf16exact"]=float(w@dbf); A["cos_fp16_minus_bf16"]=float(w@d16-w@dbf)
    A["d_fp16_vs_bf16_maxabs"]=float(np.abs(d16-dbf).max())
    A["rand_floor"]=rand_floor(w)
    sd=Xa@d16
    A["auc_probe"]=auc(ya,oof); A["auc_d"]=auc(ya,sd); A["auc_wo_trainz"]=auc(ya,wt); A["auc_wo_foldz"]=auc(ya,wf)
    A["ci"]=boot(ya,sa,{"probe":oof,"d":sd,"wo_trainz":wt,"wo_foldz":wf})
    if a=="all":
        A["oof_vs_pod3_maxabs"]=float(np.abs(oof-Z["pod3_oof"]).max())
        A["projd_vs_pod3_maxabs"]=float(np.abs(sd-Z["pod3_proj_d"]).max())
        A["projd_bf16_vs_pod3_maxabs"]=float(np.abs(Xa@dbf-Z["pod3_proj_d"]).max())
        A["wo_trainz_vs_pod3_maxabs"]=float(np.abs(wt-Z["pod3_perp_trainz"]).max())
        A["wo_foldz_vs_pod3_maxabs"]=float(np.abs(wf-Z["pod3_perp_z"]).max())
        A["pyes_vs_saved_maxabs"]=float(np.abs(pyes16-Z["p_yes"]).max())
    # Mac folds (sklearn 1.7.2 on the arm64 Mac), same Linux fit
    fm=Z[f"macfold_{a}"].astype(int)
    oofm,wm,_,_=fit_folds(Xa,ya,fm,d16)
    A["macfolds_auc"]=auc(ya,oofm); A["macfolds_cos"]=float(wm@d16); A["macfolds_ci"]=boot(ya,sa,{"probe":oofm})["probe"]
    mates=lambda f:{s:frozenset(sa[f==f[np.where(sa==s)[0][0]]]) for s in np.unique(sa)}
    ml,mm=mates(fl),mates(fm); A["spk_new_foldmates_mac_vs_linux"]=int(sum(ml[s]!=mm[s] for s in ml))
    if sk172 and a in sk172:
        f7=np.array(sk172[a]); oof7,w7,_,_=fit_folds(Xa,ya,f7,d16)
        A["sk172linux_auc"]=auc(ya,oof7); m7=mates(f7)
        A["sk172linux_equal_macfolds"]=bool((f7==fm).all()); A["sk172linux_equal_191"]=bool((f7==fl).all())
        A["spk_new_foldmates_sk172linux_vs_mac"]=int(sum(m7[s]!=mm[s] for s in ml))
    for j,i in enumerate(ix):
        rows.append((a,str(Z["name"][i]),sa[j],int(ya[j]),int(fl[j]),int(fm[j]),repr(float(oof[j])),repr(float(sd[j])),repr(float(wt[j])),repr(float(wf[j])),repr(float(oofm[j]))))
    np.save(f"/workspace/out/vf_{a}_d_fp16.npy",d16); np.save(f"/workspace/out/vf_{a}_w.npy",w)
    R["arms"][a]=A; print(a,json.dumps({k:v for k,v in A.items() if k!="ci"}),A["ci"],time.time()-t0,flush=True)
np.save("/workspace/out/vf_W_yesno_rows.npy",Wb[np.r_[YES,NO]])
import csv
with open("/workspace/out/vf_main_perclip.csv","w",newline="") as fh:
    wr=csv.writer(fh); wr.writerow(["arm","name","speaker","label","fold_linux191","fold_mac172","oof","proj_d","wo_trainz","wo_foldz","oof_macfolds"]); wr.writerows(rows)
json.dump(R,open("/workspace/out/vf_main.json","w"),indent=1)
print("DONE",time.time()-t0,flush=True)
