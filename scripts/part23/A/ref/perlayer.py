import csv, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
D="/workspace/edaicfull"; OUT=f"{D}/out"
def auc_rank(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return float("nan")
    o=np.argsort(s,kind="mergesort"); r=np.empty(len(s),float); sv=s[o]; i=0
    while i<len(s):
        j=i
        while j+1<len(s) and sv[j+1]==sv[i]: j+=1
        r[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    return float((r[y==1].sum()-n1*(n1+1)/2.0)/(n1*n0))
sc=list(csv.DictReader(open(f"{OUT}/full_zeroshot_scores.csv")))
clips=[r["clip"] for r in sc]; y=np.array([int(r["label"]) for r in sc]); spk=np.array([r["speaker"] for r in sc])
S={k:[] for k in ("enc","proj","llm","ans")}
for c in clips:
    z=np.load(f"{OUT}/shards/{c}.npz")
    for k in S: S[k].append(z[k])
for k in S: S[k]=np.stack(S[k])
def layer_curve(X,L):
    o=np.zeros(len(y)); fold=[]
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=spk):
        ss=StandardScaler().fit(X[tr,L,:])
        lr=LogisticRegression(max_iter=2000,class_weight="balanced",C=1.0).fit(ss.transform(X[tr,L,:]),y[tr])
        p=lr.predict_proba(ss.transform(X[te,L,:]))[:,1]; o[te]=p; fold.append(auc_rank(y[te],p))
    return float(np.mean(fold)), float(np.std(fold)), auc_rank(y,o)
for key,ref in [("ans","o25_full_ans_perlayer.csv"),("llm","o25_full_llm_perlayer.csv"),("enc","o25_full_encoder_perlayer.csv")]:
    X=S[key]; nL=X.shape[1]
    res=Parallel(n_jobs=24,prefer="processes")(delayed(layer_curve)(X,L) for L in range(nL))
    mine_oof=np.array([r[2] for r in res]); mine_mean=np.array([r[0] for r in res])
    R=list(csv.DictReader(open(f"{D}/ref_perlayer/{ref}")))
    ref_oof=np.array([float(r["auc_oof"]) for r in R]); ref_mean=np.array([float(r["auc_mean"]) for r in R])
    with open(f"{OUT}/{key}_perlayer_rebuilt.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["stage","auc_mean","auc_std","auc_oof","ref_auc_mean","ref_auc_oof"])
        for L in range(nL): w.writerow([L,res[L][0],res[L][1],res[L][2],ref_mean[L],ref_oof[L]])
    print(f"=== {key}  n_layers {nL} ===")
    print(f"  my   peak auc_oof {mine_oof.max():.4f} @ {int(mine_oof.argmax())}   ref peak {ref_oof.max():.4f} @ {int(ref_oof.argmax())}")
    print(f"  curve pearson r (auc_oof) {np.corrcoef(mine_oof,ref_oof)[0,1]:.4f}   mean|diff| {np.abs(mine_oof-ref_oof).mean():.4f}   max|diff| {np.abs(mine_oof-ref_oof).max():.4f}")
    print(f"  curve pearson r (auc_mean){np.corrcoef(mine_mean,ref_mean)[0,1]:.4f}   mean|diff| {np.abs(mine_mean-ref_mean).mean():.4f}")
