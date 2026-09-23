"""Mac-side third check of the 3b cosine from the 10 lm_head rows only (fp16 dump, as POD3 used) plus the per-clip
log-sum-exp saved by the pod (the full-vocab softmax normaliser). Folds: the Linux pod folds (sklearn 1.9.1)."""
import numpy as np, csv, json, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
G="scores/leftovers_23sep/pod3b/lo-pod3b-1/"
v=np.load(G+"pod3b_primary_fp16_vectors.npz")
z=np.load("<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_pitt_states.npz",allow_pickle=True)
X=z["ans"][:,-1,:].astype(np.float64); y=z["label"].astype(int); nm=z["name"].astype(str)
fold={r["name"]:int(r["fold_linux"]) for r in csv.DictReader(open(G+"pod3b_primary_fp16_perclip.csv")) if r["arm"]=="all"}
F=np.array([fold[n] for n in nm])
Wy=v["W_yes"].astype(np.float64); Wn=v["W_no"].astype(np.float64); lse=v["lse"]
py=np.exp(X@Wy.T-lse[:,None]); pn=np.exp(X@Wn.T-lse[:,None])
wy=py.mean(0); wn=pn.mean(0); wy/=wy.sum(); wn/=wn.sum()
d=Wy.T@wy-Wn.T@wn; d/=np.linalg.norm(d)
ws=[]
for k in range(5):
    tr=F!=k; sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr]),y[tr]); ws.append(lr.coef_[0]/sc.scale_)
w=np.mean(ws,0); w/=np.linalg.norm(w)
out={"cosine_mac_from_rows":float(w@d),"d_vs_pod_dvec_maxabs":float(np.abs(d-v["all_dvec"]).max()),"w_raw_vs_pod_maxabs":float(np.abs(w-v["all_w_raw"]).max())}
print(out); json.dump(out,open("mac_cosine_from_rows.json","w"),indent=1)
