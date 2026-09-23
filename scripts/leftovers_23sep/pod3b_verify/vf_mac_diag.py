# Mac diagnostic (own code): Mac sklearn 1.7.2 fit on (a) the sklearn>=1.9 stable-rule folds, (b) the Mac 1.7.2 folds.
import sys, json, numpy as np, sklearn, platform
sys.path.insert(0,"scripts")
from vf_common import greedy_groupkfold, fit_folds, auc, boot
Z=np.load("mac/vf_inputs.npz",allow_pickle=True)
X=Z["X"].astype(np.float64); y=Z["y"]; spk=Z["spk"].astype(str); arm=Z["arm"].astype(str)
R={"sklearn":sklearn.__version__,"numpy":np.__version__,"machine":platform.machine()}
dd=np.zeros(X.shape[1]); dd[0]=1.0
for a in ("all","conflict","agreement"):
    m=np.ones(len(y),bool) if a=="all" else arm==a
    fl=greedy_groupkfold(spk[m]); fm=Z[f"macfold_{a}"].astype(int)
    oL,_,_,_=fit_folds(X[m],y[m],fl,dd); oM,_,_,_=fit_folds(X[m],y[m],fm,dd)
    r={"auc_macfit_linuxfolds":auc(y[m],oL),"auc_macfit_macfolds":auc(y[m],oM)}
    if a=="all":
        r["oof_macfit_linuxfolds_vs_pod3_maxabs"]=float(np.abs(oL-Z["pod3_oof"]).max())
        np.save("mac/vf_mac_oof_linuxfolds_all.npy",oL)
        r["ci_macfit_macfolds"]=boot(y[m],spk[m],{"p":oM})["p"]
    R[a]=r; print(a,r,flush=True)
json.dump(R,open("mac/vf_mac_diag.json","w"),indent=1)
