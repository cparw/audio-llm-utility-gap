"""Run the REAL sklearn 1.7.2 GroupKFold (installed to /workspace/sk172, numpy 2.1.2, Linux x86) on the 3b clips
and save its fold ids. Then (second call, main env sklearn 1.9.1) refit the probe on those folds.
Twin for the verifier's re-implementation of the 1.7.2 rule on Linux (legacyrule_linux_*)."""
import sys, json, numpy as np
mode=sys.argv[1]
z=np.load("/workspace/q3o_pitt_states.npz",allow_pickle=True)
y_all=z["label"].astype(int); spk_all=z["spk"].astype(str); nm=z["name"].astype(str); arms=np.array([n.split("_")[0] for n in nm])
if mode=="folds":
    import sklearn; from sklearn.model_selection import GroupKFold
    out={"sklearn":sklearn.__version__,"sklearn_file":sklearn.__file__,"numpy":np.__version__}
    for arm in ("all","conflict","agreement"):
        m=np.ones(len(y_all),bool) if arm=="all" else arms==arm
        f=np.full(m.sum(),-1)
        for k,(tr,te) in enumerate(GroupKFold(n_splits=5).split(np.zeros((m.sum(),1)),groups=spk_all[m])): f[te]=k
        out[arm]=f.tolist()
    json.dump(out,open("/workspace/out/sk172_linux_folds.json","w")); print(out["sklearn"],out["sklearn_file"],flush=True)
else:
    import sklearn, warnings; warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
    X_all=z["ans"][:,-1,:].astype(np.float64); F=json.load(open("/workspace/out/sk172_linux_folds.json"))
    mac=np.load("/workspace/mac_folds.npz",allow_pickle=True)
    def auc(y,s):
        p=s[y==1]; n=s[y==0]; return float(((p[:,None]>n[None,:]).sum()+0.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
    res={"fit_sklearn":sklearn.__version__,"folds_from":F["sklearn"]+" "+F["sklearn_file"]}
    for arm in ("all","conflict","agreement"):
        m=np.ones(len(y_all),bool) if arm=="all" else arms==arm
        X=X_all[m]; y=y_all[m]; f=np.array(F[arm]); oof=np.zeros(len(y))
        for k in range(5):
            tr=f!=k; te=f==k; sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr]),y[tr]); oof[te]=lr.predict_proba(sc.transform(X[te]))[:,1]
        res[arm]={"auc_probe":auc(y,oof),"same_as_mac_folds":bool((f==mac[arm]).all())}
        print(arm,res[arm],flush=True)
    json.dump(res,open("/workspace/out/sk172_linux_refit.json","w"),indent=1)
