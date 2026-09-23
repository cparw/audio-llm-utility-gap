# Independent recompute of the triage REAL and file-copy rows (read only; nothing written under omni_final).
import numpy as np, pandas as pd, os
from sklearn.metrics import roc_auc_score as A
R="<local data dir>/release"
def boot(df, spk, fn, B=2000):
    rng=np.random.default_rng(0)
    sp=np.array(sorted(df[spk].astype(str).unique()))
    groups={s:np.where(df[spk].astype(str).values==s)[0] for s in sp}
    vals=[]
    for _ in range(B):
        idx=rng.choice(len(sp), size=len(sp), replace=True)
        rows=np.concatenate([groups[sp[i]] for i in idx])
        d=df.iloc[rows]
        if d.label.nunique()<2: continue
        vals.append(fn(d))
    return fn(df), np.percentile(vals,2.5), np.percentile(vals,97.5), len(vals), len(df), len(sp)
def show(tag,r): print(f"{tag}: {r[0]:.4f} [{r[1]:.4f}, {r[2]:.4f}] n={r[4]} spk={r[5]} draws={r[3]}")
def paired(ft, zs, key_ft, key_zs):
    a=pd.read_csv(ft); b=pd.read_csv(zs)
    a["k"]=a[key_ft].astype(str).map(os.path.basename); b["k"]=b[key_zs].astype(str).map(os.path.basename)
    m=a[["k","speaker","label","p_yes"]].merge(b[["k","label","p_yes"]], on="k", suffixes=("_ft","_zs"))
    assert (m.label_ft==m.label_zs).all() and len(m)==len(a)==len(b)
    m["label"]=m.label_ft
    return boot(m,"speaker",lambda d: A(d.label,d.p_yes_ft)-A(d.label,d.p_yes_zs))
OF=f"{R}/omni_final"
show("gain E-DAIC 30 s", paired(f"{OF}/omnisft_edaic_oof.csv", f"{OF}/omni_edaic_zeroshot_scores.csv","path","clip"))
show("gain E-DAIC full (variants)", paired(f"{R}/edaic_rerun/variants/sft_full_oof.csv", f"{R}/edaic_rerun/variants/o25_full_zeroshot_scores.csv","path","clip"))
for ds in ("pitt","adresso","adress2020"):
    show(f"gain {ds}", paired(f"{OF}/omnisft_{ds}_oof.csv", f"{OF}/omni_{ds}_zeroshot_scores.csv","path","clip"))
seg=pd.read_csv(f"{R}/edaic_rerun/pitt_all_segments.csv", dtype={"spk":str})
print("orig_set values:", seg.orig_set.fillna("NA").value_counts().to_dict(), "tertile:", seg.tertile.value_counts().to_dict())
raw=A(seg.label,seg.rate); print("rate raw 708", round(raw,4))
show("rate alone 708 (1-AUC)", boot(seg,"spk",lambda d: 1-A(d.label,d.rate)))
for c in ["orig_set"]:
    kept=seg[seg[c].notna()]
    if len(kept)==468:
        print("kept by orig_set notna:", len(kept)); show("rate alone 468 (1-AUC)", boot(kept,"spk",lambda d: 1-A(d.label,d.rate)))
for tag in ("part3","part10"):
    d=pd.read_csv(f"{R}/overnight2/{tag}/af3_pitt.csv")
    d["arm"]=d.clip_path.map(os.path.basename).str.split("_").str[0]
    for arm,dd in d.groupby("arm"):
        show(f"AF3 Pitt {tag} {arm}", boot(dd.reset_index(drop=True),"speaker_id",lambda x: A(x.label,x.p_yes)))
print("--- speaker = grp + spk")
seg["sid2"]=seg.grp.astype(str)+"_"+seg.spk.astype(str)
show("rate alone 708 (1-AUC)", boot(seg,"sid2",lambda d: 1-A(d.label,d.rate)))
kept=seg[seg.orig_set.isin(["agreement","conflict"])].reset_index(drop=True)
print("kept raw", round(A(kept.label,kept.rate),4))
show("rate alone 468 (1-AUC)", boot(kept,"sid2",lambda d: 1-A(d.label,d.rate)))
