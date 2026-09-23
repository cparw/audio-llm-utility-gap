import sys, os, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_auc import auc_rank, boot_paired, boot_auc          # primary
from verify_pod1 import auc_pairwise, paired_ci, ci_single   # independent
OUT="<local data dir>/Desktop/release/edaic_rerun/part16/POD1"
def f4(x): return f"{x:.4f}"
res=[]
def cmp_row(rid, what, a, b, n, nspk):
    ok = f4(a)==f4(b)
    res.append(dict(id=rid, what=what, primary=f4(a), verify=f4(b), agree=ok, n=n, n_spk=nspk))
    return ok

# --- arm AUCs + paired CIs for every scored per-clip csv ---
SCORED=[("VALIDATION","p14_thr070_check.csv","0.70 rebuild, original wording (=Table 2 audio)"),
        ("1d","p14_phq10_omni.csv","PHQ-8>=10 cutoff"),
        ("1e060","p14_thr060_omni.csv","sentiment 0.60"),
        ("1e080","p14_thr080_omni.csv","sentiment 0.80"),
        ("1b_p2","p14_o25_audio_p2.csv","prompt wording 2"),
        ("1b_p3","p14_o25_audio_p3.csv","prompt wording 3"),
        ("1c","p14_o25_sftfull.csv","sft_full retrain, OOF by speaker")]
for rid, fn, lab in SCORED:
    p=f"{OUT}/{fn}"
    if not os.path.exists(p): print(f"SKIP {rid} ({fn} not present yet)"); continue
    d=pd.read_csv(p)
    pc,pa,_=paired_ci(d.label.values,d.p_yes.values,d.speaker.values,d.arm.values,2000,0)
    vc=boot_paired(d.label.values,d.p_yes.values,d.speaker.values,d.arm.values,2000,0)
    for arm,vci,pci in (("conflict",vc["conflict"],pc),("agreement",vc["agreement"],pa)):
        s=d[d.arm==arm]
        cmp_row(f"{rid}_{arm}_auc",f"{lab} {arm} AUC",auc_rank(s.label,s.p_yes),auc_pairwise(s.label,s.p_yes),len(s),s.speaker.nunique())
        cmp_row(f"{rid}_{arm}_lo",f"{lab} {arm} CI lo",vci[0],pci[0],len(s),s.speaker.nunique())
        cmp_row(f"{rid}_{arm}_hi",f"{lab} {arm} CI hi",vci[1],pci[1],len(s),s.speaker.nunique())

# --- 1a completion: text arms, pearson, same-decision ---
man=pd.read_csv("manifests/part14_manifest_new.csv")
for tag in ("o25","q2a"):
    p=f"{OUT}/p14_{tag}_1a_joined.csv"
    if not os.path.exists(p): continue
    d=pd.read_csv(p)
    pc,pa,_=paired_ci(d.label.values,d.p_yes_text.values,d.speaker_id.values,d.arm.values,2000,0)
    vc=boot_paired(d.label.values,d.p_yes_text.values,d.speaker_id.values,d.arm.values,2000,0)
    for arm,vci,pci in (("conflict",vc["conflict"],pc),("agreement",vc["agreement"],pa)):
        s=d[d.arm==arm]
        cmp_row(f"1a_{tag}_text_{arm}_auc",f"1a {tag} text {arm} AUC",auc_rank(s.label,s.p_yes_text),auc_pairwise(s.label,s.p_yes_text),len(s),s.speaker_id.nunique())
        cmp_row(f"1a_{tag}_text_{arm}_lo",f"1a {tag} text {arm} CI lo",vci[0],pci[0],len(s),s.speaker_id.nunique())
        cmp_row(f"1a_{tag}_text_{arm}_hi",f"1a {tag} text {arm} CI hi",vci[1],pci[1],len(s),s.speaker_id.nunique())
    x=d.p_yes_text.values.astype(float); y=d.p_yes_audio.values.astype(float)
    r1=float(np.corrcoef(x,y)[0,1])
    xm,ym=x-x.mean(),y-y.mean()
    r2=float((xm*ym).sum()/np.sqrt((xm**2).sum()*(ym**2).sum()))   # independent Pearson
    cmp_row(f"1a_{tag}_pearson",f"1a {tag} Pearson r text vs audio",r1,r2,len(d),d.speaker_id.nunique())
    s1=float(((x>0.5)==(y>0.5)).mean())
    s2=float(np.count_nonzero((x>0.5).astype(int)==(y>0.5).astype(int))/len(x))
    cmp_row(f"1a_{tag}_same_decision",f"1a {tag} same-decision share",s1,s2,len(d),d.speaker_id.nunique())

# --- 1f ---
p=f"{OUT}/p14_o25_greedy.csv"
if os.path.exists(p):
    g=pd.read_csv(p)
    fw=g.first_word.fillna("").astype(str).str.replace(r"[^A-Za-z]","",regex=True).str.lower()
    yn=fw.isin(["yes","no"])
    cmp_row("1f_share_yesno_all","1f share first word Yes/No",float(yn.mean()),float(np.count_nonzero(yn.values)/len(g)),len(g),g.spk.nunique())
    dec=g.logit_decision.str.lower()
    a1=float((fw[yn]==dec[yn]).mean())
    a2=float(np.count_nonzero(fw.values[yn.values]==dec.values[yn.values])/int(yn.sum()))
    cmp_row("1f_agree_gen_logit_all","1f agreement generated vs logit",a1,a2,int(yn.sum()),g.spk.nunique())
    for arm in ("conflict","agreement"):
        m=(g.arm==arm).values
        s1=float(yn.values[m].mean()); s2=float(np.count_nonzero(yn.values[m])/m.sum())
        cmp_row(f"1f_share_yesno_{arm}",f"1f share Yes/No {arm}",s1,s2,int(m.sum()),g[g.arm==arm].spk.nunique())
        mm=m&yn.values
        b1=float((fw.values[mm]==dec.values[mm]).mean()); b2=float(np.count_nonzero(fw.values[mm]==dec.values[mm])/mm.sum())
        cmp_row(f"1f_agree_gen_logit_{arm}",f"1f agreement {arm}",b1,b2,int(mm.sum()),g[g.arm==arm].spk.nunique())

V=pd.DataFrame(res)
V.to_csv(f"{OUT}/POD1_VERIFY.csv",index=False)
print(V.to_string(index=False))
print(f"\nVERIFIED {V.agree.sum()}/{len(V)} values agree to 4 decimals")
bad=V[~V.agree]
if len(bad): print("\nDISAGREEMENTS:\n"+bad.to_string(index=False))
