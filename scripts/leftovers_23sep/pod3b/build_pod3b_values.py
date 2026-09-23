"""Build pod3b_values.tsv from the pulled pod outputs. Primary values come from lo-pod3b-1 (unmodified
direction_by_arm.py + pod3b_primary.py) and lo-pod3b-3 (pod3b_gap.py); verifier values from lo-pod3b-2
(pod3b_verify.py run 2) and lo-pod3b-4 (pod3b_verify_spread.py). Primary intervals are recomputed here from the
pods' per-clip columns with the leftovers bootstrap (fresh default_rng(0) per cell, idx = rng.choice(N, N, replace=True)
over sorted unique speaker ids, 2000 draws, 2.5/97.5). Bootstrap and AUC are deterministic arithmetic, not refits."""
import json, csv, numpy as np
R="<local data dir>/leftovers_23sep/pod3b/"
P1=R+"pull/lo-pod3b-1/out/"; P2=R+"pull/lo-pod3b-2/out/"; P3=R+"pull/lo-pod3b-3/out/"; P4=R+"pull/lo-pod3b-4/out/"; M=R+"mac/"
C1="audit_p16 item 2 / STATUS_LEDGER 5.3 and 6.5: 3b cosine 0.0051 had no twin (lm_head never reached the Mac)"
C2="audit_p16 item 6: POD3 cosines (3 rows) had no twin"
C3="audit_p16 item 2 / partF: 3b auc_d and auc_without_d UNVERIFIABLE ON MAC"
C4="audit_p16 item 6: POD3 by-arm without-d (2 rows) had no twin"
C5="audit_p16 item 6: POD3 by-arm probe 0.6145 and 0.9349 disagreed with the Mac verifier (0.5520, 0.9314)"
C6="STATUS_LEDGER 5.3 and 6.5 / audit_p16 left list: Mac refit 0.7864 vs 0.8307 not logged or explained"
def auc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); o=np.argsort(s,kind="mergesort"); r=np.empty(len(s)); ss=s[o]; i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        r[o[i:j+1]]=(i+j)/2+1; i=j+1
    n1=(y==1).sum(); n0=(y==0).sum(); return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def boot(y,spk,s):
    us=np.unique(spk); N=len(us); pos={u:np.flatnonzero(spk==u) for u in us}; rng=np.random.default_rng(0); v=[]
    for _ in range(2000):
        idx=rng.choice(N,size=N,replace=True); ii=np.concatenate([pos[us[k]] for k in idx])
        if len(np.unique(y[ii]))<2: continue
        v.append(auc(y[ii],s[ii]))
    return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))
def load(p): return list(csv.DictReader(open(p)))
pc=load(P1+"pod3b_primary_fp16_perclip.csv"); gp=load(P3+"pod3b_gap_perclip.csv")
pj={t:json.load(open(P1+f"pod3b_primary_{t}.json")) for t in ("fp16","bf16exact")}
pa={t:{a["arm"]:a for a in pj[t]["arms"]} for t in pj}
orig={r["arm"]:r for r in load(P1+"orig_direction_by_arm_fp16.csv")}
origb={r["arm"]:r for r in load(P1+"orig_direction_by_arm_bf16exact.csv")}
gap=json.load(open(P3+"pod3b_gap.json")); gAB={r["arm"]:r for r in gap["AB"]}
V=json.load(open(P2+"pod3b_verify.json")); V1=json.load(open(P2+"pod3b_verify_run1_legacyfolds.json"))
sk=json.load(open(P2+"sk172_linux_refit.json")); mc=json.load(open(M+"mac_crosscheck.json")); mr=json.load(open(M+"mac_cosine_from_rows.json"))
PUB={"all":dict(cos=0.0051,rand=(0.0177,0.0008,0.0498),probe=(0.8307,0.7833,0.8741),d=(0.7656,0.7088,0.8179),wo=(0.8305,0.7826,0.8737),fz=0.8276),
     "conflict":dict(cos=-0.0067,rand=(0.0177,0.0008,0.0516),probe=(0.6145,0.5021,0.7246),d=(0.6029,0.4965,0.7025),wo=(0.6196,0.5094,0.7283)),
     "agreement":dict(cos=0.0086,rand=(0.0172,0.0007,0.0480),probe=(0.9349,0.9034,0.9614),d=(0.8264,0.7741,0.8731),wo=(0.9341,0.9021,0.9604))}
NN={"all":(468,228),"conflict":(146,100),"agreement":(322,175)}
rows=[]
def r4(x): return None if x is None else (x if isinstance(x,str) else round(float(x),4))
def add(id,what,val,lo,hi,vv,vlo,vhi,pub,arm,src,vsrc,closes,note=""):
    agree = r4(val)==r4(vv) and r4(lo)==r4(vlo) and r4(hi)==r4(vhi)
    pm = "" if pub is None else ("yes" if r4(val)==r4(pub) else "NO")
    n,ns=NN.get(arm,("",""))
    fm=lambda x: (f"{x:.2e}" if 0<abs(x)<1e-3 else f"{x:.6f}") if isinstance(x,float) else x
    rows.append([id,what,fm(val),"" if lo is None else f"{lo:.4f}","" if hi is None else f"{hi:.4f}",
                 fm(vv),"" if vlo is None else f"{vlo:.4f}","" if vhi is None else f"{vhi:.4f}",
                 "yes" if agree else "NO", "" if pub is None else pub, pm, n, ns, src, vsrc, closes, note])
for arm in ("all","conflict","agreement"):
    cr=[r for r in pc if r["arm"]==arm]; y=np.array([int(r["label"]) for r in cr]); spk=np.array([r["speaker"] for r in cr])
    col=lambda c: np.array([float(r[c]) for r in cr])
    vf=V[f"fp16_{arm}"]; vb=V[f"bf16exact_{arm}"]; pf=pa["fp16"][arm]; pb=pa["bf16exact"][arm]; pub=PUB[arm]
    CL = C1 if arm=="all" else C2
    add(f"3b_{arm}_cosine","cosine of probe weight with the mass-weighted Yes minus No readout direction (lm_head as POD3 dumped it, float16)",
        pf["cosine_w_d"],None,None,vf["cosine"],None,None,pub["cos"],arm,"lo-pod3b-1 pod3b_primary_fp16.json (orig_direction_by_arm_fp16.csv prints %s)"%orig[arm]["cosine"],
        "lo-pod3b-2 pod3b_verify.json fp16_%s"%arm,CL, ("Mac check from the 10 lm_head rows + saved log-sum-exp: %.6f"%mr["cosine_mac_from_rows"]) if arm=="all" else "")
    add(f"3b_{arm}_cosine_bf16exact","same cosine with the exact bf16 lm_head (no float16 rounding)",pb["cosine_w_d"],None,None,vb["cosine"],None,None,pub["cos"],arm,
        "lo-pod3b-1 pod3b_primary_bf16exact.json (orig_direction_by_arm_bf16exact.csv prints %s)"%origb[arm]["cosine"],"lo-pod3b-2 pod3b_verify.json bf16exact_%s"%arm,CL)
    add(f"3b_{arm}_cosine_random_floor","mean |cosine| of the probe weight with 2000 random unit directions, seed 0, and its 2.5/97.5 percentiles",
        float(orig[arm]["cosine_random"]),float(orig[arm]["cos_rand_lo"]),float(orig[arm]["cos_rand_hi"]),vf["rand_mean"],vf["rand_lo"],vf["rand_hi"],pub["rand"][0],arm,
        "lo-pod3b-1 orig_direction_by_arm_fp16.csv (4 dp as the original script prints)","lo-pod3b-2 pod3b_verify.json fp16_%s"%arm,CL)
    for key,c,vkey,vci,pubk,what,cl in (("auc_probe_oof","oof","auc_probe","auc_probe_ci","probe","probe OOF AUC, GroupKFold(5) by speaker, sklearn 1.9.1 folds",C1 if arm=="all" else C5),
                                  ("auc_d","proj_d","auc_d","auc_d_ci","d","AUC of the projection on the readout direction alone",C3),
                                  ("auc_probe_without_d_trainz","perp_trainz","auc_wo_trainz","auc_wo_trainz_ci","wo","probe AUC with d removed, held-out scores standardised with the TRAINING fold (as shipped)",C3 if arm=="all" else C4),
                                  ("auc_probe_without_d_foldz","perp_foldz","auc_wo_foldz","auc_wo_foldz_ci","fz","probe AUC with d removed, standardised WITHIN the held-out fold (canonical)",C3 if arm=="all" else C4)):
        v=pf[key]; s=col(c); assert abs(auc(y,s)-v)<1e-12; lo,hi=boot(y,spk,s)
        p=pub.get(pubk); p=p[0] if isinstance(p,tuple) else p
        add(f"3b_{arm}_{key}",what,v,lo,hi,vf[vkey],vf[vci][0],vf[vci][1],p,arm,"lo-pod3b-1 pod3b_primary_fp16_perclip.csv col %s"%c,"lo-pod3b-2 pod3b_verify.json fp16_%s"%arm,cl)
# refit with the Mac's folds (the 0.7864), and the partition diagnostics
for arm in ("all","conflict","agreement"):
    cr=[r for r in gp if r["arm"]==arm]; y=np.array([int(r["label"]) for r in cr]); spk=np.array([r["speaker"] for r in cr])
    sM=np.array([float(r["oof_mac_folds"]) for r in cr]); sL=np.array([float(r["oof_linux_folds"]) for r in cr])
    g=gAB[arm]; vm=V[f"macfolds_{arm}"]; lo,hi=boot(y,spk,sM)
    add(f"refit_{arm}_macfolds_linuxfit","probe OOF AUC refit on Linux (sklearn 1.9.1) with the Mac's sklearn 1.7.2 GroupKFold folds",g["auc_mac_folds"],lo,hi,vm["auc_probe"],vm["auc_probe_ci"][0],vm["auc_probe_ci"][1],
        {"all":0.7864,"conflict":0.5520,"agreement":0.9314}[arm],arm,"lo-pod3b-3 pod3b_gap_perclip.csv col oof_mac_folds","lo-pod3b-2 pod3b_verify.json macfolds_%s"%arm,C6 if arm=="all" else C5,
        "published column holds the Mac partF value (POD3_partF_out.json)")
    add(f"refit_{arm}_macfolds_cosine","cosine with the Mac's folds (fp16 lm_head)",g["cos_mac_folds"],None,None,vm["cosine"],None,None,None,arm,"lo-pod3b-3 pod3b_gap.json","lo-pod3b-2 pod3b_verify.json macfolds_%s"%arm,C6)
    add(f"refit_{arm}_linuxfolds_gapfit","probe OOF AUC with the Linux sklearn 1.9.1 folds, refit by the gap script",g["auc_linux_folds"],*boot(y,spk,sL),V[f"fp16_{arm}"]["auc_probe"],*V[f"fp16_{arm}"]["auc_probe_ci"],PUB[arm]["probe"][0],arm,
        "lo-pod3b-3 pod3b_gap_perclip.csv col oof_linux_folds","lo-pod3b-2 pod3b_verify.json fp16_%s"%arm,C6)
    add(f"folds_{arm}_speakers_new_foldmates","speakers whose fold-mates differ between the Mac folds and the Linux folds",g["speakers_with_different_foldmates"],None,None,vm["speakers_with_different_foldmates"],None,None,None,arm,
        "lo-pod3b-3 pod3b_gap.json","lo-pod3b-2 pod3b_verify.json macfolds_%s"%arm,C6,"out of %d speakers"%NN[arm][1])
    add(f"refit_{arm}_sk172rule_on_linux","probe OOF AUC with folds from the real sklearn 1.7.2 GroupKFold run on Linux x86 numpy 2.1.2",sk[arm]["auc_probe"],None,None,V[f"legacyrule_linux_{arm}"]["auc_probe"],None,None,None,arm,
        "lo-pod3b-2 sk172_linux_refit.json (sklearn 1.7.2 installed to a --target dir)","lo-pod3b-2 pod3b_verify.json legacyrule_linux_%s (re-implementation of the 1.7.2 rule)"%arm,C6,
        "folds identical to the Mac's: %s"%sk[arm]["same_as_mac_folds"])
    add(f"mac_{arm}_macfit_macfolds","Mac diagnostic: sklearn 1.7.2 fit, Mac folds (reproduces partF)",mc[arm]["mac_folds"],None,None,g["auc_mac_folds"],None,None,{"all":0.7864,"conflict":0.5520,"agreement":0.9314}[arm],arm,
        "mac/mac_crosscheck.json","lo-pod3b-3 pod3b_gap.json auc_mac_folds (Linux fit, same folds)",C6)
    add(f"mac_{arm}_macfit_linuxfolds","Mac diagnostic: sklearn 1.7.2 fit, Linux folds",mc[arm]["linux_folds"],None,None,g["auc_linux_folds"],None,None,PUB[arm]["probe"][0],arm,
        "mac/mac_crosscheck.json","lo-pod3b-3 pod3b_gap.json auc_linux_folds (Linux fit, same folds)",C6)
# spread over 200 tie orders of the same greedy rule
for arm in ("all","conflict","agreement"):
    A=load(P3+f"pod3b_gap_spread_{arm}.csv"); B=load(P4+f"verify_spread_{arm}.csv")
    a=np.array([float(r["auc_probe"]) for r in A]); b=np.array([float(r["auc_probe"]) for r in B]); ca=np.array([float(r["cosine"]) for r in A]); cb=np.array([float(r["cosine"]) for r in B])
    pubp=pa["fp16"][arm]["auc_probe_oof"]; macv=gAB[arm]["auc_mac_folds"]; fl=float(orig[arm]["cos_rand_hi"])
    stats=[("mean",np.mean),("sd",lambda x: np.std(x,ddof=1)),("p2_5",lambda x: np.percentile(x,2.5)),("p97_5",lambda x: np.percentile(x,97.5)),("min",np.min),("max",np.max)]
    for nm_,f in stats:
        add(f"spread_{arm}_auc_{nm_}",f"probe OOF AUC over 200 speaker-grouped greedy partitions that differ only in tie order: {nm_}",float(f(a)),None,None,float(f(b)),None,None,None,arm,
            f"lo-pod3b-3 pod3b_gap_spread_{arm}.csv",f"lo-pod3b-4 verify_spread_{arm}.csv",C6)
    add(f"spread_{arm}_n_at_or_above_published","partitions (of 200) with AUC at or above the pod value %.4f"%pubp,int((a>=pubp-1e-12).sum()),None,None,int((b>=pubp-1e-12).sum()),None,None,None,arm,f"lo-pod3b-3 pod3b_gap_spread_{arm}.csv",f"lo-pod3b-4 verify_spread_{arm}.csv",C6)
    add(f"spread_{arm}_n_at_or_below_mac","partitions (of 200) with AUC at or below the Mac-fold value %.4f"%macv,int((a<=macv+1e-12).sum()),None,None,int((b<=macv+1e-12).sum()),None,None,None,arm,f"lo-pod3b-3 pod3b_gap_spread_{arm}.csv",f"lo-pod3b-4 verify_spread_{arm}.csv",C6)
    add(f"spread_{arm}_cos_min","cosine over the 200 partitions: min",float(ca.min()),None,None,float(cb.min()),None,None,None,arm,f"lo-pod3b-3 pod3b_gap_spread_{arm}.csv",f"lo-pod3b-4 verify_spread_{arm}.csv",C1)
    add(f"spread_{arm}_cos_max","cosine over the 200 partitions: max",float(ca.max()),None,None,float(cb.max()),None,None,None,arm,f"lo-pod3b-3 pod3b_gap_spread_{arm}.csv",f"lo-pod3b-4 verify_spread_{arm}.csv",C1)
    add(f"spread_{arm}_n_abs_cos_above_floor_hi","partitions (of 200) whose |cosine| exceeds the random-floor 97.5 percentile %.4f"%fl,int((np.abs(ca)>fl).sum()),None,None,int((np.abs(cb)>fl).sum()),None,None,None,arm,f"lo-pod3b-3 pod3b_gap_spread_{arm}.csv",f"lo-pod3b-4 verify_spread_{arm}.csv",C1)
# probe and probe-without-d over the same 200 partitions (lo-pod3b-7 primary, lo-pod3b-6 verifier)
P6=R+"pull/lo-pod3b-6/out/"; P7=R+"pull/lo-pod3b-7/out/"
for arm in ("all","conflict","agreement"):
    A=load(P7+f"spread_wo_{arm}.csv"); B=load(P6+f"verify_spread_wo_{arm}.csv")
    ga=load(P3+f"pod3b_gap_spread_{arm}.csv"); assert all(abs(float(x["auc_probe"])-float(y_["auc_probe"]))<1e-12 for x,y_ in zip(ga,A))
    pw={"trainz":pa["fp16"][arm]["auc_probe_without_d_trainz"],"foldz":pa["fp16"][arm]["auc_probe_without_d_foldz"]}
    for c in ("trainz","foldz"):
        a=np.array([float(r[f"auc_wo_{c}"]) for r in A]); b=np.array([float(r[f"auc_wo_{c}"]) for r in B])
        pa_=np.array([float(r["auc_probe"]) for r in A]); pb_=np.array([float(r["auc_probe"]) for r in B])
        for nm_,f in (("mean",np.mean),("p2_5",lambda x: np.percentile(x,2.5)),("p97_5",lambda x: np.percentile(x,97.5)),("min",np.min),("max",np.max)):
            add(f"spread_{arm}_wo_{c}_{nm_}",f"probe AUC with d removed ({'train-fold z, as shipped' if c=='trainz' else 'held-out-fold z, canonical'}) over the 200 partitions: {nm_}",float(f(a)),None,None,float(f(b)),None,None,None,arm,
                f"lo-pod3b-7 spread_wo_{arm}.csv",f"lo-pod3b-6 verify_spread_wo_{arm}.csv",C4 if arm!="all" else C3)
        add(f"spread_{arm}_wo_{c}_n_at_or_above_published","partitions (of 200) with d-removed AUC at or above the POD3-split value %.4f"%pw[c],int((a>=pw[c]-1e-12).sum()),None,None,int((b>=pw[c]-1e-12).sum()),None,None,None,arm,
            f"lo-pod3b-7 spread_wo_{arm}.csv",f"lo-pod3b-6 verify_spread_wo_{arm}.csv",C3)
        dA=pa_-a; dB=pb_-b
        add(f"spread_{arm}_probe_minus_wo_{c}_mean","probe AUC minus d-removed AUC, mean over the 200 partitions",float(dA.mean()),float(np.percentile(dA,2.5)),float(np.percentile(dA,97.5)),float(dB.mean()),float(np.percentile(dB,2.5)),float(np.percentile(dB,97.5)),None,arm,
            f"lo-pod3b-7 spread_wo_{arm}.csv",f"lo-pod3b-6 verify_spread_wo_{arm}.csv",C3,"lo/hi here are the 2.5/97.5 percentiles across partitions, not a bootstrap")
# lm_head identity checks
lf=json.load(open(P1+"lmh_fetch.json"))
add("lmh_sha256_bf16","sha256 of thinker.lm_head.weight raw bf16 bytes, Qwen3-Omni-30B-A3B-Instruct rev 26291f79, shard 13",lf["sha256_raw_bf16_bytes"],None,None,V["lm_head_sha256_bf16"],None,None,None,"all",
    "lo-pod3b-1 lmh_fetch.json (HTTP range read, manual bf16 parse)","lo-pod3b-2 pod3b_verify.json (full shard via huggingface_hub, safetensors/torch)",C1)
add("lmh_projd_maxabs_vs_pod","max |X.d - proj_d saved by POD3 in q3o_perp_cols.npz| (d rebuilt from the downloaded lm_head)",pa["fp16"]["all"]["proj_d_vs_pod_perp_cols"]["max_abs_diff"],None,None,V["fp16_projd_maxabs_vs_pod"],None,None,None,"all",
    "lo-pod3b-1 pod3b_primary_fp16.json","lo-pod3b-2 pod3b_verify.json",C1,"relative size %.1e; the pod's d and this d are the same vector"%pa["fp16"]["all"]["proj_d_vs_pod_perp_cols"]["max_rel"])
add("lmh_pyes_maxabs_vs_saved","max |p_yes from float64 softmax(X W^T) - p_yes saved from the bf16 GPU forward pass|",pj["fp16"]["p_yes_check"]["max_abs_diff"],None,None,V["fp16_pyes_maxabs_vs_saved"],None,None,None,"all",
    "lo-pod3b-1 pod3b_primary_fp16.json","lo-pod3b-2 pod3b_verify.json",C1,"median %.4f, Pearson %.4f; bf16 logit rounding on the GPU, not a different head"%(pj["fp16"]["p_yes_check"]["median_abs_diff"],pj["fp16"]["p_yes_check"]["pearson"]))
add("refit_oof_maxabs_vs_pod","max |probe OOF from the Linux refit - oof saved by POD3 in q3o_perp_cols.npz|",pa["fp16"]["all"]["oof_vs_pod_perp_cols"]["max_abs_diff"],None,None,V["fp16_all"]["oof_maxabs_vs_pod"],None,None,None,"all",
    "lo-pod3b-1 pod3b_primary_fp16.json","lo-pod3b-2 pod3b_verify.json",C6)
H=["id","what","value","lo","hi","value_verified","lo_verified","hi_verified","agree_4dp","published","matches_published_4dp","n","n_speakers","file","verifier_file","closes","note"]
with open(R+"pod3b_values.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(H); w.writerows(rows)
bad=[r for r in rows if r[8]!="yes"]; pubbad=[r for r in rows if r[10]=="NO"]
print(len(rows),"rows;",len(bad),"not agreeing at 4 dp;",len(pubbad),"differ from published")
for r in bad+pubbad: print(r[0],r[2],r[5],r[9])
