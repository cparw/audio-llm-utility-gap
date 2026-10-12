import csv, json, numpy as np
from sklearn.metrics import roc_auc_score
R="/project2/msoleyma_946/speech_health/results_chaitanya"
print("INDEPENDENT RECOMPUTE from *_clips.csv vs *_summary.json")
print("tag                 flip csv/json     acNeut csv/json    behavDiag csv/json   verdict")
for tag in ["kcl_pd_read","ita_pd_read","nv_pd_read","pcg_pd_read","dcaps_dep",
            "omni_kcl_pd_read","omni_ita_pd_read","omni_nv_pd_read","omni_pcg_pd_read","omni_dcaps_dep"]:
    try:
        rows=list(csv.DictReader(open(R+"/reliance/"+tag+"_clips.csv")))
        s=json.load(open(R+"/reliance/"+tag+"_summary.json"))
    except Exception as e:
        print(tag.ljust(20)+"(not present yet)")
        continue
    y=np.array([int(r["label"]) for r in rows])
    sick=np.array([float(r["rel_sick"]) for r in rows]); heal=np.array([float(r["rel_healthy"]) for r in rows])
    neut=np.array([float(r["rel_neutral"]) for r in rows]); bd=np.array([float(r["behav_diag"]) for r in rows])
    flip=float(np.mean([int((si>=0.5)!=(hi>=0.5)) for si,hi in zip(sick,heal)]))
    try: acn=roc_auc_score(y,neut)
    except Exception: acn=float("nan")
    try: bda=roc_auc_score(y,bd)
    except Exception: bda=float("nan")
    def cl(a,b,t=0.01):
        if a!=a and b!=b: return "ok"
        if a!=a or b!=b: return "DIFF"
        return "ok" if abs(a-b)<=t else "DIFF"
    m=[cl(flip,s["flip_rate"]),cl(acn,s["acoustic_neutral_AUC"]),cl(bda,s["behavioral_diag_AUC"])]
    verdict="ALL OK" if all(x=="ok" for x in m) else ",".join(m)
    print("%-20s %.3f/%.3f       %.3f/%.3f        %.3f/%.3f          %s" % (
        tag, flip, s["flip_rate"], acn, s["acoustic_neutral_AUC"], bda, s["behavioral_diag_AUC"], verdict))
