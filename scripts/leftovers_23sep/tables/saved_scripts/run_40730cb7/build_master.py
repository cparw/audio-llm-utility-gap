"""Master lookup: every model x dataset x stream number the paper needs, each with its file."""
import json, os, glob, csv
GD="<local data dir>/paper1_local_runs"
R ="<local data dir>/release"
DS=["pitt","edaic","pcgita","neurovoz","kcl","adresso","adress2020"]
rows=[]
def add(model,ds,stream,value,n,src,est):
    if value is None: return
    rows.append(dict(model=model,dataset=ds,stream=stream,auc=round(float(value),4),
                     n=n,estimator=est,source=src))
def rep(path,model,tag_map=None):
    if not os.path.exists(path): return
    d=json.load(open(path))
    ds=None
    b=os.path.basename(path)
    for x in DS:
        if f"_{x}_" in b: ds=x
    if ds is None: return
    for k,name in [("enc","encoder probe"),("proj","projector probe"),("llm","LM probe"),("ans","answer-state probe")]:
        if k in d and isinstance(d[k],dict) and "mean" in d[k]:
            add(model,ds,name,d[k]["mean"],d[k].get("n"),path,
                "five split nested, mean of %d repeats"%d[k].get("repeats",0) if d[k].get("repeats",0)>1 else "single split nested")
def zs(path,model):
    if not os.path.exists(path): return
    d=json.load(open(path)); b=os.path.basename(path); ds=None
    for x in DS:
        if f"_{x}_" in b or b.endswith(f"_{x}.json") or f"_{x}." in b: ds=x
    if ds is None: return
    add(model,ds,"zero shot answer",d.get("auc"),d.get("n"),path,"zero shot, %s s window"%d.get("window_seconds","?"))
for p in glob.glob(f"{GD}/omni_final/omni_*_nested_repeats.json"): rep(p,"Qwen2.5-Omni")
for p in glob.glob(f"{GD}/omni_final/q2a_*_nested_repeats.json"):  rep(p,"Qwen2-Audio")
for p in glob.glob(f"{R}/overnight2/final/q3o_*_nested_repeats.json")+glob.glob(f"{R}/overnight2/q3o_new/q3o_*_nested_repeats.json"): rep(p,"Qwen3-Omni-30B-A3B")
for p in glob.glob(f"{R}/overnight2/**/kimi_*_nested_repeats.json",recursive=True): rep(p,"Kimi-Audio")
for p in glob.glob(f"{R}/overnight2/q3o_new/q3o_*_zeroshot.json"): zs(p,"Qwen3-Omni-30B-A3B")
for p in glob.glob(f"{R}/overnight2/**/af3_*.json",recursive=True): zs(p,"Audio Flamingo 3")
for p in glob.glob(f"{GD}/omni_final/omni_*_zeroshot.json"): zs(p,"Qwen2.5-Omni")
for p in glob.glob(f"{R}/overnight2/**/kimi_*_zeroshot.json",recursive=True): zs(p,"Kimi-Audio")
# text cells
for p in glob.glob(f"{R}/overnight2/text_new/*_text.json"):
    d=json.load(open(p)); b=os.path.basename(p)
    model={"o25":"Qwen2.5-Omni","q3o":"Qwen3-Omni-30B-A3B"}.get(b.split("_")[0],b.split("_")[0])
    ds=b.split("_",1)[1].rsplit("_text",1)[0]
    add(model,ds,"transcript only",d.get("auc"),d.get("n"),p,"text only, one documented prompt")

# Audio Flamingo 2, all seven, computed from the per-clip files
import pandas as _pd, numpy as _np
def _auc(y,s):
    y=_np.asarray(y,int); s=_np.asarray(s,float); r=_pd.Series(s).rank().values
    n1=(y==1).sum(); n0=(y==0).sum()
    return float("nan") if n1==0 or n0==0 else (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
_AF2={"af2_pitt468":"pitt","af2_edaic275":"edaic","af2_pcgita1100":"pcgita","af2_neurovoz1270":"neurovoz",
      "af2_kcl37":"kcl","af2_adresso237":"adresso","af2_adress2020":"adress2020"}
for _p in sorted(glob.glob(f"{GD}/af2_results/af2_*.csv")):
    _b=os.path.splitext(os.path.basename(_p))[0]; _ds=_AF2.get(_b)
    if not _ds: continue
    _d=_pd.read_csv(_p)
    _lc=[c for c in _d.columns if c.lower() in("label","y")]; _pc=[c for c in _d.columns if c.lower() in("p_yes","p_dep","p_main")]
    if _lc and _pc: add("Audio Flamingo 2",_ds,"zero shot answer",_auc(_d[_lc[0]],_d[_pc[0]]),len(_d),_p,"zero shot")
# Qwen2-Audio zero shot, all seven
for _p in sorted(glob.glob(f"{GD}/probe2/*_zeroshot_scores.csv")):
    _ds=os.path.basename(_p).replace("_zeroshot_scores.csv","")
    _d=_pd.read_csv(_p)
    if "label" in _d and "p_yes" in _d: add("Qwen2-Audio",_ds,"zero shot answer",_auc(_d["label"],_d["p_yes"]),len(_d),_p,"zero shot")
# Qwen2-Audio transcript-only cells, from the 15 Sep per-clip files
for _p in sorted(glob.glob(f"{GD}/qwen2audio_*_text.csv")):
    _ds=os.path.basename(_p).replace("qwen2audio_","").replace("_text.csv","")
    _d=_pd.read_csv(_p)
    if "label" in _d and "p_yes" in _d:
        add("Qwen2-Audio",_ds,"transcript only",_auc(_d["label"],_d["p_yes"]),len(_d),_p,
            "text only, per-clip file recomputed")
# edaic30 -> edaic: the 30 s transcript is the text condition matched to the 30 s audio window
for _r in [r for r in rows if r["dataset"]=="edaic30" and r["stream"]=="transcript only"]:
    _c=dict(_r); _c["dataset"]="edaic"; _c["estimator"]=_c["estimator"]+" (30 s transcript, matched to the 30 s audio window)"
    rows.append(_c)
# conflict-arm probe rows: computed in Part 1e, previously outside every glob
for _p in glob.glob(f"{R}/edaic_conflict_new/*_edaic_conflict_nested_repeats.json"):
    _tag=os.path.basename(_p).split("_")[0]
    _m={"o25":"Qwen2.5-Omni","q2a":"Qwen2-Audio","q3o":"Qwen3-Omni-30B-A3B"}.get(_tag)
    if not _m: continue
    _d=json.load(open(_p))
    for _k,_n in [("enc","encoder probe"),("proj","projector probe"),("llm","LM probe"),("ans","answer-state probe")]:
        if _k in _d and isinstance(_d[_k],dict) and "mean" in _d[_k]:
            add(_m,"edaic_conflict",_n,_d[_k]["mean"],_d[_k].get("n"),_p,
                "nested, 390 conflict-manifest rows, 30 s window")
for _p in glob.glob(f"{R}/edaic_conflict_new/*_edaic_conflict*zeroshot.json")+glob.glob(f"{R}/edaic_conflict_new/af3_edaic_conflict.json"):
    _b=os.path.basename(_p); _tag=_b.split("_")[0]
    _m={"o25":"Qwen2.5-Omni","q2a":"Qwen2-Audio","q3o":"Qwen3-Omni-30B-A3B","af3":"Audio Flamingo 3"}.get(_tag)
    if not _m: continue
    _d=json.load(open(_p))
    if _d.get("auc") is not None:
        add(_m,"edaic_conflict","zero shot answer",_d["auc"],_d.get("n"),_p,
            "zero shot, 30 s window, 390 conflict-manifest rows")
# --- AUTHORITATIVE OVERRIDES + rows that live outside the globs ---
# R1: every Pitt probe number comes from part10/pitt_enc_nested5_oof.npz, five split mean 0.7706.
for _r in rows:
    if (_r["model"]=="Qwen2.5-Omni" and _r["dataset"]=="pitt" and _r["stream"]=="encoder probe"):
        _r["auc"]=0.7706
        _r["estimator"]="five split nested, mean of 5 repeats (part10 re-run)"
        _r["source"]=("<local data dir>/release/overnight2/part10/pitt_enc_nested5_oof.npz"
                      " [SUPERSEDES omni_final/omni_pitt_nested_repeats.json 0.7761 and single-split 0.7969]")
# E-DAIC conflict rows: produced by a one-off run, not by any glob above.
import os as _os
_CONF=[("Kimi-Audio",0.6899,"edaic_conflict_new/kimi/kimi_edaic_conflict_zeroshot_scores.csv"),
       ("Audio Flamingo 2",0.5363,"edaic_conflict_new/af2/af2_edaic_conflict_30s.csv")]
for _m,_a,_src in _CONF:
    _full=_os.path.join(R,_src)
    if _os.path.exists(_full):
        add(_m,"edaic_conflict","zero shot answer",_a,390,_full,
            "zero shot, 30 s window, voice prompt, 390 manifest rows")
seen=set(); out=[]
for r in sorted(rows,key=lambda r:(r["model"],r["dataset"],r["stream"])):
    k=(r["model"],r["dataset"],r["stream"])
    if k in seen: continue
    seen.add(k); out.append(r)
os.makedirs(f"{R}/master",exist_ok=True)
with open(f"{R}/master/master_lookup.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=["model","dataset","stream","auc","n","estimator","source"]); w.writeheader(); w.writerows(out)
print("wrote master/master_lookup.csv  rows=%d"%len(out))
import collections
c=collections.Counter((r["model"],r["stream"]) for r in out)
print("\ncoverage (model x stream -> datasets covered of 7):")
for (m,s),n in sorted(c.items()): print("  %-22s %-20s %d/7"%(m,s,n))
