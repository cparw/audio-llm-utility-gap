"""POD3 mechanism demonstration for the HIGH discrepancy.
Scores the SAME 468 Pitt clips with two attention backends, everything else held fixed.
If swapping only a low-level kernel moves p_yes by roughly the same amount as the
shipped-vs-pod gap, that gap is explained by numeric environment, not by configuration."""
import os, csv, json, time, datetime
import numpy as np, torch, librosa

MANIFEST="/workspace/mf_pitt468.csv"
MID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
P="Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
WIN=30.0
YES=[7414,9454,9693,9834,14004]; NO=[902,2152,2308,2753,8996]
rows=list(csv.DictReader(open(MANIFEST)))

def auc_pairwise(y,s):
    y=np.asarray(y); s=np.asarray(s,float)
    a=s[y==1]; b=s[y==0]
    return (float((a[:,None]>b[None,:]).sum())+0.5*float((a[:,None]==b[None,:]).sum()))/(len(a)*len(b))

from transformers import Qwen3OmniMoeProcessor as Proc, Qwen3OmniMoeForConditionalGeneration as Model
proc=Proc.from_pretrained(MID)
res={}
for attn in ("sdpa","eager"):
    t0=time.time()
    try:
        full=Model.from_pretrained(MID, dtype=torch.bfloat16, attn_implementation=attn)
    except Exception as e:
        print(f"BACKEND {attn} UNAVAILABLE: {type(e).__name__}: {e}", flush=True)
        res[attn]={"error":f"{type(e).__name__}: {e}"}
        continue
    net=full.thinker
    for a in ("talker","code2wav","token2wav"):
        if hasattr(full,a): delattr(full,a)
    if hasattr(net,"visual"): del net.visual
    net=net.to("cuda").eval()
    print(f"BACKEND {attn} loaded {time.time()-t0:.0f}s", flush=True)
    py_all=[]
    for i,r in enumerate(rows):
        x,_=librosa.load(r["path"],sr=16000); x=x[:int(16000*WIN)]
        conv=[{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
        text=proc.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
        inp=proc(text=text,audio=[x],sampling_rate=16000,return_tensors="pt",padding=True)
        inp={k:(v.to("cuda") if hasattr(v,"to") else v) for k,v in inp.items()}
        inp.pop("use_audio_in_video",None)
        for _k,_v in list(inp.items()):
            if torch.is_tensor(_v) and _v.dtype==torch.float32: inp[_k]=_v.to(torch.bfloat16)
        with torch.no_grad():
            o=net(**inp)
            pr=torch.softmax(o.logits[0,-1].float(),dim=-1).cpu().numpy()
            a_,b_=float(pr[YES].sum()),float(pr[NO].sum())
        py_all.append(a_/(a_+b_+1e-12))
        if i%150==0: print(f"  [{attn}] {i+1}/{len(rows)} {time.time()-t0:.0f}s",flush=True)
    res[attn]={"p_yes":py_all}
    del net, full; torch.cuda.empty_cache()
    print(f"BACKEND {attn} done {time.time()-t0:.0f}s",flush=True)

y=np.array([int(r["label"]) for r in rows])
summary={"model":MID,"manifest":MANIFEST,"prompt":P,"window_seconds":WIN,"n":len(y),
 "transformers":__import__("transformers").__version__,"torch":torch.__version__,
 "gpu":torch.cuda.get_device_name(0),
 "date":datetime.datetime.now().isoformat(timespec="seconds")}
have=[k for k in res if "p_yes" in res[k]]
for k in have:
    summary[f"auc_{k}"]=round(auc_pairwise(y,res[k]["p_yes"]),6)
    print(f"AUC {k} = {summary[f'auc_{k}']:.6f}",flush=True)
if len(have)==2:
    pa=np.array(res[have[0]]["p_yes"]); pb=np.array(res[have[1]]["p_yes"])
    d=np.abs(pa-pb)
    cl=lambda v: np.clip(v,1e-9,1-1e-9)
    la=np.log(cl(pa)/(1-cl(pa))); lb=np.log(cl(pb)/(1-cl(pb)))
    summary.update({
      "compared":have,
      "identical_clips":int((d==0).sum()),
      "frac_diff_gt_0.001":round(float((d>1e-3).mean()),4),
      "frac_diff_gt_0.1":round(float((d>0.1).mean()),4),
      "max_abs_dp_yes":round(float(d.max()),4),
      "mean_abs_dp_yes":round(float(d.mean()),5),
      "mean_abs_dlogit":round(float(np.abs(lb-la).mean()),4),
      "mean_signed_dlogit":round(float((lb-la).mean()),4),
      "spearman":round(float(np.corrcoef(np.argsort(np.argsort(pa)),np.argsort(np.argsort(pb)))[0,1]),4),
      "auc_delta":round(summary[f"auc_{have[1]}"]-summary[f"auc_{have[0]}"],6)})
    print(f"{have[0]} vs {have[1]}: identical {int((d==0).sum())}/{len(d)} "
          f"mean|dp| {d.mean():.5f} mean|dlogit| {np.abs(lb-la).mean():.4f} "
          f"spearman {summary['spearman']:.4f} AUC delta {summary['auc_delta']:+.6f}",flush=True)
with open("/workspace/out/q3o_pitt_attn_backend.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["clip","speaker","label"]+[f"p_yes_{k}" for k in have])
    for i,r in enumerate(rows):
        w.writerow([os.path.basename(r["path"]),r["speaker"],r["label"]]+[f"{res[k]['p_yes'][i]:.10f}" for k in have])
json.dump(summary,open("/workspace/out/q3o_pitt_attn_backend.json","w"),indent=1)
print("WROTE attn backend outputs",flush=True)
