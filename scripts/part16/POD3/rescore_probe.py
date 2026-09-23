"""POD3: (1) dump thinker lm_head weights for the by-arm direction test,
(2) re-score all 468 Pitt clips TWICE in one process to measure run-to-run determinism
    of this MoE in the CURRENT environment. Scoring path copied from extract_probe_layers.py."""
import os, csv, sys, json, time, datetime
import numpy as np, torch, librosa

MANIFEST="/workspace/mf_pitt468.csv"
MID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
P="Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
WIN=30.0
t0=time.time()
from transformers import Qwen3OmniMoeProcessor as Proc, Qwen3OmniMoeForConditionalGeneration as Model
proc=Proc.from_pretrained(MID)
full=Model.from_pretrained(MID, dtype=torch.bfloat16)
net=full.thinker
for attr in ("talker","code2wav","token2wav"):
    if hasattr(full,attr): delattr(full,attr)
if hasattr(net,"visual"): del net.visual

# --- dump lm_head once (float16 to keep it small) for the direction test ---
W=net.lm_head.weight.detach().float().cpu().numpy()
np.save("/workspace/out/q3o_lm_head.npy", W.astype(np.float16))
print("lm_head dumped", W.shape, W.dtype, flush=True)

net=net.to("cuda").eval()
tok=proc.tokenizer
cfg=net.config
AUD=next((v for v in (getattr(cfg,"audio_token_id",None), getattr(cfg,"audio_token_index",None)) if v is not None),
         tok.convert_tokens_to_ids("<|AUDIO|>"))
def ids_first(words):
    s=set()
    for w in words:
        t=tok.encode(w, add_special_tokens=False)
        if t: s.add(t[0])
    return sorted(s)
def ids_single(words):
    out=[]
    for w in words:
        t=tok.encode(w, add_special_tokens=False)
        if len(t)==1: out.append(t[0])
    return sorted(set(out))
YES_F,NO_F=ids_first(["Yes"," Yes","yes"," yes","YES"]), ids_first(["No"," No","no"," no","NO"])
YES_S,NO_S=ids_single(["Yes"," Yes","yes"," yes","YES"]), ids_single(["No"," No","no"," no","NO"])
print("yes ids first-token",YES_F,"single-token",YES_S,flush=True)
print("no  ids first-token",NO_F,"single-token",NO_S,flush=True)
print("ID SETS IDENTICAL:", YES_F==YES_S and NO_F==NO_S, flush=True)
YES,NO=YES_F,NO_F
rows=list(csv.DictReader(open(MANIFEST)))
print(f"model ready {time.time()-t0:.0f}s, {len(rows)} clips", flush=True)

def score_all(tag):
    out=[]
    for i,r in enumerate(rows):
        x,_=librosa.load(r["path"], sr=16000)
        x=x[:int(16000*WIN)] if WIN>0 else x
        conv=[{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
        text=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp=proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inp={k:(v.to("cuda") if hasattr(v,"to") else v) for k,v in inp.items()}
        inp.pop("use_audio_in_video",None)
        for _k,_v in list(inp.items()):
            if torch.is_tensor(_v) and _v.dtype==torch.float32: inp[_k]=_v.to(torch.bfloat16)
        with torch.no_grad():
            o=net(**inp)
            pr=torch.softmax(o.logits[0,-1].float(),dim=-1).cpu().numpy()
            py,pn=float(pr[YES].sum()),float(pr[NO].sum())
        out.append((os.path.basename(r["path"]), str(r["speaker"]), int(r["label"]),
                    py/(py+pn+1e-12), py+pn))
        if i%100==0: print(f"  [{tag}] {i+1}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    return out

def rank_auc(yv,s):
    yv=np.asarray(yv); s=np.asarray(s,float)
    n1=int((yv==1).sum()); n0=int((yv==0).sum())
    o=np.argsort(s,kind="mergesort"); ss=s[o]; rk=np.empty(len(s))
    i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        rk[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    return float((rk[yv==1].sum()-n1*(n1+1)/2.0)/(n1*n0))

A=score_all("runA"); B=score_all("runB")
y=np.array([m[2] for m in A])
pa=np.array([m[3] for m in A]); pb=np.array([m[3] for m in B])
aucA,aucB=rank_auc(y,pa),rank_auc(y,pb)
d=np.abs(pa-pb)
print(f"\nRERUN A auc {aucA:.6f}   RERUN B auc {aucB:.6f}", flush=True)
print(f"A vs B  identical clips {int((d==0).sum())}/{len(d)}  max|d| {d.max():.3e}  mean|d| {d.mean():.3e}", flush=True)
with open("/workspace/out/q3o_pitt_rescore_runA.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["clip","speaker","label","p_yes","mass","prompt"])
    for m in A: w.writerow(list(m)+[P])
with open("/workspace/out/q3o_pitt_rescore_runB.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["clip","speaker","label","p_yes","mass","prompt"])
    for m in B: w.writerow(list(m)+[P])
json.dump({"model":MID,"manifest":MANIFEST,"prompt":P,"window_seconds":WIN,
 "n":len(y),"auc_runA":round(aucA,6),"auc_runB":round(aucB,6),
 "runA_vs_runB_identical_clips":int((d==0).sum()),
 "runA_vs_runB_max_abs_diff":float(d.max()),"runA_vs_runB_mean_abs_diff":float(d.mean()),
 "transformers":__import__("transformers").__version__,"torch":torch.__version__,
 "librosa":__import__("librosa").__version__,
 "yes_ids":YES,"no_ids":NO,"dtype":"bfloat16","device":"cuda",
 "date":datetime.datetime.now().isoformat(timespec="seconds")},
 open("/workspace/out/q3o_pitt_rescore.json","w"),indent=1)
print("WROTE rescore outputs", flush=True)
