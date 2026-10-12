#!/usr/bin/env python3
"""Few shot on the Alzheimer's clips: one dementia and one control example, then the test clip."""
import csv, os, time
import numpy as np, torch, librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
B = os.path.expanduser("~/Desktop/Hard drive data for paper/DementiaBank")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", dtype=torch.float16).to(dev).eval()
tok = proc.tokenizer
def wids(ws):
    s = set()
    for w in ws:
        for v in (w, " "+w):
            e = tok(v, add_special_tokens=False).input_ids
            if e: s.add(e[0])
    return sorted(s)
YES, NO = wids(["Yes","yes","YES"]), wids(["No","no","NO"])
PROMPT = ("You heard three recordings. The first speaker has dementia. The second speaker "
          "is healthy. Based only on the third recording, does the third speaker show signs "
          "of dementia? Answer with one word, Yes or No.")
rows = [r for r in csv.DictReader(open(f"{B}/pitt_conflict_manifest.csv")) if r.get("segment_path")]
def load(r, sec):
    x,_ = librosa.load(f"{B}/{r['segment_path']}", sr=16000); return x[:sec*16000]
out = open(f"{B}/ad_fewshot_scores.csv", "w", newline="")
w = csv.writer(out); w.writerow(["segment_path","set","label","spk","p_yes","mass"])
t0 = time.time()
for i, r in enumerate(rows):
    spkr = r["grp"]+r["spk"]
    pd_ref = next(x for x in rows if x["label"]=="1" and x["grp"]+x["spk"] != spkr)
    hc_ref = next(x for x in rows if x["label"]=="0" and x["grp"]+x["spk"] != spkr)
    try:
        auds = [load(pd_ref,15), load(hc_ref,15), load(r,30)]
        conv = [{"role":"user","content":[{"type":"audio","audio":auds[0]},{"type":"audio","audio":auds[1]},
                                          {"type":"audio","audio":auds[2]},{"type":"text","text":PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=auds, sampling_rate=16000, return_tensors="pt", padding=True)
        inp = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}
        with torch.no_grad(): o = model(**inp)
        p = torch.softmax(o.logits[0,-1].float().cpu(), -1)
        py, pn = float(p[YES].sum()), float(p[NO].sum()); t = py+pn
        w.writerow([r["segment_path"], r["set"], r["label"], spkr, py/t if t>0 else 0.5, t]); out.flush()
    except Exception as e:
        print(f"fail {r['segment_path']}: {repr(e)[:100]}", flush=True)
    if (i+1)%50==0: print(f"{i+1}/{len(rows)} {(time.time()-t0)/60:.1f} min", flush=True)
out.close()
import numpy as np
rr = list(csv.DictReader(open(f"{B}/ad_fewshot_scores.csv")))
def auc(y,p):
    y,p = np.asarray(y,float), np.asarray(p,float)
    o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1,len(p)+1)
    n1,n0 = y.sum(),(1-y).sum(); return (rk[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
y = [int(r["label"]) for r in rr]; p = [float(r["p_yes"]) for r in rr]
print(f"AD FEWSHOT overall AUC {auc(y,p):.3f} (zero shot 0.618), yes rate {np.mean(np.array(p)>0.5):.2f}", flush=True)
for s in ("agreement","conflict"):
    m = [r["set"]==s for r in rr]
    print(f"  {s}: {auc(np.array(y)[m], np.array(p)[m]):.3f}", flush=True)
