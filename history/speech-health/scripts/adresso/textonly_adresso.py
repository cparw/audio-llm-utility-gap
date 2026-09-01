#!/usr/bin/env python3
"""Text only on the adresso clips: qwen2 audio gets just the whisper transcript, no audio."""
import csv, os, time
import numpy as np, torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
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
rows = list(csv.DictReader(open(f"{D}/adresso/adresso_transcripts.csv")))
out = open(f"{D}/adresso/adresso_textonly_scores.csv", "w", newline="")
w = csv.writer(out); w.writerow(["spk","label","p_yes","mass"])
t0 = time.time()
for i, r in enumerate(rows):
    txt = (r.get("text") or "").strip()[:2000]
    prompt = (f'Here is a transcript of what a person said during a picture description task: "{txt}" '
              "Based only on this transcript, does this speaker show signs of dementia? "
              "Answer with one word, Yes or No.")
    try:
        conv = [{"role":"user","content":[{"type":"text","text":prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, return_tensors="pt", padding=True)
        inp = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}
        with torch.no_grad(): o = model(**inp)
        p = torch.softmax(o.logits[0,-1].float().cpu(), -1)
        py, pn = float(p[YES].sum()), float(p[NO].sum()); t = py+pn
        w.writerow([r["spk"], r["label"], py/t if t>0 else 0.5, t]); out.flush()
    except Exception as e:
        print(f"fail {r['spk']}: {repr(e)[:100]}", flush=True)
    if (i+1)%50==0: print(f"{i+1}/{len(rows)} {(time.time()-t0)/60:.1f} min", flush=True)
out.close()
rr = list(csv.DictReader(open(f"{D}/adresso/adresso_textonly_scores.csv")))
def auc(y,p):
    y,p = np.asarray(y,float), np.asarray(p,float)
    o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1,len(p)+1)
    n1,n0 = y.sum(),(1-y).sum(); return (rk[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
y = [int(r["label"]) for r in rr]; p = [float(r["p_yes"]) for r in rr]
print(f"ADRESSO TEXTONLY overall AUC {auc(y,p):.3f} (audio answer 0.636, probe 0.904)", flush=True)
