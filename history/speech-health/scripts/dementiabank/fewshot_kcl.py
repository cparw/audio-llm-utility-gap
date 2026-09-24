#!/usr/bin/env python3
"""Few shot on KCL: two reference voices in the prompt, one sick one healthy,
then the test clip. Same p_yes metric as everything else."""
import csv, os, glob, time
import numpy as np, torch, librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

D = os.path.expanduser("~/Desktop/Hard drive data for paper")
KCL = f"{D}/KCL/extracted"
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

files = sorted(glob.glob(f"{KCL}/*/*/*/*.wav"))
rows = []
for f in files:
    rows.append(dict(path=f, task=("read" if "ReadText" in f else "spontaneous"),
                     label=1 if "/PD/" in f else 0,
                     spk=os.path.basename(f).split("_")[0]))
print(f"{len(rows)} clips", flush=True)

PROMPT = ("You heard three recordings. The first speaker has Parkinson's disease. "
          "The second speaker is healthy. Based only on how the third speaker's voice "
          "sounds, does the third speaker show signs of Parkinson's disease? "
          "Answer with one word, Yes or No.")

def load(p, sec=30):
    x, _ = librosa.load(p, sr=16000)
    return x[:sec*16000]

out_path = f"{D}/fewshot/kcl_fewshot_scores.csv"
fh = open(out_path, "w", newline="")
w = csv.writer(fh); w.writerow(["path","task","spk","label","p_yes","mass","pd_ref","hc_ref"])
t0 = time.time()
for i, r in enumerate(rows):
    pd_ref = next(x for x in rows if x["task"]==r["task"] and x["label"]==1 and x["spk"]!=r["spk"])
    hc_ref = next(x for x in rows if x["task"]==r["task"] and x["label"]==0 and x["spk"]!=r["spk"])
    try:
        auds = [load(pd_ref["path"], 15), load(hc_ref["path"], 15), load(r["path"], 30)]
        conv = [{"role":"user","content":[
            {"type":"audio","audio":auds[0]}, {"type":"audio","audio":auds[1]},
            {"type":"audio","audio":auds[2]}, {"type":"text","text":PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=auds, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        p = torch.softmax(out.logits[0,-1].float().cpu(), -1)
        py, pn = float(p[YES].sum()), float(p[NO].sum()); t = py+pn
        w.writerow([os.path.basename(r["path"]), r["task"], r["spk"], r["label"],
                    py/t if t>0 else 0.5, t,
                    os.path.basename(pd_ref["path"]), os.path.basename(hc_ref["path"])])
        fh.flush()
    except Exception as e:
        print(f"fail {r['path']}: {repr(e)[:120]}", flush=True)
    if (i+1)%20==0: print(f"{i+1}/{len(rows)} {(time.time()-t0)/60:.1f} min", flush=True)
fh.close()
rows2 = list(csv.DictReader(open(out_path)))
def rank_auc(y, p):
    y, p = np.asarray(y,float), np.asarray(p,float)
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1,len(p)+1)
    n1, n0 = y.sum(), (1-y).sum()
    return (ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
for t in ("read","spontaneous"):
    rr = [r for r in rows2 if r["task"]==t]
    if not rr: continue
    auc = rank_auc([int(r["label"]) for r in rr], [float(r["p_yes"]) for r in rr])
    yr = np.mean([float(r["p_yes"])>0.5 for r in rr])
    print(f"FEWSHOT {t}: n {len(rr)}, AUC {auc:.3f}, yes rate {yr:.2f}", flush=True)
print("zero shot was read 0.646 / spont 0.740", flush=True)
