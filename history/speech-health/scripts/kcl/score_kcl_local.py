#!/usr/bin/env python3
"""Score fresh KCL audio with qwen2 audio, voice prompt, both tasks.
Read task doubles as a check against the stored 0.658."""
import csv, os, glob, time
import numpy as np, torch, librosa
D = "/Volumes/G-Drive Pro/KCL"
os.environ.setdefault("HF_HOME", "/Volumes/G-Drive Pro/DementiaBank/hf_cache")
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

PROMPT = ("Based only on how this person's voice sounds, does this speaker show signs "
          "of Parkinson's disease? Answer with one word, Yes or No.")
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

files = sorted(glob.glob(f"{D}/extracted/*/*/*/*.wav"))
print(f"{len(files)} files", flush=True)
fh = open(f"{D}/kcl_local_scores.csv", "w", newline="")
w = csv.writer(fh); w.writerow(["file","task","speaker","label","p_voice","mass"])
t0 = time.time()
for i, f in enumerate(files):
    parts = f.split("/")
    task = "read" if "ReadText" in parts else "spontaneous"
    label = 1 if "/PD/" in f else 0
    spk = os.path.basename(f).split("_")[0]
    try:
        x, _ = librosa.load(f, sr=16000)
        x = x[:30*16000]
        conv = [{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        p = torch.softmax(out.logits[0,-1].float().cpu(), -1)
        py, pn = float(p[YES].sum()), float(p[NO].sum()); t = py+pn
        w.writerow([os.path.basename(f), task, spk, label, py/t if t>0 else 0.5, t]); fh.flush()
    except Exception as e:
        print(f"fail {f}: {repr(e)[:100]}", flush=True)
    if (i+1) % 20 == 0: print(f"{i+1}/{len(files)} {(time.time()-t0)/60:.1f} min", flush=True)
fh.close()
import csv as c2
rows = list(c2.DictReader(open(f"{D}/kcl_local_scores.csv")))
from collections import defaultdict
for t in ("read","spontaneous"):
    rr = [r for r in rows if r["task"]==t]
    y = np.array([int(r["label"]) for r in rr]); p = np.array([float(r["p_voice"]) for r in rr])
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p)+1)
    n1, n0 = y.sum(), (1-y).sum()
    auc = (ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
    print(f"{t}: n {len(rr)}, AUC {auc:.3f}, yes rate {(p>0.5).mean():.2f}", flush=True)
