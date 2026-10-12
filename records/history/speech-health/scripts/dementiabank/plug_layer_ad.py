#!/usr/bin/env python3
"""Task 4 from the Aug 21 meeting: feed the probe's best encoder layer (24)
into the projector instead of the final encoder layer, no training, and
re-score the AD segments. Baseline answer AUC 0.618 (agreement 0.699 / conflict 0.413)."""
import csv, os, time, types
import numpy as np, torch, librosa
D = "/Volumes/G-Drive Pro/DementiaBank"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache")
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

PLUG = int(os.environ.get("PLUG_LAYER", "24"))
P_MAIN = ("Based only on this recording, does this speaker show signs of dementia? "
          "Answer with one word, Yes or No.")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", dtype=torch.float16).to(dev).eval()
tok = proc.tokenizer
tower = model.model.audio_tower
orig_forward = tower.forward

def plugged_forward(self, *args, **kwargs):
    kwargs["output_hidden_states"] = True
    out = orig_forward(*args, **kwargs)
    hs = out.hidden_states[PLUG]
    x = hs
    if hasattr(self, "avg_pooler") and self.avg_pooler is not None and hs.shape[1] != out.last_hidden_state.shape[1]:
        x = self.avg_pooler(x.permute(0, 2, 1)).permute(0, 2, 1)
    if hasattr(self, "layer_norm") and self.layer_norm is not None:
        x = self.layer_norm(x)
    assert x.shape == out.last_hidden_state.shape, (x.shape, out.last_hidden_state.shape)
    out.last_hidden_state = x
    return out
tower.forward = types.MethodType(plugged_forward, tower)

def wids(ws):
    s = set()
    for w in ws:
        for v in (w, " " + w):
            e = tok(v, add_special_tokens=False).input_ids
            if e: s.add(e[0])
    return sorted(s)
YES, NO = wids(["Yes","yes","YES"]), wids(["No","no","NO"])
print(f"device {dev}, plug layer {PLUG}", flush=True)

rows = list(csv.DictReader(open(f"{D}/pitt_conflict_manifest.csv")))
out_path = f"{D}/ad_scores_plug{PLUG}.csv"
done = set()
if os.path.exists(out_path):
    done = {r["segment_path"] for r in csv.DictReader(open(out_path))}
mode = "a" if done else "w"
fh = open(out_path, mode, newline="")
w = csv.DictWriter(fh, fieldnames=["segment_path","set","label","spk","p_main","mass_main"])
if not done: w.writeheader()

t0 = time.time()
for i, r in enumerate(rows):
    sp = r.get("segment_path","")
    if not sp or sp in done: continue
    try:
        x,_ = librosa.load(f"{D}/{sp}", sr=16000)
        conv = [{"role":"user","content":[{"type":"audio","audio":x},
                                          {"type":"text","text":P_MAIN}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        p = torch.softmax(out.logits[0,-1].float().cpu(), -1)
        py, pn = float(p[YES].sum()), float(p[NO].sum()); t = py+pn
        w.writerow(dict(segment_path=sp, set=r["set"], label=r["label"],
                        spk=r["grp"]+r["spk"], p_main=(py/t if t>0 else 0.5), mass_main=t))
        fh.flush()
    except Exception as e:
        print(f"fail {sp}: {repr(e)[:120]}", flush=True)
    if (i+1) % 50 == 0:
        print(f"progress {i+1}/{len(rows)}  {(time.time()-t0)/60:.1f} min", flush=True)
print("done", flush=True)
