"""Zero-shot audio scoring, p_yes identical to the paper (extract_probe_layers.py):
p_yes = P(Yes)/(P(Yes)+P(No)) at the FIRST answer position, summing the bare and leading-space
variants of Yes/yes/YES and No/no/NO, single-token encodings only. 30 s window, bf16, greedy/deterministic.
usage: score_audio.py MANIFEST(path,label,speaker,arm,id) OUT.csv [MODEL_ID]   env: PROMPT_OVERRIDE, WINDOW_S"""
import os, csv, sys, json, time, datetime, numpy as np, torch, librosa
MAN, OUT = sys.argv[1:3]
MID = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-Omni-7B"
P = os.environ.get("PROMPT_OVERRIDE",
    "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.")
WIN = float(os.environ.get("WINDOW_S", "30"))
dev = "cuda"; t0 = time.time()
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=torch.bfloat16)
net = full.thinker
for a in ("talker", "token2wav"):
    if hasattr(full, a): delattr(full, a)
if hasattr(net, "visual"): del net.visual
net = net.to(dev).eval()
tok = proc.tokenizer
def ids(ws):
    s = set()
    for w in ws:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1: s.add(t[0])
    return sorted(s)
YES, NO = ids(["Yes"," Yes","yes"," yes","YES"]), ids(["No"," No","no"," no","NO"])
print(f"MODEL {MID} {time.time()-t0:.0f}s | yes {YES} | no {NO}\nPROMPT {P!r}", flush=True)
rows = list(csv.DictReader(open(MAN)))
fh = open(OUT, "w", newline=""); w = csv.writer(fh)
w.writerow(["id","speaker","arm","label","p_yes","answer_mass","prompt"])
for i, r in enumerate(rows):
    x, _ = librosa.load(r["path"], sr=16000)
    x = x[:int(16000*WIN)] if WIN > 0 else x
    conv = [{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}
    inp.pop("use_audio_in_video", None)
    for _k,_v in list(inp.items()):
        if torch.is_tensor(_v) and _v.dtype == torch.float32: inp[_k] = _v.to(torch.bfloat16)
    with torch.no_grad():
        pr = torch.softmax(net(**inp).logits[0,-1].float(), -1).cpu().numpy()
    py, pn = float(pr[YES].sum()), float(pr[NO].sum()); mass = py+pn
    w.writerow([r["id"], r["speaker"], r.get("arm",""), r["label"], f"{py/(mass+1e-12):.10f}", f"{mass:.10f}", P])
    if i % 100 == 0 or i == len(rows)-1:
        fh.flush(); print(f"{i+1}/{len(rows)} {r['id']} {time.time()-t0:.0f}s", flush=True)
fh.close()
json.dump({"checkpoint":MID,"clip_list":os.path.abspath(MAN),"prompt":P,"n":len(rows),
  "window_seconds":WIN,"dtype":"bfloat16","device":dev,"yes_ids":YES,"no_ids":NO,
  "p_yes":"P(Yes)/(P(Yes)+P(No)) at first answer position, single-token Yes/No variants",
  "command":f"PROMPT_OVERRIDE=... WINDOW_S={WIN} python3 score_audio.py {MAN} {OUT} {MID}",
  "date":datetime.date.today().isoformat()}, open(OUT.replace(".csv",".json"),"w"), indent=1)
print(f"DONE {len(rows)} -> {OUT} {time.time()-t0:.0f}s", flush=True)
