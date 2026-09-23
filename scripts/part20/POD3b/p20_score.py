"""PART20 POD3b. Zero-shot audio scoring, core computation copied line for line from
release/edaic_rerun/part16/POD1/score_audio.py (validated in Part 16: reproduced Part 14 Omni 0.2218/0.9482).
Added only: resume from an existing output (checkpoint = flush after every clip) and sharding.
usage: p20_score.py MANIFEST(path,id,speaker,arm,label) OUT.csv SHARD NSHARD   env: PROMPT_OVERRIDE, WINDOW_S"""
import os, csv, sys, json, time, datetime, hashlib, numpy as np, torch, librosa
MAN, OUT = sys.argv[1:3]
SH, NSH = int(sys.argv[3]), int(sys.argv[4])
MID = "Qwen/Qwen2.5-Omni-7B"
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
rows = [r for i, r in enumerate(csv.DictReader(open(MAN))) if i % NSH == SH]
done = set()
if os.path.exists(OUT):
    done = {r["id"] for r in csv.DictReader(open(OUT))}
    print(f"RESUME {len(done)} already scored", flush=True)
fh = open(OUT, "a", newline=""); w = csv.writer(fh)
if not done: w.writerow(["id","speaker","arm","label","p_yes","answer_mass","wav_sha256_1mb","n_samples_scored","prompt"])
for i, r in enumerate(rows):
    if r["id"] in done: continue
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
    sha = hashlib.sha256(open(r["path"],"rb").read(1<<20)).hexdigest()
    w.writerow([r["id"], r["speaker"], r.get("arm",""), r["label"], f"{py/(mass+1e-12):.10f}", f"{mass:.10f}", sha, len(x), P])
    fh.flush()
    if i % 100 == 0 or i == len(rows)-1:
        print(f"{i+1}/{len(rows)} {r['id']} {time.time()-t0:.0f}s", flush=True)
fh.close()
json.dump({"checkpoint":MID,"clip_list":os.path.abspath(MAN),"shard":SH,"nshard":NSH,"prompt":P,"n":len(rows),
  "window_seconds":WIN,"dtype":"bfloat16","device":dev,"yes_ids":YES,"no_ids":NO,
  "torch":torch.__version__,"transformers":__import__("transformers").__version__,
  "p_yes":"P(Yes)/(P(Yes)+P(No)) at first answer position, single-token Yes/No variants",
  "command":f"HF_HOME=/workspace/hf WINDOW_S={WIN} python3 p20_score.py {MAN} {OUT} {SH} {NSH}",
  "date":datetime.date.today().isoformat()}, open(OUT.replace(".csv",".json"),"w"), indent=1)
print(f"DONE {len(rows)} -> {OUT} {time.time()-t0:.0f}s", flush=True)
