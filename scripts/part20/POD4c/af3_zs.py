"""AF3 zero-shot, Pitt recording prompt. Same pipeline as p15/af3_score.py
(librosa 16 kHz -> optional cut to WINDOW_S -> temp wav -> processor chat template ->
logits at the last prompt position -> p_yes over single-token Yes/No variants), bf16.
Checkpointed: appends one row per clip, resumes by clip_id.
usage: af3_zs.py MANIFEST OUT.csv   (manifest cols: clip_id,path,label,speaker,arm[,orig_clip_id])"""
import os, sys, csv, time, tempfile
import numpy as np, torch, librosa, soundfile as sf
MAN, OUT = sys.argv[1:3]
WIN = float(os.environ.get("WINDOW_S", "30"))
MID = "nvidia/audio-flamingo-3-hf"
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
import transformers
print("transformers", transformers.__version__, "torch", torch.__version__, flush=True)
dev = "cuda"
proc = AutoProcessor.from_pretrained(MID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
tok = getattr(proc, "tokenizer", proc)
def ids_for(ws):
    out = []
    for w in ws:
        for f in (w, " " + w):
            t = tok.encode(f, add_special_tokens=False)
            if len(t) == 1: out.append(t[0])
    return sorted(set(out))
YES = ids_for(["Yes", "yes", "YES"]); NO = ids_for(["No", "no", "NO"])
print("YES ids", YES, [tok.decode([i]) for i in YES], "NO ids", NO, [tok.decode([i]) for i in NO], flush=True)
rows = list(csv.DictReader(open(MAN)))
done = set()
if os.path.exists(OUT):
    done = {r["clip_id"] for r in csv.DictReader(open(OUT))}
fh = open(OUT, "a", newline=""); w = csv.writer(fh)
if not done:
    w.writerow(["clip_id", "orig_clip_id", "speaker_id", "label", "arm", "clip_path", "dur_s", "scored_s",
                "p_yes", "answer_mass", "prompt"]); fh.flush()
TMP = tempfile.mkdtemp(prefix="af3_"); t0 = time.time(); n = 0
print(f"{len(rows)} rows, {len(done)} already done, WINDOW_S={WIN}", flush=True)
for i, r in enumerate(rows):
    if r["clip_id"] in done: continue
    x, _ = librosa.load(r["path"], sr=16000); dur = len(x) / 16000
    if WIN > 0: x = x[:int(16000 * WIN)]
    cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
    conv = [{"role": "user", "content": [{"type": "audio", "path": cut}, {"type": "text", "text": P}]}]
    inp = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt")
    inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
    for k, v in list(inp.items()):
        if torch.is_tensor(v) and v.dtype == torch.float32: inp[k] = v.to(torch.bfloat16)
    with torch.no_grad():
        lg = model(**inp).logits[0, -1].float()
    pr = torch.softmax(lg, -1); py = float(pr[YES].sum()); pn = float(pr[NO].sum())
    w.writerow([r["clip_id"], r.get("orig_clip_id", r["clip_id"]), r["speaker"], r["label"], r["arm"], r["path"],
                round(dur, 3), round(len(x) / 16000, 3), py / (py + pn + 1e-12), py + pn, P]); fh.flush(); n += 1
    if n % 50 == 0 or i + 1 == len(rows):
        print(f"{i+1}/{len(rows)} p_yes {py/(py+pn+1e-12):.4f} mass {py+pn:.4f} {time.time()-t0:.0f}s", flush=True)
fh.close()
print("AF3_DONE", OUT, flush=True)
