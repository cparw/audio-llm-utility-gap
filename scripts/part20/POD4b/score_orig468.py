import os, csv, sys, json, time, datetime, hashlib
os.environ.setdefault("HF_HOME", "/workspace/hf")
import numpy as np, torch, librosa

MID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
WIN = 30.0
BASE = "/workspace/p20"
OUT = "/workspace/scores/part20/POD4b"
VAL_DIR = f"{BASE}/val"
NOINV = f"{BASE}/noinv"
t0 = time.time()

from transformers import Qwen3OmniMoeProcessor as Proc, Qwen3OmniMoeForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=torch.bfloat16)
net = full.thinker
for attr in ("talker", "code2wav", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to("cuda").eval()
tok = proc.tokenizer
print("attn impl:", getattr(net.config, "_attn_implementation", None), flush=True)

def ids_single(words):
    out = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1: out.append(t[0])
    return sorted(set(out))
YES = ids_single(["Yes", " Yes", "yes", " yes", "YES", " YES"])
NO = ids_single(["No", " No", "no", " no", "NO", " NO"])
YES5 = ids_single(["Yes", " Yes", "yes", " yes", "YES"])
NO5 = ids_single(["No", " No", "no", " no", "NO"])
print("YES", YES, "NO", NO, "| ref5 YES", YES5, "NO", NO5, flush=True)
print(f"model ready {time.time()-t0:.0f}s", flush=True)

def score(path):
    x, _ = librosa.load(path, sr=16000)
    dur = len(x) / 16000.0
    x = x[:int(16000 * WIN)] if WIN > 0 else x
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    for _k, _v in list(inp.items()):
        if torch.is_tensor(_v) and _v.dtype == torch.float32: inp[_k] = _v.to(torch.bfloat16)
    with torch.no_grad():
        o = net(**inp)
        pr = torch.softmax(o.logits[0, -1].float(), dim=-1).cpu().numpy()
    py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    py5, pn5 = float(pr[YES5].sum()), float(pr[NO5].sum())
    top = int(pr.argmax())
    return dict(p_yes=py / (py + pn + 1e-12), answer_mass=py + pn,
                p_yes_ref5=py5 / (py5 + pn5 + 1e-12), answer_mass_ref5=py5 + pn5,
                top_token=tok.decode([top]), dur_s=round(dur, 4), dur_scored_s=round(min(dur, WIN), 4))

# ---------------- same-pod re-score of the 468 ORIGINAL windows (same code path as the noinv run) ----------------
ODIR = f"{BASE}/orig/segments"
folds = {r["clip_id"]: r for r in csv.DictReader(open(f"{BASE}/pitt_groupkfold5_pod.csv"))}
sets = {r["clip_id"]: r for r in csv.DictReader(open(f"{BASE}/pitt_sets.csv"))}
part = f"{OUT}/q3o_pitt_orig468_samepod_zeroshot_scores.partial.csv"
cols = ["clip", "speaker", "label", "set", "p_yes", "answer_mass", "p_yes_ref5", "answer_mass_ref5", "top_token",
        "dur_s", "dur_scored_s", "prompt"]
done = set()
if os.path.exists(part):
    done = {r["clip"] for r in csv.DictReader(open(part))}
    print(f"RESUME: {len(done)} clips already in checkpoint", flush=True)
fh = open(part, "a", newline=""); w = csv.DictWriter(fh, fieldnames=cols)
if not done: w.writeheader(); fh.flush()
keys = sorted(sets)
for i, oc in enumerate(keys):
    if oc in done: continue
    s = score(f"{ODIR}/{oc}")
    w.writerow(dict(clip=oc, speaker=folds[oc]["speaker_id"], label=int(sets[oc]["label"]), set=sets[oc]["set"],
                    p_yes=s["p_yes"], answer_mass=s["answer_mass"], p_yes_ref5=s["p_yes_ref5"],
                    answer_mass_ref5=s["answer_mass_ref5"], top_token=s["top_token"], dur_s=s["dur_s"],
                    dur_scored_s=s["dur_scored_s"], prompt=P))
    fh.flush()
    if i % 50 == 0: print(f"  orig {i+1}/{len(keys)} {time.time()-t0:.0f}s p_yes {s['p_yes']:.4f}", flush=True)
fh.close()
rows = list(csv.DictReader(open(part)))
assert len(rows) == 468 == len({r["clip"] for r in rows}), len(rows)
os.replace(part, f"{OUT}/q3o_pitt_orig468_samepod_zeroshot_scores.csv")
print("ORIG SCORING DONE", len(rows), flush=True)
