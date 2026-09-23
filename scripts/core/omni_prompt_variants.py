"""Omni zero-shot under an alternative prompt wording. Same scoring as the paper: p_yes = P(Yes)/(P(Yes)+P(No))
at the first answer position, summing Yes/No variants with and without a leading space. 30 s window, bf16.
usage: omni_prompt_variants.py MANIFEST OUT.csv COND(ad|pd|mdd) VARIANT(p2|p3)"""
import os, csv, sys, json, time, datetime, numpy as np, torch, librosa
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
MAN, OUT, COND, VAR = sys.argv[1:5]
MID = "Qwen/Qwen2.5-Omni-7B"
CONDNAME = {"pd": "Parkinson's disease", "mdd": "depression", "ad": "dementia"}[COND]
P = {"p2": f"Listen to this recording. Does the speaker have {CONDNAME}? Answer Yes or No.",
     "p3": f"From the audio alone, is this speaker likely to have {CONDNAME}? Reply with one word, Yes or No."}[VAR]
proc = Qwen2_5OmniProcessor.from_pretrained(MID)
full = Qwen2_5OmniForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16)
th = full.thinker
for a in ("talker", "token2wav"):
    if hasattr(full, a): delattr(full, a)
if hasattr(th, "visual"): del th.visual
th = th.to("cuda").eval()
tok = proc.tokenizer
def ids(ws):
    s = set()
    for w in ws:
        t = tok.encode(w, add_special_tokens=False)
        if t: s.add(t[0])
    return sorted(s)
YES, NO = ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
rows = list(csv.DictReader(open(MAN))); res = []; t0 = time.time()
for i, r in enumerate(rows):
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}; inp.pop("use_audio_in_video", None)
    with torch.no_grad(): pr = torch.softmax(th(**inp).logits[0, -1].float(), -1).cpu().numpy()
    py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    res.append((os.path.basename(r["path"]), r["speaker"], r["label"], py / (py + pn + 1e-12), py + pn, P))
    if i % 100 == 0: print(f"{i+1}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_yes", "mass", "prompt"]); w.writerows(res)
from sklearn.metrics import roc_auc_score
y = [int(a[2]) for a in res]; p = [a[3] for a in res]
AUC = float(roc_auc_score(y, p))
json.dump({"checkpoint": MID, "clip_list": os.path.abspath(MAN), "condition": COND, "variant": VAR, "prompt": P,
           "n": len(res), "auc": round(AUC, 4), "yes_rate": round(sum(1 for v in p if v > .5) / len(p), 4),
           "median_mass": round(float(np.median([a[4] for a in res])), 4), "window_seconds": 30,
           "dtype": "bfloat16", "date": datetime.date.today().isoformat()}, open(OUT.replace(".csv", ".json"), "w"), indent=1)
print(f"RESULT prompt {VAR} {os.path.basename(MAN)} n={len(res)} AUC={AUC:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
