"""Audio Flamingo 3 zero shot, one 30.0 s window per clip so the multi-window path cannot fire.
Every clip is pre-cut to a temporary 30.0 s wav before scoring.
p_yes = P(Yes)/(P(Yes)+P(No)) at the first answer position, Yes/No summing variants with and without a leading space.
usage: af3_score.py MANIFEST OUT.csv COND(ad|pd|mdd) [--nocut]"""
import os, csv, sys, json, time, datetime, tempfile
import numpy as np, torch, librosa, soundfile as sf
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
MAN, OUT, COND = sys.argv[1:4]
NOCUT = "--nocut" in sys.argv
MID = "nvidia/audio-flamingo-3-hf"
P = {"pd": "Based only on how this person's voice sounds, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.",
     "mdd": "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.",
     "ad": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."}[COND]
t0 = time.time()
proc = AutoProcessor.from_pretrained(MID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16).to("cuda").eval()
tok = proc.tokenizer
def ids(ws):
    s = set()
    for w in ws:
        t = tok.encode(w, add_special_tokens=False)
        if t: s.add(t[0])
    return sorted(s)
YES, NO = ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
print(f"AF3 loaded in {time.time()-t0:.0f}s | sound token {tok.convert_tokens_to_ids('<sound>')} | yes {YES} no {NO}", flush=True)
rows = list(csv.DictReader(open(MAN))); res = []; TMP = tempfile.mkdtemp(prefix="af3_30s_"); durs = []
for i, r in enumerate(rows):
    x, _ = librosa.load(r["path"], sr=16000)
    durs.append(len(x) / 16000)
    if not NOCUT: x = x[:16000 * 30]
    cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
    conv = [{"role": "user", "content": [{"type": "audio", "path": cut}, {"type": "text", "text": P}]}]
    inp = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt")
    inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}
    for k, v in list(inp.items()):
        if torch.is_tensor(v) and v.dtype == torch.float32: inp[k] = v.to(torch.bfloat16)
    with torch.no_grad():
        pr = torch.softmax(model(**inp).logits[0, -1].float(), dim=-1).cpu().numpy()
    py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    res.append((r.get("speaker", ""), r["path"], r["label"], py / (py + pn + 1e-12), py + pn, P))
    if i % 50 == 0 or i == len(rows) - 1:
        print(f"{i+1}/{len(rows)} {os.path.basename(r['path'])} p_yes {res[-1][3]:.3f} mass {res[-1][4]:.3f} {time.time()-t0:.0f}s", flush=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["speaker_id", "clip_path", "label", "p_yes", "answer_mass", "prompt"]); w.writerows(res)
from sklearn.metrics import roc_auc_score
y = [int(float(a[2])) for a in res]; p = [a[3] for a in res]
AUC = float(roc_auc_score(y, p))
json.dump({"model": MID, "prompt": P, "script": os.path.abspath(__file__), "clip_list": os.path.abspath(MAN),
           "n": len(res), "n_speakers": len({a[0] for a in res}), "window_seconds": None if NOCUT else 30,
           "pre_cut_to_single_window": not NOCUT, "auc": round(AUC, 4),
           "yes_rate": round(sum(1 for v in p if v > .5) / len(p), 4),
           "median_answer_mass": round(float(np.median([a[4] for a in res])), 4),
           "median_source_duration_s": round(float(np.median(durs)), 2),
           "max_source_duration_s": round(float(max(durs)), 2),
           "dtype": "bfloat16", "date": datetime.date.today().isoformat()}, open(OUT.replace(".csv", ".json"), "w"), indent=1)
print(f"RESULT af3 {os.path.basename(MAN)} n={len(res)} AUC={AUC:.4f} yes={sum(1 for v in p if v>.5)/len(p):.4f} "
      f"mass={np.median([a[4] for a in res]):.4f} {(time.time()-t0)/60:.1f} min", flush=True)
