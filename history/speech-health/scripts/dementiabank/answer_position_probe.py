"""Reviewer's test: probe the hidden state AT THE ANSWER POSITION (last prompt token, where Yes/No is
read off), every LLM layer, on the 468 pitt segments. If it reads high, the deficit is the readout;
if it reads like the answer, the language model never integrated the signal."""
import csv, os, numpy as np, torch, librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
B = os.path.expanduser("~/Desktop/Hard drive data for paper/DementiaBank")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", dtype=torch.float16).to(dev).eval()
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
rows = [r for r in csv.DictReader(open(f"{B}/pitt_conflict_manifest.csv")) if r.get("segment_path")]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["grp"]+r["spk"] for r in rows]); arm = np.array([r["set"] for r in rows])
npz = f"{B}/ad_answer_position_states.npz"
if not os.path.exists(npz):
    feats = []
    for i, r in enumerate(rows):
        x, _ = librosa.load(f"{B}/{r['segment_path']}", sr=16000)
        conv = [{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inp = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        feats.append(torch.stack([h[0, -1].float().cpu() for h in out.hidden_states]).numpy())  # [layers, dim] at the answer position
        if (i+1) % 50 == 0: print(f"extract {i+1}/{len(rows)}", flush=True)
    np.savez_compressed(npz, feats=np.stack(feats), label=y, spk=spk, arm=arm)
z = np.load(npz, allow_pickle=True); F = z["feats"]
res = []
with open(f"{B}/ad_answer_position_probe.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["llm_layer","auc_overall","auc_agreement","auc_conflict"])
    for L in range(F.shape[1]):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(F[:,L,:], y, groups=spk):
            sc = StandardScaler().fit(F[tr][:,L,:]); c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(F[tr][:,L,:]), y[tr])
            oof[te] = c.decision_function(sc.transform(F[te][:,L,:]))
        a = roc_auc_score(y, oof); ag = roc_auc_score(y[arm=="agreement"], oof[arm=="agreement"]); cf = roc_auc_score(y[arm=="conflict"], oof[arm=="conflict"])
        w.writerow([L, f"{a:.3f}", f"{ag:.3f}", f"{cf:.3f}"]); res.append((L, a, ag, cf))
        print(f"llm layer {L:2d} at answer position: overall {a:.3f} agreement {ag:.3f} conflict {cf:.3f}", flush=True)
best = max(res, key=lambda t: t[1]); last = res[-1]
print(f"ANSWER-POSITION PROBE: best layer {best[0]} {best[1]:.3f}; final layer {last[1]:.3f} (the model's own answer 0.618, audio-token probe up to 0.86)", flush=True)
