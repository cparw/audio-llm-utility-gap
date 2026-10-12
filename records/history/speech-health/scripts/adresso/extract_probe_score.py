"""ADReSSo: encoder states, per layer probe, and the model's own answers. One chain."""
import csv, os, numpy as np, torch, librosa, time
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", dtype=torch.float16).to(dev).eval()
tower = model.model.audio_tower
tok = proc.tokenizer
rows = list(csv.DictReader(open(f"{D}/adresso/adresso_manifest.csv")))
print(f"{len(rows)} clips, device {dev}", flush=True)

# 1. encoder states
npz = f"{D}/adresso/adresso_encoder_states.npz"
if not os.path.exists(npz):
    feats, keep = [], []
    for i, r in enumerate(rows):
        try:
            x,_ = librosa.load(f"{D}/{r['segment_path']}", sr=16000)
            inp = proc.feature_extractor([x], sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                out = tower(inp["input_features"].to(dev).half(), output_hidden_states=True)
            feats.append(torch.stack([h.mean(dim=1).squeeze(0).float().cpu() for h in out.hidden_states]).numpy())
            keep.append(r)
        except Exception as e:
            print(f"fail {r['spk']}: {repr(e)[:80]}", flush=True)
        if (i+1)%25==0: print(f"extract {i+1}/{len(rows)}", flush=True)
    np.savez_compressed(npz, feats=np.stack(feats),
        spk=[r["spk"] for r in keep], label=[int(r["label"]) for r in keep])
    print("states saved", np.stack(feats).shape, flush=True)

# 2. per layer probe (one clip per speaker, so folds are speaker disjoint by design)
z = np.load(npz, allow_pickle=True)
F, y = z["feats"], z["label"].astype(int)
with open(f"{D}/adresso/adresso_encoder_perlayer.csv","w",newline="") as fh:
    w = csv.writer(fh); w.writerow(["layer","auc_mean","auc_std"])
    for L in range(F.shape[1]):
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(F[:,L,:], y):
            X = StandardScaler().fit(F[tr][:,L,:]).transform(F[:,L,:])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        w.writerow([L, f"{np.mean(aucs):.3f}", f"{np.std(aucs):.3f}"])
        print(f"probe layer {L}: {np.mean(aucs):.3f}", flush=True)

# 3. the model's own answers
def wids(ws):
    s = set()
    for wd in ws:
        for v in (wd, " "+wd):
            e = tok(v, add_special_tokens=False).input_ids
            if e: s.add(e[0])
    return sorted(s)
YES, NO = wids(["Yes","yes","YES"]), wids(["No","no","NO"])
P = ("Based only on this recording, does this speaker show signs of dementia? "
     "Answer with one word, Yes or No.")
out_path = f"{D}/adresso/adresso_scores_qwen2audio.csv"
done = set()
if os.path.exists(out_path):
    done = {r["spk"] for r in csv.DictReader(open(out_path))}
fh = open(out_path, "a" if done else "w", newline="")
w = csv.DictWriter(fh, fieldnames=["spk","label","p_main","mass_main"])
if not done: w.writeheader()
t0 = time.time()
for i, r in enumerate(rows):
    if r["spk"] in done: continue
    try:
        x,_ = librosa.load(f"{D}/{r['segment_path']}", sr=16000)
        conv = [{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        p = torch.softmax(out.logits[0,-1].float().cpu(), -1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        t = py+pn
        w.writerow(dict(spk=r["spk"], label=r["label"], p_main=py/t if t>0 else 0.5, mass_main=t)); fh.flush()
    except Exception as e:
        print(f"score fail {r['spk']}: {repr(e)[:80]}", flush=True)
    if (i+1)%25==0: print(f"answers {i+1}/{len(rows)}  {(time.time()-t0)/60:.1f} min", flush=True)
fh.close()
sc = list(csv.DictReader(open(out_path)))
ys = np.array([int(r["label"]) for r in sc]); ps = np.array([float(r["p_main"]) for r in sc])
print(f"ANSWER AUC adresso: {roc_auc_score(ys, ps):.3f} on {len(sc)} clips", flush=True)
