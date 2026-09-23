"""AF2 zero-shot, Pitt recording prompt, via the lab AF2Adapter (the code behind
af2_results/af2_pitt468.csv: official NVIDIA af2_code loaded by af2_import.load_af2(),
fp32, <audio>{prompt lowercased}<SEP>, 10 s window grid, peak norm, int16 round trip).
Load path identical to the 20 Sep pod af2_run.py: librosa.load(sr=16000, mono=True), then CUT to the first WINDOW_S s (default 30)
(control: original windows on the same first-30-s span the interviewer-free cut was built from).
p_yes (canonical) = adapter.score_yes_no rule: Yes/yes/No/no, bare + leading space, first token id.
p_yes_rule3 = same forward pass, Yes/yes/YES + No/no/NO, bare + leading space, single-token encodings only.
Checkpointed by clip_id.  usage: af2_zs.py MANIFEST OUT.csv"""
import os, sys, csv, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("AF2_CODE_DIR", "/workspace/af2code/af2/af2_code")
os.environ.setdefault("AF2_CKPT_DIR", "/workspace/hf/hub/models--nvidia--audio-flamingo-2/snapshots/3a7f4aea17e72c7367940affea6749069a7095fc")
sys.path.insert(0, "/workspace/af2code/af2"); sys.path.insert(0, "/workspace/af2code/speech-health")
import torch, librosa, numpy as np
import transformers
print("python", sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__, flush=True)
MAN, OUT = sys.argv[1:3]
WIN = float(os.environ.get("WINDOW_S", "30"))
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
from src.models.af2_adapter import AF2Adapter
t0 = time.time()
ad = AF2Adapter(model_name="af2", device=torch.device("cuda"))
print(f"AF2 READY {time.time()-t0:.0f}s dtype {next(ad.model.parameters()).dtype}", flush=True)
tok = getattr(ad.processor, "tokenizer", ad.processor)
yc, nc = ad._yes_no_ids()
def ids3(ws):
    out = []
    for w in ws:
        for f in (w, " " + w):
            t = tok.encode(f, add_special_tokens=False)
            if len(t) == 1: out.append(t[0])
    return sorted(set(out))
Y3 = ids3(["Yes", "yes", "YES"]); N3 = ids3(["No", "no", "NO"])
print("canonical YES", yc, [tok.decode([i]) for i in yc], "NO", nc, [tok.decode([i]) for i in nc], flush=True)
print("rule3 YES", Y3, [tok.decode([i]) for i in Y3], "NO", N3, [tok.decode([i]) for i in N3], flush=True)
rows = list(csv.DictReader(open(MAN)))
done = set()
if os.path.exists(OUT):
    done = {r["clip_id"] for r in csv.DictReader(open(OUT))}
fh = open(OUT, "a", newline=""); w = csv.writer(fh)
if not done:
    w.writerow(["clip_id", "orig_clip_id", "speaker_id", "label", "arm", "clip_path", "dur_s", "scored_s",
                "p_yes", "answer_mass", "p_yes_rule3", "answer_mass_rule3", "prompt"]); fh.flush()
t1 = time.time(); n = 0
print(f"{len(rows)} rows, {len(done)} already done, WINDOW_S={WIN}", flush=True)
for i, r in enumerate(rows):
    if r["clip_id"] in done: continue
    x, _ = librosa.load(r["path"], sr=16000, mono=True)
    dur = len(x) / 16000
    if WIN > 0: x = x[:int(16000 * WIN)]
    wav = torch.from_numpy(x).reshape(1, -1)
    with torch.no_grad():
        audio_np = wav.reshape(-1).cpu().numpy().astype(np.float32)
        inputs = ad._build_answer_inputs(audio_np, 16000, P)
        out = ad.model(**inputs)
        full = torch.softmax(out.logits[:, -1, :].float()[0], dim=-1)
    ym = float(full[list(yc)].sum()); nm = float(full[list(nc)].sum()); den = ym + nm
    py = ym / den if den > 0 else 0.5
    y3 = float(full[Y3].sum()); n3 = float(full[N3].sum())
    w.writerow([r["clip_id"], r.get("orig_clip_id", r["clip_id"]), r["speaker"], r["label"], r["arm"], r["path"],
                round(dur, 3), round(len(x) / 16000, 3), py, den, y3 / (y3 + n3 + 1e-12), y3 + n3, P]); fh.flush(); n += 1
    if n % 50 == 0 or i + 1 == len(rows):
        print(f"{i+1}/{len(rows)} p_yes {py:.4f} mass {den:.4f} {time.time()-t1:.0f}s", flush=True)
fh.close()
print("AF2_DONE", OUT, flush=True)
