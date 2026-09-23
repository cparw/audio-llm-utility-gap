"""Part 20 POD4 zero-shot scorer. The scoring path is copied line for line from the paper's
extract_probe_layers.py (p15 PATCHED COPY): same processor call, same chat template call,
same float32 -> model dtype cast, same YES/NO id sets (first token of each variant), same
softmax at the last prompt position, same 30 s cap. The hidden-state hooks are left out
(they do not touch the logits). Extra columns p_yes_r3 / mass_r3 use the Part 20 rule-3 id
set: bare and leading-space Yes/yes/YES and No/no/NO, single-token encodings only.

usage: score_zs.py MODEL_ID DTYPE JOB [JOB ...]   JOB = MANIFEST,OUT_CSV,PROMPT_KEY[,LIMIT]
  MANIFEST columns: clip (key), path, speaker, label, arm, src_clip
  PROMPT_KEY: recording | voice
  DTYPE: bf16 | fp16 (the paper used fp16 for Qwen2-Audio, bf16 for Omni)
Checkpoint: rows are appended to OUT_CSV.partial as they are scored; a restart skips them.
"""
import os, sys, csv, time, hashlib
import numpy as np, torch, librosa

MID, DT = sys.argv[1:3]
JOBS = [j.split(",") for j in sys.argv[3:]]
PROMPTS = {
    "recording": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No.",
    "voice": "Based only on how this person's voice sounds, does this speaker show signs of dementia? Answer with one word, Yes or No.",
}
WIN = 30.0
IS_OMNI = "Omni" in MID
MDTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}[DT]
dev = "cuda"
OHS = os.environ.get("OHS", "1") == "1"   # the paper call passed output_hidden_states=True; kept by default
t0 = time.time()
if IS_OMNI:
    from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID)
    full = Model.from_pretrained(MID, dtype=MDTYPE)
    net = full.thinker
    for attr in ("talker", "token2wav"):
        if hasattr(full, attr): delattr(full, attr)
    if hasattr(net, "visual"): del net.visual
    net = net.to(dev).eval()
else:
    from transformers import AutoProcessor as Proc, Qwen2AudioForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID)
    net = Model.from_pretrained(MID, dtype=MDTYPE).to(dev).eval()
tok = proc.tokenizer
cfg = net.config
AUD = next((v for v in (getattr(cfg, "audio_token_id", None), getattr(cfg, "audio_token_index", None)) if v is not None),
           tok.convert_tokens_to_ids("<|AUDIO|>"))

def ids(words):  # paper id set: first token of each variant
    s = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if t: s.add(t[0])
    return sorted(s)

def ids_single(words):  # rule-3 id set: single-token encodings only
    s = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1: s.add(t[0])
    return sorted(s)

YES, NO = ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
YES3 = ids_single(["Yes", " Yes", "yes", " yes", "YES", " YES"])
NO3 = ids_single(["No", " No", "no", " no", "NO", " NO"])
enc_dump = {w: tok.encode(w, add_special_tokens=False) for w in ["Yes", " Yes", "yes", " yes", "YES", " YES", "No", " No", "no", " no", "NO", " NO"]}
import transformers
print(f"MODEL {MID} dtype {DT} loaded in {time.time()-t0:.0f}s transformers {transformers.__version__} torch {torch.__version__}", flush=True)
print(f"audio token {AUD} | paper yes ids {YES} no ids {NO} | rule3 yes ids {YES3} no ids {NO3}", flush=True)
print(f"ENCODINGS {enc_dump}", flush=True)
print(f"output_hidden_states={OHS} attn_implementation={getattr(net.config, '_attn_implementation', None)}", flush=True)

def run_job(MANIFEST, OUT, PKEY, LIMIT=None):
    P = PROMPTS[PKEY]
    print(f"JOB {MANIFEST} -> {OUT} | PROMPT {P}", flush=True)
    rows = list(csv.DictReader(open(MANIFEST)))
    if LIMIT: rows = rows[:LIMIT]
    PART = OUT + ".partial"
    done = {}
    if os.path.exists(PART):
        for r in csv.DictReader(open(PART)):
            done[r["clip"]] = r
        print(f"RESUME {len(done)} rows already in {PART}", flush=True)
    cols = ["clip", "src_clip", "speaker", "label", "arm", "p_yes", "mass", "p_yes_r3", "mass_r3",
            "n_audio_tokens", "sec_used", "wav_sha256", "prompt", "model", "dtype"]
    new = not os.path.exists(PART)
    fh = open(PART, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if new: w.writeheader(); fh.flush()
    for i, r in enumerate(rows):
        if r["clip"] in done: continue
        x, _ = librosa.load(r["path"], sr=16000)
        x = x[:int(16000 * WIN)] if WIN > 0 else x
        conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
        inp.pop("use_audio_in_video", None)
        for _k, _v in list(inp.items()):
            if torch.is_tensor(_v) and _v.dtype == torch.float32: inp[_k] = _v.to(MDTYPE)
        with torch.no_grad():
            out = net(**inp, output_hidden_states=True) if OHS else net(**inp)
            nat = int((inp["input_ids"][0] == AUD).sum()); assert nat > 0, "no audio tokens in the prompt"
            pr = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        py, pn = float(pr[YES].sum()), float(pr[NO].sum())
        py3, pn3 = float(pr[YES3].sum()), float(pr[NO3].sum())
        sha = hashlib.sha256(open(r["path"], "rb").read()).hexdigest()
        row = dict(clip=r["clip"], src_clip=r["src_clip"], speaker=r["speaker"], label=r["label"], arm=r["arm"],
                   p_yes=repr(py / (py + pn + 1e-12)), mass=repr(py + pn), p_yes_r3=repr(py3 / (py3 + pn3 + 1e-12)),
                   mass_r3=repr(py3 + pn3), n_audio_tokens=nat, sec_used=round(len(x) / 16000, 4), wav_sha256=sha,
                   prompt=P, model=MID, dtype=DT)
        w.writerow(row); fh.flush(); done[r["clip"]] = row
        if i % 50 == 0 or i == len(rows) - 1:
            print(f"{i+1}/{len(rows)} {r['clip']} label {r['label']} p_yes {py/(py+pn+1e-12):.4f} mass {py+pn:.4f} {time.time()-t0:.0f}s", flush=True)
    fh.close()
    with open(OUT, "w", newline="") as fo:
        ww = csv.DictWriter(fo, fieldnames=cols); ww.writeheader()
        for r in rows: ww.writerow({c: done[r["clip"]][c] for c in cols})
    print(f"WROTE {OUT} n={len(rows)} {time.time()-t0:.0f}s", flush=True)

for j in JOBS:
    run_job(j[0], j[1], j[2], int(j[3]) if len(j) > 3 else None)
print("ALL JOBS DONE", flush=True)
