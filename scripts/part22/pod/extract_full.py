"""E-DAIC FULL window: Qwen2.5-Omni zero-shot + per-layer probe states.
Layout matches p15/extract_probe_layers.py exactly (same hooks, same pooling, same Yes/No scoring).
Additions: WINDOW_S=0 (full clip), per-clip checkpoint shards (resume-safe), context-fit accounting.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import csv, sys, json, time, datetime, numpy as np, torch, librosa

MANIFEST, OUTPREFIX, COND = sys.argv[1:4]
MID = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen2.5-Omni-7B"
SHARD = os.path.dirname(OUTPREFIX) + "/shards"
os.makedirs(SHARD, exist_ok=True)

P = {"pd": "Based only on how this person's voice sounds, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.",
     "mdd": "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.",
     "ad": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."}[COND]
P = os.environ.get("PROMPT_OVERRIDE", P)
WIN = float(os.environ.get("WINDOW_S", "0"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
MDTYPE = torch.bfloat16
t0 = time.time()

from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=torch.bfloat16)
net = full.thinker
for attr in ("talker", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to(dev).eval()
tower = net.audio_tower; enc_layers = tower.layers; projector = tower.proj

tok = proc.tokenizer
cfg = net.config
AUD = next((v for v in (getattr(cfg, "audio_token_id", None), getattr(cfg, "audio_token_index", None)) if v is not None),
           tok.convert_tokens_to_ids("<|AUDIO|>"))
def ids(words):
    s = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if t: s.add(t[0])
    return sorted(s)
YES, NO = ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
n_enc, n_llm = len(enc_layers), cfg.text_config.num_hidden_layers + 1
MAXPOS = getattr(cfg.text_config, "max_position_embeddings", 32768)
print(f"MODEL {MID} loaded in {time.time()-t0:.0f}s on {dev}", flush=True)
print(f"STAGES enc {n_enc} (+1 proj) | llm stages {n_llm} | max_position_embeddings {MAXPOS}", flush=True)
print(f"audio token {AUD} | yes {YES} | no {NO} | WINDOW_S {WIN}", flush=True)

grab = {}
def hook(name):
    def f(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        grab[name] = h.reshape(-1, h.shape[-1]).mean(dim=0).float().cpu().numpy()
    return f
handles = [enc_layers[i].register_forward_hook(hook(f"enc{i}")) for i in range(n_enc)]
handles.append(projector.register_forward_hook(hook("proj")))

rows = list(csv.DictReader(open(MANIFEST)))
SCSV = OUTPREFIX + "_zeroshot_scores.csv"
done = set()
if os.path.exists(SCSV):
    for r in csv.DictReader(open(SCSV)): done.add(r["clip"])
    print(f"RESUME: {len(done)} clips already scored", flush=True)
else:
    with open(SCSV, "w", newline="") as fh:
        csv.writer(fh).writerow(["clip","speaker","label","p_yes","mass","n_audio_tok","seq_len","fit_in_context","dur_s","sec","prompt"])

for i, r in enumerate(rows):
    name = os.path.basename(r["path"])
    if name in done: continue
    tc = time.time()
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
        out = net(**inp, output_hidden_states=True)
        mask = (inp["input_ids"][0] == AUD)
        n_aud = int(mask.sum()); seq = int(inp["input_ids"].shape[1])
        assert n_aud > 0, "no audio tokens in the prompt"
        llm = torch.stack([h[0][mask].mean(dim=0).float().cpu() for h in out.hidden_states]).numpy()
        ans = torch.stack([h[0][-1].float().cpu() for h in out.hidden_states]).numpy()
        pr = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    del out
    E = np.stack([grab[f"enc{j}"] for j in range(n_enc)])
    np.savez_compressed(f"{SHARD}/{name}.npz", enc=E, proj=grab["proj"][None, :], llm=llm, ans=ans)
    pyes = py / (py + pn + 1e-12); mass = py + pn
    dur = len(x) / 16000.0
    fit = int(seq <= MAXPOS)
    with open(SCSV, "a", newline="") as fh:
        csv.writer(fh).writerow([name, r["speaker"], r["label"], pyes, mass, n_aud, seq, fit,
                                 round(dur,3), round(time.time()-tc,2), P])
    if i % 5 == 0 or i == len(rows)-1:
        print(f"{i+1}/{len(rows)} {name} lab {r['label']} p_yes {pyes:.3f} mass {mass:.3f} "
              f"dur {dur:.0f}s aud_tok {n_aud} seq {seq} fit {fit} {time.time()-tc:.1f}s  tot {time.time()-t0:.0f}s", flush=True)
    torch.cuda.empty_cache()

for h in handles: h.remove()
print("EXTRACTION DONE", time.time()-t0, flush=True)
