"""Per-layer probe states for an audio LLM: audio encoder stages, the projector, the language model stages at the
audio token positions, and the answer position. One script for both backbones.
  Qwen2-Audio      audio_tower -> multi_modal_projector -> language_model
  Qwen2.5-Omni     thinker.audio_tower(.layers) -> thinker.audio_tower.proj -> thinker.model
usage: extract_probe_layers.py MANIFEST(path,label,speaker) OUTPREFIX COND(ad|pd|mdd) [MODEL_ID]
writes OUTPREFIX_states.npz, OUTPREFIX_zeroshot_scores.csv (with a prompt column) and OUTPREFIX_zeroshot.json."""
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import csv, sys, json, time, datetime, numpy as np, torch, librosa
MANIFEST, OUTPREFIX, COND = sys.argv[1:4]
MID = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen2.5-Omni-7B"
P = {"pd": "Based only on how this person's voice sounds, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.",
     "mdd": "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.",
     "ad": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."}[COND]
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
t0 = time.time()
IS_OMNI = "Omni" in MID
if IS_OMNI:
    from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID)
    full = Model.from_pretrained(MID, dtype=torch.bfloat16)
    net = full.thinker
    for attr in ("talker", "token2wav"):
        if hasattr(full, attr): delattr(full, attr)
    if hasattr(net, "visual"): del net.visual
    net = net.to(dev).eval()
    tower = net.audio_tower; enc_layers = tower.layers; projector = tower.proj
else:
    from transformers import AutoProcessor as Proc, Qwen2AudioForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID)
    net = Model.from_pretrained(MID, dtype=torch.float16).to(dev).eval()
    tower = getattr(net, "audio_tower", None) or net.model.audio_tower
    enc_layers = tower.layers
    projector = getattr(net, "multi_modal_projector", None) or net.model.multi_modal_projector
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
n_enc, n_llm = len(enc_layers), cfg.text_config.num_hidden_layers + 1 if IS_OMNI else cfg.text_config.num_hidden_layers + 1
print(f"MODEL {MID} loaded in {time.time()-t0:.0f}s on {dev}", flush=True)
print(f"STAGES audio encoder layers {n_enc} (+1 projector) | language model stages {n_llm} (embeddings + {n_llm-1} layers)", flush=True)
print(f"audio token {AUD} | yes ids {YES} | no ids {NO}", flush=True)
grab = {}
def hook(name):
    def f(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        grab[name] = h.reshape(-1, h.shape[-1]).mean(dim=0).float().cpu().numpy()
    return f
handles = [enc_layers[i].register_forward_hook(hook(f"enc{i}")) for i in range(n_enc)]
handles.append(projector.register_forward_hook(hook("proj")))
rows = list(csv.DictReader(open(MANIFEST)))
enc_f, prj_f, llm_f, ans_f, meta = [], [], [], [], []
for i, r in enumerate(rows):
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    with torch.no_grad():
        out = net(**inp, output_hidden_states=True)
        mask = (inp["input_ids"][0] == AUD); assert int(mask.sum()) > 0, "no audio tokens in the prompt"
        llm = torch.stack([h[0][mask].mean(dim=0).float().cpu() for h in out.hidden_states]).numpy()
        ans = torch.stack([h[0][-1].float().cpu() for h in out.hidden_states]).numpy()
        pr = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    enc_f.append(np.stack([grab[f"enc{j}"] for j in range(n_enc)])); prj_f.append(grab["proj"])
    llm_f.append(llm); ans_f.append(ans)
    meta.append((os.path.basename(r["path"]), str(r["speaker"]), int(r["label"]), py / (py + pn + 1e-12), py + pn))
    if i % 25 == 0 or i == len(rows) - 1:
        print(f"{i+1}/{len(rows)} {meta[-1][0]} label {meta[-1][2]} p_yes {meta[-1][3]:.3f} mass {meta[-1][4]:.3f}  {time.time()-t0:.0f}s", flush=True)
for h in handles: h.remove()
E, PJ, L, A = np.stack(enc_f), np.stack(prj_f)[:, None, :], np.stack(llm_f), np.stack(ans_f)
y = np.array([m[2] for m in meta]); spk = np.array([m[1] for m in meta]); pyes = np.array([m[3] for m in meta])
np.savez_compressed(OUTPREFIX + "_states.npz", enc=E, proj=PJ, llm=L, ans=A, label=y, spk=spk, p_yes=pyes, name=[m[0] for m in meta])
with open(OUTPREFIX + "_zeroshot_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_yes", "mass", "prompt"])
    for m in meta: w.writerow(list(m) + [P])
from sklearn.metrics import roc_auc_score
AUC = float(roc_auc_score(y, pyes))
json.dump({"checkpoint": MID, "clip_list": os.path.abspath(MANIFEST), "condition": COND, "prompt": P,
           "n": len(y), "n_positive": int(y.sum()), "auc": round(AUC, 4), "median_mass": round(float(np.median([m[4] for m in meta])), 4),
           "window_seconds": 30, "dtype": "bfloat16" if IS_OMNI else "float16", "device": dev,
           "encoder_layers": n_enc, "llm_stages": n_llm, "date": datetime.date.today().isoformat()},
          open(OUTPREFIX + "_zeroshot.json", "w"), indent=1)
print(f"RESULT zero-shot {os.path.basename(MANIFEST)} n={len(y)} AUC={AUC:.4f} "
      f"median_mass={np.median([m[4] for m in meta]):.3f}  {time.time()-t0:.0f}s", flush=True)
print("STATES", {k: list(v.shape) for k, v in (("enc", E), ("proj", PJ), ("llm", L), ("ans", A))}, flush=True)
