"""PART20 POD5. Copy of p15/extract_probe_layers.py (Qwen2.5-Omni branch, same hooks, same pooling, same prompt call),
changed only in: (1) p_yes per the part20 rule = P(Yes)/(P(Yes)+P(No)) over the SINGLE-TOKEN bare and leading-space
variants of Yes/yes/YES and No/no/NO at the first answer position, full-vocab softmax, answer_mass recorded; the
extract_probe_layers.py convention (first token of 5 variants) is kept beside it as p_yes_legacy; (2) checkpoint every
50 clips, resumable; (3) the lm_head rows of every id used plus the final norm are saved for the direction test.
usage: extract_p20.py MANIFEST(path,label,speaker,orig_clip_id,set) OUTPREFIX COND"""
import os, csv, sys, json, time, datetime, numpy as np, torch, librosa
MANIFEST, OUTPREFIX, COND = sys.argv[1:4]
MID = "Qwen/Qwen2.5-Omni-7B"
P = {"ad": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."}[COND]
WIN = float(os.environ.get("WINDOW_S", "30"))
dev = "cuda"; MDTYPE = torch.bfloat16; t0 = time.time()
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=torch.bfloat16)
net = full.thinker
for attr in ("talker", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to(dev).eval()
tower = net.audio_tower; enc_layers = tower.layers; projector = tower.proj
tok = proc.tokenizer; cfg = net.config
AUD = next((v for v in (getattr(cfg, "audio_token_id", None), getattr(cfg, "audio_token_index", None)) if v is not None),
           tok.convert_tokens_to_ids("<|AUDIO|>"))
def ids_legacy(words):
    s = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if t: s.add(t[0])
    return sorted(s)
def ids_rule(ws):
    out, forms = [], {}
    for w in ws:
        for f in (w, " " + w):
            t = tok.encode(f, add_special_tokens=False); forms[f] = t
            if len(t) == 1: out.append(t[0])
    return sorted(set(out)), forms
YES_L, NO_L = ids_legacy(["Yes", " Yes", "yes", " yes", "YES"]), ids_legacy(["No", " No", "no", " no", "NO"])
(YES, fy), (NO, fn) = ids_rule(["Yes", "yes", "YES"]), ids_rule(["No", "no", "NO"])
print("rule ids yes", YES, "no", NO, "| forms", {**fy, **fn}, "| legacy yes", YES_L, "no", NO_L, flush=True)
lm_w = net.lm_head.weight.detach().float().cpu()
allids = sorted(set(YES + NO + YES_L + NO_L))
norm = net.model.norm
torch.save({"ids": allids, "rows": lm_w[allids].clone(), "yes_rule": YES, "no_rule": NO, "yes_legacy": YES_L, "no_legacy": NO_L,
            "final_norm_weight": norm.weight.detach().float().cpu(), "norm_type": type(norm).__name__,
            "norm_eps": float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0))), "forms": {**fy, **fn}},
           OUTPREFIX + "_readout.pt")
del lm_w
n_enc, n_llm = len(enc_layers), cfg.text_config.num_hidden_layers + 1
print(f"MODEL {MID} loaded in {time.time()-t0:.0f}s | enc {n_enc} +proj | llm stages {n_llm} | audio token {AUD}", flush=True)
grab = {}
def hook(name):
    def f(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        grab[name] = h.reshape(-1, h.shape[-1]).mean(dim=0).float().cpu().numpy()
    return f
handles = [enc_layers[i].register_forward_hook(hook(f"enc{i}")) for i in range(n_enc)]
handles.append(projector.register_forward_hook(hook("proj")))
rows = list(csv.DictReader(open(MANIFEST)))
CK = OUTPREFIX + "_ckpt"; os.makedirs(CK, exist_ok=True)
CH = 50
for c0 in range(0, len(rows), CH):
    cf = f"{CK}/chunk_{c0:04d}.npz"
    if os.path.exists(cf): print("skip done chunk", c0, flush=True); continue
    enc_f, prj_f, llm_f, ans_f, meta = [], [], [], [], []
    for i in range(c0, min(c0 + CH, len(rows))):
        r = rows[i]
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
            mask = (inp["input_ids"][0] == AUD); assert int(mask.sum()) > 0, "no audio tokens in the prompt"
            llm = torch.stack([h[0][mask].mean(dim=0).float().cpu() for h in out.hidden_states]).numpy()
            ans = torch.stack([h[0][-1].float().cpu() for h in out.hidden_states]).numpy()
            pr = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        py, pn = float(pr[YES].sum()), float(pr[NO].sum()); pyl, pnl = float(pr[YES_L].sum()), float(pr[NO_L].sum())
        top = int(pr.argmax())
        enc_f.append(np.stack([grab[f"enc{j}"] for j in range(n_enc)])); prj_f.append(grab["proj"]); llm_f.append(llm); ans_f.append(ans)
        meta.append(dict(clip=os.path.basename(r["path"]), orig_clip_id=r["orig_clip_id"], speaker=r["speaker"], label=int(r["label"]), set=r["set"],
                         p_yes=py / (py + pn), answer_mass=py + pn, p_yes_legacy=pyl / (pyl + pnl + 1e-12), mass_legacy=pyl + pnl,
                         top1_token=tok.decode([top]), n_audio_tokens=int(mask.sum()), dur_s=round(len(x) / 16000, 3)))
        if i % 25 == 0: print(f"{i+1}/{len(rows)} {meta[-1]['clip']} label {meta[-1]['label']} p_yes {meta[-1]['p_yes']:.3f} mass {meta[-1]['answer_mass']:.3f} {time.time()-t0:.0f}s", flush=True)
    np.savez(cf + ".tmp.npz", enc=np.stack(enc_f), proj=np.stack(prj_f)[:, None, :], llm=np.stack(llm_f), ans=np.stack(ans_f), meta=json.dumps(meta))
    os.replace(cf + ".tmp.npz", cf)
    print(f"CHECKPOINT chunk {c0} saved {time.time()-t0:.0f}s", flush=True)
for h in handles: h.remove()
parts = [np.load(f"{CK}/chunk_{c0:04d}.npz", allow_pickle=True) for c0 in range(0, len(rows), CH)]
E = np.concatenate([p["enc"] for p in parts]); PJ = np.concatenate([p["proj"] for p in parts])
L = np.concatenate([p["llm"] for p in parts]); A = np.concatenate([p["ans"] for p in parts])
meta = sum([json.loads(str(p["meta"])) for p in parts], [])
assert len(meta) == len(rows) and all(m["clip"] == os.path.basename(r["path"]) for m, r in zip(meta, rows))
y = np.array([m["label"] for m in meta]); spk = np.array([m["speaker"] for m in meta]); pyes = np.array([m["p_yes"] for m in meta])
np.savez(OUTPREFIX + "_states.npz", enc=E, proj=PJ, llm=L, ans=A, label=y, spk=spk, p_yes=pyes,
         name=np.array([m["orig_clip_id"] for m in meta]), cut_name=np.array([m["clip"] for m in meta]), set=np.array([m["set"] for m in meta]))
with open(OUTPREFIX + "_zeroshot_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh); cols = list(meta[0].keys()); w.writerow(cols + ["prompt"])
    for m in meta: w.writerow([m[c] for c in cols] + [P])
from scipy.stats import rankdata
def rauc(y, s):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
AUC = rauc(y, pyes)
json.dump({"checkpoint": MID, "clip_list": os.path.abspath(MANIFEST), "condition": COND, "prompt": P, "n": len(y), "n_positive": int(y.sum()),
           "auc_rule_pyes": round(AUC, 4), "auc_legacy_pyes": round(rauc(y, np.array([m["p_yes_legacy"] for m in meta])), 4),
           "median_answer_mass": round(float(np.median([m["answer_mass"] for m in meta])), 4), "window_seconds": WIN, "dtype": "bfloat16",
           "yes_ids_rule": YES, "no_ids_rule": NO, "yes_ids_legacy": YES_L, "no_ids_legacy": NO_L, "encoder_layers": n_enc, "llm_stages": n_llm,
           "date": datetime.date.today().isoformat()}, open(OUTPREFIX + "_zeroshot.json", "w"), indent=1)
print(f"RESULT zero-shot n={len(y)} AUC={AUC:.4f} {time.time()-t0:.0f}s", flush=True)
print("STATES", {k: list(v.shape) for k, v in (("enc", E), ("proj", PJ), ("llm", L), ("ans", A))}, flush=True)
print("EXTRACT DONE", flush=True)
