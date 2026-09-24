"""PART26 Kimi-Audio ENCODER states. NEW SCRIPT, written 24 Sep 2026 for PART26 (no earlier run extracted Kimi encoder
states; every earlier Kimi run hooked only the language model).

What is new: forward hooks on the 32 layers of the Whisper-large-v3 encoder inside kimia_infer
(km.prompt_manager.whisper_model.speech_encoder.layers). This is the encoder that builds history.continuous_feature,
the continuous path into the LM. The discrete path (GLM-4-Voice speech tokens) is not hooked.

What is copied line for line from release/podD2_final/kimi_probe.py (the script behind the Kimi zero-shot files):
KimiAudio(model_path, load_detokenizer=False); librosa 16 kHz load, first 30 s cut x[:16000*30] written to a temp wav;
the same prompt text per condition; prompt_manager.get_prompt([text prompt, audio], output_type="text");
alm.forward(input_ids, text_input_ids, whisper_input_feature, is_continuous_mask, position_ids); softmax of the text
logits at the last position; the same yes/no id sets (first token of Yes/ Yes/yes/ yes/YES and No...).
The language model hooks of kimi_probe.py are dropped (they do not touch the logits).

Pooling: kimia pads each <=30 s segment to 3000 mel frames (1500 encoder frames) and keeps the first token_len*4
encoder frames, token_len = (L-1)//1280 + 1, L = samples (whisper.py WhisperEncoder.forward). enc = mean over those
kept frames of each layer output: the frames Kimi passes on, the analogue of the Qwen2.5-Omni encoder states (hook
output mean, and the Omni encoder sees no padding). enc_allframes = mean over all 1500 frames (padding included),
saved for reference only.
Per-clip hook check: layer_norm(layer 31 output)[kept frames] mean must equal the mean of history.continuous_feature
(reshaped back to 1280-d frames); the max abs difference is saved as hookcheck_maxdiff.
usage: kimi_enc_extract.py MANIFEST(path,label,speaker) OUTPREFIX COND(ad|pd|mdd) [N_FIRST]"""
import os, sys, csv, json, time, glob, datetime, tempfile, hashlib, platform, collections
import numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, "/workspace/kimi_src")
from kimia_infer.api.kimia import KimiAudio
MAN, OUTP, COND = sys.argv[1:4]
NFIRST = int(sys.argv[4]) if len(sys.argv) > 4 else None
P = {"pd": "Based only on how this person's voice sounds, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.",
     "mdd": "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.",
     "ad": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."}[COND]
CACHE = glob.glob("/workspace/hf/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/*")[0]
t0 = time.time()
km = KimiAudio(model_path=CACHE, load_detokenizer=False)
alm = km.alm
wenc = km.prompt_manager.whisper_model.speech_encoder
layers = wenc.layers; NL = len(layers)
print(f"Kimi loaded in {time.time()-t0:.0f}s from {CACHE} | whisper encoder layers {NL} d_model {wenc.config.d_model} "
      f"dtype {next(wenc.parameters()).dtype} | LM dtype {next(alm.parameters()).dtype}", flush=True)
def yes_no_ids():
    cands = {}
    for name, words in (("yes", ["Yes", " Yes", "yes", " yes", "YES"]), ("no", ["No", " No", "no", " no", "NO"])):
        s = set()
        for w in words:
            try:
                t = km.prompt_manager._tokenize_text(w)
                if t: s.add(t[0])
            except Exception: pass
        cands[name] = sorted(s)
    return cands["yes"], cands["no"]
YES, NO = yes_no_ids()
print("yes ids", YES, "no ids", NO, flush=True)
grab, calls = {}, collections.Counter()
def hook(i):
    def f(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        grab[i] = h.detach(); calls[i] += 1
    return f
handles = [layers[i].register_forward_hook(hook(i)) for i in range(NL)]
rows = list(csv.DictReader(open(MAN)))
if NFIRST: rows = rows[:NFIRST]
enc_f, encall_f, fin_f, meta = [], [], [], []
TMP = tempfile.mkdtemp(prefix="kimi30_")
dev = torch.cuda.current_device()
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
for n, r in enumerate(rows):
    x, _ = librosa.load(r["path"], sr=16000); dfile = len(x) / 16000; x = x[:16000 * 30]
    cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
    grab.clear(); calls.clear()
    chats = [{"role": "user", "message_type": "text", "content": P},
             {"role": "user", "message_type": "audio", "content": cut}]
    history = km.prompt_manager.get_prompt(chats, output_type="text")
    assert all(calls[i] == 1 for i in range(NL)), f"whisper encoder layers ran {dict(calls)} times, expected once each"
    L = len(librosa.load(cut, sr=16000)[0])          # the samples kimia's extract_whisper_feat read from the cut wav
    kf = ((L - 1) // 1280 + 1) * 4                    # encoder frames kimia keeps (whisper.py: token_len * 4)
    T = grab[0].shape[1]
    assert grab[0].shape[0] == 1 and T == 1500 and kf <= T, (tuple(grab[0].shape), kf)
    with torch.no_grad():
        enc = torch.stack([grab[i][0, :kf].float().mean(dim=0) for i in range(NL)]).cpu().numpy()
        encall = torch.stack([grab[i][0].float().mean(dim=0) for i in range(NL)]).cpu().numpy()
        cf = history.continuous_feature
        cf0 = cf[0] if isinstance(cf, (list, tuple)) else cf
        cf_frames = cf0.reshape(-1, wenc.config.d_model)
        assert cf_frames.shape[0] == kf, (tuple(cf0.shape), kf)
        fin = cf_frames.float().mean(dim=0).cpu().numpy()
        ln = wenc.layer_norm(grab[NL - 1][0, :kf]).float().mean(dim=0).cpu().numpy()
        hk = float(np.abs(ln - fin).max())
    aids, tids, cont_mask, _, _ = history.to_tensor()
    feat = history.continuous_feature
    aids = aids.to(dev); tids = tids.to(dev); cont_mask = cont_mask.to(dev)
    pos = torch.arange(0, aids.shape[1], device=dev).unsqueeze(0).long()
    with torch.no_grad():
        res = alm.forward(input_ids=aids, text_input_ids=tids,
                          whisper_input_feature=feat, is_continuous_mask=cont_mask,
                          position_ids=pos, past_key_values=None)
        lg = res.logits if hasattr(res, "logits") else res
        text_logits = lg[1] if isinstance(lg, (tuple, list)) else lg
    pr = torch.softmax(text_logits[0, -1].float(), dim=-1).cpu().numpy()
    py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    assert np.isfinite(enc).all() and np.isfinite(encall).all(), "non-finite encoder state"
    enc_f.append(enc); encall_f.append(encall); fin_f.append(fin)
    meta.append(dict(name=os.path.basename(r["path"]), spk=str(r["speaker"]), label=int(r["label"]),
                     p_yes=py / (py + pn + 1e-12), mass=py + pn, kf=kf, L=L, dfile=dfile, hk=hk, sha=sha256(r["path"])))
    if n % 25 == 0 or n == len(rows) - 1:
        m = meta[-1]
        print(f"{n+1}/{len(rows)} {m['name']} p_yes {m['p_yes']:.4f} mass {m['mass']:.3f} kept_frames {kf} "
              f"hookcheck {hk:.2e} {time.time()-t0:.0f}s", flush=True)
for h in handles: h.remove()
E, EA, F = np.stack(enc_f), np.stack(encall_f), np.stack(fin_f)
g = lambda k: np.array([m[k] for m in meta])
y, spk, pyes = g("label").astype(int), g("spk").astype(str), g("p_yes").astype(float)
np.savez_compressed(OUTP + "_encstates.npz", enc=E, enc_allframes=EA, enc_final_ln=F, label=y, spk=spk, name=g("name").astype(str),
                    p_yes=pyes, mass=g("mass"), kept_frames=g("kf"), samples_heard=g("L"), dur_file_s=g("dfile"),
                    hookcheck_maxdiff=g("hk"), wav_sha256=g("sha").astype(str))
with open(OUTP + "_zeroshot_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_yes", "mass", "kept_frames", "samples_heard", "dur_file_s", "wav_sha256", "prompt"])
    for m in meta: w.writerow([m["name"], m["spk"], m["label"], repr(m["p_yes"]), repr(m["mass"]), m["kf"], m["L"], round(m["dfile"], 4), m["sha"], P])
from sklearn.metrics import roc_auc_score
AUC = float(roc_auc_score(y, pyes)) if len(set(y)) > 1 else None
import transformers
vers = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0)}
try:
    import flash_attn; vers["flash_attn"] = flash_attn.__version__
except Exception: pass
json.dump({"checkpoint": "moonshotai/Kimi-Audio-7B-Instruct", "snapshot": os.path.basename(CACHE), "clip_list": os.path.abspath(MAN),
           "clip_list_sha256": sha256(MAN), "condition": COND, "prompt": P, "n": len(y), "n_positive": int(y.sum()),
           "zeroshot_auc_this_run": AUC, "window_seconds": 30, "dtype": "bfloat16",
           "encoder": "kimia_infer prompt_manager.whisper_model.speech_encoder (Whisper-large-v3 encoder inside Kimi-Audio, the continuous-feature path)",
           "encoder_layers": NL, "enc_shape": list(E.shape),
           "pooling": "enc: mean over the token_len*4 encoder frames kimia keeps, token_len=(L-1)//1280+1; enc_allframes: mean over all 1500 frames",
           "hookcheck_max_over_clips": float(g("hk").max()), "new_script": True,
           "copied_from": "release/podD2_final/kimi_probe.py (forward pass, prompt, yes/no ids, 30 s cut)",
           "versions": vers, "seconds": round(time.time() - t0, 1), "command": " ".join(sys.argv),
           "date_utc": datetime.datetime.utcnow().isoformat() + "Z"}, open(OUTP + "_encstates.json", "w"), indent=1)
print(f"RESULT kimi encoder states {os.path.basename(MAN)} n={len(y)} enc {list(E.shape)} zero-shot AUC this run "
      f"{AUC if AUC is None else round(AUC, 4)} hookcheck max {g('hk').max():.2e} {time.time()-t0:.0f}s", flush=True)
