"""PART 26: Kimi-Audio Whisper ENCODER states for the encoder probe, plus the zero-shot answer on the same forward pass.

Built from release/podD2_final/kimi_probe.py (the script behind every Kimi-Audio zero-shot file). Everything that
script does is kept as it was: the same checkpoint load, the same prompt per condition, the same 30 s window
(librosa.load at 16 kHz, x[:16000*30], written to a temp wav because Kimi's tokenizer reads a path), the same
yes/no id sets, the same alm.forward call and the same p_yes = P(yes) / (P(yes) + P(no)) at the last position.

NEW in PART 26 (no earlier script did this): forward hooks on the 32 layers of the Whisper-large-v3 encoder inside
kimia_infer (km.prompt_manager.whisper_model.speech_encoder.layers). That encoder builds history.continuous_feature,
the continuous path that Kimi adds to the discrete GLM-4 speech tokens before the LM. The discrete-token tokenizer
(GLM-4-Voice, a separate module) is not hooked.
Pooling, as the Qwen2.5-Omni states: each layer's output hidden state (layer output, before the encoder's final
layer_norm) averaged over time. Kimi pads every clip to 30 s (1500 frames) and then keeps the first token_len*4
frames, token_len = (L-1)//1280 + 1, which is exactly what it passes to the LM. The saved `enc` averages over
those kept frames (Qwen2.5-Omni's encoder has no padding, so its mean is over real frames too). `enc_pad1500`
averages over all 1500 frames, padding included, kept only as a sensitivity copy.
Checks per clip: every hooked layer fired exactly once; kept frames == continuous_feature frames * 4 == token_len*4.
usage: kimi_encstates26.py MANIFEST(path,label,speaker) OUTPREFIX COND(ad|pd|mdd)"""
import os, sys, csv, json, time, glob, datetime, tempfile, hashlib, platform
import numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, "/workspace/kimi_src")
from kimia_infer.api.kimia import KimiAudio
MAN, OUTP, COND = sys.argv[1:4]
P = {"pd": "Based only on how this person's voice sounds, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.",
     "mdd": "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.",
     "ad": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."}[COND]
CACHE = glob.glob("/workspace/hf_cache/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/*")[0]
t0 = time.time()
km = KimiAudio(model_path=CACHE, load_detokenizer=False)
alm = km.alm
lm = alm.model if hasattr(alm, "model") else alm.transformer
layers = lm.layers
enc = km.prompt_manager.whisper_model.speech_encoder
enc_layers = enc.layers
NE = len(enc_layers)
print(f"Kimi loaded in {time.time()-t0:.0f}s | language model layers {len(layers)} | whisper encoder layers {NE} "
      f"({type(enc).__module__}.{type(enc).__name__}) dtype {next(enc.parameters()).dtype}", flush=True)
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
grab, calls = {}, {}
def ehook(i):
    def f(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        grab[i] = h.detach()
        calls[i] = calls.get(i, 0) + 1
    return f
handles = [enc_layers[i].register_forward_hook(ehook(i)) for i in range(NE)]
rows = list(csv.DictReader(open(MAN)))
enc_f, encpad_f, nfr, meta = [], [], [], []
TMP = tempfile.mkdtemp(prefix="kimi30_")
for n, r in enumerate(rows):
    # identical to kimi_probe.py: Kimi's tokenizer reads a path and consumes the whole file, so cut the same 30 s window
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
    chats = [{"role": "user", "message_type": "text", "content": P},
             {"role": "user", "message_type": "audio", "content": cut}]
    grab.clear(); calls.clear()
    history = km.prompt_manager.get_prompt(chats, output_type="text")
    assert sorted(calls) == list(range(NE)) and all(v == 1 for v in calls.values()), f"encoder hooks fired {calls}"
    aids, tids, cont_mask, _, _ = history.to_tensor()
    feat = history.continuous_feature
    cf = feat[0] if isinstance(feat, (list, tuple)) else feat
    L = int(librosa.load(cut, sr=16000)[0].shape[0])          # what extract_whisper_feat reads back from the temp wav
    token_len = (L - 1) // (160 * 8) + 1
    keep = token_len * 4
    assert keep == int(cf.shape[1]) * 4, f"kept frames {keep} vs continuous_feature {tuple(cf.shape)}"
    assert grab[0].shape[1] == 1500 and keep <= 1500, grab[0].shape
    e_keep = torch.stack([grab[i][0, :keep].float().mean(dim=0) for i in range(NE)]).cpu().numpy()
    e_pad = torch.stack([grab[i][0].float().mean(dim=0) for i in range(NE)]).cpu().numpy()
    dev = torch.cuda.current_device()
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
    enc_f.append(e_keep); encpad_f.append(e_pad); nfr.append(keep)
    meta.append((os.path.basename(r["path"]), str(r["speaker"]), int(r["label"]), py / (py + pn + 1e-12), py + pn, L))
    if n % 25 == 0 or n == len(rows) - 1:
        print(f"{n+1}/{len(rows)} {meta[-1][0]} samples {L} kept frames {keep} p_yes {meta[-1][3]:.3f} mass {meta[-1][4]:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
for h in handles: h.remove()
E, EP = np.stack(enc_f).astype(np.float32), np.stack(encpad_f).astype(np.float32)
assert np.isfinite(E).all() and np.isfinite(EP).all()
y = np.array([m[2] for m in meta]); spk = np.array([m[1] for m in meta]); pyes = np.array([m[3] for m in meta])
np.savez_compressed(OUTP + "_enc_states.npz", enc=E, enc_pad1500=EP, enc_frames=np.array(nfr), n_samples=np.array([m[5] for m in meta]),
                    label=y, spk=spk, p_yes=pyes, mass=np.array([m[4] for m in meta]), name=np.array([m[0] for m in meta]))
with open(OUTP + "_rerun_zeroshot_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_yes", "mass", "n_samples", "enc_frames", "prompt"])
    for m, k in zip(meta, nfr): w.writerow([m[0], m[1], m[2], repr(m[3]), repr(m[4]), m[5], k, P])
from sklearn.metrics import roc_auc_score
AUC = float(roc_auc_score(y, pyes))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
import transformers
json.dump({"checkpoint": "moonshotai/Kimi-Audio-7B-Instruct", "snapshot": CACHE, "clip_list": os.path.abspath(MAN),
           "clip_list_sha256": sha(MAN), "condition": COND, "prompt": P, "n": len(y), "rerun_answer_auc": AUC,
           "encoder": "kimia_infer Whisper-large-v3 encoder (prompt_manager.whisper_model.speech_encoder), forward hooks on all "
                      f"{NE} layers, layer outputs before the final encoder layer_norm",
           "enc_pooling": "mean over the first token_len*4 frames (the frames Kimi passes to the LM), token_len=(L-1)//1280+1",
           "enc_pad1500_pooling": "mean over all 1500 frames of the 30 s padded input (sensitivity copy only)",
           "enc_shape": list(E.shape), "window_seconds": 30, "dtype": "bfloat16",
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
                        "gpu": torch.cuda.get_device_name(0)},
           "seconds": round(time.time() - t0, 1), "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "command": " ".join(sys.argv)}, open(OUTP + "_enc_states.json", "w"), indent=1)
print(f"RESULT kimi enc states {os.path.basename(MAN)} n={len(y)} enc {list(E.shape)} rerun answer AUC={AUC:.4f} {time.time()-t0:.0f}s", flush=True)
