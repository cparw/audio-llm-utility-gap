"""PART26 Kimi-Audio encoder states. A copy of release/podD2_final/kimi_probe.py (the script behind the saved Kimi zero-shot
files) with ONE addition: forward hooks on the 32 Whisper-large-v3 encoder layers inside kimia_infer
(km.prompt_manager.whisper_model.speech_encoder.layers). That encoder produces history.continuous_feature, the whisper path
Kimi adds to the discrete speech tokens before the language model.
Pooling: kimia_infer runs the encoder on the 30 s padded mel and keeps the first token_len*4 frames,
token_len = (L-1)//1280 + 1 (WhisperEncoder.forward). Each layer output is cut to those same frames and mean-pooled over them
with h.mean(dim=0).float(), the pooling of the Qwen2.5-Omni encoder states (forward hook on each encoder layer, mean over the
audio frames; part25/A/podfiles/extract_probe_layers.py). enc = the 32 layer outputs, as for Qwen2.5-Omni.
Also saved for reference, not probed: enc_ln = the final encoder layer_norm output (what Kimi passes on) pooled the same way,
and the max abs difference between that layer_norm output and the continuous feature Kimi built (0 means the hooks sit on the
exact path Kimi uses).
Everything else (30 s window, prompt, LM hooks, p_yes) is unchanged, so the saved Kimi zero-shot scores and LM states are a
reproduction check on this run.
usage: kimi_enc_probe.py MANIFEST(path,label,speaker) OUTPREFIX COND(ad|pd|mdd)"""
import os, sys, csv, json, time, glob, datetime, tempfile
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
print(f"Kimi loaded in {time.time()-t0:.0f}s | language model layers {len(layers)} | snapshot {CACHE}", flush=True)
tok = km.prompt_manager.text_tokenizer if hasattr(km.prompt_manager, "text_tokenizer") else None
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
grab = {}
def hook(i):
    def f(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        grab[i] = h.detach()
    return f
handles = [layers[i].register_forward_hook(hook(i)) for i in range(len(layers))]
# ---- PART26 addition: Whisper encoder hooks ----
SE = km.prompt_manager.whisper_model.speech_encoder
ENC_LAYERS = SE.layers
N_ENC = len(ENC_LAYERS)
print(f"whisper encoder {type(SE).__module__}.{type(SE).__name__} layers {N_ENC} d_model {SE.config.d_model} "
      f"dtype {next(SE.parameters()).dtype}", flush=True)
egrab = {}
def ehook(key):
    def f(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        egrab.setdefault(key, []).append(h.detach())
    return f
handles += [ENC_LAYERS[i].register_forward_hook(ehook(i)) for i in range(N_ENC)]
handles.append(SE.layer_norm.register_forward_hook(ehook("ln")))
# ------------------------------------------------
rows = list(csv.DictReader(open(MAN)))
llm_f, ans_f, enc_f, encln_f, meta, vframes, lnfeat_diff = [], [], [], [], [], [], []
TMP = tempfile.mkdtemp(prefix="kimi30_")
for n, r in enumerate(rows):
    # Kimi's tokenizer reads a path and consumes the whole file, so cut the same 30 s window every other model used
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
    chats = [{"role": "user", "message_type": "text", "content": P},
             {"role": "user", "message_type": "audio", "content": cut}]
    egrab.clear()
    history = km.prompt_manager.get_prompt(chats, output_type="text")
    aids, tids, cont_mask, _, _ = history.to_tensor()
    feat = history.continuous_feature
    dev = torch.cuda.current_device()
    aids = aids.to(dev); tids = tids.to(dev); cont_mask = cont_mask.to(dev)
    pos = torch.arange(0, aids.shape[1], device=dev).unsqueeze(0).long()
    with torch.no_grad():
        res = alm.forward(input_ids=aids, text_input_ids=tids,
                                                   whisper_input_feature=feat, is_continuous_mask=cont_mask,
                                                   position_ids=pos, past_key_values=None)
        lg = res.logits if hasattr(res, "logits") else res     # Kimi: logits=(audio_logits, text_logits)
        text_logits = lg[1] if isinstance(lg, (tuple, list)) else lg
    m = cont_mask[0].bool()                     # the audio span: positions carrying continuous whisper features
    llm = torch.stack([grab[i][0][m].mean(dim=0).float().cpu() for i in range(len(layers))]).numpy()
    ans = torch.stack([grab[i][0][-1].float().cpu() for i in range(len(layers))]).numpy()
    pr = torch.softmax(text_logits[0, -1].float(), dim=-1).cpu().numpy()
    py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    # ---- PART26 addition: encoder pooling over the frames Kimi keeps ----
    Lk = len(librosa.load(cut, sr=16000)[0])            # the samples extract_whisper_feat reads from the cut file
    token_len = (Lk - 1) // (160 * 8) + 1; T = token_len * 4
    assert all(len(egrab.get(i, [])) == 1 for i in range(N_ENC)) and len(egrab.get("ln", [])) == 1, \
        f"encoder ran {[len(egrab.get(i, [])) for i in range(N_ENC)]} times, expected once (one 30 s segment)"
    assert egrab[0][0].shape[0] == 1 and egrab[0][0].shape[1] >= T, f"unexpected encoder output {tuple(egrab[0][0].shape)} T={T}"
    f0 = feat[0] if isinstance(feat, (list, tuple)) else feat
    assert f0.shape[-2] == token_len, f"continuous feature {tuple(f0.shape)} vs token_len {token_len}"
    enc = torch.stack([egrab[i][0][0, :T].mean(dim=0).float().cpu() for i in range(N_ENC)]).numpy()
    lnh = egrab["ln"][0][0, :T]
    encln = lnh.mean(dim=0).float().cpu().numpy()
    d = float((lnh.reshape(1, T // 4, -1).float().cpu() - f0.float().cpu()).abs().max())
    enc_f.append(enc); encln_f.append(encln); vframes.append(T); lnfeat_diff.append(d)
    # ----------------------------------------------------------------------
    llm_f.append(llm); ans_f.append(ans)
    meta.append((os.path.basename(r["path"]), str(r["speaker"]), int(r["label"]), py / (py + pn + 1e-12), py + pn))
    if n % 25 == 0 or n == len(rows) - 1:
        print(f"{n+1}/{len(rows)} {meta[-1][0]} p_yes {meta[-1][3]:.3f} mass {meta[-1][4]:.3f} enc_frames {T} of "
              f"{egrab[0][0].shape[1]} ln_vs_feat {d:.3g} {time.time()-t0:.0f}s", flush=True)
for h in handles: h.remove()
L, A = np.stack(llm_f), np.stack(ans_f)
E, EL = np.stack(enc_f), np.stack(encln_f)[:, None, :]
y = np.array([m[2] for m in meta]); spk = np.array([m[1] for m in meta]); pyes = np.array([m[3] for m in meta])
np.savez_compressed(OUTP + "_states.npz", enc=E, enc_ln=EL, llm=L, ans=A, label=y, spk=spk, p_yes=pyes,
                    name=[m[0] for m in meta], enc_frames=np.array(vframes), ln_vs_feat_maxabs=np.array(lnfeat_diff))
with open(OUTP + "_zeroshot_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_yes", "mass", "prompt"])
    for m in meta: w.writerow(list(m) + [P])
from sklearn.metrics import roc_auc_score
AUC = float(roc_auc_score(y, pyes))
json.dump({"checkpoint": "moonshotai/Kimi-Audio-7B-Instruct", "snapshot": CACHE, "clip_list": os.path.abspath(MAN), "condition": COND,
           "prompt": P, "n": len(y), "auc": round(AUC, 4), "llm_stages": int(L.shape[1]),
           "encoder": "Whisper-large-v3 encoder inside kimia_infer (prompt_manager.whisper_model.speech_encoder), bf16",
           "encoder_layers": int(E.shape[1]), "encoder_dim": int(E.shape[2]),
           "encoder_pooling": "forward hook on each encoder layer, output cut to the token_len*4 frames Kimi keeps, h.mean(dim=0).float()",
           "enc_frames_min_max": [int(min(vframes)), int(max(vframes))],
           "ln_vs_continuous_feature_maxabs": float(max(lnfeat_diff)),
           "note": "PART26: kimi_probe.py plus Whisper encoder hooks; LM stages and zero-shot unchanged",
           "window_seconds": 30, "dtype": "bfloat16",
           "date": datetime.date.today().isoformat()}, open(OUTP + "_zeroshot.json", "w"), indent=1)
print(f"RESULT kimi zero-shot {os.path.basename(MAN)} n={len(y)} AUC={AUC:.4f}  llm stages {L.shape[1]}  {time.time()-t0:.0f}s", flush=True)
print("STATES", {k: list(v.shape) for k, v in (("enc", E), ("enc_ln", EL), ("llm", L), ("ans", A))},
      "ln_vs_feat_max", max(lnfeat_diff), flush=True)
