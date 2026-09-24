"""PART 26 Kimi-Audio: the release kimi_probe.py (podD2_final, unchanged in every step it already had) plus NEW forward
hooks on the Whisper encoder that builds Kimi's continuous features (history.continuous_feature):
  km.prompt_manager.whisper_model.speech_encoder.layers[0..31]  (Whisper-large-v3 weights shipped in the Kimi checkpoint,
  folder whisper-large-v3, run in bfloat16 exactly as kimia_infer runs it).
Each layer output is mean-pooled over the positions Kimi keeps (the first token_len*4 frames, token_len=(L-1)//1280+1;
1500 frames for a full 30 s clip) with h.mean(dim=0).float(), the same pooling op as the Qwen2.5-Omni hook in
extract_probe_layers.py (h.reshape(-1, D).mean(dim=0).float()). Nothing else changes: same prompt, same 30 s cut
(librosa.load(sr=16000) then x[:16000*30], rewritten to a temp wav because Kimi reads a path), same forward, same
yes/no ids, same llm/ans states. The hooks only read tensors; they do not change any output.
The GLM-4-Voice tokenizer (discrete speech tokens) has its own Whisper-VQ encoder; it is NOT hooked.
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
print(f"Kimi loaded in {time.time()-t0:.0f}s | language model layers {len(layers)}", flush=True)
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
# ---- NEW: Whisper encoder hooks (continuous feature path) ----
WENC = km.prompt_manager.whisper_model.speech_encoder
enc_layers = WENC.layers
n_enc = len(enc_layers)
print(f"whisper encoder layers {n_enc} | d_model {WENC.config.d_model} | dtype {next(WENC.parameters()).dtype}", flush=True)
egrab, ecount, NVALID = {}, {}, {"n": None}
def ehook(i):
    def f(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out          # [1, 1500, 1280] (no input_seq_lens: full padded window)
        assert h.dim() == 3 and h.shape[0] == 1, h.shape
        nv = NVALID["n"]; assert nv is not None and nv <= h.shape[1]
        egrab[i] = h[0, :nv].mean(dim=0).float().cpu().numpy()
        ecount[i] = ecount.get(i, 0) + 1
    return f
ehandles = [enc_layers[i].register_forward_hook(ehook(i)) for i in range(n_enc)]
rows = list(csv.DictReader(open(MAN)))
enc_f, llm_f, ans_f, meta, nvalid_l = [], [], [], [], []
TMP = tempfile.mkdtemp(prefix="kimi30_")
for n, r in enumerate(rows):
    # Kimi's tokenizer reads a path and consumes the whole file, so cut the same 30 s window every other model used
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
    # positions Kimi keeps after the Whisper encoder, from the length Kimi itself will load (librosa.load of the cut)
    Lk = len(librosa.load(cut, sr=16000)[0]); assert Lk <= 480000, Lk
    NVALID["n"] = ((Lk - 1) // (160 * 8) + 1) * 4
    egrab.clear(); ecount.clear()
    chats = [{"role": "user", "message_type": "text", "content": P},
             {"role": "user", "message_type": "audio", "content": cut}]
    history = km.prompt_manager.get_prompt(chats, output_type="text")
    assert sorted(ecount) == list(range(n_enc)) and all(v == 1 for v in ecount.values()), f"encoder hooks fired {ecount}"
    aids, tids, cont_mask, _, _ = history.to_tensor()
    feat = history.continuous_feature
    assert int(feat[0].shape[1]) * 4 == NVALID["n"], (feat[0].shape, NVALID["n"])   # Kimi keeps exactly these frames
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
    enc_f.append(np.stack([egrab[j] for j in range(n_enc)])); nvalid_l.append(NVALID["n"])
    llm_f.append(llm); ans_f.append(ans)
    meta.append((os.path.basename(r["path"]), str(r["speaker"]), int(r["label"]), py / (py + pn + 1e-12), py + pn))
    if n % 25 == 0 or n == len(rows) - 1:
        print(f"{n+1}/{len(rows)} {meta[-1][0]} p_yes {meta[-1][3]:.3f} mass {meta[-1][4]:.3f} enc_frames {NVALID['n']} {time.time()-t0:.0f}s", flush=True)
for h in handles + ehandles: h.remove()
E, L, A = np.stack(enc_f), np.stack(llm_f), np.stack(ans_f)
assert np.isfinite(E).all() and np.isfinite(L).all(), "non-finite states"
y = np.array([m[2] for m in meta]); spk = np.array([m[1] for m in meta]); pyes = np.array([m[3] for m in meta])
np.savez_compressed(OUTP + "_states.npz", enc=E, llm=L, ans=A, label=y, spk=spk, p_yes=pyes, name=[m[0] for m in meta],
                    enc_frames=np.array(nvalid_l))
with open(OUTP + "_zeroshot_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_yes", "mass", "prompt"])
    for m in meta: w.writerow(list(m) + [P])
from sklearn.metrics import roc_auc_score
AUC = float(roc_auc_score(y, pyes))
import transformers
json.dump({"checkpoint": "moonshotai/Kimi-Audio-7B-Instruct", "snapshot": CACHE, "clip_list": os.path.abspath(MAN), "condition": COND,
           "prompt": P, "n": len(y), "auc": round(AUC, 4), "llm_stages": int(L.shape[1]), "encoder_layers": int(E.shape[1]),
           "encoder": "Kimi continuous-feature Whisper encoder (prompt_manager.whisper_model.speech_encoder, whisper-large-v3 folder of the Kimi checkpoint), outputs of layers 0..31 before the final layer_norm",
           "encoder_pooling": "per layer h[0, :token_len*4].mean(dim=0).float() in the model dtype (bfloat16), token_len=(L-1)//1280+1 from the cut wav as Kimi loads it",
           "enc_frames_min": int(min(nvalid_l)), "enc_frames_max": int(max(nvalid_l)),
           "note": "PART 26 NEW encoder hook added to release podD2_final/kimi_probe.py; every other step unchanged",
           "window_seconds": 30, "dtype": "bfloat16", "torch": torch.__version__, "transformers": transformers.__version__,
           "librosa": librosa.__version__, "gpu": torch.cuda.get_device_name(0),
           "date": datetime.date.today().isoformat(), "seconds": round(time.time() - t0, 1)}, open(OUTP + "_zeroshot.json", "w"), indent=1)
print(f"RESULT kimi zero-shot {os.path.basename(MAN)} n={len(y)} AUC={AUC:.4f}  llm stages {L.shape[1]} enc layers {E.shape[1]}  {time.time()-t0:.0f}s", flush=True)
print("STATES", {k: list(v.shape) for k, v in (("enc", E), ("llm", L), ("ans", A))}, flush=True)
