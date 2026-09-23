"""Kimi-Audio language model stages at the audio token positions and the answer position.
Kimi has one language model even though its input is two summed paths (discrete speech tokens + whisper features),
so the encoder side is skipped and only the LM is probed, exactly as the paper's Method describes for the LLM half.
usage: kimi_probe.py MANIFEST(path,label,speaker) OUTPREFIX COND(ad|pd|mdd)"""
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
rows = list(csv.DictReader(open(MAN)))
llm_f, ans_f, meta = [], [], []
TMP = tempfile.mkdtemp(prefix="kimi30_")
for n, r in enumerate(rows):
    # Kimi's tokenizer reads a path and consumes the whole file, so cut the same 30 s window every other model used
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
    chats = [{"role": "user", "message_type": "text", "content": P},
             {"role": "user", "message_type": "audio", "content": cut}]
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
    llm_f.append(llm); ans_f.append(ans)
    meta.append((os.path.basename(r["path"]), str(r["speaker"]), int(r["label"]), py / (py + pn + 1e-12), py + pn))
    if n % 25 == 0 or n == len(rows) - 1:
        print(f"{n+1}/{len(rows)} {meta[-1][0]} p_yes {meta[-1][3]:.3f} mass {meta[-1][4]:.3f} {time.time()-t0:.0f}s", flush=True)
for h in handles: h.remove()
L, A = np.stack(llm_f), np.stack(ans_f)
y = np.array([m[2] for m in meta]); spk = np.array([m[1] for m in meta]); pyes = np.array([m[3] for m in meta])
np.savez_compressed(OUTP + "_states.npz", llm=L, ans=A, label=y, spk=spk, p_yes=pyes, name=[m[0] for m in meta])
with open(OUTP + "_zeroshot_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_yes", "mass", "prompt"])
    for m in meta: w.writerow(list(m) + [P])
from sklearn.metrics import roc_auc_score
AUC = float(roc_auc_score(y, pyes))
json.dump({"checkpoint": "moonshotai/Kimi-Audio-7B-Instruct", "clip_list": os.path.abspath(MAN), "condition": COND,
           "prompt": P, "n": len(y), "auc": round(AUC, 4), "llm_stages": int(L.shape[1]),
           "note": "language model stages only; Kimi sums discrete speech tokens and whisper features before the LM so there is no single encoder path to probe",
           "window_seconds": 30, "dtype": "bfloat16",
           "date": datetime.date.today().isoformat()}, open(OUTP + "_zeroshot.json", "w"), indent=1)
print(f"RESULT kimi zero-shot {os.path.basename(MAN)} n={len(y)} AUC={AUC:.4f}  llm stages {L.shape[1]}  {time.time()-t0:.0f}s", flush=True)
