"""Diagnose the val10 mismatch: per clip, score twice in one process, save discrete audio token ids,
whisper feature checksum, and per-layer mean over the audio span (same hook as kimi_probe.py) to compare with
the canonical overnight2/kimi/kimi_pitt_states.npz rows."""
import os, sys, csv, glob, time, tempfile, hashlib, json
import numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, "/workspace/kimi_src")
from kimia_infer.api.kimia import KimiAudio
MAN, OUTP = sys.argv[1:3]
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
CACHE = glob.glob("/workspace/hf/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/*")[0]
km = KimiAudio(model_path=CACHE, load_detokenizer=False); alm = km.alm
lm = alm.model if hasattr(alm, "model") else alm.transformer; layers = lm.layers
YES = [7414, 9454, 9693, 9834, 14004]; NO = [902, 2152, 2308, 2753, 8996]
grab = {}
def hook(i):
    def f(_m, _i, out): grab[i] = (out[0] if isinstance(out, tuple) else out).detach()
    return f
hs = [layers[i].register_forward_hook(hook(i)) for i in range(len(layers))]
TMP = tempfile.mkdtemp(); dev = torch.cuda.current_device()
res_all = {}
print("GPU", torch.cuda.get_device_name(0), "torch", torch.__version__, "cudnn", torch.backends.cudnn.version(), flush=True)
for r in csv.DictReader(open(MAN)):
    out = []
    for rep in range(2):
        x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
        cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
        h = km.prompt_manager.get_prompt([{"role": "user", "message_type": "text", "content": P},
                                           {"role": "user", "message_type": "audio", "content": cut}], output_type="text")
        aids, tids, cm, _, _ = h.to_tensor(); feat = h.continuous_feature
        aids = aids.to(dev); tids = tids.to(dev); cm = cm.to(dev)
        pos = torch.arange(0, aids.shape[1], device=dev).unsqueeze(0).long()
        with torch.no_grad():
            o = alm.forward(input_ids=aids, text_input_ids=tids, whisper_input_feature=feat, is_continuous_mask=cm, position_ids=pos, past_key_values=None)
            lg = o.logits if hasattr(o, "logits") else o; tl = lg[1] if isinstance(lg, (tuple, list)) else lg
        pr = torch.softmax(tl[0, -1].float(), -1).cpu().numpy(); py, pn = pr[YES].sum(), pr[NO].sum()
        m = cm[0].bool()
        llm = torch.stack([grab[i][0][m].mean(0).float().cpu() for i in range(len(layers))]).numpy()
        fl = [f.float().cpu().numpy() if torch.is_tensor(f) else np.asarray(f) for f in (feat if isinstance(feat, (list, tuple)) else [feat])]
        out.append(dict(p_yes=float(py / (py + pn)), aids=aids[0].cpu().numpy().tolist(), wav_sha=hashlib.sha256(x.tobytes()).hexdigest(),
                        feat_sum=[float(np.abs(f).sum()) for f in fl], llm=llm))
    same_ids = out[0]["aids"] == out[1]["aids"]
    print(f"{r['clip']} rep0 {out[0]['p_yes']:.6f} rep1 {out[1]['p_yes']:.6f} same_audio_ids {same_ids} n_tok {len(out[0]['aids'])} feat {out[0]['feat_sum']} vs {out[1]['feat_sum']}", flush=True)
    res_all[r["clip"]] = out
np.save(OUTP, np.array([res_all], dtype=object), allow_pickle=True)
print("DIAG DONE", flush=True)
