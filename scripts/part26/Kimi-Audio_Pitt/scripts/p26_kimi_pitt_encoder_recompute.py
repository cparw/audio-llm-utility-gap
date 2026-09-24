"""PART 26 Kimi-Audio Pitt: independent recompute of the Kimi Whisper-encoder states for a few clips on the Mac.
Adapted from part26/Kimi-Audio_E-DAIC/scripts/p26_kimi_edaic_encoder_recompute.py (same method, Pitt paths and clips).
Different code path from the pod: Hugging Face transformers WhisperModel (eager attention, float32, CPU) loaded from the
whisper-large-v3 folder of the same Kimi-Audio-7B-Instruct snapshot (9a82a84c..., model.safetensors sha256 d677ab65...,
checked below), Hugging Face WhisperFeatureExtractor (128 mel bins) for the log-mel, audio from the Mac pitt468 files with
librosa.load(sr=16000)[:480000]. The pod used Kimi's own whisper code (bfloat16, GPU) and its own log-mel.
For each clip and each of the 32 layers, the layer output mean over the 1500 frames is compared with the pod enc state
(cosine, relative L2), and the best-matching pod layer is checked to be the same layer index. The final layer_norm output
mean is compared with the pod enc_final_ln (the mean of Kimi's continuous feature).
usage: /usr/local/bin/python3 p26_kimi_pitt_encoder_recompute.py"""
import os, sys, json, datetime, hashlib
import numpy as np, torch, librosa, transformers
from transformers import WhisperModel, WhisperFeatureExtractor
C = "scores/part26/Kimi-Audio_Pitt"
W = f"{C}/verify/hf_whisper/whisper-large-v3"
W_SHA = "d677ab655d1916439c5868c819a0e48cdac574defab83c69b0bbc2b7b31a9f06"  # HF LFS oid of Kimi-Audio-7B-Instruct@9a82a84c whisper-large-v3/model.safetensors
SRC = "<local data dir>/paper1_local_runs/clips_local/clips_for_af3/pitt468"
STATES = f"{C}/pull/p26-kimi-pitt-gpu/out/kimi_pitt_encstates.npz"
torch.set_num_threads(8)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
z = np.load(STATES, allow_pickle=True); names = [str(v) for v in z["name"]]; E = z["enc"]; FL = z["enc_final_ln"]
PICK = [0, 117, 234, 351, 467]
CLIPS = [names[i] for i in PICK]
w_sha = sha(f"{W}/model.safetensors")
fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000)
enc = WhisperModel.from_pretrained(W, torch_dtype=torch.float32, attn_implementation="eager").encoder.eval()
grab = {}
hs = [l.register_forward_hook((lambda i: (lambda m, a, o: grab.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].mean(0).numpy())))(i))
      for i, l in enumerate(enc.layers)]
res = {}
for c in CLIPS:
    x = librosa.load(os.path.join(SRC, c), sr=16000)[0][:480000]
    feats = fe(x, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad(): out = enc(feats)
    H = np.stack([grab[i] for i in range(len(enc.layers))]); k = names.index(c); P = E[k].astype(np.float64)
    fin = out.last_hidden_state[0].mean(0).numpy().astype(np.float64); pf = FL[k].astype(np.float64)
    cos = [float(H[i] @ P[i] / (np.linalg.norm(H[i]) * np.linalg.norm(P[i]))) for i in range(32)]
    rel = [float(np.linalg.norm(H[i] - P[i]) / np.linalg.norm(P[i])) for i in range(32)]
    best = [int(np.argmin([np.linalg.norm(H[i] - P[j]) for j in range(32)])) for i in range(32)]
    fcos = float(fin @ pf / (np.linalg.norm(fin) * np.linalg.norm(pf)))
    res[c] = {"row": k, "samples": int(len(x)), "min_cosine": min(cos), "median_cosine": float(np.median(cos)), "max_rel_l2": max(rel),
              "median_rel_l2": float(np.median(rel)), "best_match_is_same_layer": best == list(range(32)),
              "final_ln_cosine": fcos, "final_ln_rel_l2": float(np.linalg.norm(fin - pf) / np.linalg.norm(pf)),
              "cosine_per_layer": cos, "rel_l2_per_layer": rel}
    print(c, f"min cos {min(cos):.5f} max rel {max(rel):.4f} same-layer match {best == list(range(32))} final_ln cos {fcos:.5f}", flush=True)
out = {"created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "script": os.path.abspath(__file__),
       "method": __doc__, "weights": W, "weights_sha256": w_sha, "weights_sha256_equals_kimi_snapshot": w_sha == W_SHA,
       "transformers": transformers.__version__, "torch": torch.__version__, "librosa": librosa.__version__,
       "states": STATES, "states_sha256": sha(STATES), "clips": res,
       "pass": (w_sha == W_SHA) and all(r["best_match_is_same_layer"] and r["min_cosine"] > 0.99 and r["samples"] == 480000 for r in res.values())}
os.makedirs(f"{C}/verify", exist_ok=True)
json.dump(out, open(f"{C}/verify/p26_kimi_pitt_encoder_recompute.json", "w"), indent=1)
print("ENC_RECOMPUTE pass", out["pass"])
