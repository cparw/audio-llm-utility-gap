"""PART 26 Kimi-Audio E-DAIC: independent recompute of the Kimi Whisper-encoder states for a few clips on the Mac.
Different code path from the pod: Hugging Face transformers WhisperModel (eager attention, float32, CPU) loaded from the
whisper-large-v3 folder of the same Kimi-Audio-7B-Instruct snapshot (9a82a84c...), Hugging Face WhisperFeatureExtractor
(128 mel bins) for the log-mel, audio from the Mac dcaps_proc files with librosa.load(sr=16000)[:480000].
The pod used Kimi's own modeling_whisper (flash attention, bfloat16) and its torch log-mel. For each clip and each of the
32 layers, the layer output mean over the 1500 frames is compared with the pod enc state (cosine, relative L2), and the
best-matching pod layer is checked to be the same layer index.
usage: /usr/local/bin/python3 p26_kimi_edaic_encoder_recompute.py STATES_NPZ"""
import os, sys, json, datetime
import numpy as np, torch, librosa, transformers
from transformers import WhisperModel, WhisperFeatureExtractor
C = "scores/part26/Kimi-Audio_E-DAIC"
W = f"{C}/verify/hf_whisper/whisper-large-v3"
SRC = "<local data dir>/paper1_local_runs/drive_data/dcaps_proc"
CLIPS = ["300.wav", "717.wav", "450.wav", "600.wav"]
z = np.load(sys.argv[1], allow_pickle=True); names = [str(v) for v in z["name"]]; E = z["enc"]
fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000)
enc = WhisperModel.from_pretrained(W, torch_dtype=torch.float32, attn_implementation="eager").encoder.eval()
grab = {}
hs = [l.register_forward_hook((lambda i: (lambda m, a, o: grab.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].mean(0).numpy())))(i))
      for i, l in enumerate(enc.layers)]
res = {}
for c in CLIPS:
    x = librosa.load(os.path.join(SRC, c), sr=16000)[0][:480000]
    feats = fe(x, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad(): enc(feats)
    H = np.stack([grab[i] for i in range(len(enc.layers))]); P = E[names.index(c)].astype(np.float64)
    cos = [float(H[i] @ P[i] / (np.linalg.norm(H[i]) * np.linalg.norm(P[i]))) for i in range(32)]
    rel = [float(np.linalg.norm(H[i] - P[i]) / np.linalg.norm(P[i])) for i in range(32)]
    best = [int(np.argmin([np.linalg.norm(H[i] - P[j]) for j in range(32)])) for i in range(32)]
    res[c] = {"min_cosine": min(cos), "median_cosine": float(np.median(cos)), "max_rel_l2": max(rel), "median_rel_l2": float(np.median(rel)),
              "best_match_is_same_layer": best == list(range(32)), "cosine_per_layer": cos, "rel_l2_per_layer": rel}
    print(c, f"min cos {min(cos):.5f} max rel {max(rel):.4f} same-layer match {best == list(range(32))}", flush=True)
out = {"created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "script": os.path.abspath(__file__), "method": __doc__,
       "weights": W, "transformers": transformers.__version__, "torch": torch.__version__, "librosa": librosa.__version__,
       "states": os.path.abspath(sys.argv[1]), "clips": res,
       "pass": all(r["best_match_is_same_layer"] and r["min_cosine"] > 0.99 for r in res.values())}
json.dump(out, open(f"{C}/verify/p26_kimi_edaic_encoder_recompute.json", "w"), indent=1)
print("ENC_RECOMPUTE pass", out["pass"])
