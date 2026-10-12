"""Independent re-run of af3_score.py lines 28 to 34 (processor only, no model weights).
For each clip: load like line 28, cut like line 30, write a temp wav like line 31, then call
proc.apply_chat_template exactly like lines 32 to 34. Record what the feature extractor receives
(a wrapper around feature_extractor.__call__), then read the returned tensors.
usage: indep_af3_length.py OUT.csv CLIP_DIR clip1.wav clip2.wav ...
"""
import os, sys, csv, tempfile
import numpy as np, torch, librosa, soundfile as sf, transformers
from transformers import AutoProcessor

OUT, CLIP_DIR, CLIPS = sys.argv[1], sys.argv[2], sys.argv[3:]
MID, REV = "nvidia/audio-flamingo-3-hf", "7d4bae64ee29878af6504ae6f6bb3e40492838ad"
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
proc = AutoProcessor.from_pretrained(MID, revision=REV)
fe = proc.feature_extractor
seen = []
orig_fe_call = type(fe).__call__
def spy(self, raw, *a, **k):
    seen.append([int(np.asarray(c).shape[0]) for c in raw])
    return orig_fe_call(self, raw, *a, **k)
type(fe).__call__ = spy
snd = proc.audio_token_id
print("transformers", transformers.__version__, "torch", torch.__version__, "librosa", librosa.__version__,
      "| processor", type(proc).__name__, "chunk_length", fe.chunk_length, "hop", fe.hop_length, "max_audio_len", proc.max_audio_len, flush=True)
TMP = tempfile.mkdtemp(prefix="af3chk_")
rows = []
for name in CLIPS:
    path = os.path.join(CLIP_DIR, name)
    for mode in ("as_run_cut30", "nocut_contrast"):
        x, _ = librosa.load(path, sr=16000)          # line 28
        src = len(x)
        if mode == "as_run_cut30": x = x[:16000 * 30]  # line 30 (NOCUT is False: no --nocut on line 52)
        cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)  # line 31
        conv = [{"role": "user", "content": [{"type": "audio", "path": cut}, {"type": "text", "text": P}]}]  # line 32
        seen.clear()
        inp = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")  # lines 33-34
        chunks = seen[0]
        feat = tuple(inp["input_features"].shape)
        mask = inp["input_features_mask"].sum(-1).tolist()
        ntok = int((inp["input_ids"] == snd).sum())
        secs = sum(mask) * fe.hop_length / 16000
        r = dict(clip=name, mode=mode, source_samples=src, source_s=round(src / 16000, 3),
                 samples_to_feature_extractor=sum(chunks), chunks=len(chunks), chunk_samples="+".join(map(str, chunks)),
                 input_features="x".join(map(str, feat)), mask_frames="+".join(map(str, mask)),
                 sound_tokens=ntok, seconds_reaching_model=round(secs, 3))
        rows.append(r); print(r, flush=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
