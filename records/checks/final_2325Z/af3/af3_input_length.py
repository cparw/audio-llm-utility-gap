"""What did AF3 receive per Pitt clip in the part10 run?
Reproduces af3_score.py lines 28-34 exactly (librosa.load sr=16000 -> x[:16000*30] unless --nocut ->
sf.write temp wav -> proc.apply_chat_template with {"type":"audio","path":cut}), then reads what the
processor built. Processor only, no model weights. Also runs the same clip with no cut, for contrast.
usage: venv/bin/python af3_input_length.py OUT.csv CLIP [CLIP ...]"""
import os, sys, csv, tempfile, hashlib
import numpy as np, librosa, soundfile as sf, torch, transformers
from transformers import AutoProcessor
MID = "nvidia/audio-flamingo-3-hf"
REV = "7d4bae64ee29878af6504ae6f6bb3e40492838ad"   # HF main since 2026-04-13, so what snapshot_download got on 20 Sep
D = "<local data dir>/paper1_local_runs/clips_local/clips_for_af3/pitt468"
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
OUT, CLIPS = sys.argv[1], sys.argv[2:]
print("transformers", transformers.__version__, "torch", torch.__version__, "librosa", librosa.__version__, "soundfile", sf.__version__, flush=True)
proc = AutoProcessor.from_pretrained(MID, revision=REV)
fe = proc.feature_extractor
print(type(proc).__name__, type(fe).__name__, f"chunk_length={fe.chunk_length}s n_samples={fe.n_samples} hop_length={fe.hop_length} "
      f"sampling_rate={fe.sampling_rate} max_audio_len={proc.max_audio_len}s audio_token={proc.audio_token} id={proc.audio_token_id}", flush=True)
# record the waveform lengths the processor itself receives (after the chat template loads the temp wav)
seen = []
_orig_call = type(proc).__call__
def spy(self, text, audio=None, **kw):
    if audio is not None:
        for a in (audio if isinstance(audio, (list, tuple)) else [audio]): seen.append(int(np.asarray(a).shape[0]))
    return _orig_call(self, text, audio=audio, **kw)
type(proc).__call__ = spy
TMP = tempfile.mkdtemp(prefix="af3_30s_")
rows = []
for c in CLIPS:
    path = os.path.join(D, c)
    for mode in ("as_run_cut30", "nocut_contrast"):
        x, _ = librosa.load(path, sr=16000)                      # af3_score.py line 28
        dur = len(x) / 16000                                     # line 29
        if mode == "as_run_cut30": x = x[:16000 * 30]            # line 30: if not NOCUT: x = x[:16000 * 30]
        cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)   # line 31
        conv = [{"role": "user", "content": [{"type": "audio", "path": cut}, {"type": "text", "text": P}]}]   # line 32
        seen.clear()
        inp = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")          # lines 33-34
        feats, mask = inp["input_features"], inp["input_features_mask"]
        n_win = feats.shape[0]; frames = int(mask.sum()); per_win = [int(v) for v in mask.sum(-1)]
        n_tok = int((inp["input_ids"] == proc.audio_token_id).sum())
        post = [int(proc._get_audio_token_length(torch.tensor(v))) for v in per_win]
        r = dict(clip=c, mode=mode, segment_dur_s=round(dur, 3), samples_sliced=len(x), temp_wav_frames=sf.info(cut).frames,
                 samples_passed_to_processor=seen[0] if seen else None, input_features_shape="x".join(map(str, feats.shape)),
                 n_windows=n_win, mask_frames_per_window="+".join(map(str, per_win)), feature_mask_sum_frames=frames,
                 audio_tokens_per_window="+".join(map(str, post)), audio_tokens_in_input_ids=n_tok,
                 input_ids_len=int(inp["input_ids"].shape[1]), seconds_reaching_model=round(frames * fe.hop_length / fe.sampling_rate, 3))
        rows.append(r)
        print(" | ".join(f"{k}={v}" for k, v in r.items()), flush=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("WROTE", OUT)
