"""PART 23 verifier, item 1: run the Qwen2.5-Omni processor myself on the whole-interview windows.
Paper call: proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)
Audio: librosa.load(<cut>/<pid>.wav, sr=16000). Runs on the pod, CPU only, no model weights loaded.
usage: python3 pod_tokens_verify.py <cut_dir> <windows_full.csv> <out_csv> [pid ...]
"""
import sys, csv, json, time
import numpy as np
import librosa
import transformers
from transformers import AutoProcessor

CUT, WINCSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
only = sys.argv[4:]
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
tok = proc.tokenizer
AUD = tok.convert_tokens_to_ids("<|AUDIO|>")
win = {r["pid"]: r for r in csv.DictReader(open(WINCSV))}
pids = only if only else sorted(win, key=int)


def formula(frames):
    a = (frames - 1) // 2 + 1
    return (a - 2) // 2 + 1


rows = []
t0 = time.time()
for pid in pids:
    x, sr = librosa.load(f"{CUT}/{pid}.wav", sr=16000)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)
    ids = inp["input_ids"][0]
    fam = inp["feature_attention_mask"][0]
    n_tok = int((ids == AUD).sum())
    frames = int(fam.sum())
    rows.append(dict(pid=pid, n_samples=len(x), dur_s=len(x) / 16000.0, win_dur_s=float(win[pid]["win_dur_s"]),
                     capped=int(win[pid]["capped"]), feat_frames=frames, feat_T=int(inp["input_features"].shape[-1]),
                     samples_div_160=len(x) // 160, n_audio_tok=n_tok, tok_formula_from_frames=int(formula(frames)),
                     seq_len=int(ids.shape[0])))
    print(pid, rows[-1]["dur_s"], frames, n_tok, round(time.time() - t0, 1), flush=True)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
json.dump(dict(transformers=transformers.__version__, numpy=np.__version__, librosa=librosa.__version__,
               audio_token_id=AUD, n=len(rows), call='proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)',
               sec=time.time() - t0), open(OUT + ".json", "w"), indent=1)
print("DONE", len(rows), flush=True)
