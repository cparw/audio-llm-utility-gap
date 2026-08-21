#!/usr/bin/env python3
"""LLM-stage states for every AD segment: embedding row after audio merge
(= projector output at audio positions, stage 0) then each decoder layer,
mean-pooled over the audio token positions only. Same prompt as scoring."""
import csv, os, time
import numpy as np, torch, librosa
D = "/Volumes/G-Drive Pro/DementiaBank"
os.environ.setdefault("HF_HOME", f"{D}/hf_cache")
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

P_MAIN = ("Based only on this recording, does this speaker show signs of dementia? "
          "Answer with one word, Yes or No.")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", dtype=torch.float16).to(dev).eval()
aud_id = getattr(model.config, "audio_token_index", None) or getattr(model.config, "audio_token_id", None)
if aud_id is None:
    aud_id = proc.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
print(f"device {dev}, audio token id {aud_id}", flush=True)

rows = list(csv.DictReader(open(f"{D}/pitt_conflict_manifest.csv")))
feats, meta = [], []
t0 = time.time()
for i, r in enumerate(rows):
    sp = r.get("segment_path", "")
    if not sp: continue
    try:
        x, _ = librosa.load(f"{D}/{sp}", sr=16000)
        conv = [{"role":"user","content":[{"type":"audio","audio":x},
                                          {"type":"text","text":P_MAIN}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[x], sampling_rate=16000,
                      return_tensors="pt", padding=True)
        inputs = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inputs.items()}
        mask = (inputs["input_ids"][0] == aud_id)
        assert int(mask.sum()) > 0, "no audio positions found"
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = torch.stack([h[0][mask].mean(dim=0).float().cpu() for h in out.hidden_states])
        feats.append(hs.half().numpy())
        meta.append((sp, r["set"], r["label"], r["grp"]+r["spk"]))
    except Exception as e:
        print(f"fail {sp}: {repr(e)[:120]}", flush=True)
    if (i+1) % 25 == 0:
        el = time.time()-t0
        print(f"progress {i+1}/{len(rows)}  {el/60:.1f} min  stages {feats[-1].shape if feats else None}", flush=True)
    if (i+1) % 100 == 0 and feats:
        np.savez_compressed(f"{D}/ad_llm_states_partial.npz",
            feats=np.stack(feats), path=[m[0] for m in meta], set_=[m[1] for m in meta],
            label=[int(m[2]) for m in meta], spk=[m[3] for m in meta])
np.savez_compressed(f"{D}/ad_llm_states.npz",
    feats=np.stack(feats), path=[m[0] for m in meta], set_=[m[1] for m in meta],
    label=[int(m[2]) for m in meta], spk=[m[3] for m in meta])
print(f"saved {len(meta)} segments, shape {np.stack(feats).shape}", flush=True)
