"""PART 22 step 1 extra: does Qwen2_5OmniProcessor.from_pretrained(MID, chunk_length=900) lift the cut at load time?"""
import numpy as np, transformers
from transformers import Qwen2_5OmniProcessor as Proc
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
proc = Proc.from_pretrained("Qwen/Qwen2.5-Omni-7B", chunk_length=900)
fe = proc.feature_extractor
print("transformers", transformers.__version__, "| after from_pretrained(chunk_length=900):",
      {k: getattr(fe, k) for k in ("chunk_length", "n_samples", "nb_max_frames", "sampling_rate")})
AUD = proc.tokenizer.convert_tokens_to_ids(proc.audio_token)
for d in (200.0, 900.0):
    x = np.zeros(int(d * 16000), dtype=np.float32) + 0.01
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    print(f"dur {d}s feat {list(inp['input_features'].shape)} mask {int(inp['feature_attention_mask'].sum())} audio_tok {int((inp['input_ids'][0]==AUD).sum())}")
