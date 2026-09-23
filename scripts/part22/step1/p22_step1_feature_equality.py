"""PART 22 step 1: are the features the model receives identical between the paper call and truncation off,
(a) for a clip under 300 s, (b) for the first 300 s of a 900 s clip, (c) for a 900 s clip vs the same clip cut at 300 s (paper call)?"""
import numpy as np, torch
from transformers import Qwen2_5OmniProcessor as Proc
proc = Proc.from_pretrained("Qwen/Qwen2.5-Omni-7B")
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
rng = np.random.default_rng(0)
def feats(x, **kw):
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, **kw)
    m = inp["feature_attention_mask"][0].bool()
    return inp["input_features"][0][:, m].numpy(), inp["input_ids"]
# speech-like level changes so the per-clip log-mel max differs between the first 300 s and the whole clip
x200 = (0.05 * rng.standard_normal(200 * 16000)).astype(np.float32)
x900 = (0.05 * rng.standard_normal(900 * 16000)).astype(np.float32); x900[600 * 16000:] *= 20.0
a, ia = feats(x200); c, ic = feats(x200, audio_kwargs={"truncation": False})
print("200 s clip: paper vs truncation off: same shape", a.shape == c.shape, "max|diff|", float(np.abs(a - c).max()), "input_ids equal", torch.equal(ia, ic))
full_paper, i1 = feats(x900); cut_paper, i2 = feats(x900[:300 * 16000])
print("900 s clip vs same clip cut at 300 s, both paper call: shape", full_paper.shape, cut_paper.shape, "max|diff|", float(np.abs(full_paper - cut_paper).max()), "input_ids equal", torch.equal(i1, i2))
off, _ = feats(x900, audio_kwargs={"truncation": False})
d = np.abs(off[:, :30000] - full_paper)
print("900 s clip, truncation off, first 30000 frames vs paper call: max|diff|", float(d.max()), "frac frames changed", float((d.max(0) > 0).mean()))
x900b = (0.05 * rng.standard_normal(900 * 16000)).astype(np.float32)
p2, _ = feats(x900b); o2, _ = feats(x900b, audio_kwargs={"truncation": False})
print("900 s stationary clip, truncation off, first 30000 frames vs paper call: max|diff|", float(np.abs(o2[:, :30000] - p2).max()))
print("FEATURE_EQUALITY_DONE")
