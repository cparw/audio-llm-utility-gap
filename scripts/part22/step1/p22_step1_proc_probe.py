"""PART 22 step 1: run the Qwen2.5-Omni processor (transformers 5.17.0, the version the paper's E-DAIC pod used)
on synthetic audio of known length and count what the model would receive. No model weights are loaded.
The paper-call line is copied verbatim from edaic_rerun/part16/POD4B/scripts_recovered/extract_probe_layers.py:82."""
import os, sys, json, numpy as np, transformers, torch
from transformers import Qwen2_5OmniProcessor as Proc
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
MID = "Qwen/Qwen2.5-Omni-7B"
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
print("transformers", transformers.__version__, transformers.__file__, "| torch", torch.__version__)
proc = Proc.from_pretrained(MID)
fe = proc.feature_extractor
print("feature_extractor class:", type(fe).__module__ + "." + type(fe).__name__)
for k in ("chunk_length", "sampling_rate", "n_samples", "nb_max_frames", "hop_length", "n_fft", "feature_size",
          "padding_side", "padding_value", "return_attention_mask", "dither"):
    print(f"  fe.{k} = {getattr(fe, k, 'MISSING')!r}")
AUDIO_KEYS = set(transformers.processing_utils.AudioKwargs.__annotations__)
ti = proc.tokenizer.init_kwargs
print("tokenizer.init_kwargs keys that are also AudioKwargs keys:", sorted(AUDIO_KEYS & set(ti)))
mk = proc._merge_kwargs(Qwen2_5OmniProcessorKwargs, tokenizer_init_kwargs=proc.tokenizer.init_kwargs,
                        sampling_rate=16000, return_tensors="pt", padding=True)
print("merged audio_kwargs for the paper call (before __call__ line 143 forces padding):", mk["audio_kwargs"])
AUD = proc.tokenizer.convert_tokens_to_ids(proc.audio_token)
print("audio token", proc.audio_token, AUD)
rng = np.random.default_rng(0)
def count(x, **kw):
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, **kw)
    n_aud = int((inp["input_ids"][0] == AUD).sum())
    return dict(n_samples_in=len(x), dur_s=len(x) / 16000, feat_shape=list(inp["input_features"].shape),
                mask_frames=int(inp["feature_attention_mask"].sum()), n_audio_tok=n_aud, seq_len=int(inp["input_ids"].shape[1]))
res = {}
durs = [200.0, 299.99, 300.0, 300.01, 600.0, 900.0]
cases = [("A_paper_call", {}),
         ("B_flat_truncation_False", {"truncation": False}),
         ("C_audio_kwargs_truncation_False", {"audio_kwargs": {"truncation": False}}),
         ("D_audio_kwargs_max_length_900s", {"audio_kwargs": {"max_length": 900 * 16000}}),
         ("E_audio_kwargs_trunc_False_and_max_length_900s", {"audio_kwargs": {"truncation": False, "max_length": 900 * 16000}})]
for d in durs:
    x = (0.05 * rng.standard_normal(int(round(d * 16000)))).astype(np.float32)
    for name, kw in cases:
        r = count(x, **kw); res[f"{name}|{d}"] = r
        print(f"{name:48s} dur {d:8.2f}s -> feat {r['feat_shape']} mask_frames {r['mask_frames']:6d} audio_tok {r['n_audio_tok']:6d} seq {r['seq_len']}", flush=True)
# F: override the instance attributes instead of passing kwargs
fe.n_samples = 900 * 16000; fe.nb_max_frames = fe.n_samples // fe.hop_length; fe.chunk_length = 900
for d in (200.0, 900.0):
    x = (0.05 * rng.standard_normal(int(round(d * 16000)))).astype(np.float32)
    r = count(x); res[f"F_override_fe_n_samples_900s|{d}"] = r
    print(f"{'F_override_fe_n_samples_900s':48s} dur {d:8.2f}s -> feat {r['feat_shape']} mask_frames {r['mask_frames']:6d} audio_tok {r['n_audio_tok']:6d} seq {r['seq_len']}", flush=True)
# derivation check from the processor's own formula (processing_qwen2_5_omni.py:151-152)
for fr in (20000, 29999, 30000, 60000, 90000):
    il = (fr - 1) // 2 + 1; print(f"formula frames {fr} -> after conv {il} -> audio tokens {(il - 2) // 2 + 1}")
json.dump(res, open(sys.argv[1], "w"), indent=1)
