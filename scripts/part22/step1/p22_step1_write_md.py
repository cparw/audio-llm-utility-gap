"""PART 22 step 1: write P22_step1_processor_config.md. Every code quote is read from the file at the stated
line range at write time (no hand-typed code), so the quotes are verbatim by construction."""
import json, os, sys, hashlib, datetime

R = "<local data dir>"
S = f"{R}/scores/part22/src"
T = f"{S}/tf5170/transformers"
OUT = f"{R}/scores/part22/P22_step1_processor_config.md"

def rel(p):
    return p.replace(f"{S}/tf5170/", "").replace(f"{S}/", "src/").replace(f"{R}/", "")

def q(path, a, b, lang="python"):
    lines = open(path).read().split("\n")
    body = "\n".join(f"{i:5d}  {lines[i-1]}" for i in range(a, b + 1))
    return f"`{rel(path)}:{a}-{b}`\n```{lang}\n{body}\n```\n"

probe = json.load(open(f"{R}/scores/part22/P22_step1_proc_probe.json"))
def pr(key):
    r = probe[key]
    return f"| {key.split('|')[0]} | {float(key.split('|')[1]):.2f} | {r['feat_shape']} | {r['mask_frames']} | {r['n_audio_tok']} | {r['seq_len']} |"

md = []
A = md.append
A("# PART 22 step 1: Qwen2.5-Omni processor config, the 300 s cut, and how to lift it\n")
A(f"Written {datetime.datetime.now().isoformat(timespec='seconds')} on the Mac. Source of every quote: the transformers 5.17.0 wheel "
  f"(`pip download transformers==5.17.0 --no-deps`, sha256 of wheel in the sidecar) unzipped to `{S}/tf5170/`, and the "
  "Qwen/Qwen2.5-Omni-7B Hub files at revision ae9e1690543ffd5c0221dc27f79834d0294cba00 in `" + S + "/hf/`. "
  "Line numbers refer to those files.\n")

A("## 0. Which transformers version scored the paper's E-DAIC full window\n")
A("- Published run (AUC 0.8285, `edaic_rerun/variants/o25_full_zeroshot.json`) ran on podA. Its setup log "
  "`edaic_rerun/logs/podA/SETUP3.log` line 15 prints: `OK tf 5.17.0 torch 2.8.0+cu128 cuda True` "
  "(install line 11: `python3 -m pip install --break-system-packages -q 'transformers>=5.0' ...`).")
A("- Part 16 rebuild (AUC 0.8278, `edaic_rerun/part16/EDAICFULL/extract.log`) prints the 5.x-only `[transformers] ... LOAD REPORT` banner; no explicit version line is in that log or in `edaic_full_perclip.json`.")
A("- Live Part 20 pods checked by ssh tonight: 205.196.17.122:11958, 205.196.17.170:18452, 216.81.245.29:41869 all print `5.17.0 /usr/local/lib/python3.12/dist-packages/transformers/__init__.py 2.8.0+cu128 3.12.3`.")
A("- sha256 of the five files quoted below is byte-identical between the unzipped 5.17.0 wheel and the installed package on pods 205.196.17.170:18452 and 216.81.245.29:41869 (hashes in the sidecar). "
  "The pod HF cache (216.81.245.29:41869, `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B`, ref ae9e1690…) holds a `preprocessor_config.json` and `config.json` byte-identical to the Hub files fetched tonight and to the Mac cache.")
A("- Mac: `/usr/local/bin/python3` has transformers **5.5.4** (`~/Library/Python/3.10/lib/python/site-packages/transformers`). It was NOT used for any result here; the probe ran in an isolated venv with 5.17.0 (scratchpad `venv22`, `--system-site-packages`, torch 2.11.0), nothing installed globally.\n")

A("## 1. The paper's scoring call (what step 2 must reproduce)\n")
A(q(f"{R}/edaic_rerun/part16/POD4B/scripts_recovered/extract_probe_layers.py", 76, 86))
A("Same call, same line text, in the Part 16 rebuild:\n")
A(q(f"{R}/edaic_rerun/part16/EDAICFULL/extract_full.py", 73, 77))
A("The same call line is present verbatim in every copy of the scoring script on disk: release_public/scripts/core/extract_probe_layers.py:63, scripts/extract_probe_layers.py:63, podD2_final/extract_probe_layers.py:76, release_public/scripts/core/extract_probe_layers_promptoverride.py:82, scripts/extract_probe_layers_promptoverride.py:82, edaic_rerun/part16/POD4B/scripts_recovered/extract_probe_layers.py:82, edaic_rerun/part16/EDAICFULL/extract_full.py:77. The published full-window json records `window_seconds: 0.0`, which only the WINDOW_S-patched copies write (the unpatched copies hard-code `x[:16000 * 30]` and write `window_seconds: 30`).\n")
A("The rendered text for this conversation (5.17.0, printed tonight) is `<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n<|audio_bos|><|AUDIO|><|audio_eos|>Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.<|im_end|>\\n<|im_start|>assistant\\n`; the processor expands the single `<|AUDIO|>` into N copies, so seq_len = N + 48.\n")

A("## 2. Processor class and its feature extractor class\n")
A(q(f"{T}/models/qwen2_5_omni/processing_qwen2_5_omni.py", 104, 111))
A("The feature extractor is loaded through AutoFeatureExtractor (no hard-coded `feature_extractor_class` on the Omni processor):\n")
A(q(f"{T}/processing_utils.py", 1865, 1879))
A(q(f"{T}/models/auto/feature_extraction_auto.py", 63, 63))
A(q(f"{S}/hf/preprocessor_config.json", 1, 31, "json"))
A("Loaded object (probe, 5.17.0): `transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor`.\n")

A("## 3. chunk_length, sampling_rate, n_samples, nb_max_frames: class defaults vs preprocessor_config.json\n")
A("Class defaults (chunk_length=30, sampling_rate=16000) and the derived values:\n")
A(q(f"{T}/models/whisper/feature_extraction_whisper.py", 69, 94))
A("json values are passed as constructor kwargs at load time:\n")
A(q(f"{T}/feature_extraction_utils.py", 576, 576))
A("Extra json keys (`n_samples`, `nb_max_frames`) fall through `**kwargs` to the mixin, which setattr's them:\n")
A(q(f"{T}/feature_extraction_utils.py", 274, 284))
A("**Which wins:** `chunk_length: 300` and `sampling_rate: 16000` from preprocessor_config.json replace the class defaults (30, 16000) as constructor arguments. "
  "`n_samples` and `nb_max_frames` from the json are first set by the mixin (line 281, called from line 81 via `super().__init__`) and then overwritten by lines 91-92 computed from chunk_length; both routes give the same numbers: "
  "n_samples = 300 x 16000 = 4,800,000; nb_max_frames = 4,800,000 // 160 = 30,000. "
  "`max_length` has no stored default anywhere: WhisperFeatureExtractor.__call__ takes `max_length=None` and substitutes `self.n_samples`. "
  "Probe printout: `fe.chunk_length = 300, fe.sampling_rate = 16000, fe.n_samples = 4800000, fe.nb_max_frames = 30000, fe.hop_length = 160, fe.return_attention_mask = True`.\n")

A("## 4. Truncation default on the audio path\n")
A("Processor-level audio defaults: no `truncation`, no `max_length`:\n")
A(q(f"{T}/models/qwen2_5_omni/processing_qwen2_5_omni.py", 79, 101))
A("__call__ merges kwargs, forces padding, calls the feature extractor with the merged audio kwargs:\n")
A(q(f"{T}/models/qwen2_5_omni/processing_qwen2_5_omni.py", 132, 152))
A("A flat kwarg is copied into every modality whose TypedDict declares it (so the paper's `padding=True` also lands in audio_kwargs, then line 143 overwrites it):\n")
A(q(f"{T}/processing_utils.py", 1637, 1658))
A("AudioKwargs declares `max_length` and `truncation`:\n")
A(q(f"{T}/processing_utils.py", 426, 435))
A("Merged audio kwargs for the paper call, printed by the probe (`_merge_kwargs` with the paper's arguments): "
  "`{'sampling_rate': 16000, 'padding': True, 'return_attention_mask': True, 'return_tensors': 'pt'}`; tokenizer init kwargs share no key with AudioKwargs (`[]`). "
  "So `truncation` and `max_length` never reach the feature extractor, and its own signature defaults apply:\n")
A(q(f"{T}/models/whisper/feature_extraction_whisper.py", 193, 206))

A("## 5. The exact lines that cut the waveform to n_samples\n")
A("WhisperFeatureExtractor.__call__ hands `max_length=self.n_samples` and `truncation=True` to pad():\n")
A(q(f"{T}/models/whisper/feature_extraction_whisper.py", 300, 307))
A("pad() calls _truncate() per sample:\n")
A(q(f"{T}/feature_extraction_sequence_utils.py", 187, 197))
A("**The cut** (raw 16 kHz waveform, before any log-mel is computed):\n")
A(q(f"{T}/feature_extraction_sequence_utils.py", 319, 337))
A("Line 333 keeps samples [0, 4,800,000) = the first 300 s. The log-mel is then computed only on what is left, and the attention mask is rescaled to frames:\n")
A(q(f"{T}/models/whisper/feature_extraction_whisper.py", 319, 341))

A("## 6. From feature frames to audio placeholder tokens (300 s -> 7500 derived from code)\n")
A(q(f"{T}/models/qwen2_5_omni/processing_qwen2_5_omni.py", 151, 152))
A(q(f"{T}/models/qwen2_5_omni/processing_qwen2_5_omni.py", 225, 227))
A("The model uses the same arithmetic and checks it:\n")
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 921, 927))
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 1787, 1788))
A("Derivation: 300 s x 16000 = 4,800,000 samples (cap). Mask frames = 4,800,000 sampled every 160 = 30,000. "
  "After conv: (30000 - 1)//2 + 1 = 15,000. Tokens: (15000 - 2)//2 + 1 = **7,500** (= 25 tokens per second). "
  "900 s untruncated: 90,000 frames -> 45,000 -> **22,500**. "
  "Frame count rule from lines 332-340: a padded clip of L samples gives ceil(L/160) frames; an unpadded clip longer than max_length gives floor(L/160).\n")
A("Check against tonight's rebuild per-clip file (`edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv`): with L = round(win_dur_s x 16000) from `variants/windows_full.csv`, capped at 4,800,000, the formula reproduces `n_audio_tok` for **275/275** clips; all **217** windows with dur_s > 300 have n_audio_tok = 7500; max n_audio_tok 7500, max seq_len 7548. "
  "(Using the 3-decimal `dur_s` column instead gives 270/275; the 5 misses, e.g. 606.wav 65.9 vs win_dur_s 65.9001, are off by one token from the rounding of dur_s.)\n")

A("## 7. Measured with the 5.17.0 processor (synthetic audio, no model weights)\n")
A("Script `scores/part22/scripts/p22_step1_proc_probe.py`, log `scores/part22/logs/p22_step1_proc_probe.log`, json `scores/part22/P22_step1_proc_probe.json`. Each row runs the exact paper call plus the listed extra argument.\n")
A("| case | dur s | input_features | mask frames | audio tokens | seq_len |\n|---|---|---|---|---|---|")
for k in probe:
    A(pr(k))
A("")
A("Case keys: A = the paper call unchanged; B = paper call + `truncation=False` (flat); C = + `audio_kwargs={'truncation': False}`; "
  "D = + `audio_kwargs={'max_length': 900*16000}`; E = + both; F = paper call unchanged after setting `proc.feature_extractor.n_samples = 900*16000` (and nb_max_frames, chunk_length).\n")

A("## 8. What lifts it, and whether the processor forwards it\n")
A("- `truncation=False` (flat, case B) or `audio_kwargs={'truncation': False}` (case C): forwarded. `truncation` is an AudioKwargs key (processing_utils.py:430), `_merge_kwargs` copies it into `output_kwargs['audio_kwargs']` (processing_utils.py:1650-1657), and processing_qwen2_5_omni.py:144 passes `**output_kwargs['audio_kwargs']` to WhisperFeatureExtractor.__call__, whose `truncation` then reaches `_truncate` (feature_extraction_sequence_utils.py:319-320 returns early). Measured: 900 s -> 22,500 tokens, 600 s -> 15,000. Short clips (< 300 s) are still padded to 30,000 frames exactly as in the paper call, so their inputs are unchanged (200 s: 5000 tokens in A, B, C). The flat form also sends `truncation=False` to the tokenizer (the text is never truncated in the paper call either); no 'not a valid argument' warning was logged. Measured feature equality (`scripts/p22_step1_feature_equality.py`, log `logs/p22_step1_feature_equality.log`): for a 200 s clip the masked input_features of the paper call and of `audio_kwargs={'truncation': False}` differ by max|diff| 0.0 and input_ids are equal.")
A("- `audio_kwargs={'max_length': 900*16000}` (case D): forwarded the same way (AudioKwargs key at processing_utils.py:429; used at feature_extraction_whisper.py:303 `max_length if max_length else self.n_samples`). Lifts the cap to 900 s but pads every clip to 90,000 frames (valid frames unchanged, masked). Do not pass `max_length` flat: the same routing would also hand it to the tokenizer.")
A("- Instance override (case F): `proc.feature_extractor.n_samples = 900*16000` works because line 303 reads `self.n_samples` at call time. Setting `chunk_length` alone after load does nothing: n_samples is computed from it only in `__init__` (feature_extraction_whisper.py:91).")
A("- `padding` cannot be used: processing_qwen2_5_omni.py:143 forces `'max_length'` on every call.")
A("- LOADTIME_PLACEHOLDER\n")
A("Recommendation for step 3: `proc(text=text, audio=[x], sampling_rate=16000, return_tensors='pt', padding=True, audio_kwargs={'truncation': False})`. It changes nothing for clips of 300 s or less, so the 58 short windows are an exact check.\n")
A("One behaviour to know before comparing numbers: the log-mel floor is per clip. For a batched call the waveform is 2-D and\n")
A(q(f"{T}/models/whisper/feature_extraction_whisper.py", 159, 165))
A("With truncation on, the max is taken over the first 300 s; with truncation off it is taken over the whole clip, so even the first 300 s of features can shift slightly for a long clip. "
  "For step 2 (full file vs the same file cut at 300 s, both through the paper call) the feature extractor sees the identical first 4,800,000 samples either way, so the features are identical and p_yes should agree to numerical noise. Measured on synthetic audio: a 900 s clip vs the same clip cut at 300 s, both through the paper call, gives input_features max|diff| 0.0 and equal input_ids. With truncation off, the first 30,000 frames of a 900 s clip differ from the paper call by max|diff| 0.0154 for stationary noise and 0.447 (0.32% of frames touched) when the last 300 s is 20x louder, which is the per-clip floor at lines 161-162.\n")

A("## 9. Model side: is ~22,500 audio tokens (900 s) allowed?\n")
A("Config (Hub `config.json`, thinker_config.audio_config): `max_source_positions: 1500`, `n_window: 100`, `encoder_layers: 32`, `d_model: 1280`, `output_dim: 3584`; thinker_config.text_config: `max_position_embeddings: 32768`, `use_sliding_window: false`, `rope_scaling: {mrope_section: [16, 24, 24], rope_type: default}`.\n")
A("Audio encoder splits the mel into 2 s chunks of n_window*2 = 200 frames, any number of them:\n")
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 754, 762))
A("Attention is confined to each chunk (cu_seqlens per chunk; non-flash path loops chunk by chunk):\n")
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 780, 781))
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 633, 652))
A("The sinusoidal position table (1500 rows) is indexed per chunk, after conv2 (stride 2) a chunk has at most 100 positions, so 1500 is never reached however long the clip:\n")
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 846, 851))
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 899, 903))
A("Thinker positions: audio-only input (no image or video grid) takes the plain branch, positions 0..seq_len-1:\n")
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 294, 294))
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 497, 504))
A("Rotary embedding computes angles from inv_freq for whatever positions it is given (no cached table capped at max_position_embeddings):\n")
A(q(f"{T}/models/qwen2_5_omni/modeling_qwen2_5_omni.py", 1318, 1335))
A("**Verdict from code:** nothing in the 5.17.0 audio encoder or thinker caps audio length. 900 s gives 450 encoder chunks, 22,500 audio embeddings, seq_len 22,548, max position 22,547 < 32,768. "
  "So a 900 s window is allowed by the code and fits the 32k context. Not established here: memory and runtime on the H200 (step 3 measures it), and whether the model behaves well past the 300 s it ships configured for (chunk_length 300 in the Hub preprocessor_config); that is an empirical question for step 3, not a code limit.\n")

A("## 10. Key lines (one-screen summary)\n")
A("- Version: transformers 5.17.0 (podA SETUP3.log:15; three live Part 20 pods). Mac global: 5.5.4 (not used).")
A("- Feature extractor: WhisperFeatureExtractor (feature_extraction_auto.py:63; preprocessor_config.json:4).")
A("- chunk_length 300 (preprocessor_config.json:2; class default 30 at feature_extraction_whisper.py:74), sampling_rate 16000, n_samples 4,800,000 (feature_extraction_whisper.py:91), nb_max_frames 30,000 (:92), max_length None -> n_samples (:303).")
A("- truncation default True (feature_extraction_whisper.py:196); the Omni processor passes none (processing_qwen2_5_omni.py:96-100).")
A("- The cut: feature_extraction_sequence_utils.py:333 `processed_features[self.model_input_names[0]] = processed_features[self.model_input_names[0]][:max_length]`.")
A("- Tokens: processing_qwen2_5_omni.py:151-152 -> 30,000 frames -> 7,500 tokens; 90,000 -> 22,500.")
A("- Lift: `audio_kwargs={'truncation': False}` is forwarded; measured 900 s -> 22,500 tokens, seq 22,548; model code allows it.")

txt = "\n".join(md) + "\n"
lt = os.environ.get("LOADTIME_LINE", "Load-time override `Proc.from_pretrained(MID, chunk_length=900)`: not run.")
txt = txt.replace("LOADTIME_PLACEHOLDER", lt)
open(OUT, "w").write(txt)
print("wrote", OUT, len(txt), "chars")
