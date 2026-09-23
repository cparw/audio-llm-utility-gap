# PART 22 step 1: Qwen2.5-Omni processor config, the 300 s cut, and how to lift it

Written 2026-09-23T05:48:55 on the Mac. Source of every quote: the transformers 5.17.0 wheel (`pip download transformers==5.17.0 --no-deps`, sha256 of wheel in the sidecar) unzipped to `/Users/chaitanyaparwatkar/Desktop/release/scores/part22/src/tf5170/`, and the Qwen/Qwen2.5-Omni-7B Hub files at revision ae9e1690543ffd5c0221dc27f79834d0294cba00 in `/Users/chaitanyaparwatkar/Desktop/release/scores/part22/src/hf/`. Line numbers refer to those files.

## 0. Which transformers version scored the paper's E-DAIC full window

- Published run (AUC 0.8285, `edaic_rerun/variants/o25_full_zeroshot.json`) ran on podA. Its setup log `edaic_rerun/logs/podA/SETUP3.log` line 15 prints: `OK tf 5.17.0 torch 2.8.0+cu128 cuda True` (install line 11: `python3 -m pip install --break-system-packages -q 'transformers>=5.0' ...`).
- Part 16 rebuild (AUC 0.8278, `edaic_rerun/part16/EDAICFULL/extract.log`) prints the 5.x-only `[transformers] ... LOAD REPORT` banner; no explicit version line is in that log or in `edaic_full_perclip.json`.
- Live Part 20 pods checked by ssh tonight: 205.196.17.122:11958, 205.196.17.170:18452, 216.81.245.29:41869 all print `5.17.0 /usr/local/lib/python3.12/dist-packages/transformers/__init__.py 2.8.0+cu128 3.12.3`.
- sha256 of the five files quoted below is byte-identical between the unzipped 5.17.0 wheel and the installed package on pods 205.196.17.170:18452 and 216.81.245.29:41869 (hashes in the sidecar). The pod HF cache (216.81.245.29:41869, `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B`, ref ae9e1690…) holds a `preprocessor_config.json` and `config.json` byte-identical to the Hub files fetched tonight and to the Mac cache.
- Mac: `/usr/local/bin/python3` has transformers **5.5.4** (`~/Library/Python/3.10/lib/python/site-packages/transformers`). It was NOT used for any result here; the probe ran in an isolated venv with 5.17.0 (scratchpad `venv22`, `--system-site-packages`, torch 2.11.0), nothing installed globally.

## 1. The paper's scoring call (what step 2 must reproduce)

`edaic_rerun/part16/POD4B/scripts_recovered/extract_probe_layers.py:76-86`
```python
   76  enc_f, prj_f, llm_f, ans_f, meta = [], [], [], [], []
   77  for i, r in enumerate(rows):
   78      x, _ = librosa.load(r["path"], sr=16000)
   79      x = x[:int(16000 * WIN)] if WIN > 0 else x
   80      conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
   81      text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
   82      inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
   83      inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
   84      inp.pop("use_audio_in_video", None)
   85      for _k, _v in list(inp.items()):                       # audio features arrive float32; match the model dtype
   86          if torch.is_tensor(_v) and _v.dtype == torch.float32: inp[_k] = _v.to(MDTYPE)
```

Same call, same line text, in the Part 16 rebuild:

`edaic_rerun/part16/EDAICFULL/extract_full.py:73-77`
```python
   73      x, _ = librosa.load(r["path"], sr=16000)
   74      x = x[:int(16000 * WIN)] if WIN > 0 else x
   75      conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
   76      text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
   77      inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
```

The same call line is present verbatim in every copy of the scoring script on disk: release_public/scripts/core/extract_probe_layers.py:63, scripts/extract_probe_layers.py:63, podD2_final/extract_probe_layers.py:76, release_public/scripts/core/extract_probe_layers_promptoverride.py:82, scripts/extract_probe_layers_promptoverride.py:82, edaic_rerun/part16/POD4B/scripts_recovered/extract_probe_layers.py:82, edaic_rerun/part16/EDAICFULL/extract_full.py:77. The published full-window json records `window_seconds: 0.0`, which only the WINDOW_S-patched copies write (the unpatched copies hard-code `x[:16000 * 30]` and write `window_seconds: 30`).

The rendered text for this conversation (5.17.0, printed tonight) is `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.<|im_end|>\n<|im_start|>assistant\n`; the processor expands the single `<|AUDIO|>` into N copies, so seq_len = N + 48.

## 2. Processor class and its feature extractor class

`transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py:104-111`
```python
  104  @auto_docstring
  105  class Qwen2_5OmniProcessor(ProcessorMixin):
  106      valid_processor_kwargs = Qwen2_5OmniProcessorKwargs
  107  
  108      def __init__(
  109          self, image_processor=None, video_processor=None, feature_extractor=None, tokenizer=None, chat_template=None
  110      ):
  111          super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)
```

The feature extractor is loaded through AutoFeatureExtractor (no hard-coded `feature_extractor_class` on the Omni processor):

`transformers/processing_utils.py:1865-1879`
```python
 1865              elif is_primary:
 1866                  # Primary non-tokenizer sub-processor: load via Auto class
 1867                  auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
 1868                  # For backward compatibility, check if sub-processor class name is hardcoded as an attribute of the processor class.
 1869                  if hasattr(cls, sub_processor_type + "_class"):
 1870                      sub_processor_class_name = getattr(cls, sub_processor_type + "_class")
 1871                      logger.warning_once(
 1872                          f"`{cls.__name__}` defines `{sub_processor_type}_class = '{sub_processor_class_name}'`, "
 1873                          f"which is deprecated. Register the correct mapping in `{auto_processor_class.__name__}` instead.",
 1874                      )
 1875                      auto_processor_class = cls.get_possibly_dynamic_module(sub_processor_class_name)
 1876                  sub_processor = auto_processor_class.from_pretrained(
 1877                      pretrained_model_name_or_path, subfolder=subfolder, **kwargs
 1878                  )
 1879                  args.append(sub_processor)
```

`transformers/models/auto/feature_extraction_auto.py:63-63`
```python
   63          ("qwen2_5_omni", "WhisperFeatureExtractor"),
```

`src/hf/preprocessor_config.json:1-31`
```json
    1  {
    2    "chunk_length": 300,
    3    "dither": 0.0,
    4    "feature_extractor_type": "WhisperFeatureExtractor",
    5    "feature_size": 128,
    6    "hop_length": 160,
    7    "image_mean": [
    8      0.48145466,
    9      0.4578275,
   10      0.40821073
   11    ],
   12    "image_processor_type": "Qwen2VLImageProcessor",
   13    "image_std": [
   14      0.26862954,
   15      0.26130258,
   16      0.27577711
   17    ],
   18    "max_pixels": 12845056,
   19    "merge_size": 2,
   20    "min_pixels": 3136,
   21    "n_fft": 400,
   22    "n_samples": 4800000,
   23    "nb_max_frames": 30000,
   24    "padding_side": "right",
   25    "padding_value": 0.0,
   26    "patch_size": 14,
   27    "processor_class": "Qwen2_5OmniProcessor",
   28    "return_attention_mask": true,
   29    "sampling_rate": 16000,
   30    "temporal_patch_size": 2
   31  }
```

Loaded object (probe, 5.17.0): `transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor`.

## 3. chunk_length, sampling_rate, n_samples, nb_max_frames: class defaults vs preprocessor_config.json

Class defaults (chunk_length=30, sampling_rate=16000) and the derived values:

`transformers/models/whisper/feature_extraction_whisper.py:69-94`
```python
   69      def __init__(
   70          self,
   71          feature_size=80,
   72          sampling_rate=16000,
   73          hop_length=160,
   74          chunk_length=30,
   75          n_fft=400,
   76          padding_value=0.0,
   77          dither=0.0,
   78          return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
   79          **kwargs,
   80      ):
   81          super().__init__(
   82              feature_size=feature_size,
   83              sampling_rate=sampling_rate,
   84              padding_value=padding_value,
   85              return_attention_mask=return_attention_mask,
   86              **kwargs,
   87          )
   88          self.n_fft = n_fft
   89          self.hop_length = hop_length
   90          self.chunk_length = chunk_length
   91          self.n_samples = chunk_length * sampling_rate
   92          self.nb_max_frames = self.n_samples // hop_length
   93          self.sampling_rate = sampling_rate
   94          self.dither = dither
```

json values are passed as constructor kwargs at load time:

`transformers/feature_extraction_utils.py:576-576`
```python
  576          feature_extractor = cls(**feature_extractor_dict)
```

Extra json keys (`n_samples`, `nb_max_frames`) fall through `**kwargs` to the mixin, which setattr's them:

`transformers/feature_extraction_utils.py:274-284`
```python
  274      def __init__(self, **kwargs):
  275          """Set elements of `kwargs` as attributes."""
  276          # Pop "processor_class", it should not be saved in feature extractor config
  277          kwargs.pop("processor_class", None)
  278          # Additional attributes without default values
  279          for key, value in kwargs.items():
  280              try:
  281                  setattr(self, key, value)
  282              except AttributeError as err:
  283                  logger.error(f"Can't set {key} with value {value} for {self}")
  284                  raise err
```

**Which wins:** `chunk_length: 300` and `sampling_rate: 16000` from preprocessor_config.json replace the class defaults (30, 16000) as constructor arguments. `n_samples` and `nb_max_frames` from the json are first set by the mixin (line 281, called from line 81 via `super().__init__`) and then overwritten by lines 91-92 computed from chunk_length; both routes give the same numbers: n_samples = 300 x 16000 = 4,800,000; nb_max_frames = 4,800,000 // 160 = 30,000. `max_length` has no stored default anywhere: WhisperFeatureExtractor.__call__ takes `max_length=None` and substitutes `self.n_samples`. Probe printout: `fe.chunk_length = 300, fe.sampling_rate = 16000, fe.n_samples = 4800000, fe.nb_max_frames = 30000, fe.hop_length = 160, fe.return_attention_mask = True`.

## 4. Truncation default on the audio path

Processor-level audio defaults: no `truncation`, no `max_length`:

`transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py:79-101`
```python
   79  class Qwen2_5OmniProcessorKwargs(ProcessingKwargs, total=False):
   80      videos_kwargs: Qwen2_5_OmniVideosKwargs
   81  
   82      _defaults = {
   83          "text_kwargs": {
   84              "padding": False,
   85              "padding_side": "left",
   86          },
   87          "videos_kwargs": {
   88              "seconds_per_chunk": 2.0,
   89              "position_id_per_seconds": 25,
   90              "use_audio_in_video": False,
   91              "size": {
   92                  "shortest_edge": 128 * 28 * 28,
   93                  "longest_edge": 768 * 28 * 28,
   94              },
   95          },
   96          "audio_kwargs": {
   97              "sampling_rate": 16000,
   98              "padding": "max_length",
   99              "return_attention_mask": True,
  100          },
  101      }
```

__call__ merges kwargs, forces padding, calls the feature extractor with the merged audio kwargs:

`transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py:132-152`
```python
  132          output_kwargs = self._merge_kwargs(
  133              Qwen2_5OmniProcessorKwargs,
  134              tokenizer_init_kwargs=self.tokenizer.init_kwargs,
  135              **kwargs,
  136          )
  137  
  138          seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
  139          position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
  140          use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
  141  
  142          if audio is not None:
  143              output_kwargs["audio_kwargs"]["padding"] = "max_length"  # Support "max_length" padding only here
  144              audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
  145              audio_inputs["feature_attention_mask"] = audio_inputs.pop(
  146                  "attention_mask"
  147              )  # rename feature_attention_mask to prevent conflicts later on
  148              audio_inputs["input_features"] = audio_inputs.pop(
  149                  "input_features"
  150              )  # rename input_features to prevent conflicts later on
  151              input_lengths = (audio_inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
  152              audio_lengths = iter((input_lengths - 2) // 2 + 1)
```

A flat kwarg is copied into every modality whose TypedDict declares it (so the paper's `padding=True` also lands in audio_kwargs, then line 143 overwrites it):

`transformers/processing_utils.py:1637-1658`
```python
 1637              for modality_key in modality_valid_kwargs:
 1638                  # check if we received a structured kwarg dict or not to handle it correctly
 1639                  if modality in kwargs:
 1640                      kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
 1641                      # check if this key was passed as a flat kwarg.
 1642                      if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
 1643                          raise ValueError(
 1644                              f"Keyword argument {modality_key} was passed two times:\n"
 1645                              f"in a dictionary for {modality} and as a **kwarg."
 1646                          )
 1647                      # fall back to the flat kwarg when the modality dict is present but doesn't carry this key
 1648                      if kwarg_value == "__empty__" and modality_key in non_modality_kwargs:
 1649                          kwarg_value = kwargs[modality_key]
 1650                  elif modality_key in kwargs:
 1651                      # we get a modality_key instead of popping it because modality-specific processors
 1652                      # can have overlapping kwargs
 1653                      kwarg_value = kwargs.get(modality_key, "__empty__")
 1654                  else:
 1655                      kwarg_value = "__empty__"
 1656                  if not isinstance(kwarg_value, str) or kwarg_value != "__empty__":
 1657                      output_kwarg[modality_key] = kwarg_value
 1658                      used_keys.add(modality_key)
```

AudioKwargs declares `max_length` and `truncation`:

`transformers/processing_utils.py:426-435`
```python
  426      sampling_rate: Annotated[int | None, positive_int()]
  427      raw_speech: Union["np.ndarray", list[float], list["np.ndarray"], list[list[float]]] | None
  428      padding: Annotated[bool | str | PaddingStrategy | None, padding_validator()]
  429      max_length: Annotated[int | None, positive_int()]
  430      truncation: Annotated[bool | str | TruncationStrategy | None, truncation_validator()]
  431      pad_to_multiple_of: Annotated[int | None, positive_int()]
  432      return_attention_mask: bool | None
  433      device: Annotated[Union[str, "torch.device"] | None, device_validator()]
  434      return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]
  435      load_audio_backend: str | None
```

Merged audio kwargs for the paper call, printed by the probe (`_merge_kwargs` with the paper's arguments): `{'sampling_rate': 16000, 'padding': True, 'return_attention_mask': True, 'return_tensors': 'pt'}`; tokenizer init kwargs share no key with AudioKwargs (`[]`). So `truncation` and `max_length` never reach the feature extractor, and its own signature defaults apply:

`transformers/models/whisper/feature_extraction_whisper.py:193-206`
```python
  193      def __call__(
  194          self,
  195          raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
  196          truncation: bool = True,
  197          pad_to_multiple_of: int | None = None,
  198          return_tensors: str | TensorType | None = None,
  199          return_attention_mask: bool | None = None,
  200          padding: str | None = "max_length",
  201          max_length: int | None = None,
  202          sampling_rate: int | None = None,
  203          do_normalize: bool | None = None,
  204          device: str | None = "cpu",
  205          **kwargs,
  206      ) -> BatchFeature:
```

## 5. The exact lines that cut the waveform to n_samples

WhisperFeatureExtractor.__call__ hands `max_length=self.n_samples` and `truncation=True` to pad():

`transformers/models/whisper/feature_extraction_whisper.py:300-307`
```python
  300          padded_inputs = self.pad(
  301              batched_speech,
  302              padding=padding,
  303              max_length=max_length if max_length else self.n_samples,
  304              truncation=truncation,
  305              pad_to_multiple_of=pad_to_multiple_of,
  306              return_attention_mask=return_attention_mask or do_normalize,
  307          )
```

pad() calls _truncate() per sample:

`transformers/feature_extraction_sequence_utils.py:187-197`
```python
  187          truncated_inputs = []
  188          for i in range(batch_size):
  189              inputs = {k: v[i] for k, v in processed_features.items()}
  190              # truncation
  191              inputs_slice = self._truncate(
  192                  inputs,
  193                  max_length=max_length,
  194                  pad_to_multiple_of=pad_to_multiple_of,
  195                  truncation=truncation,
  196              )
  197              truncated_inputs.append(inputs_slice)
```

**The cut** (raw 16 kHz waveform, before any log-mel is computed):

`transformers/feature_extraction_sequence_utils.py:319-337`
```python
  319          if not truncation:
  320              return processed_features
  321          elif truncation and max_length is None:
  322              raise ValueError("When setting ``truncation=True``, make sure that ``max_length`` is defined.")
  323  
  324          required_input = processed_features[self.model_input_names[0]]
  325  
  326          # find `max_length` that fits `pad_to_multiple_of`
  327          if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
  328              max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
  329  
  330          needs_to_be_truncated = len(required_input) > max_length
  331  
  332          if needs_to_be_truncated:
  333              processed_features[self.model_input_names[0]] = processed_features[self.model_input_names[0]][:max_length]
  334              if "attention_mask" in processed_features:
  335                  processed_features["attention_mask"] = processed_features["attention_mask"][:max_length]
  336  
  337          return processed_features
```

Line 333 keeps samples [0, 4,800,000) = the first 300 s. The log-mel is then computed only on what is left, and the attention mask is rescaled to frames:

`transformers/models/whisper/feature_extraction_whisper.py:319-341`
```python
  319          input_features = padded_inputs.get("input_features").transpose(2, 0, 1)
  320  
  321          extract_fbank_features = (
  322              self._torch_extract_fbank_features if is_torch_available() else self._np_extract_fbank_features
  323          )
  324          input_features = extract_fbank_features(input_features[0], device)
  325  
  326          if isinstance(input_features[0], list):
  327              padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]
  328  
  329          else:
  330              padded_inputs["input_features"] = input_features
  331  
  332          if return_attention_mask:
  333              # rescale from sample (48000) to feature (3000)
  334              rescaled_attention_mask = padded_inputs["attention_mask"][:, :: self.hop_length]
  335  
  336              # The STFT computation produces L//hop_length + 1 frames, but we skip the last frame (see `_torch_extract_fbank_features`).
  337              # This means we need to trim the rescaled attention mask to match the actual number of frames (L//hop_length) when the input length
  338              # is not perfectly divisible by the hop length.
  339              if padded_inputs["attention_mask"].shape[1] % self.hop_length != 0:
  340                  rescaled_attention_mask = rescaled_attention_mask[:, :-1]
  341              padded_inputs["attention_mask"] = rescaled_attention_mask
```

## 6. From feature frames to audio placeholder tokens (300 s -> 7500 derived from code)

`transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py:151-152`
```python
  151              input_lengths = (audio_inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
  152              audio_lengths = iter((input_lengths - 2) // 2 + 1)
```

`transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py:225-227`
```python
  225              for _, special_token in positions:
  226                  if special_token == self.audio_token:
  227                      sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)
```

The model uses the same arithmetic and checks it:

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:921-927`
```python
  921      def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
  922          """
  923          Computes the output length of the convolutional layers and the output length of the audio encoder
  924          """
  925          input_lengths = (input_lengths - 1) // 2 + 1
  926          output_lengths = (input_lengths - 2) // 2 + 1
  927          return input_lengths, output_lengths
```

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:1787-1788`
```python
 1787          if audio_outputs.last_hidden_state.shape[0] != sum(audio_output_lengths.tolist()):
 1788              raise ValueError("length of audio_features should match audio_output_lengths")
```

Derivation: 300 s x 16000 = 4,800,000 samples (cap). Mask frames = 4,800,000 sampled every 160 = 30,000. After conv: (30000 - 1)//2 + 1 = 15,000. Tokens: (15000 - 2)//2 + 1 = **7,500** (= 25 tokens per second). 900 s untruncated: 90,000 frames -> 45,000 -> **22,500**. Frame count rule from lines 332-340: a padded clip of L samples gives ceil(L/160) frames; an unpadded clip longer than max_length gives floor(L/160).

Check against tonight's rebuild per-clip file (`edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv`): with L = round(win_dur_s x 16000) from `variants/windows_full.csv`, capped at 4,800,000, the formula reproduces `n_audio_tok` for **275/275** clips; all **217** windows with dur_s > 300 have n_audio_tok = 7500; max n_audio_tok 7500, max seq_len 7548. (Using the 3-decimal `dur_s` column instead gives 270/275; the 5 misses, e.g. 606.wav 65.9 vs win_dur_s 65.9001, are off by one token from the rounding of dur_s.)

## 7. Measured with the 5.17.0 processor (synthetic audio, no model weights)

Script `scores/part22/scripts/p22_step1_proc_probe.py`, log `scores/part22/logs/p22_step1_proc_probe.log`, json `scores/part22/P22_step1_proc_probe.json`. Each row runs the exact paper call plus the listed extra argument.

| case | dur s | input_features | mask frames | audio tokens | seq_len |
|---|---|---|---|---|---|
| A_paper_call | 200.00 | [1, 128, 30000] | 20000 | 5000 | 5048 |
| B_flat_truncation_False | 200.00 | [1, 128, 30000] | 20000 | 5000 | 5048 |
| C_audio_kwargs_truncation_False | 200.00 | [1, 128, 30000] | 20000 | 5000 | 5048 |
| D_audio_kwargs_max_length_900s | 200.00 | [1, 128, 90000] | 20000 | 5000 | 5048 |
| E_audio_kwargs_trunc_False_and_max_length_900s | 200.00 | [1, 128, 90000] | 20000 | 5000 | 5048 |
| A_paper_call | 299.99 | [1, 128, 30000] | 29999 | 7500 | 7548 |
| B_flat_truncation_False | 299.99 | [1, 128, 30000] | 29999 | 7500 | 7548 |
| C_audio_kwargs_truncation_False | 299.99 | [1, 128, 30000] | 29999 | 7500 | 7548 |
| D_audio_kwargs_max_length_900s | 299.99 | [1, 128, 90000] | 29999 | 7500 | 7548 |
| E_audio_kwargs_trunc_False_and_max_length_900s | 299.99 | [1, 128, 90000] | 29999 | 7500 | 7548 |
| A_paper_call | 300.00 | [1, 128, 30000] | 30000 | 7500 | 7548 |
| B_flat_truncation_False | 300.00 | [1, 128, 30000] | 30000 | 7500 | 7548 |
| C_audio_kwargs_truncation_False | 300.00 | [1, 128, 30000] | 30000 | 7500 | 7548 |
| D_audio_kwargs_max_length_900s | 300.00 | [1, 128, 90000] | 30000 | 7500 | 7548 |
| E_audio_kwargs_trunc_False_and_max_length_900s | 300.00 | [1, 128, 90000] | 30000 | 7500 | 7548 |
| A_paper_call | 300.01 | [1, 128, 30000] | 30000 | 7500 | 7548 |
| B_flat_truncation_False | 300.01 | [1, 128, 30001] | 30001 | 7500 | 7548 |
| C_audio_kwargs_truncation_False | 300.01 | [1, 128, 30001] | 30001 | 7500 | 7548 |
| D_audio_kwargs_max_length_900s | 300.01 | [1, 128, 90000] | 30001 | 7500 | 7548 |
| E_audio_kwargs_trunc_False_and_max_length_900s | 300.01 | [1, 128, 90000] | 30001 | 7500 | 7548 |
| A_paper_call | 600.00 | [1, 128, 30000] | 30000 | 7500 | 7548 |
| B_flat_truncation_False | 600.00 | [1, 128, 60000] | 60000 | 15000 | 15048 |
| C_audio_kwargs_truncation_False | 600.00 | [1, 128, 60000] | 60000 | 15000 | 15048 |
| D_audio_kwargs_max_length_900s | 600.00 | [1, 128, 90000] | 60000 | 15000 | 15048 |
| E_audio_kwargs_trunc_False_and_max_length_900s | 600.00 | [1, 128, 90000] | 60000 | 15000 | 15048 |
| A_paper_call | 900.00 | [1, 128, 30000] | 30000 | 7500 | 7548 |
| B_flat_truncation_False | 900.00 | [1, 128, 90000] | 90000 | 22500 | 22548 |
| C_audio_kwargs_truncation_False | 900.00 | [1, 128, 90000] | 90000 | 22500 | 22548 |
| D_audio_kwargs_max_length_900s | 900.00 | [1, 128, 90000] | 90000 | 22500 | 22548 |
| E_audio_kwargs_trunc_False_and_max_length_900s | 900.00 | [1, 128, 90000] | 90000 | 22500 | 22548 |
| F_override_fe_n_samples_900s | 200.00 | [1, 128, 90000] | 20000 | 5000 | 5048 |
| F_override_fe_n_samples_900s | 900.00 | [1, 128, 90000] | 90000 | 22500 | 22548 |

Case keys: A = the paper call unchanged; B = paper call + `truncation=False` (flat); C = + `audio_kwargs={'truncation': False}`; D = + `audio_kwargs={'max_length': 900*16000}`; E = + both; F = paper call unchanged after setting `proc.feature_extractor.n_samples = 900*16000` (and nb_max_frames, chunk_length).

## 8. What lifts it, and whether the processor forwards it

- `truncation=False` (flat, case B) or `audio_kwargs={'truncation': False}` (case C): forwarded. `truncation` is an AudioKwargs key (processing_utils.py:430), `_merge_kwargs` copies it into `output_kwargs['audio_kwargs']` (processing_utils.py:1650-1657), and processing_qwen2_5_omni.py:144 passes `**output_kwargs['audio_kwargs']` to WhisperFeatureExtractor.__call__, whose `truncation` then reaches `_truncate` (feature_extraction_sequence_utils.py:319-320 returns early). Measured: 900 s -> 22,500 tokens, 600 s -> 15,000. Short clips (< 300 s) are still padded to 30,000 frames exactly as in the paper call, so their inputs are unchanged (200 s: 5000 tokens in A, B, C). The flat form also sends `truncation=False` to the tokenizer (the text is never truncated in the paper call either); no 'not a valid argument' warning was logged. Measured feature equality (`scripts/p22_step1_feature_equality.py`, log `logs/p22_step1_feature_equality.log`): for a 200 s clip the masked input_features of the paper call and of `audio_kwargs={'truncation': False}` differ by max|diff| 0.0 and input_ids are equal.
- `audio_kwargs={'max_length': 900*16000}` (case D): forwarded the same way (AudioKwargs key at processing_utils.py:429; used at feature_extraction_whisper.py:303 `max_length if max_length else self.n_samples`). Lifts the cap to 900 s but pads every clip to 90,000 frames (valid frames unchanged, masked). Do not pass `max_length` flat: the same routing would also hand it to the tokenizer.
- Instance override (case F): `proc.feature_extractor.n_samples = 900*16000` works because line 303 reads `self.n_samples` at call time. Setting `chunk_length` alone after load does nothing: n_samples is computed from it only in `__init__` (feature_extraction_whisper.py:91).
- `padding` cannot be used: processing_qwen2_5_omni.py:143 forces `'max_length'` on every call.
- 

Recommendation for step 3: `proc(text=text, audio=[x], sampling_rate=16000, return_tensors='pt', padding=True, audio_kwargs={'truncation': False})`. It changes nothing for clips of 300 s or less, so the 58 short windows are an exact check.

One behaviour to know before comparing numbers: the log-mel floor is per clip. For a batched call the waveform is 2-D and

`transformers/models/whisper/feature_extraction_whisper.py:159-165`
```python
  159          log_spec = torch.clamp(mel_spec, min=1e-10).log10()
  160          if waveform.dim() == 2:
  161              max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
  162              log_spec = torch.maximum(log_spec, max_val - 8.0)
  163          else:
  164              log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
  165          log_spec = (log_spec + 4.0) / 4.0
```

With truncation on, the max is taken over the first 300 s; with truncation off it is taken over the whole clip, so even the first 300 s of features can shift slightly for a long clip. For step 2 (full file vs the same file cut at 300 s, both through the paper call) the feature extractor sees the identical first 4,800,000 samples either way, so the features are identical and p_yes should agree to numerical noise. Measured on synthetic audio: a 900 s clip vs the same clip cut at 300 s, both through the paper call, gives input_features max|diff| 0.0 and equal input_ids. With truncation off, the first 30,000 frames of a 900 s clip differ from the paper call by max|diff| 0.0154 for stationary noise and 0.447 (0.32% of frames touched) when the last 300 s is 20x louder, which is the per-clip floor at lines 161-162.

## 9. Model side: is ~22,500 audio tokens (900 s) allowed?

Config (Hub `config.json`, thinker_config.audio_config): `max_source_positions: 1500`, `n_window: 100`, `encoder_layers: 32`, `d_model: 1280`, `output_dim: 3584`; thinker_config.text_config: `max_position_embeddings: 32768`, `use_sliding_window: false`, `rope_scaling: {mrope_section: [16, 24, 24], rope_type: default}`.

Audio encoder splits the mel into 2 s chunks of n_window*2 = 200 frames, any number of them:

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:754-762`
```python
  754      chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
  755      chunk_lengths = torch.full((chunk_num.sum(),), n_window * 2, dtype=torch.long, device=feature_lens.device)
  756      tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
  757      chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
  758      chunk_lengths = torch.where(chunk_lengths == 0, n_window * 2, chunk_lengths)
  759  
  760      chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
  761      padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
  762      return padded_feature, chunk_lengths
```

Attention is confined to each chunk (cu_seqlens per chunk; non-flash path loops chunk by chunk):

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:780-781`
```python
  780      after_conv1 = (chunk_lengths - 1) // 2 + 1
  781      return F.pad(after_conv1.cumsum(0), (1, 0), value=0).to(torch.int32)
```

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:633-652`
```python
  633              # Other implementations: Process each chunk separately
  634              lengths = cu_seqlens[1:] - cu_seqlens[:-1]
  635              splits = [
  636                  torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
  637              ]
  638              attn_outputs = [
  639                  attention_interface(
  640                      self,
  641                      q,
  642                      k,
  643                      v,
  644                      attention_mask=None,
  645                      scaling=self.scaling,
  646                      dropout=0.0 if not self.training else self.attention_dropout,
  647                      is_causal=False,
  648                      **kwargs,
  649                  )[0]
  650                  for q, k, v in zip(*splits)
  651              ]
  652              attn_output = torch.cat(attn_outputs, dim=1)
```

The sinusoidal position table (1500 rows) is indexed per chunk, after conv2 (stride 2) a chunk has at most 100 positions, so 1500 is never reached however long the clip:

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:846-851`
```python
  846          self.max_source_positions = config.max_source_positions
  847          self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
  848          self.n_window = config.n_window
  849          self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
  850          self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
  851          self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
```

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:899-903`
```python
  899          padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
  900          padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)
  901          padded_embed = padded_embed + self.positional_embedding.positional_embedding[
  902              : padded_embed.shape[1], :
  903          ].unsqueeze(0).to(padded_embed.dtype)
```

Thinker positions: audio-only input (no image or video grid) takes the plain branch, positions 0..seq_len-1:

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:294-294`
```python
  294          if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
```

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:497-504`
```python
  497          else:
  498              position_ids = attention_mask.long().cumsum(-1) - 1
  499              position_ids.masked_fill_(attention_mask == 0, 1)
  500              position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
  501              max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
  502              mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
  503  
  504              return position_ids, mrope_position_deltas
```

Rotary embedding computes angles from inv_freq for whatever positions it is given (no cached table capped at max_position_embeddings):

`transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py:1318-1335`
```python
 1318  class Qwen2_5OmniRotaryEmbedding(nn.Module):
 1319      @deprecate_kwarg("device", version="5.18")
 1320      def __init__(self, config: Qwen2_5OmniConfig, device=None):
 1321          super().__init__()
 1322          self.max_seq_len_cached = config.max_position_embeddings
 1323          self.original_max_seq_len = config.max_position_embeddings
 1324  
 1325          self.config = config
 1326  
 1327          self.rope_type = self.config.rope_parameters["rope_type"]
 1328          rope_init_fn: Callable = self.compute_default_rope_parameters
 1329          if self.rope_type != "default":
 1330              rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
 1331          inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
 1332  
 1333          self.inv_freq = nn.Buffer(inv_freq, persistent=False)
 1334          self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
 1335          self.mrope_section = config.rope_parameters.get("mrope_section", [16, 24, 24])
```

**Verdict from code:** nothing in the 5.17.0 audio encoder or thinker caps audio length. 900 s gives 450 encoder chunks, 22,500 audio embeddings, seq_len 22,548, max position 22,547 < 32,768. So a 900 s window is allowed by the code and fits the 32k context. Not established here: memory and runtime on the H200 (step 3 measures it), and whether the model behaves well past the 300 s it ships configured for (chunk_length 300 in the Hub preprocessor_config); that is an empirical question for step 3, not a code limit.

## 10. Key lines (one-screen summary)

- Version: transformers 5.17.0 (podA SETUP3.log:15; three live Part 20 pods). Mac global: 5.5.4 (not used).
- Feature extractor: WhisperFeatureExtractor (feature_extraction_auto.py:63; preprocessor_config.json:4).
- chunk_length 300 (preprocessor_config.json:2; class default 30 at feature_extraction_whisper.py:74), sampling_rate 16000, n_samples 4,800,000 (feature_extraction_whisper.py:91), nb_max_frames 30,000 (:92), max_length None -> n_samples (:303).
- truncation default True (feature_extraction_whisper.py:196); the Omni processor passes none (processing_qwen2_5_omni.py:96-100).
- The cut: feature_extraction_sequence_utils.py:333 `processed_features[self.model_input_names[0]] = processed_features[self.model_input_names[0]][:max_length]`.
- Tokens: processing_qwen2_5_omni.py:151-152 -> 30,000 frames -> 7,500 tokens; 90,000 -> 22,500.
- Lift: `audio_kwargs={'truncation': False}` is forwarded; measured 900 s -> 22,500 tokens, seq 22,548; model code allows it.
