"""PART 23 item 6, pod C. Processor-level truncation check for Qwen3-Omni and Audio Flamingo 3 on real E-DAIC full windows.
usage: python3 p23c_proc.py q3o|af3
Runs the paper's processor call on the 5 bracket windows, writes one csv + json per model to /workspace/p23c/out as it goes."""
import os, sys, csv, json, time, hashlib, tempfile, math, datetime
import numpy as np, torch, librosa, soundfile as sf, transformers
from huggingface_hub import hf_hub_download, HfApi
WHICH = sys.argv[1]
OUT = "/workspace/p23c/out"; os.makedirs(OUT, exist_ok=True)
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
PIDS = ["612", "417", "359", "300", "716"]
WIN = {r["pid"]: r for r in csv.DictReader(open("/workspace/edaicfull/windows_full.csv"))}
TFDIR = os.path.dirname(transformers.__file__)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    open(f"{OUT}/{WHICH}_proc.log", "a").write(s + "\n")
res = {"model": None, "transformers": transformers.__version__, "torch": torch.__version__,
       "librosa": librosa.__version__, "date_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"), "files": []}
if WHICH == "q3o":
    MID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"; REV = HfApi().model_info(MID).sha
    from transformers import Qwen3OmniMoeProcessor as Proc
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs as PK, _get_feat_extract_output_lengths as q3len
    proc = Proc.from_pretrained(MID, revision=REV)
    srcs = ["models/qwen3_omni_moe/processing_qwen3_omni_moe.py", "models/qwen3_omni_moe/modeling_qwen3_omni_moe.py"]
    hubfiles = ["preprocessor_config.json", "config.json"]
    AUD = proc.tokenizer.convert_tokens_to_ids(proc.audio_token)
else:
    MID = "nvidia/audio-flamingo-3-hf"; REV = "7d4bae64ee29878af6504ae6f6bb3e40492838ad"   # snapshot on podA, logs/podA/dl_af3.log
    from transformers import AutoProcessor
    from transformers.models.audioflamingo3.processing_audioflamingo3 import AudioFlamingo3ProcessorKwargs as PK
    proc = AutoProcessor.from_pretrained(MID, revision=REV)
    srcs = ["models/audioflamingo3/processing_audioflamingo3.py", "models/audioflamingo3/modeling_audioflamingo3.py"]
    hubfiles = ["preprocessor_config.json", "processor_config.json", "config.json"]
    AUD = proc.audio_token_id
srcs += ["models/whisper/feature_extraction_whisper.py", "feature_extraction_sequence_utils.py", "processing_utils.py"]
res["model"] = MID; res["revision"] = REV
res["installed_src_sha256"] = {s: sha(os.path.join(TFDIR, s)) for s in srcs}
res["hub_files"] = {}
for f in hubfiles:
    try:
        p = hf_hub_download(MID, f, revision=REV); res["hub_files"][f] = {"sha256": sha(p), "json": json.load(open(p))}
    except Exception as e:
        res["hub_files"][f] = {"error": repr(e)[:200]}
fe = proc.feature_extractor
res["processor_class"] = type(proc).__name__; res["feature_extractor_class"] = type(fe).__name__
res["fe_attrs"] = {k: getattr(fe, k, None) for k in ("chunk_length", "n_samples", "nb_max_frames", "hop_length", "sampling_rate", "feature_size", "n_fft")}
res["proc_attrs"] = {k: getattr(proc, k, None) for k in ("max_audio_len", "audio_token")}
res["audio_token_id"] = int(AUD)
res["processor_kwargs_defaults_audio"] = PK._defaults.get("audio_kwargs")
try:
    mk = proc._merge_kwargs(PK, tokenizer_init_kwargs=proc.tokenizer.init_kwargs)
    res["merged_audio_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in mk["audio_kwargs"].items()}
except Exception as e:
    res["merged_audio_kwargs"] = {"error": repr(e)[:300]}
say("MODEL", MID, "rev", REV, "tf", transformers.__version__, "| proc", res["processor_class"], "| fe", res["feature_extractor_class"])
say("FE", json.dumps(res["fe_attrs"]), "| PROC", json.dumps(res["proc_attrs"]), "| audio_token_id", AUD)
say("DEFAULT audio_kwargs", json.dumps(res["processor_kwargs_defaults_audio"]), "| MERGED", json.dumps(res["merged_audio_kwargs"]))
for f, v in res["hub_files"].items():
    j = v.get("json", {})
    if f == "config.json":
        ac = (j.get("thinker_config", {}) or {}).get("audio_config") or j.get("audio_config") or {}
        say("HUB config.json audio_config", json.dumps({k: ac.get(k) for k in ("n_window", "n_window_infer", "max_source_positions", "conv_chunksize", "num_mel_bins", "output_dim")}))
    else:
        say("HUB", f, json.dumps({k: j.get(k) for k in ("feature_extractor_type", "processor_class", "chunk_length", "n_samples", "nb_max_frames", "hop_length", "sampling_rate", "max_audio_len", "feature_size") if k in j}))
TMP = tempfile.mkdtemp(prefix="p23c_")
rows = []
for pid in PIDS:
    path = f"/workspace/edaicfull/cut/{pid}.wav"
    x, _ = librosa.load(path, sr=16000)                       # WINDOW_S=0 in the paper runs: no crop before the processor
    ns = len(x); win_s = float(WIN[pid]["win_dur_s"])
    if WHICH == "q3o":   # extract_probe_layers_promptoverride.py:78-82 verbatim
        conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        frames = int(inp["feature_attention_mask"].sum()); feat_shape = list(inp["input_features"].shape)
        frames_full = math.ceil(ns / fe.hop_length)
        tok_full = int(q3len(torch.tensor([frames_full]), 50)[0])
    else:                # af3_zs.py:42-45 verbatim with WINDOW_S=0
        cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
        conv = [{"role": "user", "content": [{"type": "audio", "path": cut}, {"type": "text", "text": P}]}]
        inp = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        frames = int(inp["input_features_mask"].sum()); feat_shape = list(inp["input_features"].shape)
        x2, _ = sf.read(cut, dtype="float32"); ns_file = len(x2)
        keep = proc.max_audio_len; proc.max_audio_len = 10 ** 6   # counterfactual: same call with the cap lifted, for the uncut token count
        inp2 = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        proc.max_audio_len = keep
        frames_full = int(inp2["input_features_mask"].sum()); tok_full = int((inp2["input_ids"][0] == AUD).sum())
    ntok = int((inp["input_ids"][0] == AUD).sum())
    heard_s = frames * fe.hop_length / 16000
    cutflag = int(ntok < tok_full)
    row = dict(pid=pid, window_s=win_s, samples_in=ns, audio_s_in=round(ns / 16000, 4), input_features_shape=feat_shape,
               feature_frames=frames, feature_frames_if_uncut=frames_full, heard_s=round(heard_s, 3),
               audio_placeholder_tokens=ntok, tokens_if_uncut=tok_full, input_ids_len=int(inp["input_ids"].shape[1]), cut=cutflag)
    rows.append(row); res["files"].append(row)
    say("FILE", json.dumps(row))
    json.dump(res, open(f"{OUT}/{WHICH}_proc.json", "w"), indent=1)
with open(f"{OUT}/{WHICH}_proc.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
say("PROC_DONE", WHICH)
