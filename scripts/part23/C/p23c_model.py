"""PART 23 item 6, pod C. One model forward on the longest E-DAIC full window (716, 900 s) to confirm how many audio
tokens reach the language model. Mirrors the paper's model load and call. usage: python3 p23c_model.py af3|q3o"""
import os, sys, json, time, datetime, tempfile, numpy as np, torch, librosa, soundfile as sf, transformers
WHICH = sys.argv[1]; OUT = "/workspace/p23c/out"; os.makedirs(OUT, exist_ok=True)
PID = "716"; PATH = f"/workspace/edaicfull/cut/{PID}.wav"
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    open(f"{OUT}/{WHICH}_model.log", "a").write(s + "\n")
res = {"which": WHICH, "pid": PID, "transformers": transformers.__version__, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0), "date_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")}
def dump(): json.dump(res, open(f"{OUT}/{WHICH}_model.json", "w"), indent=1)
cap = {}
def first_text_layer(m):
    for n, mod in m.named_modules():
        if n.endswith("layers.0") and "audio" not in n and "tower" not in n and "visual" not in n:
            return n, mod
def pre(_m, a, kw):
    h = a[0] if a else kw.get("hidden_states"); cap["text_layer0_input_shape"] = list(h.shape)
t0 = time.time()
x, _ = librosa.load(PATH, sr=16000)                          # WINDOW_S=0: whole window
res["samples_in"] = len(x); res["audio_s_in"] = len(x) / 16000
if WHICH == "af3":
    MID = "nvidia/audio-flamingo-3-hf"; REV = "7d4bae64ee29878af6504ae6f6bb3e40492838ad"
    from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MID, revision=REV)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MID, revision=REV, dtype=torch.bfloat16).to("cuda").eval()
    tok = proc.tokenizer
    def ids_for(ws):                                          # af3_zs.py:19-25
        out = []
        for w in ws:
            for f in (w, " " + w):
                t = tok.encode(f, add_special_tokens=False)
                if len(t) == 1: out.append(t[0])
        return sorted(set(out))
    YES = ids_for(["Yes", "yes", "YES"]); NO = ids_for(["No", "no", "NO"])
    AUD = model.config.audio_token_id
    base = model.model; orig = base.get_audio_features
    def gaf(*a, **k):
        o = orig(*a, **k); cap["audio_features_rows"] = int(o.pooler_output.shape[0]); return o
    base.get_audio_features = gaf
    ln, lmod = first_text_layer(base.language_model); lmod.register_forward_pre_hook(pre, with_kwargs=True)
    say("loaded", MID, REV, f"{time.time()-t0:.0f}s", "| text layer hooked:", ln, "| audio_token_id", AUD)
    TMP = tempfile.mkdtemp(prefix="p23c_"); cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
    conv = [{"role": "user", "content": [{"type": "audio", "path": cut}, {"type": "text", "text": P}]}]
    inp = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inp = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inp.items()}
    for k, v in list(inp.items()):
        if torch.is_tensor(v) and v.dtype == torch.float32: inp[k] = v.to(torch.bfloat16)
    res["placeholder_tokens_in_input_ids"] = int((inp["input_ids"][0] == AUD).sum()); res["input_ids_len"] = int(inp["input_ids"].shape[1])
    res["input_features_shape"] = list(inp["input_features"].shape); res["feature_frames"] = int(inp["input_features_mask"].sum())
    try:
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        pr = torch.softmax(lg, -1); py = float(pr[YES].sum()); pn = float(pr[NO].sum())
        res.update(forward_ok=1, p_yes=py / (py + pn + 1e-12), answer_mass=py + pn, **cap)
    except Exception as e:
        res.update(forward_ok=0, error=repr(e)[:500], **cap)
else:
    MID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    from huggingface_hub import HfApi
    REV = HfApi().model_info(MID).sha
    from transformers import Qwen3OmniMoeProcessor as Proc, Qwen3OmniMoeForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID, revision=REV)             # extract_probe_layers_promptoverride.py:25-33
    full = Model.from_pretrained(MID, revision=REV, dtype=torch.bfloat16)
    net = full.thinker
    for attr in ("talker", "code2wav", "token2wav"):
        if hasattr(full, attr): delattr(full, attr)
    if hasattr(net, "visual"): del net.visual
    net = net.to("cuda").eval()
    tok = proc.tokenizer; cfg = net.config
    AUD = next((v for v in (getattr(cfg, "audio_token_id", None), getattr(cfg, "audio_token_index", None)) if v is not None),
               tok.convert_tokens_to_ids("<|AUDIO|>"))
    def ids(words):
        s = set()
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if t: s.add(t[0])
        return sorted(s)
    YES, NO = ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
    orig = net.get_audio_features
    def gaf(*a, **k):
        o = orig(*a, **k); cap["audio_features_rows"] = int(o.last_hidden_state.shape[0]); return o
    net.get_audio_features = gaf
    ln, lmod = first_text_layer(net.model); lmod.register_forward_pre_hook(pre, with_kwargs=True)
    say("loaded", MID, REV, f"{time.time()-t0:.0f}s", "| text layer hooked:", ln, "| audio_token_id", AUD, "| yes", YES, "no", NO)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    for _k, _v in list(inp.items()):
        if torch.is_tensor(_v) and _v.dtype == torch.float32: inp[_k] = _v.to(torch.bfloat16)
    res["placeholder_tokens_in_input_ids"] = int((inp["input_ids"][0] == AUD).sum()); res["input_ids_len"] = int(inp["input_ids"].shape[1])
    res["input_features_shape"] = list(inp["input_features"].shape); res["feature_frames"] = int(inp["feature_attention_mask"].sum())
    res["revision"] = REV
    try:
        with torch.no_grad():
            out = net(**inp, output_hidden_states=True)
            pr = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
            py, pn = float(pr[YES].sum()), float(pr[NO].sum())
        res.update(forward_ok=1, p_yes=py / (py + pn + 1e-12), answer_mass=py + pn, **cap)
    except Exception as e:
        res.update(forward_ok=0, error=repr(e)[:500], **cap)
        if "out of memory" in repr(e).lower():                 # fall back to the audio tower alone
            torch.cuda.empty_cache()
            try:
                with torch.no_grad():
                    o = orig(inp["input_features"], inp["feature_attention_mask"], None, return_dict=True)
                res["audio_tower_only_rows"] = int(o.last_hidden_state.shape[0])
            except Exception as e2:
                res["audio_tower_only_error"] = repr(e2)[:300]
res["model"] = MID; res.setdefault("revision", REV); res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
res["seconds"] = round(time.time() - t0, 1)
dump(); say("MODEL_RESULT", json.dumps(res))
