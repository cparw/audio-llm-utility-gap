"""PART 23 A, items 1-3 extraction (pod). E-DAIC WHOLE interview window (900 s cap), Qwen2.5-Omni-7B thinker, bf16.
Model load, prompt, chat template, dtype casting and Yes/No scoring are those of
edaic_rerun/part16/EDAICFULL/extract_full.py; the processor call is the PART 22 lifted call
    proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)
ONE forward pass per file gives: the zero-shot p_yes (item 2) and all per-stage states (item 3):
  enc  (32, 1280)  each audio-encoder layer output, mean over time   (same hook as extract_full.py)
  proj (1, 3584)   audio_tower.proj output, mean over audio positions (same hook as extract_full.py)
  llm  (29, 3584)  hidden_states[0..28] mean over the <|AUDIO|> token positions
  ans  (29, 3584)  hidden_states[0..28] at the last input position (= first answer position)
Pooling is done exactly as extract_full.py does it (mean in the model dtype, then .float()); stored as float32.
Per-clip npz is written immediately (resume-safe); a checkpoint marker is written every 25 clips.
usage: python3 p23a_extract.py
"""
import os, csv, sys, json, time, hashlib, numpy as np, torch, librosa
import transformers

MID = "Qwen/Qwen2.5-Omni-7B"
D = "/workspace/p23a"
CUT = f"{D}/edaicfull/cut"
SH = f"{D}/shards"
os.makedirs(SH, exist_ok=True)
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
PAPER_YES = [7414, 9454, 9693, 9834, 14004]
PAPER_NO = [902, 2152, 2308, 2753, 8996]
RULE_YES = PAPER_YES + [14080]
RULE_NO = PAPER_NO + [5664]
dev = "cuda"; MDTYPE = torch.bfloat16
t0 = time.time()

from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=torch.bfloat16)
net = full.thinker
for attr in ("talker", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to(dev).eval()
tower = net.audio_tower; enc_layers = tower.layers; projector = tower.proj
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
assert YES == PAPER_YES and NO == PAPER_NO, (YES, NO)
assert tok.encode(" YES", add_special_tokens=False) == [14080] and tok.encode(" NO", add_special_tokens=False) == [5664]
n_enc, n_llm = len(enc_layers), cfg.text_config.num_hidden_layers + 1
MAXPOS = getattr(cfg.text_config, "max_position_embeddings", 32768)
LIB = dict(transformers=transformers.__version__, torch=torch.__version__, numpy=np.__version__, librosa=librosa.__version__,
           python=sys.version.split()[0], cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
           attn=getattr(cfg, "_attn_implementation", None))
print(f"MODEL {MID} loaded in {time.time()-t0:.0f}s | {LIB}", flush=True)
print(f"enc layers {n_enc} | llm stages {n_llm} | max_position_embeddings {MAXPOS} | audio token {AUD} | yes {YES} no {NO}", flush=True)

grab = {}
def hook(name):
    def f(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        grab[name] = h.reshape(-1, h.shape[-1]).mean(dim=0).float().cpu().numpy()
        grab[name + "_n"] = int(h.reshape(-1, h.shape[-1]).shape[0])
    return f
handles = [enc_layers[i].register_forward_hook(hook(f"enc{i}")) for i in range(n_enc)]
handles.append(projector.register_forward_hook(hook("proj")))

def sha1mb(p):
    with open(p, "rb") as fh: return hashlib.sha256(fh.read(1 << 20)).hexdigest()

def tok_formula(n_samples):
    # WhisperFeatureExtractor (hop 160, center STFT, last frame dropped) -> n_samples // 160 mel frames;
    # then Qwen2.5-Omni _get_feat_extract_output_lengths: (L-1)//2+1, then (L-2)//2+1
    F = n_samples // 160
    a = (F - 1) // 2 + 1
    return F, (a - 2) // 2 + 1

while not os.path.exists(f"{D}/edaicfull/build_report.csv"):   # wait for build_windows_p23a.py to finish all 275 cuts
    time.sleep(10)
WIN = list(csv.DictReader(open(f"{D}/edaicfull/windows_full.csv")))
SCSV = f"{D}/A_extract_perclip.csv"
cols = ["pid", "win_dur_s", "capped", "wave_samples", "wave_s", "feat_frames_formula", "feat_frames_observed",
        "audio_tok_expected_from_formula", "audio_tok_observed", "enc_positions", "proj_positions", "seq_len", "fit_in_32k",
        "p_yes", "answer_mass", "p_yes_rule", "answer_mass_rule", "cut_wav_sha256_first1MB", "peak_gpu_gib", "model_sec"]
done = set()
if os.path.exists(SCSV):
    done = {r["pid"] for r in csv.DictReader(open(SCSV)) if os.path.exists(f"{SH}/{r['pid']}.npz")}
    print(f"RESUME: {len(done)} clips already done", flush=True)
    # rewrite csv keeping only rows whose npz exists (a crash between npz and csv row is covered by the npz-first order)
    keep = [r for r in csv.DictReader(open(SCSV)) if r["pid"] in done]
    with open(SCSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in keep]
else:
    with open(SCSV, "w", newline="") as fh: csv.writer(fh).writerow(cols)

for i, r in enumerate(WIN):
    pid = r["pid"]
    if pid in done: continue
    tc = time.time()
    path = f"{CUT}/{pid}.wav"
    x, _ = librosa.load(path, sr=16000)   # identical to extract_full.py / p22_run.py
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)
    feat_obs = int(inp["feature_attention_mask"].sum())
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    for _k, _v in list(inp.items()):
        if torch.is_tensor(_v) and _v.dtype == torch.float32: inp[_k] = _v.to(MDTYPE)
    grab.clear()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tm = time.time()
    with torch.no_grad():
        out = net(**inp, output_hidden_states=True)
        mask = (inp["input_ids"][0] == AUD)
        n_aud = int(mask.sum()); seq = int(inp["input_ids"].shape[1])
        assert n_aud > 0
        assert len(out.hidden_states) == n_llm
        llm = torch.stack([h[0][mask].mean(dim=0).float().cpu() for h in out.hidden_states]).numpy()
        ans = torch.stack([h[0][-1].float().cpu() for h in out.hidden_states]).numpy()
        pr = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
    del out
    torch.cuda.synchronize(); msec = time.time() - tm; peak = torch.cuda.max_memory_allocated() / 2**30
    py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    ry, rn = float(pr[RULE_YES].sum()), float(pr[RULE_NO].sum())
    E = np.stack([grab[f"enc{j}"] for j in range(n_enc)]).astype(np.float32)
    np.savez_compressed(f"{SH}/{pid}.npz", enc=E, proj=grab["proj"][None, :].astype(np.float32),
                        llm=llm.astype(np.float32), ans=ans.astype(np.float32))
    Ff, tf = tok_formula(len(x))
    row = dict(pid=pid, win_dur_s=r["win_dur_s"], capped=r["capped"], wave_samples=len(x), wave_s=round(len(x) / 16000, 4),
               feat_frames_formula=Ff, feat_frames_observed=feat_obs, audio_tok_expected_from_formula=tf, audio_tok_observed=n_aud,
               enc_positions=grab["enc0_n"], proj_positions=grab["proj_n"], seq_len=seq, fit_in_32k=int(seq <= MAXPOS),
               p_yes=repr(py / (py + pn + 1e-12)), answer_mass=repr(py + pn), p_yes_rule=repr(ry / (ry + rn + 1e-12)),
               answer_mass_rule=repr(ry + rn), cut_wav_sha256_first1MB=sha1mb(path), peak_gpu_gib=round(peak, 3), model_sec=round(msec, 3))
    with open(SCSV, "a", newline="") as fh: csv.DictWriter(fh, fieldnames=cols).writerow(row)
    done.add(pid); n = len(done)
    if n % 25 == 0 or n <= 3 or n == len(WIN):
        print(f"{n}/{len(WIN)} pid {pid} dur {float(r['win_dur_s']):.0f}s tok {n_aud} (formula {tf}) seq {seq} p_yes {py/(py+pn):.4f} "
              f"mass {py+pn:.4f} peak {peak:.1f}GiB {time.time()-tc:.1f}s tot {time.time()-t0:.0f}s", flush=True)
        open(f"{D}/extract_checkpoint_{n:03d}.RESULT", "w").write(f"{n} clips extracted\n")
    torch.cuda.empty_cache()

for h in handles: h.remove()
json.dump(dict(libs=LIB, model_id=MID, prompt_verbatim=P, dtype="bfloat16", yes_ids=YES, no_ids=NO, rule_yes_ids=RULE_YES, rule_no_ids=RULE_NO,
               audio_token_id=AUD, max_position_embeddings=MAXPOS, n_enc=n_enc, n_llm=n_llm,
               processor_call='proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)',
               audio_load='librosa.load(cut_wav, sr=16000)',
               pooling="enc/proj: forward hook h.reshape(-1,d).mean(0).float(); llm: hidden_states[s][0][audio_mask].mean(0).float(); ans: hidden_states[s][0][-1].float()",
               command="cd /workspace/p23a && nohup python3 p23a_extract.py > extract.log 2>&1 &"),
          open(f"{D}/A_extract_meta.json", "w"), indent=1)
open(f"{D}/A_extract.RESULT", "w").write(f"{len(done)} clips\n")
print("EXTRACTION DONE", time.time() - t0, flush=True)
