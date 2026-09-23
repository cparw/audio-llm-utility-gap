"""PART 22 steps 2 and 3 (pod). Same model load, prompt, processor call and Yes/No scoring as
edaic_rerun/part16/EDAICFULL/extract_full.py (the scorer that produced 0.8278). Only additions:
per-call diagnostics, a 300 s cut variant, and a truncation=False variant of the SAME processor call.
usage: python3 p22_run.py step2|probe3|all3
"""
import os, csv, sys, json, time, hashlib, numpy as np, torch, librosa
import transformers
from transformers.models.whisper import feature_extraction_whisper as FEW

MODE = sys.argv[1]
MID = "Qwen/Qwen2.5-Omni-7B"
CUT = "/workspace/edaicfull/cut"
OUT = "/workspace/p22"
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
dev = "cuda"; MDTYPE = torch.bfloat16
t0 = time.time()

# record exactly what reaches WhisperFeatureExtractor.__call__ (pass-through wrapper, no change to behaviour)
_orig_call = FEW.WhisperFeatureExtractor.__call__
LAST_FE_KWARGS = {}
def _logged_call(self, raw_speech, *a, **kw):
    LAST_FE_KWARGS.clear(); LAST_FE_KWARGS.update({k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in kw.items()})
    LAST_FE_KWARGS["_n_samples_in"] = int(len(raw_speech[0])) if isinstance(raw_speech, list) else int(len(raw_speech))
    return _orig_call(self, raw_speech, *a, **kw)
FEW.WhisperFeatureExtractor.__call__ = _logged_call

from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=torch.bfloat16)
net = full.thinker
for attr in ("talker", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to(dev).eval()
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
def rule_ids(words):   # task rule set: bare and leading-space Yes/yes/YES, No/no/NO, single-token encodings only
    s, multi = set(), []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        (s.add(t[0]) if len(t) == 1 else multi.append((w, t)))
    return sorted(s), multi
(RYES, my), (RNO, mn) = rule_ids(["Yes", " Yes", "yes", " yes", "YES", " YES"]), rule_ids(["No", " No", "no", " no", "NO", " NO"])
MAXPOS = getattr(cfg.text_config, "max_position_embeddings", 32768)
fe = proc.feature_extractor
print(f"MODEL {MID} loaded in {time.time()-t0:.0f}s | transformers {transformers.__version__} torch {torch.__version__}", flush=True)
print(f"feature_extractor {type(fe).__name__} chunk_length {fe.chunk_length} sampling_rate {fe.sampling_rate} "
      f"n_samples {fe.n_samples} nb_max_frames {fe.nb_max_frames} hop {fe.hop_length}", flush=True)
print(f"rule yes {RYES} rule no {RNO} multi-token excluded {my + mn}", flush=True)
print(f"audio token {AUD} | yes {YES} | no {NO} | max_position_embeddings {MAXPOS} | attn {getattr(cfg, '_attn_implementation', None)}", flush=True)

def load(pid):
    x, _ = librosa.load(f"{CUT}/{pid}.wav", sr=16000)   # identical to extract_full.py
    return x

def prep(x, lifted=False):
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if not lifted:
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)   # extract_full.py call, verbatim
    else:
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)
    fek = dict(LAST_FE_KWARGS)
    return inp, fek

def score(inp):
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    for _k, _v in list(inp.items()):
        if torch.is_tensor(_v) and _v.dtype == torch.float32: inp[_k] = _v.to(MDTYPE)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tc = time.time()
    with torch.no_grad():
        out = net(**inp, output_hidden_states=True)
        pr = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        py, pn = float(pr[YES].sum()), float(pr[NO].sum())
        ry, rn = float(pr[RYES].sum()), float(pr[RNO].sum())
    del out
    torch.cuda.synchronize()
    sec = time.time() - tc; peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.empty_cache()
    return dict(p_yes=py / (py + pn + 1e-12), answer_mass=py + pn, p_yes_rule=ry / (ry + rn), answer_mass_rule=ry + rn, peak_gpu_gib=round(peak, 3), model_sec=round(sec, 3))

def describe(x, inp, fek):
    ii = inp["input_ids"][0]
    return dict(wave_samples=int(len(x)), wave_s=round(len(x) / 16000, 4),
                input_features_shape=list(inp["input_features"].shape),
                feature_attention_mask_sum=int(inp["feature_attention_mask"].sum()),
                n_audio_tok=int((ii == AUD).sum()), seq_len=int(ii.shape[0]),
                fit_in_context=int(ii.shape[0] <= MAXPOS), fe_kwargs=fek)

def same(a, b):
    return dict(input_ids_equal=bool(torch.equal(a["input_ids"], b["input_ids"])),
                input_features_equal=bool(a["input_features"].shape == b["input_features"].shape and torch.equal(a["input_features"], b["input_features"])),
                feature_attention_mask_equal=bool(a["feature_attention_mask"].shape == b["feature_attention_mask"].shape and torch.equal(a["feature_attention_mask"], b["feature_attention_mask"])))

REF = {r["pid"]: r for r in csv.DictReader(open(f"{OUT}/edaic_full_perclip.csv"))}
WIN = {r["pid"]: r for r in csv.DictReader(open("/workspace/edaicfull/windows_full.csv"))}
LIB = dict(transformers=transformers.__version__, torch=torch.__version__, numpy=np.__version__, librosa=librosa.__version__,
           python=sys.version.split()[0], cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))

if MODE == "step2":
    PIDS = ["305", "679", "466", "705"]
    res = []
    for pid in PIDS:
        x = load(pid)
        x300 = x[:300 * 16000]
        a, fa = prep(x); b, fb = prep(x300)
        da, db = describe(x, a, fa), describe(x300, b, fb)
        eq = same(a, b)
        sa, sb = score(a), score(b)
        long_ = len(x) > 300 * 16000
        tok_same = da["n_audio_tok"] == db["n_audio_tok"]
        p6 = f"{sa['p_yes']:.6f}" == f"{sb['p_yes']:.6f}"
        verdict = ("TRUNCATION PROVEN" if (long_ and tok_same and p6) else
                   ("CONTROL: window shorter than 300 s, full and 300 s cut are the same input" if (not long_ and tok_same and p6 and eq["input_features_equal"]) else "NOT PROVEN"))
        r = dict(pid=pid, src_dur_s=float(WIN[pid]["src_dur_s"]), win_dur_s=float(WIN[pid]["win_dur_s"]), capped=int(WIN[pid]["capped"]),
                 label=int(REF[pid]["label"]), ref_p_yes_answer=float(REF[pid]["p_yes_answer"]),
                 full=dict(**da, **sa), cut300=dict(**db, **sb), identity=eq,
                 p_yes_full_6dp=f"{sa['p_yes']:.6f}", p_yes_cut300_6dp=f"{sb['p_yes']:.6f}",
                 abs_diff_p_yes=abs(sa["p_yes"] - sb["p_yes"]), verdict=verdict)
        res.append(r)
        print(json.dumps(r), flush=True)
    longs = [r for r in res if r["pid"] != "705"]
    overall = "TRUNCATION PROVEN" if all(r["verdict"] == "TRUNCATION PROVEN" for r in longs) else "NOT PROVEN"
    json.dump(dict(step="PART22 step2", results=res, overall_verdict=overall, model_id=MID, prompt_verbatim=P,
                   dtype="bfloat16", libs=LIB, yes_ids=YES, no_ids=NO, rule_yes_ids=RYES, rule_no_ids=RNO, audio_token_id=AUD,
                   processor_call_verbatim='proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)',
                   cut_rule="x300 = x[:300 * 16000]  (x from librosa.load(path, sr=16000))",
                   command="python3 /workspace/p22/p22_run.py step2"),
              open(f"{OUT}/step2_truncation_proof.json", "w"), indent=1)
    print("OVERALL", overall, flush=True)
    open(f"{OUT}/step2_truncation_proof.RESULT", "w").write(overall + "\n")

elif MODE == "probe3":
    pid = "305"
    x = load(pid)
    d, fd = prep(x); l, fl = prep(x, lifted=True)
    dd, dl = describe(x, d, fd), describe(x, l, fl)
    print("DEFAULT", json.dumps(dd), flush=True)
    print("LIFTED ", json.dumps(dl), flush=True)
    rec = dict(step="PART22 step3 probe", pid=pid, default=dd, lifted=dl,
               lifted_call_verbatim='proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)',
               model_id=MID, prompt_verbatim=P, libs=LIB, yes_ids=YES, no_ids=NO, rule_yes_ids=RYES, rule_no_ids=RNO,
               command="python3 /workspace/p22/p22_run.py probe3")
    try:
        sd = score(d); rec["default"].update(sd); print("DEFAULT SCORE", sd, flush=True)
        sl = score(l); rec["lifted"].update(sl); print("LIFTED SCORE", sl, flush=True)
        rec["model_ran"] = True
    except Exception as e:
        import traceback; tb = traceback.format_exc()
        rec["model_ran"] = False; rec["error"] = tb; print(tb, flush=True)
    json.dump(rec, open(f"{OUT}/step3_probe.json", "w"), indent=1)
    open(f"{OUT}/step3_probe.RESULT", "w").write(json.dumps(dict(n_audio_tok_lifted=dl["n_audio_tok"], model_ran=rec["model_ran"])) + "\n")

elif MODE == "all3":
    SCSV = f"{OUT}/lifted_all275.csv"
    cols = ["pid", "label", "win_dur_s", "capped", "ref_p_yes_answer",
            "def_p_yes", "def_mass", "def_n_audio_tok", "def_seq_len", "def_peak_gib", "def_sec",
            "lift_p_yes", "lift_mass", "lift_n_audio_tok", "lift_seq_len", "lift_feat_frames", "lift_fit", "lift_peak_gib", "lift_sec",
            "inputs_identical", "wave_samples", "def_p_yes_rule", "def_mass_rule", "lift_p_yes_rule", "lift_mass_rule"]
    done = set()
    if os.path.exists(SCSV):
        done = {r["pid"] for r in csv.DictReader(open(SCSV))}
        print(f"RESUME: {len(done)} clips already scored", flush=True)
    else:
        with open(SCSV, "w", newline="") as fh: csv.writer(fh).writerow(cols)
    pids = [r["pid"] for r in csv.DictReader(open("/workspace/edaicfull/windows_full.csv"))]
    for i, pid in enumerate(pids):
        if pid in done: continue
        tc = time.time()
        x = load(pid)
        d, fd = prep(x); l, fl = prep(x, lifted=True)
        dd, dl = describe(x, d, fd), describe(x, l, fl)
        eq = same(d, l)
        sd = score(d); sl = score(l)
        ident = int(eq["input_ids_equal"] and eq["input_features_equal"] and eq["feature_attention_mask_equal"])
        row = [pid, REF[pid]["label"], WIN[pid]["win_dur_s"], WIN[pid]["capped"], REF[pid]["p_yes_answer"],
               repr(sd["p_yes"]), repr(sd["answer_mass"]), dd["n_audio_tok"], dd["seq_len"], sd["peak_gpu_gib"], sd["model_sec"],
               repr(sl["p_yes"]), repr(sl["answer_mass"]), dl["n_audio_tok"], dl["seq_len"], dl["feature_attention_mask_sum"], dl["fit_in_context"],
               sl["peak_gpu_gib"], sl["model_sec"], ident, dd["wave_samples"],
               repr(sd["p_yes_rule"]), repr(sd["answer_mass_rule"]), repr(sl["p_yes_rule"]), repr(sl["answer_mass_rule"])]
        with open(SCSV, "a", newline="") as fh: csv.writer(fh).writerow(row)
        n = len(done) + 1; done.add(pid)
        if n % 25 == 0 or n <= 3 or n == len(pids):
            print(f"{n}/{len(pids)} pid {pid} dur {float(WIN[pid]['win_dur_s']):.0f}s def tok {dd['n_audio_tok']} p {sd['p_yes']:.4f} | "
                  f"lift tok {dl['n_audio_tok']} seq {dl['seq_len']} p {sl['p_yes']:.4f} peak {sl['peak_gpu_gib']:.1f}GiB "
                  f"{time.time()-tc:.1f}s tot {time.time()-t0:.0f}s", flush=True)
            open(f"{OUT}/all3_checkpoint_{n:03d}.RESULT", "w").write(f"{n} clips scored\n")
    json.dump(dict(libs=LIB, model_id=MID, prompt_verbatim=P, yes_ids=YES, no_ids=NO, rule_yes_ids=RYES, rule_no_ids=RNO, audio_token_id=AUD, max_position_embeddings=MAXPOS,
                   default_call='proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)',
                   lifted_call='proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)',
                   command="python3 /workspace/p22/p22_run.py all3"), open(f"{OUT}/lifted_all275.meta.json", "w"), indent=1)
    open(f"{OUT}/lifted_all275.RESULT", "w").write(f"{len(done)} clips\n")
    print("ALL3 DONE", time.time() - t0, flush=True)
