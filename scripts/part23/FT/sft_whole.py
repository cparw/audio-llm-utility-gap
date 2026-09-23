"""PART23 item 4. One outer fold of the Qwen2.5-Omni projector fine-tune, copied from p15/sft_projector.py (Omni branch):
thinker.audio_tower.proj only trainable (float32 via the same _fp32_forward wrap), encoder and LM frozen, bf16 model,
AdamW lr 1e-4, 3 epochs, one clip per update, order RandomState(fold*10+ep).permutation(tr), cross entropy over the
' Yes' and ' No' logits at the last position, target 0 if label==1 else 1, eval p_yes = softmax over those two logits.
Changed only in:
  MODE=whole : the recipe's x = x[:16000*30] slice is removed and the processor call gets truncation=False, so the
               model hears the whole window (windows_full.csv, cap 900 s, up to 22,500 audio tokens).
  MODE=ctrl300: the 30 s slice is removed, processor default (Qwen2.5-Omni feature extractor caps at n_samples = 300 s).
  folds are READ from FOLDFILE (clip_id <pid>.wav), never recomputed; this process trains folds != FOLD, predicts FOLD.
  eval also records p_yes_rule (single-token Yes/yes/YES vs No/no/NO, bare and leading space, full softmax),
  answer_mass and audio_tok beside the recipe p_yes. Checkpoint (projector + AdamW state) at every epoch end and
  every CK steps; per-clip eval rows written as they are scored. Optional gradient checkpointing on the frozen LM
  (GC=auto: only if a probe forward+backward on the longest training clip does not fit; the probe's grads are zeroed
  and no optimizer step is taken, so it does not change the math).
usage: sft_whole.py MANIFEST(path,label,speaker,pid) FOLDFILE FOLD MODE(whole|ctrl300) OUTDIR"""
import os, csv, sys, time, copy, json, datetime, hashlib, platform, numpy as np, torch, librosa
MAN, FOLDF, FOLD, MODE, OUTD = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
assert MODE in ("whole", "ctrl300")
MID = "Qwen/Qwen2.5-Omni-7B"; EPOCHS, LR = 3, 1e-4
CK = int(os.environ.get("CK", "55")); GCMODE = os.environ.get("GC", "auto"); GC_LIMIT = float(os.environ.get("GC_LIMIT_GB", "128"))
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
os.makedirs(OUTD, exist_ok=True)
TAG = f"{MODE}_fold{FOLD}"; LOG = lambda *a: print(*a, flush=True)
dev = "cuda"; DT = torch.bfloat16
T_START = time.time()
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
import transformers
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=DT)
net = full.thinker
for attr in ("talker", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to(dev)
proj = net.audio_tower.proj
tok = proc.tokenizer
_orig = proj.forward
def _fp32_forward(x): return _orig(x.float()).to(DT)
proj.forward = _fp32_forward
for p in net.parameters(): p.requires_grad = False
proj_init = copy.deepcopy(proj.state_dict())
def wid(w): return tok(" " + w, add_special_tokens=False).input_ids[0]
YES, NO = wid("Yes"), wid("No")
def ids_rule(ws):
    out = []
    for w in ws:
        for f in (w, " " + w):
            t = tok.encode(f, add_special_tokens=False)
            if len(t) == 1: out.append(t[0])
    return sorted(set(out))
YR, NR = ids_rule(["Yes", "yes", "YES"]), ids_rule(["No", "no", "NO"])
cfg = net.config
AUD = next((v for v in (getattr(cfg, "audio_token_id", None), getattr(cfg, "audio_token_index", None)) if v is not None),
           tok.convert_tokens_to_ids("<|AUDIO|>"))
rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([str(r["speaker"]) for r in rows])
fd = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(FOLDF))}
fold_of = np.array([fd[r["pid"] + ".wav"][0] for r in rows])
assert all(fd[r["pid"] + ".wav"][1] == r["speaker"] for r in rows), "speaker mismatch vs fold file"
tr, te = np.where(fold_of != FOLD)[0], np.where(fold_of == FOLD)[0]
fe = proc.feature_extractor
LOG(f"MODEL {MID} | tf {transformers.__version__} torch {torch.__version__} | {len(rows)} clips {len(set(spk))} spk | fold {FOLD} "
    f"train {len(tr)} test {len(te)} | mode {MODE} | recipe ids yes {YES} no {NO} | rule ids {YR} {NR} | audio tok {AUD} | "
    f"fe n_samples {getattr(fe,'n_samples',None)} chunk_length {getattr(fe,'chunk_length',None)} | attn {getattr(cfg,'_attn_implementation',None)}")
def inputs_for(r):
    x, _ = librosa.load(r["path"], sr=16000)                      # recipe slice x[:16000*30] removed (both modes)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if MODE == "whole":
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)
    else:
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    return inp, len(x) / 16000.0
import soundfile as sf
dur = {i: sf.info(r["path"]).frames / 16000.0 for i, r in enumerate(rows)}
CKF = f"{OUTD}/{TAG}_ckpt.pt"; EVF = f"{OUTD}/{TAG}_eval_partial.csv"; STF = f"{OUTD}/{TAG}_state.json"
st = json.load(open(STF)) if os.path.exists(STF) else {"loss": {}, "steps": [], "gc": None, "probe": None, "restarts": 0, "train_s": 0.0}
torch.cuda.reset_peak_memory_stats()
def save_state(): json.dump(st, open(STF + ".tmp", "w")); os.replace(STF + ".tmp", STF)
proj.load_state_dict(proj_init)
for p in proj.parameters(): p.data = p.data.float(); p.requires_grad = True
opt = torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=LR)
ep0, pos0, tot, nb = 0, 0, 0.0, 0
if os.path.exists(CKF):
    ck = torch.load(CKF, map_location=dev, weights_only=False)
    proj.load_state_dict(ck["proj"]); opt.load_state_dict(ck["opt"])
    ep0, pos0, tot, nb = ck["ep"], ck["pos"], ck["tot"], ck["nb"]
    st["restarts"] += 1; save_state()
    LOG(f"RESUME from checkpoint ep {ep0} pos {pos0} (restart #{st['restarts']})")
def set_gc(on):
    if on:
        net.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    st["gc"] = bool(on); save_state()
def step_on(i):
    inp, _ = inputs_for(rows[i])
    out = net(**inp)
    two = out.logits[0, -1, [YES, NO]].float()
    target = torch.tensor(0 if y[i] == 1 else 1, device=dev)
    loss = torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
    loss.backward()
    ntok = int((inp["input_ids"][0] == AUD).sum()); del out, inp
    return loss, ntok
net.train()
if st["gc"] is None and ep0 == 0 and pos0 == 0 and os.path.exists(EVF) is False:
    ilong = int(tr[np.argmax([dur[i] for i in tr])])
    if GCMODE == "on": set_gc(True)
    elif GCMODE == "off": set_gc(False)
    else:
        tp = time.time(); ok = False
        try:
            loss, ntok = step_on(ilong); torch.cuda.synchronize()
            pk = torch.cuda.max_memory_allocated() / 2**30; ok = pk < GC_LIMIT
            st["probe"] = {"pid": rows[ilong]["pid"], "dur_s": dur[ilong], "audio_tok": ntok, "peak_alloc_gb_no_gc": round(pk, 2), "s": round(time.time()-tp, 2)}
        except torch.cuda.OutOfMemoryError:
            st["probe"] = {"pid": rows[ilong]["pid"], "dur_s": dur[ilong], "oom_no_gc": True}
        opt.zero_grad(); torch.cuda.empty_cache()
        if not ok:
            set_gc(True); torch.cuda.reset_peak_memory_stats(); tp = time.time()
            loss, ntok = step_on(ilong); torch.cuda.synchronize()
            st["probe"]["peak_alloc_gb_gc"] = round(torch.cuda.max_memory_allocated() / 2**30, 2); st["probe"]["s_gc"] = round(time.time()-tp, 2)
            opt.zero_grad(); torch.cuda.empty_cache()
        else: set_gc(False)
        for p in proj.parameters(): assert p.grad is None
        LOG(f"PROBE longest train clip {st['probe']} -> gradient checkpointing {'ON' if st['gc'] else 'OFF'}")
        torch.cuda.reset_peak_memory_stats()
elif st["gc"]: set_gc(True)
tr_tok_sum = sum(dur[i] for i in tr)  # seconds of audio per epoch (projection)
t_train = time.time(); train_s_before = st.get("train_s", 0.0)
for ep in range(ep0, EPOCHS):
    order = np.random.RandomState(FOLD * 10 + ep).permutation(tr)
    if ep > ep0: tot = nb = 0; pos_start = 0
    else: pos_start = pos0
    for pos in range(pos_start, len(order)):
        i = order[pos]; ts = time.time()
        loss, ntok = step_on(i); opt.step(); opt.zero_grad()
        tot += float(loss.detach()); nb += 1; dt = time.time() - ts
        st["steps"].append([ep, pos, rows[i]["pid"], round(dur[i], 2), ntok, round(dt, 3), round(float(loss.detach()), 5)])
        if len(st["steps"]) == 20 and st.get("proj20") is None:
            s = np.array([z[5] for z in st["steps"][:20]]); d = np.array([z[3] for z in st["steps"][:20]])
            per_s = s.sum() / d.sum()
            proj_len = per_s * tr_tok_sum * EPOCHS / 60; proj_mean = s.mean() * len(tr) * EPOCHS / 60
            ev = proj_len / (len(tr) * EPOCHS) * len(te) / 3
            st["proj20"] = {"mean_step_s": round(float(s.mean()), 3), "mean_dur_s_20": round(float(d.mean()), 1),
                            "mean_dur_s_train": round(tr_tok_sum / len(tr), 1), "proj_train_min_len_weighted": round(proj_len, 1),
                            "proj_train_min_mean": round(proj_mean, 1), "proj_eval_min": round(ev, 1),
                            "peak_alloc_gb_so_far": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            LOG(f"PROJ20 {TAG} {st['proj20']}"); save_state()
        if (pos + 1) % CK == 0 and pos + 1 < len(order):
            torch.save({"proj": proj.state_dict(), "opt": opt.state_dict(), "ep": ep, "pos": pos + 1, "tot": tot, "nb": nb}, CKF + ".tmp"); os.replace(CKF + ".tmp", CKF)
            st["train_s"] = train_s_before + time.time() - t_train; save_state()
            LOG(f"{TAG} ep {ep} step {pos+1}/{len(order)} loss_avg {tot/nb:.4f} last_tok {ntok} {dt:.2f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB {(time.time()-T_START)/60:.1f} min")
    st["loss"][str(ep)] = tot / nb
    torch.save({"proj": proj.state_dict(), "opt": opt.state_dict(), "ep": ep + 1, "pos": 0, "tot": 0.0, "nb": 0}, CKF + ".tmp"); os.replace(CKF + ".tmp", CKF)
    st["train_s"] = train_s_before + time.time() - t_train; save_state()
    LOG(f"fold {FOLD} epoch {ep} loss {tot/nb:.4f} {(time.time()-T_START)/60:.1f} min  EPOCH_CKPT")
if not os.path.exists(f"{OUTD}/{TAG}_proj_final.pt"): torch.save(proj.state_dict(), f"{OUTD}/{TAG}_proj_final.pt")
st["peak_alloc_gb_train"] = round(torch.cuda.max_memory_allocated() / 2**30, 2); st["peak_reserved_gb_train"] = round(torch.cuda.max_memory_reserved() / 2**30, 2); save_state()
torch.cuda.reset_peak_memory_stats()
net.eval(); t_ev = time.time()
scored = {r["pid"] for r in csv.DictReader(open(EVF))} if os.path.exists(EVF) else set()
if not os.path.exists(EVF):
    with open(EVF, "w", newline="") as fh: csv.writer(fh).writerow(["pid", "label", "fold", "p_yes", "p_yes_rule", "answer_mass", "audio_tok", "dur_s", "seq_len"])
with torch.no_grad():
    for i in te:
        if rows[i]["pid"] in scored: continue
        inp, d = inputs_for(rows[i])
        out = net(**inp)
        lg = out.logits[0, -1].float()
        pyes = float(torch.softmax(lg[[YES, NO]], -1)[0])
        pr = torch.softmax(lg, -1); py, pn = float(pr[YR].sum()), float(pr[NR].sum())
        with open(EVF, "a", newline="") as fh:
            csv.writer(fh).writerow([rows[i]["pid"], int(y[i]), FOLD, repr(pyes), repr(py / (py + pn)), repr(py + pn),
                                     int((inp["input_ids"][0] == AUD).sum()), round(d, 4), int(inp["input_ids"].shape[1])])
        del out, inp
st["eval_s"] = time.time() - t_ev; st["peak_alloc_gb_eval"] = round(torch.cuda.max_memory_allocated() / 2**30, 2); save_state()
ev = {r["pid"]: r for r in csv.DictReader(open(EVF))}
FIN = f"{OUTD}/FT_{TAG}_oof.csv"
with open(FIN, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["pid", "label", "fold", "p_yes", "p_yes_rule", "answer_mass", "audio_tok", "dur_s", "seq_len"])
    for i in te:
        e = ev[rows[i]["pid"]]; w.writerow([e[k] for k in ["pid", "label", "fold", "p_yes", "p_yes_rule", "answer_mass", "audio_tok", "dur_s", "seq_len"]])
from sklearn.metrics import roc_auc_score
yy = np.array([int(ev[rows[i]["pid"]]["label"]) for i in te]); pp = np.array([float(ev[rows[i]["pid"]]["p_yes"]) for i in te])
auc = float(roc_auc_score(yy, pp)) if len(set(yy)) > 1 else float("nan")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
cut_hash = hashlib.sha256("".join(f"{r['pid']}:{sha(r['path'])}\n" for r in rows).encode()).hexdigest()
try:
    from huggingface_hub import snapshot_download; snap = os.path.basename(snapshot_download(MID, local_files_only=True))
except Exception as e: snap = repr(e)[:80]
import sklearn, scipy
meta = {"what": f"Qwen2.5-Omni projector fine-tune, E-DAIC, {'WHOLE window (truncation lifted)' if MODE=='whole' else 'CONTROL 300 s (processor default cap)'}, outer fold {FOLD}",
        "result_file": os.path.basename(FIN), "fold": FOLD, "mode": MODE, "fold_auc_two_logit_p_yes": auc,
        "n": int(len(te)), "n_speakers": int(len(set(spk[te]))), "n_pos": int(yy.sum()), "n_train": int(len(tr)),
        "model_id": MID, "hf_snapshot": snap, "prompt": P, "seed": "training order np.random.RandomState(fold*10+epoch).permutation(train indices); no other RNG",
        "recipe": "p15/sft_projector.py Omni branch: thinker.audio_tower.proj only (float32), encoder+LM frozen bf16, AdamW lr 1e-4, 3 epochs, 1 clip/update, CE over ' Yes'/' No' logits at last position",
        "changes_from_recipe": ["removed x = x[:16000*30] (the recipe heard only the first 30 s)",
                                "processor truncation=False" if MODE == "whole" else "processor default (300 s cap)",
                                "folds read from saved fold file", "one outer fold per process", "extra eval columns p_yes_rule, answer_mass, audio_tok",
                                "checkpoint every epoch and every %d steps" % CK],
        "gradient_checkpointing_frozen_lm": st["gc"], "probe_longest_clip": st["probe"], "proj20": st.get("proj20"),
        "yes_id": YES, "no_id": NO, "rule_ids": [YR, NR], "audio_token_id": AUD,
        "feature_extractor_n_samples": getattr(fe, "n_samples", None), "attn_implementation": getattr(cfg, "_attn_implementation", None),
        "losses_per_epoch": st["loss"], "train_minutes": round(st["train_s"] / 60, 2), "eval_minutes": round(st["eval_s"] / 60, 2),
        "wall_minutes_this_process": round((time.time() - T_START) / 60, 2), "restarts_from_checkpoint": st["restarts"],
        "peak_alloc_gb_train": st["peak_alloc_gb_train"], "peak_reserved_gb_train": st["peak_reserved_gb_train"], "peak_alloc_gb_eval": st["peak_alloc_gb_eval"],
        "audio_tok_test": {"min": int(min(int(ev[rows[i]['pid']]['audio_tok']) for i in te)), "max": int(max(int(ev[rows[i]['pid']]['audio_tok']) for i in te))},
        "audio_tok_train_steps_max": int(max(z[4] for z in st["steps"])) if st["steps"] else None,
        "sources": {"manifest": [os.path.abspath(MAN), sha(MAN)], "fold_file": [os.path.abspath(FOLDF), sha(FOLDF)],
                    "script": [os.path.abspath(__file__), sha(__file__)], "cut_windows_sha256_of_list": cut_hash,
                    "windows_full.csv": ["/workspace/edaicfull/windows_full.csv", sha("/workspace/edaicfull/windows_full.csv")],
                    "build_windows.py": ["/workspace/build_windows.py", sha("/workspace/build_windows.py")]},
        "command": " ".join(["python3"] + sys.argv), "env": {"GC": GCMODE, "GC_LIMIT_GB": GC_LIMIT, "CK": CK},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                     "numpy": np.__version__, "librosa": librosa.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
                     "gpu": torch.cuda.get_device_name(0)},
        "host": platform.node(), "date_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"}
json.dump(meta, open(f"{OUTD}/FT_{TAG}_oof.json", "w"), indent=1)
open(f"{OUTD}/FT_{TAG}.RESULT", "w").write(f"{auc}\n")
LOG(f"RESULT {TAG} fold AUC {auc:.4f} n={len(te)} train {meta['train_minutes']} min eval {meta['eval_minutes']} min peak {st['peak_alloc_gb_train']} GB gc {st['gc']}")
