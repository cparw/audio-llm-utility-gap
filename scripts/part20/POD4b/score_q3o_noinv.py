"""PART20 POD4b: Qwen3-Omni-30B-A3B-Instruct zero-shot on the 468 Pitt interviewer-free clips.
Scoring path copied from part16/POD3/rescore_probe.py (same loading, same chat template call,
same 30 s truncation, thinker only, logits at the last prompt position = first answer position).
Phase 1: re-score 10 original windows (validation vs part16/pod_sync/q3o_pitt_zeroshot_scores.csv).
Phase 2: wait for /workspace/p20/noinv/READY_POD, then score all clips in manifest.csv with a
per-row checkpoint (restart skips done clips).
p_yes (rule 3): ids of single-token encodings of Yes, Yes, yes, yes, YES, YES (bare + leading space);
same for No. p_yes_ref5 = the reference script's 5+5 id set (no ' YES' / ' NO'), kept for the check.
"""
import os, csv, sys, json, time, datetime, hashlib
os.environ.setdefault("HF_HOME", "/workspace/hf")
import numpy as np, torch, librosa

MID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
WIN = 30.0
BASE = "/workspace/p20"
OUT = "/workspace/scores/part20/POD4b"
VAL_DIR = f"{BASE}/val"
NOINV = f"{BASE}/noinv"
t0 = time.time()

from transformers import Qwen3OmniMoeProcessor as Proc, Qwen3OmniMoeForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=torch.bfloat16)
net = full.thinker
for attr in ("talker", "code2wav", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to("cuda").eval()
tok = proc.tokenizer
print("attn impl:", getattr(net.config, "_attn_implementation", None), flush=True)

def ids_single(words):
    out = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1: out.append(t[0])
    return sorted(set(out))
YES = ids_single(["Yes", " Yes", "yes", " yes", "YES", " YES"])
NO = ids_single(["No", " No", "no", " no", "NO", " NO"])
YES5 = ids_single(["Yes", " Yes", "yes", " yes", "YES"])
NO5 = ids_single(["No", " No", "no", " no", "NO"])
print("YES", YES, "NO", NO, "| ref5 YES", YES5, "NO", NO5, flush=True)
print(f"model ready {time.time()-t0:.0f}s", flush=True)

def score(path):
    x, _ = librosa.load(path, sr=16000)
    dur = len(x) / 16000.0
    x = x[:int(16000 * WIN)] if WIN > 0 else x
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    for _k, _v in list(inp.items()):
        if torch.is_tensor(_v) and _v.dtype == torch.float32: inp[_k] = _v.to(torch.bfloat16)
    with torch.no_grad():
        o = net(**inp)
        pr = torch.softmax(o.logits[0, -1].float(), dim=-1).cpu().numpy()
    py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    py5, pn5 = float(pr[YES5].sum()), float(pr[NO5].sum())
    top = int(pr.argmax())
    return dict(p_yes=py / (py + pn + 1e-12), answer_mass=py + pn,
                p_yes_ref5=py5 / (py5 + pn5 + 1e-12), answer_mass_ref5=py5 + pn5,
                top_token=tok.decode([top]), dur_s=round(dur, 4), dur_scored_s=round(min(dur, WIN), 4))

# ---------------- Phase 1: validation ----------------
ref = {r["clip"]: r for r in csv.DictReader(open(f"{BASE}/ref_rerun.csv"))}
vfile = f"{OUT}/q3o_val10_original_windows.csv"
if not os.path.exists(vfile):
    rows = []
    for fn in sorted(os.listdir(VAL_DIR)):
        if not fn.endswith(".wav"): continue
        s = score(f"{VAL_DIR}/{fn}")
        rr = ref[fn]
        rows.append(dict(clip=fn, speaker_rerun=rr["speaker"], label=rr["label"],
                         p_yes=s["p_yes"], p_yes_ref5=s["p_yes_ref5"], answer_mass=s["answer_mass"],
                         rerun_p_yes=float(rr["p_yes"]), rerun_mass=float(rr["mass"]),
                         absdiff_ref5_vs_rerun=abs(s["p_yes_ref5"] - float(rr["p_yes"])),
                         absdiff_rule3_vs_rerun=abs(s["p_yes"] - float(rr["p_yes"])),
                         top_token=s["top_token"], dur_s=s["dur_s"], prompt=P))
        print(f"VAL {fn} p_yes {s['p_yes']:.6f} ref5 {s['p_yes_ref5']:.6f} rerun {float(rr['p_yes']):.6f}", flush=True)
    with open(vfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    d5 = max(r["absdiff_ref5_vs_rerun"] for r in rows); d3 = max(r["absdiff_rule3_vs_rerun"] for r in rows)
    print(f"VALIDATION max|ref5-rerun| {d5:.3e}  max|rule3-rerun| {d3:.3e}  PASS(1e-3)={d5<1e-3 and d3<1e-3}", flush=True)
    json.dump(dict(max_abs_ref5_vs_rerun=d5, max_abs_rule3_vs_rerun=d3, n=len(rows), pass_1e3=bool(d5 < 1e-3 and d3 < 1e-3),
                   yes_ids=YES, no_ids=NO, yes_ids_ref5=YES5, no_ids_ref5=NO5),
              open(f"{OUT}/q3o_val10_original_windows.check.json", "w"), indent=1)

# ---------------- Phase 2: interviewer-free clips ----------------
while not os.path.exists(f"{NOINV}/READY_POD"):
    print(f"waiting for READY_POD {time.time()-t0:.0f}s", flush=True); time.sleep(30)
man = list(csv.DictReader(open(f"{NOINV}/manifest.csv")))
folds = {r["clip_id"]: r for r in csv.DictReader(open(f"{BASE}/pitt_groupkfold5_pod.csv"))}
sets = {r["clip_id"]: r for r in csv.DictReader(open(f"{BASE}/pitt_sets.csv"))}  # from pitt_conflict_manifest.csv: basename(segment_path) -> set, label
part = f"{OUT}/q3o_pitt_noinv_zeroshot_scores.partial.csv"
cols = ["clip", "orig_clip_id", "speaker", "label", "set", "cut_arm", "p_yes", "answer_mass", "p_yes_ref5",
        "answer_mass_ref5", "top_token", "dur_s", "dur_scored_s", "prompt"]
done = set()
if os.path.exists(part):
    done = {r["clip"] for r in csv.DictReader(open(part))}
    print(f"RESUME: {len(done)} clips already in checkpoint", flush=True)
fh = open(part, "a", newline=""); w = csv.DictWriter(fh, fieldnames=cols)
if not done: w.writeheader(); fh.flush()
clipcol = "clip"
for i, r in enumerate(man):
    fn = os.path.basename(r[clipcol])
    if fn in done: continue
    s = score(f"{NOINV}/clips/{fn}")
    oc = r["orig_clip_id"]
    assert int(sets[oc]["label"]) == int(r["label"]), (oc, "label mismatch vs conflict manifest")
    w.writerow(dict(clip=fn, orig_clip_id=oc, speaker=folds[oc]["speaker_id"], label=int(r["label"]), set=sets[oc]["set"], cut_arm=r["arm"],
                    p_yes=s["p_yes"], answer_mass=s["answer_mass"], p_yes_ref5=s["p_yes_ref5"],
                    answer_mass_ref5=s["answer_mass_ref5"], top_token=s["top_token"], dur_s=s["dur_s"],
                    dur_scored_s=s["dur_scored_s"], prompt=P))
    fh.flush()
    if i % 25 == 0: print(f"  noinv {i+1}/{len(man)} {time.time()-t0:.0f}s p_yes {s['p_yes']:.4f} mass {s['answer_mass']:.4f}", flush=True)
fh.close()
rows = list(csv.DictReader(open(part)))
assert len(rows) == len(man) == len({r["clip"] for r in rows}), (len(rows), len(man))
os.replace(part, f"{OUT}/q3o_pitt_noinv_zeroshot_scores.csv")
json.dump(dict(transformers=__import__("transformers").__version__, torch=torch.__version__,
               librosa=librosa.__version__, gpu=torch.cuda.get_device_name(0), yes_ids=YES, no_ids=NO,
               yes_ids_ref5=YES5, no_ids_ref5=NO5, window_seconds=WIN, dtype="bfloat16",
               finished=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", wall_s=time.time() - t0),
          open(f"{OUT}/q3o_pitt_noinv_zeroshot_scores.env.json", "w"), indent=1)
print("SCORING DONE", len(rows), flush=True)
