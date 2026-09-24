"""
Score the DAIC conflict clips and their matched-agreement partners with
Qwen2.5-Omni-7B, audio only, same prompt as the 0.835-AUC whole-interview run.

Question: on segments where the words contradict the clinical label, does the
model go with the words or with the label?
"""
import os, csv, json, warnings
import numpy as np, pandas as pd, torch, librosa
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

R    = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT  = f"{R}/committed/paradox"
MODEL = "Qwen/Qwen2.5-Omni-7B"
SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
       "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
# identical wording to abl_dcaps_omni (AUC 0.835 on whole interviews)
P_MAIN = ("Based only on this recording, does this speaker show signs of depression? "
          "Answer with one word, Yes or No.")
# voice-only framing from the reliance run
P_VOICE = ("Based only on how this person's voice sounds, does this speaker show signs of "
           "depression? Answer with one word, Yes or No.")

proc = Qwen2_5OmniProcessor.from_pretrained(MODEL)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda").eval()
try:
    model.disable_talker(); print("talker disabled", flush=True)
except Exception as e:
    print("disable_talker skipped", repr(e)[:80], flush=True)
tok = proc.tokenizer
def wids(ws):
    s = set()
    for w in ws:
        for v in (w, " " + w):
            e = tok(v, add_special_tokens=False).input_ids
            if e: s.add(e[0])
    return sorted(s)
YES, NO = wids(["Yes", "yes", "YES"]), wids(["No", "no", "NO"])
print("yes_ids", YES, "no_ids", NO, flush=True)

def score(audio, prompt):
    conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio},
                                         {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[audio], return_tensors="pt", padding=True, use_audio_in_video=False)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=1, do_sample=False, output_scores=True,
                           return_dict_in_generate=True, return_audio=False, use_audio_in_video=False)
    p = torch.softmax(g.scores[0][0].float(), dim=-1)
    py, pn = float(p[YES].sum()), float(p[NO].sum())
    return py / (py + pn + 1e-9), py + pn

M = pd.read_csv(f"{OUT}/manifest.csv")
print("clips to score: %d (%s)" % (len(M), M.set.value_counts().to_dict()), flush=True)
rows = []
memo = {}   # agreement clips are reused across pairs; score each file once
for i, r in M.iterrows():
    try:
        if r.filepath in memo:
            pm, mm, pv = memo[r.filepath]
        else:
            a, _ = librosa.load(r.filepath, sr=16000, mono=True)
            pm, mm = score(a, P_MAIN)
            pv, _  = score(a, P_VOICE)
            memo[r.filepath] = (pm, mm, pv)
        rows.append(dict(seg_uid=r.seg_uid, set=r["set"], pair_id=r.pair_id,
                         speaker=r.speaker_id, label=r.label, sev=r.sev,
                         dur=r.clip_dur_s, val=r.val, oof=r.oof,
                         words_say_depressed=r.words_say_depressed,
                         p_dep=pm, mass=mm, p_dep_voice=pv))
    except Exception as e:
        print("FAIL", r.filepath, repr(e)[:120], flush=True)
    if (i + 1) % 40 == 0: print("  %d/%d" % (i + 1, len(M)), flush=True)
D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/omni_clip_scores.csv", index=False)
print("scored %d clips, mean answer mass %.3f" % (len(D), D.mass.mean()), flush=True)

# ---------------------------------------------------------------- thresholds
W = pd.read_csv(f"{R}/ablation/abl_dcaps_omni_clips.csv")   # 275 whole interviews, same prompt
def bal_acc(yv, pv, t):
    pr = (pv >= t).astype(int)
    return 0.5 * ((pr[yv == 1] == 1).mean() + (pr[yv == 0] == 0).mean())
grid = np.linspace(0.01, 0.99, 197)
T_SESSION = float(grid[np.argmax([bal_acc(W.label.values, W.p_orig.values, t) for t in grid])])
print("\nwhole-interview reference (n=275, same prompt): AUC %.3f, best balanced-acc threshold %.3f"
      % (roc_auc_score(W.label, W.p_orig), T_SESSION), flush=True)

def blk(d, t):
    yv, pv = d.label.values, d.p_dep.values
    pr = (pv >= t).astype(int)
    out = dict(n=len(d), speakers=d.speaker.nunique(), mean_p=round(float(pv.mean()), 3),
               acc=round(float((pr == yv).mean()), 3),
               bal_acc=round(float(bal_acc(yv, pv, t)), 3),
               pred_dep_rate=round(float(pr.mean()), 3))
    try: out["auc"] = round(float(roc_auc_score(yv, pv)), 3)
    except Exception: out["auc"] = float("nan")
    return out

res = {}
for t, tag in [(0.5, "thr0.50"), (T_SESSION, "thr_session")]:
    for s in ["conflict", "agreement"]:
        res[f"{s}|{tag}"] = blk(D[D.set == s], t)
    for s in ["conflict", "agreement"]:
        for lab, nm in [(1, "dep"), (0, "ctl")]:
            res[f"{s}|{nm}|{tag}"] = blk(D[(D.set == s) & (D.label == lab)], t)

print("\n================ RESULTS ================")
for t, tag in [(0.5, "0.50"), (T_SESSION, "%.3f (from whole interviews)" % T_SESSION)]:
    print("\n--- decision threshold p(depressed) >= %s ---" % tag)
    for s in ["conflict", "agreement"]:
        d = D[D.set == s]; b = blk(d, t)
        print("  %-10s n=%3d spk=%2d  acc %.3f  bal_acc %.3f  AUC %.3f  mean p %.3f  pred-dep rate %.3f"
              % (s, b["n"], b["speakers"], b["acc"], b["bal_acc"], b["auc"], b["mean_p"], b["pred_dep_rate"]))
        for lab, nm in [(1, "depressed"), (0, "control  ")]:
            dd = d[d.label == lab]; bb = blk(dd, t)
            print("      %s label  n=%3d  acc %.3f  mean p %.3f" % (nm, bb["n"], bb["acc"], bb["mean_p"]))

# ---------------------------------------------------------------- words vs label
t = T_SESSION
C = D[D.set == "conflict"].copy()
C["pred"] = (C.p_dep >= t).astype(int)
follow_label = float((C.pred == C.label).mean())
follow_words = float((C.pred == C.words_say_depressed).mean())
print("\n--- on CONFLICT segments, what does the model track? (thr %.3f) ---" % t)
print("  prediction matches the CLINICAL LABEL : %.3f" % follow_label)
print("  prediction matches the LEXICAL CONTENT: %.3f" % follow_words)
print("  (the two are mutually exclusive here by construction of the set)")

# ---------------------------------------------------------------- paired test
pairs = D.pivot_table(index="pair_id", columns="set", values="p_dep")
lab_p = D[D.set == "conflict"].set_index("pair_id").label
spk_p = D[D.set == "conflict"].set_index("pair_id").speaker
pairs = pairs.join(lab_p.rename("label")).join(spk_p.rename("speaker")).dropna()
cc = ((pairs.conflict >= t).astype(int) == pairs.label).values
ca = ((pairs.agreement >= t).astype(int) == pairs.label).values
b01 = int((~cc & ca).sum()); b10 = int((cc & ~ca).sum())
from scipy.stats import binomtest
mc = binomtest(b10, b10 + b01, 0.5).pvalue if (b10 + b01) else float("nan")
print("\n--- matched pairs (same speaker, duration-matched, n=%d) ---" % len(pairs))
print("  conflict correct  %.3f   agreement correct  %.3f   difference %+.3f"
      % (cc.mean(), ca.mean(), cc.mean() - ca.mean()))
print("  discordant pairs: agreement-only-correct %d, conflict-only-correct %d, exact McNemar p=%.4g"
      % (b01, b10, mc))

# speaker-clustered bootstrap on the difference
rng = np.random.default_rng(0); spks = pairs.speaker.unique(); boot = []
for _ in range(4000):
    s = rng.choice(spks, len(spks), replace=True)
    idx = np.concatenate([np.where(pairs.speaker.values == x)[0] for x in s])
    boot.append(cc[idx].mean() - ca[idx].mean())
lo, hi = np.percentile(boot, [2.5, 97.5])
print("  speaker-clustered bootstrap 95%% CI on the difference: [%+.3f, %+.3f] (4000 resamples)" % (lo, hi))

# speaker-level label shuffle control
sl = D[D.set == "conflict"].groupby("speaker").label.first()
null = []
for _ in range(2000):
    perm = pd.Series(rng.permutation(sl.values), index=sl.index)
    yv = D.loc[D.set == "conflict", "speaker"].map(perm).values
    null.append(float(((D.loc[D.set == "conflict", "p_dep"].values >= t).astype(int) == yv).mean()))
print("  speaker-level label-shuffle control on conflict acc: mean %.3f (95%% [%.3f, %.3f]) vs observed %.3f"
      % (np.mean(null), *np.percentile(null, [2.5, 97.5]), (C.pred == C.label).mean()))

# ---------------------------------------------------------------- ceiling
sub = W[W.speaker.isin(D.speaker.unique())]
print("\n--- CEILING / LENGTH CONTROL ---")
print("  same %d speakers, WHOLE 10-20 min interview, same model+prompt: AUC %.3f, acc %.3f (thr %.3f)"
      % (sub.speaker.nunique(), roc_auc_score(sub.label, sub.p_orig),
         ((sub.p_orig >= T_SESSION).astype(int) == sub.label).mean(), T_SESSION))
print("  median clip duration here: conflict %.1fs, agreement %.1fs"
      % (D[D.set == "conflict"].dur.median(), D[D.set == "agreement"].dur.median()))

json.dump(dict(model="Qwen2.5-Omni-7B", prompt=P_MAIN, threshold_session=T_SESSION,
               n_clips=int(len(D)), n_pairs=int(len(pairs)), speakers=int(D.speaker.nunique()),
               blocks=res, follow_label=follow_label, follow_words=follow_words,
               paired_conflict_acc=float(cc.mean()), paired_agreement_acc=float(ca.mean()),
               paired_diff=float(cc.mean() - ca.mean()),
               paired_diff_ci=[float(lo), float(hi)], mcnemar_p=float(mc),
               shuffle_null_mean=float(np.mean(null)),
               whole_interview_auc_same_speakers=float(roc_auc_score(sub.label, sub.p_orig)),
               whole_interview_acc_same_speakers=float(((sub.p_orig >= T_SESSION).astype(int) == sub.label).mean()),
               whole_interview_auc_all275=float(roc_auc_score(W.label, W.p_orig)),
               ), open(f"{OUT}/omni_paradox_summary.json", "w"), indent=2)
print("\nJOB_DONE", flush=True)
