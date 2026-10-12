"""Independent AF3 Pitt check (track af3, accel_24sep). Read only on every input. Writes only next to this file.
1. part10 json and the scoring script it names (provenance copy): the cut rule and how the run was launched.
2. Own reading of three clips: samples from librosa.load(sr=16000), samples after x[:16000*30], the temp wav
   round trip, and the AF3 windowing rule re-implemented here from the transformers 5.5.4 source
   (window = 16000*30 samples, n_win = ceil(n/window)), WhisperFeatureExtractor(feature_size=128) mask frames,
   and audio tokens = ((frames-1)//2+1 - 2)//2 + 1 per clip.
3. Durations of all 468 local clips, and which run each per-clip file matches.
4. AUC (sklearn roc_auc_score) per arm for part10, interviewer-free, full segment, with 2000-draw speaker
   bootstrap, fresh default_rng(0) per cell, paired differences on the same draw."""
import csv, os, sys, json, re, hashlib, tempfile, io
import numpy as np, librosa, soundfile as sf
from sklearn.metrics import roc_auc_score

OUTD = os.path.dirname(os.path.abspath(__file__))
P10J = "<local data dir>/release/overnight2/part10/af3_pitt.json"
P10 = "<local data dir>/release/overnight2/part10/af3_pitt.csv"
P10LOG = "<local data dir>/release/overnight2/part10/logs/af3_pitt.log"
DRV = "<local data dir>/release/overnight2/part10/logs/DRIVER.log"
FC = "checks/final_2325Z/af3"
SCRIPT = FC + "/provenance/af3_score.py"
FINISH = FC + "/provenance/finish_all.sh"
POD4C = "scores/part20/POD4c"
NOINV = POD4C + "/work/af3_noinv468.csv"
VAL10 = POD4C + "/work/af3_val10.csv"
HASHES = POD4C + "/work/orig_hashes.csv"
FULL = "<local data dir>/paper1_local_runs/runpod_results/pitt468_af3.csv"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
CLIPS = "<local data dir>/paper1_local_runs/clips_local/clips_for_af3/pitt468"
TEX = "<local data dir>/paper1_submission_23sep/final_overleaf_2325Z/AudioLLMHealthUtilityGap.tex"
THREE = ["agreement_Con_118-1_6475.wav", "conflict_Con_227-0_460.wav", "conflict_Dem_639-0_9189.wav"]
SR, WIN_S, HOP = 16000, 30, 160
NB = 2000

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def say(*a):
    print(*a, flush=True)

# ---------------- 1. json, script, launch line, log ----------------
say("== 1. part10 json and scoring script ==")
J = json.load(open(P10J))
say({k: J[k] for k in ("script", "clip_list", "n", "n_speakers", "window_seconds", "pre_cut_to_single_window", "auc",
                       "median_source_duration_s", "max_source_duration_s", "date")})
src = open(SCRIPT).read().splitlines()
say("script sha256", sha(SCRIPT))
for i, l in enumerate(src, 1):
    if re.search(r"NOCUT|x\[:16000 \* 30\]|window_seconds|pre_cut_to_single_window|\"type\": \"audio\"", l):
        say(f"  line {i}: {l.strip()}")
fin = [l.strip() for l in open(FINISH) if "af3_score.py" in l]
say("launch lines in finish_all.sh:", fin)
nocut_used = any("--nocut" in l for l in fin)
say("any --nocut on the launch line:", nocut_used)
log = open(P10LOG, errors="replace").read()
res = [l for l in log.splitlines() if l.startswith("RESULT")]
say("part10 log RESULT:", res)
say("DRIVER RESULT af3 pitt:", [l.strip() for l in open(DRV) if "RESULT af3 mf_pitt" in l])
# the json keys window_seconds / pre_cut_to_single_window are written only by this script version (lines 50-51)
say("json has keys written only by this script version:",
    all(k in J for k in ("window_seconds", "pre_cut_to_single_window", "median_source_duration_s")))
pip = [l.strip() for l in open(FINISH) if "pip install" in l]
say("pod pip line:", pip)

# ---------------- 2. three clips, own reading ----------------
say("\n== 2. three clips, own reading ==")
import transformers
from transformers import WhisperFeatureExtractor
say("local transformers", transformers.__version__, "librosa", librosa.__version__, "soundfile", sf.__version__)
fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=SR, hop_length=HOP, chunk_length=WIN_S, n_fft=400)
H = {r["clip"]: r for r in csv.DictReader(open(HASHES))}
def tokens_for(frames):
    conv = (frames - 1) // 2 + 1
    return (conv - 2) // 2 + 1
def af3_windows(x):
    # re-implementation of AudioFlamingo3Processor.__call__ windowing (transformers 5.5.4, lines 158-179)
    w = SR * WIN_S; n = int(x.shape[0]); nw = max(1, (n + w - 1) // w); nw = min(nw, 600 // WIN_S)
    cap = min(n, nw * w)
    return [x[i * w:min((i + 1) * w, cap)] for i in range(nw)]
tmpd = tempfile.mkdtemp(prefix="af3chk_")
three_rows = []
for c in THREE:
    p = os.path.join(CLIPS, c)
    info = sf.info(p)
    h = sha(p)
    x, sr = librosa.load(p, sr=SR)
    n_full = len(x)
    for mode in ("as_run_cut30", "no_cut"):
        y = x[:SR * WIN_S] if mode == "as_run_cut30" else x
        t = os.path.join(tmpd, "clip.wav"); sf.write(t, y, SR)
        z, _ = sf.read(t, dtype="float32")        # what apply_chat_template loads from the temp wav path
        wins = af3_windows(z)
        feats = fe(wins, sampling_rate=SR, padding="max_length", return_attention_mask=True, return_tensors="np")
        per = [int(m.sum()) for m in feats["attention_mask"]]
        frames = sum(per)
        tok = int(tokens_for(frames))
        r = dict(clip=c, mode=mode, file_sr=info.samplerate, file_frames=info.frames, file_sha256_matches_pod_hash=(h == H[c]["file_sha256"]),
                 source_samples_librosa=n_full, source_seconds=round(n_full / SR, 4), samples_after_cut=len(y),
                 temp_wav_frames=sf.info(t).frames, samples_to_processor=len(z), seconds_to_processor=round(len(z) / SR, 4),
                 n_windows=len(wins), window_samples="+".join(str(len(w)) for w in wins),
                 feature_shape="x".join(map(str, feats["input_features"].shape)), mask_frames="+".join(map(str, per)),
                 mask_frames_total=frames, audio_tokens=tok, seconds_reaching_model=round(frames * HOP / SR, 4))
        three_rows.append(r)
        say(" | ".join(f"{k}={v}" for k, v in r.items()))
with open(os.path.join(OUTD, "af3_three_clips_own_reading.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(three_rows[0])); w.writeheader(); w.writerows(three_rows)
# compare to the final_checks csv
fc = {(r["clip"], r["mode"]): r for r in csv.DictReader(open(FC + "/af3_pitt_three_clips_input_length.csv"))}
for r in three_rows:
    k = (r["clip"], "as_run_cut30" if r["mode"] == "as_run_cut30" else "nocut_contrast")
    o = fc[k]
    say(f"  vs final_checks {k}: samples {r['samples_to_processor']} vs {o['samples_passed_to_processor']}, "
        f"frames {r['mask_frames_total']} vs {o['feature_mask_sum_frames']}, tokens {r['audio_tokens']} vs {o['audio_tokens_in_input_ids']}, "
        f"seconds {r['seconds_reaching_model']} vs {o['seconds_reaching_model']}")

# ---------------- 3. per-clip files, durations, which input each run saw ----------------
say("\n== 3. per-clip files ==")
ARM = {os.path.basename(r["segment_path"]): r["set"] for r in csv.DictReader(open(MAN))}
p10 = {os.path.basename(r["clip_path"]): r for r in csv.DictReader(open(P10))}
ni = {r["orig_clip_id"]: r for r in csv.DictReader(open(NOINV))}
fu = {os.path.basename(r["clip_path"]): r for r in csv.DictReader(open(FULL))}
v10 = {r["orig_clip_id"]: r for r in csv.DictReader(open(VAL10))}
say("n part10", len(p10), "noinv", len(ni), "full", len(fu), "val10", len(v10))
assert set(p10) == set(ni) == set(fu) and len(p10) == 468 and set(p10) <= set(ARM)
lab_ok = all(int(p10[k]["label"]) == int(ni[k]["label"]) == int(fu[k]["label"]) for k in p10)
spk_ok = all(p10[k]["speaker_id"] == ni[k]["speaker_id"] == fu[k]["speaker_id"] for k in p10)
arm_prefix_ok = all(ARM[k] == k.split("_")[0] for k in p10)
say("labels agree across files:", lab_ok, "| speakers agree:", spk_ok, "| manifest set == filename prefix:", arm_prefix_ok)
dur = {k: sf.info(os.path.join(CLIPS, k)).frames / SR for k in p10}
d = np.array([dur[k] for k in sorted(p10)])
say(f"local clip durations: min {d.min():.3f} median {np.median(d):.3f} max {d.max():.3f}; over 30.0 s: {(d > 30).sum()} of 468; "
    f"json median {J['median_source_duration_s']} max {J['max_source_duration_s']}")
# val10 is a separate first-30 s run (POD4c, transformers 5.17.0); compare
dv = [abs(float(p10[k]["p_yes"]) - float(v10[k]["p_yes"])) for k in v10]
dvf = [abs(float(fu[k]["p_yes"]) - float(v10[k]["p_yes"])) for k in v10]
say(f"val10 (10 clips, scored_s=30.0 on all): max |part10 - val10| = {max(dv):.2e}; max |full - val10| = {max(dvf):.4f}, median {np.median(dvf):.4f}")
ks = sorted(p10)
dp = np.array([abs(float(p10[k]["p_yes"]) - float(fu[k]["p_yes"])) for k in ks])
for lo, hi in ((29.9, 30.1), (30.1, 31), (31, 34), (34, 40), (40, 60)):
    m = (d > lo) & (d <= hi)
    if m.sum(): say(f"  |part10 - full| p_yes, clips {lo}-{hi} s: n={m.sum()} median {np.median(dp[m]):.4f} max {dp[m].max():.4f} share>0.01 {np.mean(dp[m] > 0.01):.2f}")
from scipy.stats import spearmanr
say(f"  spearman(duration, |part10 - full|) = {spearmanr(d, dp).correlation:.3f}")
nis = np.array([float(ni[k]["scored_s"]) for k in ks]); nid = np.array([float(ni[k]["dur_s"]) for k in ks])
say(f"interviewer-free clips: dur_s min {nid.min():.3f} median {np.median(nid):.3f} max {nid.max():.3f}; scored_s max {nis.max():.3f}; clips cut: {(nis < nid - 1e-6).sum()}")
for c in THREE:
    say(f"  {c}: dur {dur[c]:.3f} s | p_yes part10 {float(p10[c]['p_yes']):.6f} full {float(fu[c]['p_yes']):.6f} noinv {float(ni[c]['p_yes']):.6f} (noinv dur {float(ni[c]['dur_s']):.3f}, scored {float(ni[c]['scored_s']):.3f})")

# ---------------- 4. AUC with speaker bootstrap ----------------
say("\n== 4. AUC and speaker bootstrap ==")
def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
def boot(spk, y, s, s2=None):
    rng = np.random.default_rng(0)
    u = np.unique(spk); idx = {k: np.flatnonzero(spk == k) for k in u}; v = []
    for _ in range(NB):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        a = auc(y[ii], s[ii])
        if s2 is not None: a = a - auc(y[ii], s2[ii])
        if not np.isnan(a): v.append(a)
    v = np.array(v); return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)
rows = []
for arm in ("all", "conflict", "agreement"):
    kk = [k for k in ks if arm == "all" or ARM[k] == arm]
    spk = np.array([p10[k]["speaker_id"] for k in kk]); y = np.array([int(p10[k]["label"]) for k in kk])
    A = np.array([float(p10[k]["p_yes"]) for k in kk]); B = np.array([float(ni[k]["p_yes"]) for k in kk]); C = np.array([float(fu[k]["p_yes"]) for k in kk])
    for cell, s, s2 in (("part10_first30s", A, None), ("interviewer_free", B, None), ("full_segment_sep16", C, None),
                        ("interviewer_free_minus_part10", B, A), ("full_segment_minus_part10", C, A)):
        val = auc(y, s) if s2 is None else auc(y, s) - auc(y, s2)
        lo, hi, nu = boot(spk, y, s, s2)
        rows.append(dict(cell=cell, arm=arm, n=len(kk), n_speakers=len(set(spk)), n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
                         value=round(val, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), usable_draws=nu))
        say(f"{cell:32s} {arm:9s} n={len(kk)} spk={len(set(spk))} {val:.4f} [{lo:.4f}, {hi:.4f}] usable={nu}")
with open(os.path.join(OUTD, "af3_pitt_auc_bootstrap_own.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# compare to the other files
for f in (FC + "/af3_pitt_bootstrap.csv", FC + "/af3_pitt_fullseg_bootstrap.csv"):
    say("reference file", f)
    for r in csv.DictReader(open(f)): say("   ", r["cell"], r["arm"], r["value"], r["ci_lo"], r["ci_hi"])
tex = open(TEX).read()
say("Table 2 AF3 row in tex:", [l.strip() for l in tex.splitlines() if l.startswith("Audio Flamingo 3")])
say("DONE")
