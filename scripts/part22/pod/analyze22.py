"""PART 22 analysis (Mac): default vs truncation-off zero-shot AUC on E-DAIC full windows.
Reads the pod's lifted_all275.csv (synced), recomputes every AUC by rank formula, 2000-draw speaker bootstrap
(numpy default_rng(0) fresh per cell, 2.5/97.5, paired = one draw per replicate). Writes json + sidecar + rows.
"""
import csv, json, sys, os, hashlib, datetime, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats22 import auc_rank, boot

BASE = "scores/part22"
SYNC = f"{BASE}/pod/sync"
SRC = f"{SYNC}/lifted_all275.csv"
REFF = "scores/part16/EDAICFULL/edaic_full_perclip.csv"
WINF = "manifests/edaic_windows_full.csv"
def sha1mb(p):
    with open(p, "rb") as f: return hashlib.sha256(f.read(1 << 20)).hexdigest()

R = list(csv.DictReader(open(SRC)))
REF = {r["pid"]: r for r in csv.DictReader(open(REFF))}
WIN = {r["pid"]: r for r in csv.DictReader(open(WINF))}
assert len(R) == 275 and len({r["pid"] for r in R}) == 275, len(R)
assert set(r["pid"] for r in R) == set(REF) == set(WIN)
R.sort(key=lambda r: int(r["pid"]))
pid = [r["pid"] for r in R]
y = np.array([int(REF[p]["label"]) for p in pid])
assert all(int(r["label"]) == int(REF[r["pid"]]["label"]) for r in R)
pref = np.array([float(REF[p]["p_yes_answer"]) for p in pid])
pdef = np.array([float(r["def_p_yes"]) for r in R])
assert np.abs(pdef - pref).max() < 1e-3, "pod default does not reproduce the reference per-clip p_yes"
plif = np.array([float(r["lift_p_yes"]) for r in R])
mdef = np.array([float(r["def_mass"]) for r in R]); mlif = np.array([float(r["lift_mass"]) for r in R])
pdef_r = np.array([float(r["def_p_yes_rule"]) for r in R]); plif_r = np.array([float(r["lift_p_yes_rule"]) for r in R])
mdef_r = np.array([float(r["def_mass_rule"]) for r in R]); mlif_r = np.array([float(r["lift_mass_rule"]) for r in R])
dur = np.array([float(WIN[p]["win_dur_s"]) for p in pid]); cap = np.array([int(WIN[p]["capped"]) for p in pid])
wav = np.array([int(r["wave_samples"]) for r in R])
tdef = np.array([int(r["def_n_audio_tok"]) for r in R]); tlif = np.array([int(r["lift_n_audio_tok"]) for r in R])
slif = np.array([int(r["lift_seq_len"]) for r in R]); flif = np.array([int(r["lift_fit"]) for r in R])
frames = np.array([int(r["lift_feat_frames"]) for r in R]); ident = np.array([int(r["inputs_identical"]) for r in R])
peak = np.array([float(r["lift_peak_gib"]) for r in R]); lsec = np.array([float(r["lift_sec"]) for r in R])

long_ = wav > 300 * 16000; short = ~long_
subsets = {"all275": np.ones(len(R), bool), "long217": long_, "capped89": cap == 1}
print("n", len(R), "pos", int(y.sum()), "| long(>300 s)", int(long_.sum()), "short(<=300 s)", int(short.sum()), "capped", int((cap == 1).sum()))
assert long_.sum() == 217 and short.sum() == 58 and (cap == 1).sum() == 89

# token accounting: expected tokens from the processor's own length formula on the untruncated frame count
# padded (<4.8M samples): attention_mask[:, ::160] keeps ceil(L/160) ones; unpadded (>4.8M): trailing partial hop is dropped -> L//160
exp_frames = np.where(wav < 4800000, -(-wav // 160), wav // 160)
exp_tok = ((exp_frames - 1) // 2 + 1 - 2) // 2 + 1
exp_def_tok = ((np.minimum(exp_frames, 30000) - 1) // 2 + 1 - 2) // 2 + 1
naive_tok = (((wav // 160) - 1) // 2 + 1 - 2) // 2 + 1
checks = dict(
    default_pod_vs_reference_max_abs_diff=float(np.abs(pdef - pref).max()),
    default_tokens_max=int(tdef.max()), default_tokens_at_7500=int((tdef == 7500).sum()),
    lifted_tokens_min=int(tlif.min()), lifted_tokens_max=int(tlif.max()),
    lifted_tokens_equal_formula_on_full_length=int((tlif == exp_tok).sum()),
    lifted_frames_equal_exact_formula=int((frames == exp_frames).sum()),
    default_tokens_equal_exact_formula_min_frames_30000=int((tdef == exp_def_tok).sum()),
    note_naive_samples_div_160_formula_misses_pids=[p for p, a, b in zip(pid, tlif, naive_tok) if a != b],
    lifted_seq_len_max=int(slif.max()), lifted_fit_in_32768=int(flif.sum()),
    lifted_tokens_capped89_all_22500=bool((tlif[cap == 1] == 22500).all()),
    short58_inputs_identical=int(ident[short].sum()),
    short58_max_abs_diff_p_yes_lifted_vs_default=float(np.abs(plif[short] - pdef[short]).max()),
    short58_reproduce_1e4=bool(np.abs(plif[short] - pdef[short]).max() < 1e-4),
    long217_inputs_identical=int(ident[long_].sum()),
    long217_mean_abs_change_p_yes=float(np.abs(plif[long_] - pdef[long_]).mean()),
    long217_pearson_default_vs_lifted=float(np.corrcoef(pdef[long_], plif[long_])[0, 1]),
    answer_mass_default_median=float(np.median(mdef)), answer_mass_lifted_median=float(np.median(mlif)),
    answer_mass_lifted_min=float(mlif.min()),
    rule_vs_paper_idset_max_abs_diff_default=float(np.abs(pdef_r - pdef).max()),
    rule_vs_paper_idset_max_abs_diff_lifted=float(np.abs(plif_r - plif).max()),
    answer_mass_rule_lifted_median=float(np.median(mlif_r)),
    lifted_peak_gpu_gib_max=float(peak.max()), lifted_model_sec_median=float(np.median(lsec)), lifted_model_sec_max=float(lsec.max()),
)
for k, v in checks.items(): print(f"  {k}: {v}")

res = {}; rows = []
def row(i, what, v, lo, hi, n, nspk):
    rows.append(dict(id=i, what=what, value=(str(int(v)) if isinstance(v, (int, np.integer)) else (f"{v:.2e}" if i.endswith("maxdiff") else f"{v:.4f}")), lo=("" if lo is None else f"{lo:.4f}"), hi=("" if hi is None else f"{hi:.4f}"),
                     n=n, n_spk=nspk, file="scores/part22/edaic_lifted_auc.json"))
    print(f"{i:36s} {what:62s} {rows[-1]['value']}" + ("" if lo is None else f" [{lo:.4f}, {hi:.4f}]"))
for name, m in subsets.items():
    yy = y[m]; sp = [p for p, k in zip(pid, m) if k]
    a_ref, a_def, a_lif = auc_rank(yy, pref[m]), auc_rank(yy, pdef[m]), auc_rank(yy, plif[m])
    b_ref = boot(yy, sp, pref[m]); b_def = boot(yy, sp, pdef[m]); b_lif = boot(yy, sp, plif[m])
    b_pair = boot(yy, sp, plif[m], pref[m])
    res[name] = dict(n=int(m.sum()), n_pos=int(yy.sum()), n_spk=len(set(sp)),
                     auc_default_reference=a_ref, ci_default_reference=[b_ref["auc_lo"], b_ref["auc_hi"]],
                     auc_default_pod=a_def, ci_default_pod=[b_def["auc_lo"], b_def["auc_hi"]],
                     auc_lifted=a_lif, ci_lifted=[b_lif["auc_lo"], b_lif["auc_hi"]],
                     diff_lifted_minus_default=a_lif - a_ref, diff_ci=[b_pair["diff_lo"], b_pair["diff_hi"]],
                     diff_boot_mean=b_pair["diff_point"], usable_draws=[b_ref["usable"], b_lif["usable"], b_pair["usable"]])
    b_dr = boot(yy, sp, pdef_r[m]); b_lr = boot(yy, sp, plif_r[m]); b_pr = boot(yy, sp, plif_r[m], pdef_r[m])
    res[name]["rule_idset"] = dict(auc_default=auc_rank(yy, pdef_r[m]), ci_default=[b_dr["auc_lo"], b_dr["auc_hi"]],
                                   auc_lifted=auc_rank(yy, plif_r[m]), ci_lifted=[b_lr["auc_lo"], b_lr["auc_hi"]],
                                   diff=auc_rank(yy, plif_r[m]) - auc_rank(yy, pdef_r[m]), diff_ci=[b_pr["diff_lo"], b_pr["diff_hi"]])
    n, ns = int(m.sum()), len(set(sp))
    row(f"P22_auc_default_{name}", f"E-DAIC zero-shot AUC, default processor (300 s cut), {name}", a_ref, b_ref["auc_lo"], b_ref["auc_hi"], n, ns)
    row(f"P22_auc_lifted_{name}", f"E-DAIC zero-shot AUC, truncation=False (whole window), {name}", a_lif, b_lif["auc_lo"], b_lif["auc_hi"], n, ns)
    row(f"P22_auc_diff_{name}", f"paired AUC lifted minus default, {name}", a_lif - a_ref, b_pair["diff_lo"], b_pair["diff_hi"], n, ns)
    rr = res[name]["rule_idset"]
    row(f"P22_auc_lifted_{name}_ruleids", f"as P22_auc_lifted_{name}, rule Yes/No id set", rr["auc_lifted"], rr["ci_lifted"][0], rr["ci_lifted"][1], n, ns)
row("P22_short58_maxdiff", "max |p_yes lifted - default| on the 58 windows <= 300 s", checks["short58_max_abs_diff_p_yes_lifted_vs_default"], None, None, 58, 58)
row("P22_lift_tok_max", "max audio tokens with truncation=False (900 s window)", checks["lifted_tokens_max"], None, None, 275, 275)
row("P22_default_tok_max", "max audio tokens with the default processor", checks["default_tokens_max"], None, None, 275, 275)

out = dict(step="PART22 step3 rescore all 275 with truncation off", created_utc=datetime.datetime.utcnow().isoformat() + "Z",
           checks=checks, results=res,
           reference_published=dict(auc=0.8285, rebuild=0.8278, rebuild_ci=[0.7703, 0.8822]),
           sources={p: sha1mb(p) for p in [SRC, REFF, WINF, f"{SYNC}/p22_run.py", f"{BASE}/pod/analyze22.py", f"{BASE}/pod/stats22.py"]},
           n=275, n_speakers=275, one_clip_per_speaker=True, seed="numpy.random.default_rng(0), fresh per cell",
           bootstrap_draws=2000, bootstrap_unit="speaker, resampled with replacement; paired = one draw per replicate, both AUCs inside the draw",
           percentiles=[2.5, 97.5], auc_method="rank formula (mid-ranks for ties)",
           default_arm="reference per-clip p_yes_answer from edaic_full_perclip.csv (the 0.8278 file); pod re-score of the same default call is also reported and matches it",
           model_id="Qwen/Qwen2.5-Omni-7B", dtype="bfloat16",
           prompt_verbatim="Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.",
           p_yes="P(Yes)/(P(Yes)+P(No)) at the first answer position; Yes ids [7414, 9454, 9693, 9834, 14004], No ids [902, 2152, 2308, 2753, 8996]",
           default_call='proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)',
           lifted_call='proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)',
           exact_command=["pod: cd /workspace/p22 && HF_HOME=/workspace/hf python3 p22_run.py all3",
                          "mac: python3 scripts/part22/pod/analyze22.py"],
           libs=json.load(open(f"{SYNC}/lifted_all275.meta.json"))["libs"], libs_mac=dict(numpy=np.__version__, python=sys.version.split()[0]))
json.dump(out, open(f"{BASE}/edaic_lifted_auc.json", "w"), indent=1)
json.dump({k: out[k] for k in ["sources", "n", "n_speakers", "seed", "bootstrap_draws", "percentiles", "model_id", "prompt_verbatim",
                               "exact_command", "libs", "libs_mac", "default_call", "lifted_call", "p_yes"]}
          | dict(results=["scores/part22/edaic_lifted_auc.json", "scores/part22/edaic_lifted_perclip.csv"], dtype="bfloat16",
                 pod="RunPod 8flz9sdjujokel H200", audio_source="https://dcapswoz.ict.usc.edu/wwwedaic/data/<pid>_P.tar.gz, windows rebuilt with edaic_rerun/part16/EDAICFULL/build_windows.py"),
          open(f"{BASE}/edaic_lifted_auc.sidecar.json", "w"), indent=1)
# per-clip table for the paper folder
with open(f"{BASE}/edaic_lifted_perclip.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["pid", "label", "win_dur_s", "capped", "p_yes_default_ref", "p_yes_default_pod", "p_yes_lifted",
                                    "mass_default", "mass_lifted", "audio_tok_default", "audio_tok_lifted", "seq_len_lifted", "peak_gpu_gib_lifted", "sec_lifted"])
    for i, r in enumerate(R):
        w.writerow([pid[i], int(y[i]), float(dur[i]), int(cap[i]), repr(float(pref[i])), repr(float(pdef[i])), repr(float(plif[i])), repr(float(mdef[i])), repr(float(mlif[i])),
                    int(tdef[i]), int(tlif[i]), int(slif[i]), float(peak[i]), float(lsec[i])])
os.makedirs(f"{BASE}/rows", exist_ok=True)
json.dump(rows, open(f"{BASE}/pod/rows_step3.json", "w"), indent=1)
print("WROTE", f"{BASE}/edaic_lifted_auc.json")
