#!/usr/bin/env python3
"""PART 12. Age and sex matched control on Pitt, Qwen2.5-Omni (and Qwen2-Audio).

Step 1 hunts for the SAVED matched list before rebuilding anything.
Step 2 restricts the 468-clip Pitt scoring set to that list and rescores.
"""
import os, json, csv, glob, subprocess, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R   = "/Volumes/G-Drive Pro/paper1_local_runs"
DB  = "/Volumes/G-Drive Pro/DementiaBank"
OUT = os.path.expanduser("~/Desktop/release/omni")
os.makedirs(OUT, exist_ok=True)

MATCHED_LIST = os.path.join(DB, "pitt_agematched_manifest.csv")
RULE_SCRIPT  = os.path.join(DB, "age_match_pitt.py")
CLIP_MANIFEST = os.path.join(DB, "pitt_conflict_manifest.csv")   # the 468 clips, carries age/sex/dx

SRC = {
  ("omni",  "nested_encoder_probe"): (os.path.join(R, "omni_final/omni_pitt_enc_nested_oof.csv"), "p_probe"),
  ("omni",  "zero_shot_answer"):     (os.path.join(R, "omni_final/omni_pitt_zeroshot_scores.csv"), "p_yes"),
  ("q2a",   "nested_encoder_probe"): (os.path.join(R, "probe2/pitt_enc_nested_oof.csv"), "p_probe"),
  ("q2a",   "zero_shot_answer"):     (os.path.join(R, "probe2/pitt_zeroshot_scores.csv"), "p_yes"),
}
MODEL_NAME = {"omni": "Qwen2.5-Omni-7B", "q2a": "Qwen2-Audio-7B-Instruct"}
N_BOOT, SEED = 2000, 0


def f4(x):
    return "NaN" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def boot_ci(labels, scores, n_boot=N_BOOT, seed=SEED):
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


print("=" * 110)
print("PART 12. Age and sex matched control on Pitt")
print("=" * 110)

# ---------------------------------------------------------------- 1. provenance hunt
print("\n--- 1. HUNT FOR THE SAVED MATCHED LIST (before rebuilding anything) ---")
hunt_dirs = [R, os.path.expanduser("~/Desktop/paper1_final"), os.path.expanduser("~/Desktop/af2_run"), DB]
found = []
for d in hunt_dirs:
    if not os.path.isdir(d):
        print(f"  {d:<55} dir not present")
        continue
    hits = []
    for pat in ("*agematch*", "*age_match*", "*matched*"):
        for depth in ("", "*/", "*/*/"):
            hits += glob.glob(os.path.join(d, depth + pat))
    hits = sorted(set(h for h in hits if os.path.isfile(h)))
    print(f"  {d:<55} {len(hits)} candidate file(s)")
    for h in hits:
        print(f"      {h}")
    found += hits

have_list = os.path.isfile(MATCHED_LIST)
print(f"\n  SAVED LIST USED : {MATCHED_LIST}   {'FOUND' if have_list else 'NOT AVAILABLE'}")
print(f"  RULE SCRIPT     : {RULE_SCRIPT}   {'FOUND' if os.path.isfile(RULE_SCRIPT) else 'NOT AVAILABLE'}")
if not have_list:
    raise SystemExit("NOT AVAILABLE: saved matched list absent and no rebuild source; stopping Part 12.")

am = pd.read_csv(MATCHED_LIST, dtype=str)
am["speaker_id"] = am["speaker_id"].str.zfill(3)
n_segments_list = len(am)
spk_pairs = am.drop_duplicates(["group", "speaker_id"])
n_spk_list = len(spk_pairs)
n_ctrl = int((spk_pairs["group"] == "Control").sum())
n_ad   = int((spk_pairs["group"] == "Dementia").sum())
n_pairs_list = min(n_ctrl, n_ad)
print(f"\n  what the saved file ACTUALLY says:")
print(f"      rows (recordings/segments) : {n_segments_list}")
print(f"      unique speakers            : {n_spk_list}  (Control {n_ctrl}, Dementia {n_ad})  -> {n_pairs_list} pairs")
_tasks = sorted(am["task"].unique()); _dx = {k: int(v) for k, v in am["diagnosis"].value_counts().items()}
print(f"      task                       : {_tasks}")
print(f"      diagnosis                  : {_dx}")
print(f"  earlier description was '64 pairs / 268 segments' -> the file does NOT say that; it says "
      f"{n_pairs_list} pairs / {n_segments_list} segments.")

# re-run the rule script to confirm the saved list is reproducible
rule_text = open(RULE_SCRIPT).read() if os.path.isfile(RULE_SCRIPT) else ""
rule_out = ""
try:
    rule_out = subprocess.run(["python3", os.path.basename(RULE_SCRIPT)], cwd=DB,
                              capture_output=True, text=True, timeout=300).stdout.strip()
    print("\n  re-running the rule script to confirm reproducibility:")
    for line in rule_out.splitlines():
        print("      " + line)
except Exception as e:
    print(f"  rule script re-run failed: {e}")

RULE_TEXT = ("Cookie task only. Dementia arm restricted to diagnosis ProbableAD or PossibleAD. "
             "One row per speaker (first cookie recording). Greedy nearest-age 1:1 matching within sex, "
             "caliper 3 years, each control used at most once. All cookie recordings of every kept speaker "
             "then form the matched manifest.")
print(f"\n  RULE (from {RULE_SCRIPT}): {RULE_TEXT}")

# ---------------------------------------------------------------- 2. restrict the 468 clips
print("\n--- 2. RESTRICT THE 468 SCORED CLIPS TO THE MATCHED LIST ---")
cm = pd.read_csv(CLIP_MANIFEST, dtype={"spk": str, "session": str})
cm["clip"] = cm["segment_path"].map(os.path.basename)
cm["spk"] = cm["spk"].str.zfill(3)
cm["sess_no"] = cm["session"].str.split("-").str[1].astype(int).astype(str)
cm["arm"] = cm["set"]

matched_spk = set(zip(spk_pairs["group"], spk_pairs["speaker_id"]))
matched_rec = set(zip(am["group"], am["speaker_id"], am["session"].astype(int).astype(str)))
cm["in_matched_speaker"]   = [ (g, s) in matched_spk for g, s in zip(cm["grp"], cm["spk"]) ]
cm["in_matched_recording"] = [ (g, s, n) in matched_rec for g, s, n in zip(cm["grp"], cm["spk"], cm["sess_no"]) ]

agree = bool((cm["in_matched_speaker"] == cm["in_matched_recording"]).all())
_n_by_spk = int(cm["in_matched_speaker"].sum()); _n_by_rec = int(cm["in_matched_recording"].sum())
print(f"  clips in 468 whose SPEAKER is in the matched list   : {_n_by_spk}")
print(f"  clips in 468 whose RECORDING is in the matched list : {_n_by_rec}")
print(f"  the two restrictions agree clip for clip            : {agree}")
sub = cm[cm["in_matched_recording"]].copy()
n_clips_sub = len(sub)
sub_spk = sub.drop_duplicates(["grp", "spk"])
n_spk_sub = len(sub_spk)
n_ctrl_sub = int((sub_spk["grp"] == "Control").sum())
n_ad_sub   = int((sub_spk["grp"] == "Dementia").sum())
print(f"\n  MATCHED SUBSET ACTUALLY SCORED: {n_clips_sub} clips, {n_spk_sub} speakers "
      f"(Control {n_ctrl_sub}, Dementia {n_ad_sub})")
_by_arm = {k: int(v) for k, v in sub["arm"].value_counts().items()}
_by_lab = {str(k): int(v) for k, v in sub["label"].value_counts().items()}
print(f"      by arm  : {_by_arm}")
print(f"      by label: {_by_lab}")
print(f"  NOTE only {n_spk_sub} of the {n_spk_list} matched speakers appear in the 468-clip scoring set, "
      f"and Control/Dementia counts are {n_ctrl_sub}/{n_ad_sub}, so the 1:1 pairing does NOT survive the "
      f"intersection. Intact pairs after restriction are not claimed.")

# age balance actually achieved in the scored subset
spk_age = sub.drop_duplicates(["grp", "spk"])[["grp", "spk", "age", "sex", "label"]]
for g in ("Control", "Dementia"):
    a = spk_age[spk_age["grp"] == g]["age"].astype(float)
    sx = {k: int(v) for k, v in spk_age[spk_age["grp"] == g]["sex"].value_counts().items()}
    print(f"      {g:<9} n_spk {len(a):3d}  mean age {a.mean():.4f}  sd {a.std(ddof=1):.4f}  sex {sx}")
age_gap = (spk_age[spk_age['grp'] == 'Dementia']['age'].astype(float).mean()
           - spk_age[spk_age['grp'] == 'Control']['age'].astype(float).mean())
print(f"      speaker-level mean age gap (AD minus control) in the SCORED subset: {age_gap:+.4f} years")

keep_clips = set(sub["clip"])

# ---------------------------------------------------------------- 3. score
print("\n--- 3. AUC ON THE MATCHED SUBSET ---")
rows = []

def add(model, measure, level, n, auc, lo, hi, path, scope):
    rows.append({"model": MODEL_NAME.get(model, model), "measure": measure, "scope": scope,
                 "level": level, "n": n, "auc": auc, "ci_lo": lo, "ci_hi": hi, "file": path})

for (model, measure), (path, col) in SRC.items():
    if not os.path.isfile(path):
        add(model, measure, "clip", 0, float("nan"), float("nan"), float("nan"), path + "  NOT AVAILABLE", "matched")
        print(f"  NOT AVAILABLE: {path}")
        continue
    df = pd.read_csv(path)
    df["clip"] = df["clip"].astype(str) if "clip" in df.columns else df["path"].map(os.path.basename)
    full = df.copy()
    m = df[df["clip"].isin(keep_clips)].copy()
    if len(m) != n_clips_sub:
        print(f"  WARNING {os.path.basename(path)}: matched {len(m)} of {n_clips_sub} expected clips")
    for scope, d in (("full468", full), ("matched", m)):
        auc_c = roc_auc_score(d["label"], d[col])
        add(model, measure, "clip", len(d), auc_c, float("nan"), float("nan"), path, scope)
        g = d.groupby("speaker")
        sl, ss = g["label"].first().astype(int).values, g[col].mean().values
        auc_s = roc_auc_score(sl, ss)
        lo, hi, _ = boot_ci(sl, ss)
        add(model, measure, "speaker", len(sl), auc_s, lo, hi, path, scope)

# age alone, model independent
age_clip = sub[["clip", "label", "age"]].copy()
auc_age_clip = roc_auc_score(age_clip["label"], age_clip["age"].astype(float))
add("age", "age_alone", "clip", len(age_clip), auc_age_clip, float("nan"), float("nan"), CLIP_MANIFEST, "matched")
sl = spk_age["label"].astype(int).values
ss = spk_age["age"].astype(float).values
auc_age_spk = roc_auc_score(sl, ss)
lo, hi, _ = boot_ci(sl, ss)
add("age", "age_alone", "speaker", len(sl), auc_age_spk, lo, hi, CLIP_MANIFEST, "matched")
# age alone on the unrestricted 468 for contrast
all_spk = cm.drop_duplicates(["grp", "spk"])[["label", "age"]]
add("age", "age_alone", "clip", len(cm), roc_auc_score(cm["label"], cm["age"].astype(float)),
    float("nan"), float("nan"), CLIP_MANIFEST, "full468")
lo2, hi2, _ = boot_ci(all_spk["label"].astype(int).values, all_spk["age"].astype(float).values)
add("age", "age_alone", "speaker", len(all_spk),
    roc_auc_score(all_spk["label"].astype(int), all_spk["age"].astype(float)), lo2, hi2, CLIP_MANIFEST, "full468")

res = pd.DataFrame(rows)

order = [("Qwen2.5-Omni-7B", "nested_encoder_probe"), ("Qwen2.5-Omni-7B", "zero_shot_answer"),
         ("age", "age_alone"),
         ("Qwen2-Audio-7B-Instruct", "nested_encoder_probe"), ("Qwen2-Audio-7B-Instruct", "zero_shot_answer")]
hdr = (f"  {'model':<26} {'measure':<22} {'lvl':<8} {'n full':>7} {'AUC full468':>12} "
       f"{'n match':>8} {'AUC matched':>12} {'95% CI matched':>22}")
print(hdr)
for model, measure in order:
    for lvl in ("clip", "speaker"):
        f_ = res[(res["model"] == model) & (res["measure"] == measure) & (res["scope"] == "full468") & (res["level"] == lvl)]
        m_ = res[(res["model"] == model) & (res["measure"] == measure) & (res["scope"] == "matched") & (res["level"] == lvl)]
        if not len(m_):
            continue
        m_ = m_.iloc[0]
        fn, fa = (int(f_.iloc[0]["n"]), f4(f_.iloc[0]["auc"])) if len(f_) else ("-", "-")
        _lo, _hi = f4(m_["ci_lo"]), f4(m_["ci_hi"])
        ci = ("[" + _lo + ", " + _hi + "]") if not np.isnan(m_["ci_lo"]) else ""
        _mn, _ma = int(m_["n"]), f4(m_["auc"])
        print(f"  {model:<26} {measure:<22} {lvl:<8} {fn:>7} {fa:>12} {_mn:>8} {_ma:>12} {ci:>22}")
print("\n  (age alone: AUC of raw age against the label; 0.5 = age carries nothing)")

# ---------------------------------------------------------------- 4. write
csv_path = os.path.join(OUT, "pitt_agematched.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "measure", "scope", "level", "n", "auc", "ci_lo", "ci_hi", "file"])
    for _, r in res.iterrows():
        w.writerow([r.model, r.measure, r.scope, r.level, int(r.n), f4(r.auc), f4(r.ci_lo), f4(r.ci_hi), r.file])

meta = {
  "part": "12. Age and sex matched control on Pitt, Qwen2.5-Omni (Qwen2-Audio alongside)",
  "date": datetime.date.today().isoformat(),
  "matched_list": {
    "status": "FOUND, saved list reused, nothing rebuilt",
    "path": MATCHED_LIST,
    "rule_script": RULE_SCRIPT,
    "rule": RULE_TEXT,
    "rule_script_rerun_stdout": rule_out,
    "built_from": os.path.join(DB, "pitt_manifest.csv"),
    "n_pairs": n_pairs_list,
    "n_segments": n_segments_list,
    "n_speakers": n_spk_list,
    "n_speakers_control": n_ctrl, "n_speakers_dementia": n_ad,
    "earlier_description_64_pairs_268_segments": (
        f"NOT what the file says. The saved file holds {n_pairs_list} pairs / {n_segments_list} segments "
        f"/ {n_spk_list} speakers, and re-running the rule script reproduces exactly that."),
  },
  "scored_subset": {
    "clip_manifest_used_for_the_join": CLIP_MANIFEST,
    "join": "clip basename of segment_path, 468 of 468 clips join to the Omni score files",
    "restriction": "clips whose (group, speaker_id, session) is in the matched list",
    "speaker_and_recording_restrictions_agree": agree,
    "n_clips": n_clips_sub, "n_speakers": n_spk_sub,
    "n_speakers_control": n_ctrl_sub, "n_speakers_dementia": n_ad_sub,
    "n_clips_by_arm": _by_arm,
    "n_clips_by_label": _by_lab,
    "caveat": (f"only {n_spk_sub} of the {n_spk_list} matched speakers are present in the 468-clip scoring set "
               f"and the surviving Control/Dementia counts are {n_ctrl_sub}/{n_ad_sub}, so 1:1 pairing does not "
               f"survive the intersection; no pair count is claimed for the scored subset"),
    "speaker_level_mean_age_gap_AD_minus_control_years": round(float(age_gap), 4),
  },
  "sources": {f"{MODEL_NAME.get(k[0], k[0])}|{k[1]}": v[0] for k, v in SRC.items()},
  "age_alone_source": CLIP_MANIFEST,
  "bootstrap": {"draws": N_BOOT, "seed": SEED, "resampled_unit": "speaker",
                "method": "percentile 2.5 / 97.5", "applies_to": "speaker level rows only"},
  "outputs": {"csv": csv_path, "json": os.path.join(OUT, "pitt_agematched.json")},
}
with open(meta["outputs"]["json"], "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"\nwrote {csv_path}")
print(f"wrote {meta['outputs']['json']}")
