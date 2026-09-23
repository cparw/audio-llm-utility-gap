#!/usr/bin/env python3
"""PART 11. Speaker level AUC on Pitt, Qwen2.5-Omni.

Three measures, each scored at clip level and at speaker level.
Speaker level rule: average the per-clip score WITHIN each speaker, then AUC over speakers.
Arm = clip name prefix (conflict_ / agreement_).
Speaker bootstrap: 2000 draws, seed 0, resample SPEAKERS with replacement.
"""
import os, json, csv, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = "/Volumes/G-Drive Pro/paper1_local_runs"
OUT = os.path.expanduser("~/Desktop/release/omni")
os.makedirs(OUT, exist_ok=True)

SRC = {
    "zero_shot_answer":      os.path.join(R, "omni_final/omni_pitt_zeroshot_scores.csv"),
    "nested_encoder_probe":  os.path.join(R, "omni_final/omni_pitt_enc_nested_oof.csv"),
    "fine_tuned_answer":     os.path.join(R, "omni_final/omnisft_pitt_oof.csv"),
}
SCORE_COL = {"zero_shot_answer": "p_yes", "nested_encoder_probe": "p_probe", "fine_tuned_answer": "p_yes"}

N_BOOT, SEED = 2000, 0


def load(measure):
    """Return DataFrame clip, speaker, label, score, arm."""
    path = SRC[measure]
    df = pd.read_csv(path)
    if "clip" in df.columns:
        clip = df["clip"].astype(str)
    else:                                   # SFT file stores a full path
        clip = df["path"].astype(str).map(os.path.basename)
    out = pd.DataFrame({
        "clip":    clip,
        "speaker": df["speaker"].astype(str),
        "label":   df["label"].astype(int),
        "score":   df[SCORE_COL[measure]].astype(float),
    })
    out["arm"] = np.where(out["clip"].str.startswith("conflict_"), "conflict",
                  np.where(out["clip"].str.startswith("agreement_"), "agreement", "UNKNOWN"))
    return out


def speaker_table(df):
    """Average per-clip score within speaker; carry the speaker label."""
    g = df.groupby("speaker", sort=True)
    return pd.DataFrame({
        "label": g["label"].first().astype(int),
        "score": g["score"].mean(),
        "n_clips": g.size(),
    }).reset_index()


def boot_ci(labels, scores, n_boot=N_BOOT, seed=SEED):
    """Percentile 95% CI, resampling the UNITS given (speakers) with replacement."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue                        # degenerate draw, one class only
        vals.append(roc_auc_score(yb, s[idx]))
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def f4(x):
    return "NaN" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


rows, sanity_problems, notes = [], [], []

# ---------- sanity: one label per speaker, and the three files agree ----------
frames = {m: load(m) for m in SRC}
for m, df in frames.items():
    bad = df.groupby("speaker")["label"].nunique()
    bad = bad[bad != 1]
    if len(bad):
        sanity_problems.append({"measure": m, "speakers_with_multiple_labels": sorted(bad.index.tolist())})
    unk = int((df["arm"] == "UNKNOWN").sum())
    if unk:
        sanity_problems.append({"measure": m, "clips_with_unrecognised_arm_prefix": unk})

base = frames["zero_shot_answer"][["clip", "speaker", "label"]].sort_values("clip").reset_index(drop=True)
for m in ("nested_encoder_probe", "fine_tuned_answer"):
    other = frames[m][["clip", "speaker", "label"]].sort_values("clip").reset_index(drop=True)
    if not base.equals(other):
        sanity_problems.append({"measure": m, "issue": "clip/speaker/label set differs from zero_shot_answer file"})

print("=" * 108)
print("PART 11. Speaker level AUC on Pitt, Qwen2.5-Omni")
print("=" * 108)
for m, p in SRC.items():
    d = frames[m]
    print(f"  {m:<22} n_clips={len(d):4d}  n_speakers={d['speaker'].nunique():4d}  score_col={SCORE_COL[m]:<7}  {p}")
print()
print("SANITY  every speaker must carry exactly one label:",
      "PASS (all speakers single-labelled in all three files)" if not sanity_problems else f"FAIL {sanity_problems}")
nclip_per_spk = frames["zero_shot_answer"].groupby("speaker").size()
print(f"        clips per speaker: min {nclip_per_spk.min()}, max {nclip_per_spk.max()}, "
      f"total clips {int(nclip_per_spk.sum())}, speakers {len(nclip_per_spk)}")
print()

# ---------- overall + per arm ----------
for m in SRC:
    df = frames[m]
    for arm in ("ALL", "conflict", "agreement"):
        sub = df if arm == "ALL" else df[df["arm"] == arm]
        # clip level
        auc_c = roc_auc_score(sub["label"], sub["score"])
        rows.append({"measure": m, "level": "clip", "arm": arm, "n": len(sub), "auc": auc_c,
                     "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot_used": "",
                     "file": SRC[m]})
        # speaker level
        st = speaker_table(sub)
        auc_s = roc_auc_score(st["label"], st["score"])
        lo, hi, used = boot_ci(st["label"].values, st["score"].values)
        rows.append({"measure": m, "level": "speaker", "arm": arm, "n": len(st), "auc": auc_s,
                     "ci_lo": lo, "ci_hi": hi, "n_boot_used": used,
                     "file": SRC[m]})

res = pd.DataFrame(rows)

def block(arm, title):
    print(title)
    print(f"  {'measure':<22} {'n_clips':>8} {'AUC clip':>10} {'n_spk':>6} {'AUC speaker':>12} {'95% speaker bootstrap':>26}")
    for m in SRC:
        c = res[(res.measure == m) & (res.arm == arm) & (res.level == "clip")].iloc[0]
        s = res[(res.measure == m) & (res.arm == arm) & (res.level == "speaker")].iloc[0]
        ci = f"[{f4(s.ci_lo)}, {f4(s.ci_hi)}]"
        print(f"  {m:<22} {int(c.n):>8} {f4(c.auc):>10} {int(s.n):>6} {f4(s.auc):>12} {ci:>26}")
    print()

block("ALL",       "OVERALL  (468 clips / 228 speakers)")
block("conflict",  "ARM conflict")
block("agreement", "ARM agreement")

# ---------- write ----------
csv_path = os.path.join(OUT, "pitt_speaker_level.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["measure", "level", "n", "auc", "file", "arm", "ci_lo", "ci_hi", "n_boot_used"])
    for _, r in res.iterrows():
        w.writerow([r.measure, r.level, int(r.n), f4(r.auc), r.file, r.arm,
                    f4(r.ci_lo), f4(r.ci_hi), r.n_boot_used])

zs = frames["zero_shot_answer"]
meta = {
    "part": "11. Speaker level AUC on Pitt, Qwen2.5-Omni",
    "date": datetime.date.today().isoformat(),
    "source_paths": {
        "zero_shot_answer":     SRC["zero_shot_answer"],
        "nested_encoder_probe": SRC["nested_encoder_probe"],
        "fine_tuned_answer":    SRC["fine_tuned_answer"],
    },
    "score_columns": SCORE_COL,
    "averaging_rule": ("Speaker level: the per-clip score is averaged with an unweighted arithmetic mean "
                       "WITHIN each speaker (so a speaker contributes one score regardless of how many clips "
                       "it has, 1 to 9 here), the speaker inherits its single clip label, and AUC is computed "
                       "over those speaker-level scores. Clip level: AUC over all clips directly."),
    "arm_rule": "arm = clip filename prefix, conflict_ or agreement_",
    "n_clips": int(len(zs)),
    "n_speakers": int(zs["speaker"].nunique()),
    "n_clips_by_arm": {a: int((zs["arm"] == a).sum()) for a in ("conflict", "agreement")},
    "n_speakers_by_arm": {a: int(zs[zs["arm"] == a]["speaker"].nunique()) for a in ("conflict", "agreement")},
    "clips_per_speaker": {"min": int(nclip_per_spk.min()), "max": int(nclip_per_spk.max())},
    "bootstrap": {"draws": N_BOOT, "seed": SEED, "resampled_unit": "speaker",
                  "method": "percentile 2.5 / 97.5, draws with a single class present are discarded",
                  "applies_to": "speaker level rows only"},
    "sanity_one_label_per_speaker": "PASS" if not sanity_problems else "FAIL",
    "sanity_problems": sanity_problems,
    "outputs": {"csv": csv_path, "json": os.path.join(OUT, "pitt_speaker_level.json")},
}
with open(meta["outputs"]["json"], "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"wrote {csv_path}")
print(f"wrote {meta['outputs']['json']}")
