"""Compare the independent recompute with the track csvs; check durations; check val10 and 16 Sep p_yes agreement."""
import csv, os, glob, sys
import numpy as np, soundfile as sf
C = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(os.path.dirname(C), "af3")
indep = {(r["cell"], r["arm"]): r for r in csv.DictReader(open(f"{C}/indep_af3_auc.csv"))}
name = {"part10_original_first30s": "part10", "noinv_interviewer_free": "noinv", "noinv_minus_part10_paired": "noinv_minus_part10_paired",
        "fullseg_sep16": "full", "fullseg_minus_part10_paired": "full_minus_part10_paired"}
print("== AUC / interval comparison, 4 dp")
for f in ("af3_pitt_bootstrap.csv", "af3_pitt_fullseg_bootstrap.csv"):
    for r in csv.DictReader(open(f"{A}/{f}")):
        m = indep[(name[r["cell"]], r["arm"])]
        ok = all(f"{float(r[k]):.4f}" == m[k] for k in ("value", "ci_lo", "ci_hi"))
        print(f"{'YES' if ok else 'NO ':3s} {r['cell']:28s} {r['arm']:9s} track {float(r['value']):+.4f} [{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}]  indep {m['value']} [{m['ci_lo']}, {m['ci_hi']}]")
print("== processor length comparison")
t = {(r["clip"], r["mode"]): r for r in csv.DictReader(open(f"{A}/af3_pitt_three_clips_input_length.csv"))}
for r in csv.DictReader(open(f"{C}/indep_three_clips_length.csv")):
    o = t[(r["clip"], r["mode"])]
    pairs = [("samples", o["samples_passed_to_processor"], r["samples_to_feature_extractor"]),
             ("input_features", o["input_features_shape"], r["input_features"]),
             ("mask", o["mask_frames_per_window"], r["mask_frames"]),
             ("tokens", o["audio_tokens_in_input_ids"], r["sound_tokens"]),
             ("seconds", f"{float(o['seconds_reaching_model']):.2f}", f"{float(r['seconds_reaching_model']):.2f}")]
    ok = all(a == b for _, a, b in pairs)
    print(("YES" if ok else "NO "), r["clip"], r["mode"], " | ".join(f"{k} {a} vs {b}" for k, a, b in pairs))
print("== local clip durations")
D = "<local data dir>/paper1_local_runs/clips_local/clips_for_af3/pitt468"
d = np.array([sf.info(p).frames / sf.info(p).samplerate for p in sorted(glob.glob(f"{D}/*.wav"))])
print(f"n {len(d)} min {d.min():.3f} median {np.median(d):.3f} max {d.max():.3f} n_over_30s {(d > 30).sum()} srs {set(sf.info(p).samplerate for p in glob.glob(f'{D}/*.wav'))}")
ni = list(csv.DictReader(open("scores/part20/POD4c/work/af3_noinv468.csv")))
nd = np.array([float(r["dur_s"]) for r in ni]); sc = np.array([float(r["scored_s"]) for r in ni])
print(f"noinv clip dur_s min {nd.min():.3f} max {nd.max():.3f} | scored_s == dur_s for all: {bool(np.allclose(nd, sc))}")
print("== p_yes cross-checks")
p10 = {os.path.basename(r["clip_path"]): float(r["p_yes"]) for r in csv.DictReader(open("<local data dir>/release/overnight2/part10/af3_pitt.csv"))}
v10 = list(csv.DictReader(open("scores/part20/POD4c/work/af3_val10.csv")))
print("val10 columns", list(v10[0].keys()))
kcol = "orig_clip_id" if "orig_clip_id" in v10[0] else ("clip_id" if "clip_id" in v10[0] else "clip_path")
dv = [abs(float(r["p_yes"]) - p10[os.path.basename(r[kcol])]) for r in v10]
print(f"val10 n {len(dv)} max |p_yes - part10| {max(dv):.3g}; scored_s {sorted(set(r.get('scored_s','?') for r in v10))}")
fu = {os.path.basename(r["clip_path"]): float(r["p_yes"]) for r in csv.DictReader(open("<local data dir>/paper1_local_runs/runpod_results/pitt468_af3.csv"))}
df = np.array([abs(fu[c] - p10[c]) for c in sorted(p10)])
print(f"16 Sep full vs part10: agreement_Con_118-1_6475 |d| {abs(fu['agreement_Con_118-1_6475.wav']-p10['agreement_Con_118-1_6475.wav']):.2g}; median |d| all 468 {np.median(df):.4f}")
