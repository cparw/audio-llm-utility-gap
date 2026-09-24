"""PART 25 track C (r2): assemble C_other_models.tsv, a summary and a sidecar from r2/C_r2_cells.json (master rows) and
r2/kimi_enc/C_r2_kimi_enc_cells.json (PART 26 Kimi-Audio encoder supplement, not master rows).
Independent checks used per row:
  - master rows reproduced by C_r2_recheck.py from the per-clip files (sklearn AUC plus an exact pairwise AUC);
  - the r2 bootstrap draws equal the first attempt's draws (C_build.py, other code, same rule) for every cell that the
    first attempt could compute with the mean of five; for Qwen3-Omni E-DAIC, the r2 draws are compared with the PART 26
    run's own draws and interval (part26/Qwen3-Omni-30B-A3B_E-DAIC).
The first attempt's TSV is kept as C_other_models.first_attempt_0049Z.tsv (not deleted).
usage: /usr/local/bin/python3 C_r2_assemble.py"""
import os, json, shutil, datetime, hashlib
import numpy as np, pandas as pd
C = "<local data dir>/release_from_mac/scores/part25/C"
R2 = f"{C}/r2"
P26Q = "<local data dir>/release_from_mac/scores/part26/Qwen3-Omni-30B-A3B_E-DAIC"
cells = json.load(open(f"{R2}/C_r2_cells.json"))
kim = json.load(open(f"{R2}/kimi_enc/C_r2_kimi_enc_cells.json")) if os.path.exists(f"{R2}/kimi_enc/C_r2_kimi_enc_cells.json") else []
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def f4(x): return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.4f}" if False else f"{x:.4f}"
def s4(x): return f"{x:+.4f}"
rows = []
for c in cells:
    ind = []
    if c.get("first_attempt_draws") and "singlesplit" not in c["first_attempt_draws"]:
        ind.append(f"first attempt draws (C_build.py) max abs diff {c['first_attempt_draws_maxabs']:.1e}")
    if c["model"].startswith("Qwen3") and c["ds_key"] == "edaic":
        D26 = np.load(f"{P26Q}/Q3O_EDAIC_draws.npz"); S26 = json.load(open(f"{P26Q}/Q3O_EDAIC_perclip.sidecar.json"))
        k = [k for k in D26.files if "gap" in k or "diff" in k]
        mine = np.load(c["draws_file"])["diff"]
        mx = float(np.nanmax(np.abs(D26[k[0]] - mine))) if k and D26[k[0]].shape == mine.shape else None
        ind.append(f"PART 26 run interval [{S26['gap_ci'][0]:+.4f}, {S26['gap_ci'][1]:+.4f}], PART 26 draws key '{k[0] if k else None}' max abs diff {mx}")
        c["part26_draws_maxabs"] = mx; c["part26_gap_ci"] = S26["gap_ci"]
    is_kimi = c["model"] == "Kimi-Audio"
    rows.append(dict(
        model=c["model"], dataset=c["dataset"], condition=c["condition"], source_kind="master_lookup rows 17..178",
        probe_stream=c["probe_stream"],
        encoder_probe=("none in master_lookup" if is_kimi else f4(c["master_probe"])),
        probe_used=f4(c["master_probe"]), probe_master_row=c["master_row_probe"],
        probe_recomputed=f4(c["probe_recomputed"]), per_repeat_recomputed_4dp=json.dumps(c["per_repeat_recomputed_4dp"]),
        per_repeat_original_4dp=json.dumps(c.get("original_per_repeat")),
        probe_status=("verified" if c["probe_verified"] else "NOT verified"),
        zero_shot_answer=f4(c["master_answer"]), answer_master_row=c["master_row_answer"], answer_recomputed=f4(c["answer_auc_sklearn"]),
        answer_status=("verified" if c["answer_verified"] else "NOT verified"),
        row_status=("verified" if (c["probe_verified"] and c["answer_verified"]) else "NOT verified"),
        diff_probe_minus_answer=s4(c["diff_master"]), diff_from_perclip=s4(c["diff_point"]),
        ci_lo=s4(c["diff_lo"]), ci_hi=s4(c["diff_hi"]), interval_excludes_0=("yes" if c["excludes_zero"] else "no"),
        probe_above_answer=("yes" if c["diff_master"] > 0 else "no"),
        interval_kind=c["interval_kind"], boot_usable=c["boot_usable"], n=c["n_joined"], n_speakers=c["n_spk"], n_pos=c["n_pos"],
        speaker_ids=c["speaker_ids_used"] + ("" if c["speaker_ids_identical"] else
            (f"; answer-file ids give [{c['alt_answer_speaker_ids_lo']:+.4f}, {c['alt_answer_speaker_ids_hi']:+.4f}]" if "alt_answer_speaker_ids_lo" in c else "; " + c.get("answer_file_speaker_note", ""))),
        independent_check="; ".join(ind),
        probe_perclip_source=c["probe_file"], probe_origin=c["probe_origin"], answer_perclip_source=c["answer_file"],
        perclip_file=c["perclip_file"], sidecar_file=c["sidecar_file"], draws_file=c["draws_file"]))
for k in kim:
    if k["status"] != "landed":
        rows.append(dict(model="Kimi-Audio", dataset=k["dataset"], condition=k["condition"], source_kind="PART 26 supplement, NOT master_lookup",
                         probe_stream="encoder probe (PART 26 new Whisper-encoder extraction)", encoder_probe="NOT LANDED", probe_status=k["status"]))
        continue
    rows.append(dict(
        model="Kimi-Audio", dataset=k["dataset"], condition=k["condition"], source_kind="PART 26 supplement, NOT master_lookup",
        probe_stream="encoder probe (PART 26 new Whisper-encoder extraction; Method currently says the Kimi encoder side is skipped)",
        encoder_probe=f4(k["probe_mean5"]), probe_used=f4(k["probe_mean5"]), probe_master_row="",
        probe_recomputed=f4(k["probe_mean5"]), per_repeat_recomputed_4dp=json.dumps(k["per_repeat_recomputed_4dp"]),
        per_repeat_original_4dp=json.dumps([L for _, L in k["part26_per_repeat_lists"]][:1]),
        probe_status=("recomputed; equals PART 26 at 4 dp" if k["per_repeat_equal_part26_4dp"] else "recomputed; DIFFERS from PART 26"),
        zero_shot_answer=f4(k["answer_master"]), answer_master_row=k["answer_row"], answer_recomputed=f4(k["answer_recomputed"]),
        answer_status=("verified" if k["answer_verified"] else "NOT verified"),
        row_status="not a master row (new PART 26 value, recomputed)",
        diff_probe_minus_answer=s4(k["probe_mean5"] - k["answer_master"]), diff_from_perclip=s4(k["diff_point"]),
        ci_lo=s4(k["diff_lo"]), ci_hi=s4(k["diff_hi"]), interval_excludes_0=("yes" if k["excludes_zero"] else "no"),
        probe_above_answer=("yes" if k["diff_point"] > 0 else "no"),
        interval_kind="paired speaker bootstrap, mean of the five per-repeat AUCs minus the answer AUC in each draw",
        boot_usable=k["boot_usable"], n=k["n"], n_speakers=k["n_spk"], n_pos=k["n_pos"], speaker_ids="probe file speaker column",
        independent_check=f"PART 26 interval(s) {k['part26_gap_ci']}; agrees at 4 dp: {k['part26_interval_agrees_4dp']}",
        probe_perclip_source=k["part26_perclip"], probe_origin="PART 26", answer_perclip_source=k["answer_file"],
        perclip_file=k["perclip_file"], sidecar_file=k["perclip_file"].replace("/perclip/", "/sidecars/").replace("_perclip.csv", ".sidecar.json"),
        draws_file=k["draws_file"]))
T = pd.DataFrame(rows)
for col in ("probe_master_row", "answer_master_row", "boot_usable", "n", "n_speakers", "n_pos"):
    T[col] = [("" if (v is None or v == "" or (isinstance(v, float) and np.isnan(v))) else str(int(v))) for v in T[col]]
T = T.fillna("")
out = f"{C}/C_other_models.tsv"
if os.path.exists(out) and not os.path.exists(f"{C}/C_other_models.first_attempt_0049Z.tsv"):
    shutil.copy2(out, f"{C}/C_other_models.first_attempt_0049Z.tsv")
for extra in ("C_other_models_summary.txt", "C_other_models.sidecar.json"):
    b = f"{C}/{extra}".replace("C_other_models", "C_other_models.first_attempt_0049Z", 1)
    if os.path.exists(f"{C}/{extra}") and not os.path.exists(b): shutil.copy2(f"{C}/{extra}", b)
T.to_csv(out, sep="\t", index=False)

# ---------------- summary ----------------
M = [c for c in cells]
PDAD = ["PC-GITA", "NeuroVoz", "MDVR-KCL", "Pitt", "ADReSSo", "ADReSS-2020"]
def count(sel):
    above = [c for c in sel if c["diff_master"] > 0]; exc = [c for c in above if c["diff_lo"] > 0]
    return len(sel), len(above), len(exc)
enc12 = [c for c in M if c["model"] != "Kimi-Audio" and c["dataset"] in PDAD]
kimi_lm6 = [c for c in M if c["model"] == "Kimi-Audio" and c["dataset"] in PDAD]
L = []
L.append(f"PART 25 track C (second attempt, r2), written {now} from {out}")
L.append("probe = master_lookup encoder probe (mean of five per-repeat nested AUCs); answer = master_lookup zero-shot answer AUC; diff = probe minus answer.")
L.append("interval = paired speaker bootstrap, 2000 draws, fresh default_rng(0) per cell, mean of the five per-repeat AUCs per draw, 2.5/97.5 percentiles.")
L.append("")
for c in M:
    L.append(f"{c['model']:18s} {c['dataset']:11s} {('encoder' if c['model']!='Kimi-Audio' else 'LM (no encoder row)'):19s} probe {c['master_probe']:.4f} "
             f"({'verified' if c['probe_verified'] else 'NOT verified'})  answer {c['master_answer']:.4f} ({'verified' if c['answer_verified'] else 'NOT verified'})  "
             f"diff {c['diff_master']:+.4f} [{c['diff_lo']:+.4f}, {c['diff_hi']:+.4f}]")
landed = [k for k in kim if k["status"] == "landed"]
if kim:
    L.append("")
    L.append("Kimi-Audio encoder probe, PART 26 new extraction (NOT master_lookup; the Method says the Kimi encoder side is skipped; the authors decide):")
    for k in kim:
        if k["status"] == "landed":
            L.append(f"  Kimi-Audio {k['dataset']:11s} encoder {k['probe_mean5']:.4f}  answer {k['answer_master']:.4f}  diff {k['diff_point']:+.4f} [{k['diff_lo']:+.4f}, {k['diff_hi']:+.4f}]"
                     f"  (PART 26 per-repeat reproduced: {k['per_repeat_equal_part26_4dp']}; PART 26 interval agrees: {k['part26_interval_agrees_4dp']})")
        else:
            L.append(f"  Kimi-Audio {k['dataset']:11s} {k['status']}")
L.append("")
n, a, e = count(enc12)
L.append(f"1. Encoder probe, Qwen2-Audio and Qwen3-Omni, 12 PD/AD cells: probe above answer in {a} of {n}; interval excludes 0 in {e} of those {a}.")
n2, a2, e2 = count(kimi_lm6)
L.append(f"   Kimi-Audio has no encoder probe in master_lookup. With its LM probe in place: probe above answer in {a2} of {n2}; interval excludes 0 in {e2} of those {a2}.")
L.append(f"   All 18 PD/AD cells (encoder for the Qwen models, LM probe for Kimi-Audio): probe above answer in {a+a2} of 18; interval excludes 0 in {e+e2} of those {a+a2}.")
kl = [k for k in landed if k["dataset"] in PDAD]
if kl:
    ka = [k for k in kl if k["diff_point"] > 0]; ke = [k for k in ka if k["diff_lo"] > 0]
    L.append(f"   Kimi-Audio PART 26 encoder probe (not master): landed for {len(kl)} of 6 PD/AD cells; probe above answer in {len(ka)} of {len(kl)}; interval excludes 0 in {len(ke)} of those {len(ka)}.")
    if len(kl) == 6:
        L.append(f"   All 18 PD/AD cells with the PART 26 Kimi encoder in place of the Kimi LM probe: probe above answer in {a+len(ka)} of 18; interval excludes 0 in {e+len(ke)} of those {a+len(ka)}.")
L.append("")
L.append("2. E-DAIC (MDD):")
for c in [c for c in M if c["dataset"] == "E-DAIC"]:
    side = "answer above probe" if c["diff_master"] < 0 else "probe above answer"
    L.append(f"   {c['model']}: {'encoder' if c['model']!='Kimi-Audio' else 'LM'} probe {c['master_probe']:.4f} vs answer {c['master_answer']:.4f}, diff {c['diff_master']:+.4f} "
             f"[{c['diff_lo']:+.4f}, {c['diff_hi']:+.4f}]; {side}; interval {'excludes' if c['excludes_zero'] else 'includes'} 0")
for k in landed:
    if k["dataset"] == "E-DAIC":
        L.append(f"   Kimi-Audio PART 26 encoder (not master): {k['probe_mean5']:.4f} vs answer {k['answer_master']:.4f}, diff {k['diff_point']:+.4f} [{k['diff_lo']:+.4f}, {k['diff_hi']:+.4f}]")
open(f"{C}/C_other_models_summary.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
side = dict(created_utc=now, script=os.path.abspath(__file__), method=__doc__, inputs=[f"{R2}/C_r2_cells.json", f"{R2}/kimi_enc/C_r2_kimi_enc_cells.json"],
            outputs=[out, f"{C}/C_other_models_summary.txt"], first_attempt_kept=f"{C}/C_other_models.first_attempt_0049Z.tsv",
            verified_rule=("a master value is 'verified' when C_r2_recheck.py reproduces it at 4 dp from its per-clip file; for the probe the five "
                           "per-repeat AUCs must also equal the per-repeat AUCs the original run saved (nested_repeats json or POD3 sidecar) at 4 dp"),
            cells=cells, kimi_encoder_supplement=kim)
json.dump(side, open(f"{C}/C_other_models.sidecar.json", "w"), indent=1, default=str)
print("ASSEMBLED", len(T), "rows")
