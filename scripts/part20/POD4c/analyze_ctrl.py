"""PART20 POD4c analysis: AUC (scipy rankdata, ties averaged) + speaker bootstrap
(2000 draws, fresh default_rng(0) per cell, speakers with replacement, percentile 2.5/97.5;
PAIRED = one speaker draw per replicate, both AUCs recomputed on it).
CONTROL: AF2 on the ORIGINAL windows cut to their first 30 s (the span the interviewer-free clips come from).
usage: analyze_ctrl.py af2 NOINV_SCORES.csv ORIG30_SCORES.csv"""
import os, sys, csv, json, hashlib, datetime
import numpy as np
from scipy.stats import rankdata

MODEL, NOINV, ORIG30 = sys.argv[1], sys.argv[2], sys.argv[3]
assert MODEL == "af2"
OUTDIR = os.environ.get("OUTDIR", "/workspace/scores/part20/POD4c")
ROWS = os.path.join(OUTDIR, "POD4c_provisional_rows.tsv")
CMD = " ".join(["python3"] + sys.argv)
NB = 2000
PROMPT = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
MAC = {  # pod copy -> Mac absolute path (the sources as they sit on the Mac)
    "/workspace/refs/af3_pitt_part3.csv": "<local data dir>/release/overnight2/part3/af3_pitt.csv",
    "/workspace/refs/af3_pitt_part10.csv": "<local data dir>/release/overnight2/part10/af3_pitt.csv",
    "/workspace/refs/af2_pitt468.csv": "<local data dir>/paper work/paper1_local_runs/af2_results/af2_pitt468.csv",
    "/workspace/refs/pitt_conflict_manifest.csv": "<local data dir>/DementiaBank/pitt_conflict_manifest.csv",
    "/workspace/cut/manifest.csv": "manifests/part20/pitt_noinv_cut/manifest.csv",
    "/workspace/orig30/orig_hashes.csv": "manifests/part20/POD4c/orig_hashes.csv (originals: <local data dir>/Desktop/clips_for_af3/pitt468/, FLAC-transported, int16 sample sha256 checked)",
}
MID = {"af3": "nvidia/audio-flamingo-3-hf",
       "af2": "nvidia/audio-flamingo-2 (snapshot 3a7f4aea17e72c7367940affea6749069a7095fc) + Qwen/Qwen2.5-3B"}[MODEL]

SCORE_SCRIPT = "/workspace/af2_zs_win.py"
SCORE_CMD = ("interviewer-free: cd /workspace && HF_HOME=/workspace/hf /workspace/af2venv/bin/python af2_zs.py cut/score_manifest.csv work/af2_noinv468.csv ; "
             "original first 30 s: cd /workspace && HF_HOME=/workspace/hf WINDOW_S=30 /workspace/af2venv/bin/python af2_zs_win.py orig30/score_manifest.csv work/af2_orig30_468.csv")
SCORE_ENV = {"af3": "POD4c H200, system python 3.12.3, torch 2.8.0+cu128, transformers 5.17.0, AudioFlamingo3ForConditionalGeneration, bf16, 30 s cut (no clip exceeds 29.3 s)",
             "af2": ("POD4c H200, venv /workspace/af2venv (--system-site-packages: python 3.12.3, torch 2.8.0+cu128) + transformers 4.46.3 / tokenizers 0.20.3, "
                     "the same software as the 20 Sep pod run that wrote af2_pitt468.csv; lab AF2Adapter + official NVIDIA af2_code from ~/Desktop/af2_run; "
                     "AF2_CODE_DIR=/workspace/af2code/af2/af2_code, AF2_CKPT_DIR=nvidia/audio-flamingo-2 snapshot 3a7f4aea; fp32; no cut (AF2 10 s window grid, max 12)")}[MODEL]

def sha1mb(p):
    with open(p, "rb") as f: return hashlib.sha256(f.read(1 << 20)).hexdigest()

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def boot(spk, y, s, s2=None):
    rng = np.random.default_rng(0)
    u = np.unique(spk); idx = {k: np.where(spk == k)[0] for k in u}
    v = []
    for _ in range(NB):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[k] for k in pick])
        a = auc(y[ii], s[ii])
        if s2 is not None:
            b = auc(y[ii], s2[ii])
            if np.isnan(a) or np.isnan(b): continue
            v.append(a - b)
        else:
            if np.isnan(a): continue
            v.append(a)
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

# --- arms from the conflict manifest (key basename(segment_path), column 'set')
ARM = {os.path.basename(r["segment_path"]): r["set"] for r in csv.DictReader(open("/workspace/refs/pitt_conflict_manifest.csv"))}
# --- new scores
new = {r["orig_clip_id"]: r for r in csv.DictReader(open(NOINV))}
assert len(new) == 468, len(new)
# --- original (paper) score files
ORIG = {"af3": [("part3", "/workspace/refs/af3_pitt_part3.csv"), ("part10", "/workspace/refs/af3_pitt_part10.csv")],
        "af2": [("af2pitt468", "/workspace/refs/af2_pitt468.csv")]}[MODEL]
orig = {}
for tag, p in ORIG:
    d = {os.path.basename(r["clip_path"]): r for r in csv.DictReader(open(p))}
    assert set(d) == set(new), (tag, len(set(d) ^ set(new)))
    for k in d:
        assert int(d[k]["label"]) == int(new[k]["label"]), ("label", k)
        assert d[k]["speaker_id"] == new[k]["speaker_id"], ("spk", k, d[k]["speaker_id"], new[k]["speaker_id"])
    orig[tag] = d
for k in new: assert ARM[k] == new[k]["arm"], ("arm", k)

keys_all = sorted(new)
ARMS = {"all": keys_all, "conflict": [k for k in keys_all if ARM[k] == "conflict"],
        "agreement": [k for k in keys_all if ARM[k] == "agreement"]}
SRC_COMMON = [NOINV, "/workspace/refs/pitt_conflict_manifest.csv", "/workspace/cut/manifest.csv"]

def emit(rid, what, val, lo, hi, n, nspk, nuse, csvrows, header, sources, extra, diff=False):
    pc = os.path.join(OUTDIR, rid + ".csv"); sj = os.path.join(OUTDIR, rid + ".json")
    with open(pc, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(csvrows)
    srcs = []
    for p in sources:
        srcs.append({"pod_path": os.path.abspath(p), "mac_path": MAC.get(p, "(pod-produced; synced to <local data dir>/release/scores/part20/POD4c/work/)"),
                     "sha256_first_1MB": sha1mb(p)})
    two = f"{val:.2f}"
    if diff:
        verdict = "interval excludes zero" if (lo > 0 or hi < 0) else "interval includes zero"
    else:
        verdict = "above 0.5" if val > 0.5 else "below 0.5" if val < 0.5 else "at 0.5"
    side = {"id": rid, "what": what, "value": round(val, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n": n, "n_speakers": nspk, "bootstrap": {"draws": NB, "usable_draws": nuse, "seed": 0,
            "rng": "numpy.random.default_rng(0) fresh per cell", "unit": "speaker (with replacement)",
            "percentile": [2.5, 97.5], "paired": diff},
            "auc": "rank formula, scipy.stats.rankdata, ties averaged",
            "model_id": MID, "prompt_verbatim": PROMPT, "command": CMD, "sources": srcs,
            "per_clip_csv": pc, "two_dp": two, "verdict": verdict,
            "scoring_command": SCORE_CMD, "scoring_env": SCORE_ENV,
            "scoring_script_sha256": hashlib.sha256(open(SCORE_SCRIPT, "rb").read()).hexdigest(), "scoring_script": SCORE_SCRIPT,
            "date_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
    side.update(extra)
    json.dump(side, open(sj, "w"), indent=1)
    open(os.path.join(OUTDIR, rid + ".RESULT"), "w").close()
    print(f"PASTE {rid}: {val:.4f} [{lo:.4f}, {hi:.4f}] n={n} n_spk={nspk} file={pc} | {two} {verdict}", flush=True)
    new_file = not os.path.exists(ROWS)
    with open(ROWS, "a") as f:
        if new_file: f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\ttwo_dp\tvs_half\n")
        f.write(f"{rid}\t{what}\t{val:.4f}\t{lo:.4f}\t{hi:.4f}\t{n}\t{nspk}\t{pc}\t{two}\t{verdict}\n")

o30 = {r["orig_clip_id"]: r for r in csv.DictReader(open(ORIG30))}
assert set(o30) == set(new)
for k in new: assert o30[k]["label"] == new[k]["label"] and o30[k]["speaker_id"] == new[k]["speaker_id"]
full = orig["af2pitt468"]
SRCS = [ORIG30, "/workspace/orig30/orig_hashes.csv", "/workspace/refs/pitt_conflict_manifest.csv"]
for arm, ks in ARMS.items():
    spk = np.array([new[k]["speaker_id"] for k in ks]); y = np.array([int(new[k]["label"]) for k in ks])
    s30 = np.array([float(o30[k]["p_yes"]) for k in ks]); sn = np.array([float(new[k]["p_yes"]) for k in ks])
    sf_ = np.array([float(full[k]["p_yes"]) for k in ks]); m30 = np.array([float(o30[k]["answer_mass"]) for k in ks])
    nspk = len(np.unique(spk))
    a30 = auc(y, s30); lo, hi, nu = boot(spk, y, s30)
    emit(f"af2_orig30_{arm}", f"AF2 zero-shot AUC, original windows cut to first 30 s (control), {arm}", a30, lo, hi, len(ks), nspk, nu,
         [[k, o30[k]["speaker_id"], o30[k]["label"], ARM[k], o30[k]["dur_s"], o30[k]["scored_s"], o30[k]["p_yes"], o30[k]["answer_mass"], o30[k]["p_yes_rule3"]] for k in ks],
         ["orig_clip_id", "speaker_id", "label", "arm", "dur_s", "scored_s", "p_yes", "answer_mass", "p_yes_rule3"], SRCS,
         {"arm": arm, "median_answer_mass": round(float(np.median(m30)), 4), "yes_rate": round(float((s30 > 0.5).mean()), 4),
          "auc_rule3_ids": round(auc(y, np.array([float(o30[k]["p_yes_rule3"]) for k in ks])), 4),
          "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())})
    an = auc(y, sn); d = a30 - an; lo, hi, nu = boot(spk, y, s30, sn)
    emit(f"af2_diff_orig30_minus_noinv_{arm}", f"AF2 paired AUC difference, original first 30 s minus interviewer-free (same span), {arm}", d, lo, hi, len(ks), nspk, nu,
         [[k, new[k]["clip_id"], new[k]["speaker_id"], new[k]["label"], ARM[k], o30[k]["p_yes"], new[k]["p_yes"]] for k in ks],
         ["orig_clip_id", "clip_id", "speaker_id", "label", "arm", "p_yes_orig30", "p_yes_noinv"], SRCS + [NOINV],
         {"arm": arm, "auc_orig30": round(a30, 4), "auc_noinv": round(an, 4), "direction": "original first 30 s minus interviewer-free"}, diff=True)
    af = auc(y, sf_); d = af - a30; lo, hi, nu = boot(spk, y, sf_, s30)
    emit(f"af2_diff_origfull_minus_orig30_{arm}", f"AF2 paired AUC difference, original full window (af2_pitt468.csv) minus original first 30 s, {arm}", d, lo, hi, len(ks), nspk, nu,
         [[k, o30[k]["speaker_id"], o30[k]["label"], ARM[k], o30[k]["dur_s"], full[k]["p_yes"], o30[k]["p_yes"]] for k in ks],
         ["orig_clip_id", "speaker_id", "label", "arm", "dur_s", "p_yes_full", "p_yes_orig30"], SRCS + ["/workspace/refs/af2_pitt468.csv"],
         {"arm": arm, "auc_full": round(af, 4), "auc_orig30": round(a30, 4), "direction": "full window minus first 30 s"}, diff=True)
