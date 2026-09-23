"""PART20 POD4c analysis: AUC (scipy rankdata, ties averaged) + speaker bootstrap
(2000 draws, fresh default_rng(0) per cell, speakers with replacement, percentile 2.5/97.5;
PAIRED = one speaker draw per replicate, both AUCs recomputed on it).
usage: analyze.py MODEL(af3|af2) NOINV_SCORES.csv   -> writes results under OUTDIR."""
import os, sys, csv, json, hashlib, datetime
import numpy as np
from scipy.stats import rankdata

MODEL, NOINV = sys.argv[1], sys.argv[2]
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
}
MID = {"af3": "nvidia/audio-flamingo-3-hf",
       "af2": "nvidia/audio-flamingo-2 (snapshot 3a7f4aea17e72c7367940affea6749069a7095fc) + Qwen/Qwen2.5-3B"}[MODEL]

SCORE_SCRIPT = {"af3": "/workspace/af3_zs.py", "af2": "/workspace/af2_zs.py"}[MODEL]
SCORE_CMD = {"af3": "cd /workspace && HF_HOME=/workspace/hf WINDOW_S=30 python3 af3_zs.py cut/score_manifest.csv work/af3_noinv468.csv",
             "af2": "cd /workspace && HF_HOME=/workspace/hf /workspace/af2venv/bin/python af2_zs.py cut/score_manifest.csv work/af2_noinv468.csv"}[MODEL]
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

PAPER = {"af3": {"part3": (0.5885, 0.6169, 0.5822), "part10": (None, 0.6075, 0.5855)},
         "af2": {"af2pitt468": (0.5724, 0.6310, 0.5544)}}  # as quoted in the task brief (overall/conflict/agreement); part10 brief quotes conflict/agreement only
summary = {}
for arm, ks in ARMS.items():
    spk = np.array([new[k]["speaker_id"] for k in ks]); y = np.array([int(new[k]["label"]) for k in ks])
    sn = np.array([float(new[k]["p_yes"]) for k in ks]); mn = np.array([float(new[k]["answer_mass"]) for k in ks])
    a = auc(y, sn); lo, hi, nu = boot(spk, y, sn)
    extra = {"arm": arm, "median_answer_mass": round(float(np.median(mn)), 4), "yes_rate": round(float((sn > 0.5).mean()), 4),
             "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
             "clips": "Pitt interviewer-free cut (PAR-only, first min(window,30 s)), <local data dir>/release/scores/part20/pitt_noinv_cut/"}
    if MODEL == "af2":
        s3 = np.array([float(new[k]["p_yes_rule3"]) for k in ks])
        extra["auc_rule3_ids"] = round(auc(y, s3), 4)
        extra["note_scoring"] = ("p_yes uses the canonical af2_pitt468.csv rule (AF2Adapter.score_yes_no: Yes/yes vs No/no, bare + "
                                 "leading space, first token id; fp32). auc_rule3_ids = same forward pass with Yes/yes/YES vs No/no/NO single-token ids.")
    rows = [[k, new[k]["clip_id"], new[k]["speaker_id"], new[k]["label"], ARM[k], new[k]["p_yes"], new[k]["answer_mass"]] +
            ([new[k]["p_yes_rule3"]] if MODEL == "af2" else []) for k in ks]
    hdr = ["orig_clip_id", "clip_id", "speaker_id", "label", "arm", "p_yes", "answer_mass"] + (["p_yes_rule3"] if MODEL == "af2" else [])
    emit(f"{MODEL}_noinv_{arm}", f"{MODEL.upper()} zero-shot AUC, Pitt interviewer-free cut, {arm}", a, lo, hi, len(ks), len(np.unique(spk)), nu,
         rows, hdr, SRC_COMMON, extra)
    summary[("noinv", arm)] = a
    for tag, p in ORIG:
        d = orig[tag]
        so = np.array([float(d[k]["p_yes"]) for k in ks])
        ao = auc(y, so); lo2, hi2, nu2 = boot(spk, y, so)
        idx = {"all": 0, "conflict": 1, "agreement": 2}[arm]
        ex2 = {"arm": arm, "paper_value_quoted_in_brief": PAPER[MODEL][tag][idx], "recomputed_minus_quoted": (None if PAPER[MODEL][tag][idx] is None else round(ao - PAPER[MODEL][tag][idx], 4)),
               "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
        rows2 = [[k, d[k]["speaker_id"], d[k]["label"], ARM[k], d[k]["p_yes"], d[k]["answer_mass"]] for k in ks]
        emit(f"{MODEL}_orig_{tag}_{arm}", f"{MODEL.upper()} zero-shot AUC, original 468 windows ({tag} file), {arm} (recomputed)", ao, lo2, hi2,
             len(ks), len(np.unique(spk)), nu2, rows2, ["orig_clip_id", "speaker_id", "label", "arm", "p_yes", "answer_mass"],
             [p, "/workspace/refs/pitt_conflict_manifest.csv"], ex2)
        summary[(tag, arm)] = ao
        # paired original minus interviewer-free
        dv = ao - a; lo3, hi3, nu3 = boot(spk, y, so, sn)
        rows3 = [[k, new[k]["clip_id"], new[k]["speaker_id"], new[k]["label"], ARM[k], d[k]["p_yes"], new[k]["p_yes"]] for k in ks]
        emit(f"{MODEL}_diff_{tag}_minus_noinv_{arm}", f"{MODEL.upper()} paired AUC difference, original ({tag}) minus interviewer-free, {arm}",
             dv, lo3, hi3, len(ks), len(np.unique(spk)), nu3, rows3,
             ["orig_clip_id", "clip_id", "speaker_id", "label", "arm", "p_yes_original", "p_yes_noinv"],
             [p] + SRC_COMMON, {"arm": arm, "auc_original": round(ao, 4), "auc_noinv": round(a, 4), "direction": "original minus interviewer-free"}, diff=True)
print("SUMMARY", {f"{k[0]}_{k[1]}": round(v, 4) for k, v in summary.items()}, flush=True)
