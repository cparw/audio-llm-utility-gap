#!/usr/bin/env python3
"""PART20 POD4b analysis: Qwen3-Omni zero-shot AUC on the 468 Pitt interviewer-free clips, overall and by arm,
beside the original-window files, plus PAIRED original-minus-interviewer-free per arm.
AUC: rank formula, scipy.stats.rankdata (ties averaged).
Bootstrap: 2000 draws, fresh numpy.random.default_rng(0) per cell, speakers resampled with replacement,
percentile 2.5/97.5; draws whose resampled set lacks a class are skipped (count recorded).
PAIRED: one speaker draw per replicate, both AUCs recomputed on that draw.
Speakers: speaker_id from folds/pitt_groupkfold5_pod.csv keyed by the
original clip_id (rule 1). Arms: 'set' from pitt_conflict_manifest.csv keyed by basename(segment_path).
"""
import csv, json, os, sys, hashlib, datetime
import numpy as np
from scipy.stats import rankdata

NB = 2000
OUT = "/workspace/scores/part20/POD4b"
MAC_OUT = "scores/part20/POD4b"
MID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PROMPT = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
CMD = " ".join(["python3"] + sys.argv)
B = "/workspace/p20"
# pod path -> Mac absolute source path (for the sidecar)
SRC = {
    "noinv": (f"{OUT}/q3o_pitt_noinv_zeroshot_scores.csv", f"{MAC_OUT}/q3o_pitt_noinv_zeroshot_scores.csv"),
    "shipped": (f"{B}/ref_shipped.csv", "<local data dir>/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"),
    "rerun": (f"{B}/ref_rerun.csv", "<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_pitt_zeroshot_scores.csv"),
    "samepod": (f"{OUT}/q3o_pitt_orig468_samepod_zeroshot_scores.csv", f"{MAC_OUT}/q3o_pitt_orig468_samepod_zeroshot_scores.csv"),
    "folds": (f"{B}/pitt_groupkfold5_pod.csv", "folds/pitt_groupkfold5_pod.csv"),
    "sets": (f"{B}/pitt_conflict_manifest.csv", "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"),
    "cutman": (f"{B}/noinv/manifest.csv", "manifests/part20/pitt_noinv_cut/manifest.csv"),
}

SCORING = dict(
    noinv=dict(command="cd /workspace/p20 && HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 nohup python3 score_q3o_noinv.py > score.log 2>&1",
               script_mac="scripts/part20/POD4b/score_q3o_noinv.py",
               audio="/workspace/p20/noinv/clips (pitt_noinv_clips.tgz sha256 b995bd9afe8a45b1c08472f7333c5855767d60de23307a7881b5f000c73d8a07; 468/468 wav sha256 checked against manifest.csv)"),
    samepod=dict(command="cd /workspace/p20 && HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 python3 score_orig468.py > score_orig.log 2>&1",
                 script_mac="scripts/part20/POD4b/score_orig468.py",
                 audio="/workspace/p20/orig/segments = <local data dir>/DementiaBank/segments (468/468 wav sha256 checked against pitt_noinv_cut/source_segments.sha256)"),
    reference_files_note="the shipped (overnight2/q3o_new) and rerun (part16/pod_sync) p_yes were scored by earlier runs with 5+5 ids (no ' YES' 14080 / ' NO' 5664); this pod's files use 6+6 and keep the 5+5 value in p_yes_ref5 (max per-clip difference 4.2e-08 on the interviewer-free clips)",
    pod="RunPod spkhephddpcaom, NVIDIA H200, runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, transformers 5.17.0, torch 2.8.0+cu128, librosa 1.0.0, numpy 2.1.2, sdpa attention, bf16, thinker only, first 30 s of each clip")

def sha1mb(p):
    with open(p, "rb") as fh: return hashlib.sha256(fh.read(1 << 20)).hexdigest()

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

folds = {r["clip_id"]: r["speaker_id"] for r in csv.DictReader(open(SRC["folds"][0]))}
sets = {os.path.basename(r["segment_path"]): (r["set"], int(r["label"])) for r in csv.DictReader(open(SRC["sets"][0]))}
assert len(folds) == 468 and len(sets) == 468

def load(tag, col="p_yes"):
    p = SRC[tag][0]
    if not os.path.exists(p): return None
    d = {}
    for r in csv.DictReader(open(p)):
        oc = r.get("orig_clip_id") or r["clip"]
        st, lab = sets[oc]
        assert int(r["label"]) == lab, (tag, oc)
        d[oc] = dict(clip=r["clip"], speaker=folds[oc], label=lab, set=st, score=float(r[col]),
                     mass=float(r.get("answer_mass") or r.get("mass") or "nan"))
    assert len(d) == 468, (tag, len(d))
    return d

def boot(spk, y, s):
    rng = np.random.default_rng(0)
    u = np.unique(spk); idx = {q: np.where(spk == q)[0] for q in u}
    v = []
    for _ in range(NB):
        ii = np.concatenate([idx[q] for q in rng.choice(u, size=len(u), replace=True)])
        a = auc(y[ii], s[ii])
        if not np.isnan(a): v.append(a)
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def boot_paired(spk, y, s_orig, s_new):
    rng = np.random.default_rng(0)
    u = np.unique(spk); idx = {q: np.where(spk == q)[0] for q in u}
    v = []
    for _ in range(NB):
        ii = np.concatenate([idx[q] for q in rng.choice(u, size=len(u), replace=True)])
        a1, a2 = auc(y[ii], s_orig[ii]), auc(y[ii], s_new[ii])
        if not (np.isnan(a1) or np.isnan(a2)): v.append(a1 - a2)
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

rows_out = []
def emit(rid, what, value, lo, hi, n, nspk, draws, perclip_rows, sources, extra, is_diff=False):
    pc = f"{OUT}/{rid}.perclip.csv"; sc = f"{OUT}/{rid}.sidecar.json"
    with open(pc, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(perclip_rows[0].keys())); w.writeheader(); w.writerows(perclip_rows)
    side = dict(id=rid, what=what, value=round(value, 4), ci95=[round(lo, 4), round(hi, 4)], n=n, n_speakers=nspk,
                usable_draws=draws, bootstrap=dict(draws=NB, rng="numpy.random.default_rng(0) fresh per cell",
                unit="speaker, with replacement", percentiles=[2.5, 97.5], paired=is_diff),
                auc="rank formula, scipy.stats.rankdata average ties", seed=0, model_id=MID, prompt_verbatim=PROMPT,
                p_yes="P(Yes)/(P(Yes)+P(No)) at the first answer position over single-token ids of Yes, Yes, yes, yes, YES, YES and No, No, no, no, NO, NO (bare and leading space); bf16",
                command=CMD, perclip_csv_mac=f"{MAC_OUT}/{rid}.perclip.csv",
                sources={k: dict(pod=SRC[k][0], mac=SRC[k][1], sha256_first_1MB=sha1mb(SRC[k][0])) for k in sources},
                scoring=SCORING, created_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", **extra)
    json.dump(side, open(sc, "w"), indent=1)
    open(f"{OUT}/{rid}.RESULT", "w").write("")
    macfile = f"{MAC_OUT}/{rid}.perclip.csv"
    if is_diff:
        two = f"{value:+.2f} [{lo:+.2f}, {hi:+.2f}], interval {'excludes' if (lo > 0 or hi < 0) else 'includes'} zero"
        vs = "excludes zero" if (lo > 0 or hi < 0) else "includes zero"
        line = f"PASTE {rid} | {what} | {value:+.4f} [{lo:+.4f}, {hi:+.4f}] | n={n} n_spk={nspk} | {macfile} | {two}"
    else:
        vs = "above 0.5" if value > 0.5 else "below 0.5"
        two = f"{value:.2f} [{lo:.2f}, {hi:.2f}], {vs}"
        line = f"PASTE {rid} | {what} | {value:.4f} [{lo:.4f}, {hi:.4f}] | n={n} n_spk={nspk} | {macfile} | {two}"
    print(line, flush=True)
    rows_out.append(dict(id=rid, what=what, value=f"{value:.4f}", lo=f"{lo:.4f}", hi=f"{hi:.4f}", n=n, n_spk=nspk,
                         file=macfile, two_dp=two, vs_half=vs))

ARMS = [("overall", None), ("conflict", "conflict"), ("agreement", "agreement")]
D = {t: load(t) for t in ("noinv", "shipped", "rerun", "samepod")}
D_ref5 = load("noinv", "p_yes_ref5")
keys_all = sorted(D["noinv"])
names = {"noinv": "interviewer-free cut (this pod)", "shipped": "original windows, shipped paper file",
         "rerun": "original windows, part16 rerun file", "samepod": "original windows re-scored on this pod"}
tagid = {"noinv": "noinv", "shipped": "orig_shipped", "rerun": "orig_rerun", "samepod": "orig_samepod"}
for t in ("noinv", "shipped", "rerun", "samepod"):
    if D[t] is None: print(f"SKIP {t}: file missing", flush=True); continue
    for arm, st in ARMS:
        ks = [k for k in keys_all if st is None or D[t][k]["set"] == st]
        spk = np.array([D[t][k]["speaker"] for k in ks]); y = np.array([D[t][k]["label"] for k in ks])
        s = np.array([D[t][k]["score"] for k in ks])
        a = auc(y, s); lo, hi, nd = boot(spk, y, s)
        pcr = [dict(clip=D[t][k]["clip"], orig_clip_id=k, speaker=D[t][k]["speaker"], label=D[t][k]["label"],
                    set=D[t][k]["set"], p_yes=D[t][k]["score"], answer_mass=D[t][k]["mass"]) for k in ks]
        ex = dict(n_positive=int(y.sum()), median_answer_mass=float(np.median([D[t][k]["mass"] for k in ks])),
                  min_answer_mass=float(np.min([D[t][k]["mass"] for k in ks])))
        if t == "noinv":
            s5 = np.array([D_ref5[k]["score"] for k in ks])
            ex.update(auc_with_reference_5plus5_ids=round(auc(y, s5), 6), max_abs_p_yes_rule3_vs_ref5=float(np.abs(s - s5).max()))
        srcs = [t, "folds", "sets"] + (["cutman"] if t == "noinv" else [])
        emit(f"P20_4b_q3o_{tagid[t]}_{arm}", f"Qwen3-Omni zero-shot AUC, Pitt, {names[t]}, {arm}", a, lo, hi, len(ks),
             int(len(np.unique(spk))), nd, pcr, srcs, ex)

for t in ("shipped", "rerun", "samepod"):
    if D[t] is None: continue
    for arm, st in ARMS:
        ks = [k for k in keys_all if st is None or D["noinv"][k]["set"] == st]
        spk = np.array([D["noinv"][k]["speaker"] for k in ks]); y = np.array([D["noinv"][k]["label"] for k in ks])
        so = np.array([D[t][k]["score"] for k in ks]); sn = np.array([D["noinv"][k]["score"] for k in ks])
        ao, an = auc(y, so), auc(y, sn); lo, hi, nd = boot_paired(spk, y, so, sn)
        pcr = [dict(orig_clip_id=k, noinv_clip=D["noinv"][k]["clip"], speaker=D["noinv"][k]["speaker"],
                    label=D["noinv"][k]["label"], set=D["noinv"][k]["set"], p_yes_original=D[t][k]["score"],
                    p_yes_noinv=D["noinv"][k]["score"]) for k in ks]
        emit(f"P20_4b_q3o_diff_{tagid[t]}_minus_noinv_{arm}",
             f"PAIRED Qwen3-Omni zero-shot AUC, {names[t]} minus interviewer-free, {arm}", ao - an, lo, hi, len(ks),
             int(len(np.unique(spk))), nd, pcr, [t, "noinv", "folds", "sets"],
             dict(auc_original=round(ao, 6), auc_noinv=round(an, 6), sign="original minus interviewer-free"), is_diff=True)

with open(f"{OUT}/POD4b_rows.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows_out)
print("ANALYSIS DONE", len(rows_out), "rows", flush=True)
