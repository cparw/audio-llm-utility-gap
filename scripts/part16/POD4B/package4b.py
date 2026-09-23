#!/usr/bin/env python3
"""POD4B packaging: per-clip csv + sidecar json for one corpus.
usage: package4b.py CORPUS
"""
import csv, json, os, sys, hashlib, subprocess, datetime
CORPUS = sys.argv[1]
ROOT, OUT = "/workspace/p16pod4b", "/workspace/p16pod4b/out"

def sha_1mb(p):
    try:
        with open(p, "rb") as f: return hashlib.sha256(f.read(1024*1024)).hexdigest()
    except Exception as e: return f"UNREADABLE: {e}"

def load(path, col, key="clip"):
    return {r[key]: r for r in csv.DictReader(open(path))}

arms = {"A": "orig", "B": "paronly", "C": "durmatch"}
zs  = {k: load(f"{OUT}/p16_{CORPUS}_{v}_zeroshot_scores.csv", "p_yes")  for k, v in arms.items()}
enc = {k: load(f"{OUT}/p16_{CORPUS}_{v}_enc_nested_oof.csv",  "p_probe") for k, v in arms.items()}
pubz = load(f"{ROOT}/pub/omni_{CORPUS}_zeroshot_scores.csv", "p_yes")
pube = load(f"{ROOT}/pub/omni_{CORPUS}_enc_nested_oof.csv",  "p_probe")
cut  = {r["clip"]: r for r in json.load(open(f"{ROOT}/cut_report_{CORPUS}.json")) if r["ok"]}
iv   = {m["clip"]: m for m in json.load(open(f"{ROOT}/par_intervals_{CORPUS}.json"))}
mf   = list(csv.DictReader(open(f"{ROOT}/mf_{CORPUS}_paronly.csv")))

rows = []
for r in mf:
    c = r["clip"]; m = iv[c]; k = cut[c]
    span = m["span_ms"] or 1
    rows.append(dict(
        clip=c, speaker=r["speaker"], label=int(r["label"]), split=r["set"],
        window_start_ms=m["start_ms"], window_span_ms=m["span_ms"],
        par_ms_in_window=m["par_ms"], inv_ms_in_window=m["inv_ms"],
        inv_share_of_window=round(m["inv_ms"]/span, 5),
        n_par_intervals=len(m["par_off"]),
        dur_A_orig_s=k["orig_s"], dur_B_paronly_s=k["par_s"], dur_C_durmatch_s=k["par_s"],
        p_yes_A=zs["A"][c]["p_yes"], answer_mass_A=zs["A"][c]["mass"],
        p_yes_B=zs["B"][c]["p_yes"], answer_mass_B=zs["B"][c]["mass"],
        p_yes_C=zs["C"][c]["p_yes"], answer_mass_C=zs["C"][c]["mass"],
        p_probe_A=enc["A"][c]["p_probe"], p_probe_B=enc["B"][c]["p_probe"], p_probe_C=enc["C"][c]["p_probe"],
        fold=enc["A"][c]["fold"],
        p_yes_published=pubz[c]["p_yes"] if c in pubz else "",
        answer_mass_published=pubz[c]["mass"] if c in pubz else "",
        p_probe_published=pube[c]["p_probe"] if c in pube else "",
    ))
outcsv = f"{OUT}/{CORPUS}_no_interviewer.csv"
with open(outcsv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {outcsv}  n={len(rows)}")

zj = json.load(open(f"{OUT}/p16_{CORPUS}_orig_zeroshot.json"))
srcs = {
 "scored_clips_arm_A": f"{ROOT}/src/{CORPUS}_orig",
 "par_intervals": f"{ROOT}/par_intervals_{CORPUS}.json",
 "manifest_arm_A": f"{ROOT}/mf_{CORPUS}_orig.csv",
 "manifest_arm_B": f"{ROOT}/mf_{CORPUS}_paronly.csv",
 "manifest_arm_C": f"{ROOT}/mf_{CORPUS}_durmatch.csv",
 "published_zeroshot": f"{ROOT}/pub/omni_{CORPUS}_zeroshot_scores.csv",
 "published_enc_probe": f"{ROOT}/pub/omni_{CORPUS}_enc_nested_oof.csv",
 "zeroshot_scores_arm_A": f"{OUT}/p16_{CORPUS}_orig_zeroshot_scores.csv",
 "zeroshot_scores_arm_B": f"{OUT}/p16_{CORPUS}_paronly_zeroshot_scores.csv",
 "zeroshot_scores_arm_C": f"{OUT}/p16_{CORPUS}_durmatch_zeroshot_scores.csv",
 "enc_oof_arm_A": f"{OUT}/p16_{CORPUS}_orig_enc_nested_oof.csv",
 "enc_oof_arm_B": f"{OUT}/p16_{CORPUS}_paronly_enc_nested_oof.csv",
 "enc_oof_arm_C": f"{OUT}/p16_{CORPUS}_durmatch_enc_nested_oof.csv",
}
seg_origin = {
 "adress2020": ("CHAT .cha transcripts of the ADReSS-2020 release, per-utterance time bullets, "
                "tiers *PAR (participant) and *INV (investigator). 0 untimed utterances in all 156 files.",
                "<local data dir>/paper work/Hard drive data for paper/DementiaBank/challenges/ADReSS-2020/"
                "ADReSS-IS2020-{train_x,test_x}/ADReSS-IS2020-data/{train,test}/transcription/"),
 "adresso":   ("The RELEASED ADReSSo21 speaker-diarised segmentation csvs (columns speaker,begin,end in ms, "
                "speaker in {PAR,INV}). No diariser was run. Extracted from the challenge archive.",
                "<local data dir>/DementiaBank/challenges/ADReSSo/ADReSSo21-diagnosis-train.tgz?f=open"
                "  ->  ADReSSo21/diagnosis/train/segmentation/{ad,cn}/<spk>.csv"),
}[CORPUS]

side = {
 "pod": "POD 4 (runpod id 1avtkbh82lpb6y), H200, working dir /workspace/p16pod4b",
 "corpus": CORPUS,
 "what": "Part 4b extended: interviewer removed, with a duration-matched control that KEEPS the interviewer",
 "model_id": zj["checkpoint"], "dtype": zj["dtype"], "device": zj["device"],
 "condition": zj["condition"],
 "prompt_verbatim": zj["prompt"],
 "window_seconds": zj["window_seconds"],
 "n": len(rows), "n_speakers": len(set(r["speaker"] for r in rows)),
 "label_distribution": {str(k): sum(1 for r in rows if r["label"] == k) for k in (0, 1)},
 "arms": {
   "A_orig": "the scored clip exactly as the paper cut it (ffmpeg -ss start_s -t 30 off the source), unchanged",
   "B_paronly": "ONLY the participant intervals inside that same window, concatenated, 5 ms raised-cosine fade at every splice boundary",
   "C_durmatch": "the first len(arm B) seconds of arm A: matched in length to B but the interviewer and the gaps are still in it",
 },
 "segmentation_source_description": seg_origin[0],
 "segmentation_source_path": seg_origin[1],
 "p_yes_definition": ("P(Yes)/(P(Yes)+P(No)) at the FIRST answer position, summing the first-token ids of "
                      "Yes/ Yes/yes/ yes/YES and No/ No/no/ no/NO, single-token encodings only. "
                      "answer_mass = P(Yes)+P(No), recorded per clip in the per-clip csv."),
 "auc": "recomputed from the per-clip scores with the rank formula in boot.py; never copied from a json",
 "bootstrap": ("2000 draws, numpy.random.default_rng(0) reseeded per cell, SPEAKERS resampled with "
               "replacement (never clips), percentile 2.5/97.5. PAIRED: the speaker list is drawn ONCE "
               "per replicate and both quantities are recomputed inside that same draw. Usable draws "
               "reported per row in boot_%s.json." % CORPUS),
 "seed": 0,
 "folds": ("No saved fold assignment exists for these clips (checked the published *_enc_nested_oof.csv, "
           "which carry no fold column). Per rule 7: sklearn GroupKFold(n_splits=5) grouped by speaker in "
           "the source csv's own order, identical across arms A/B/C, which makes the paired contrast exact. "
           "Inner layer choice GroupKFold(4) on the training speakers. Logged as POD4B-D5 (MEDIUM)."),
 "probe": "encoder stages only (enc), 32 Qwen2.5-Omni audio encoder layers, nested layer selection",
 "commands": [
   f"python3 cut_arms4b.py {CORPUS}",
   f"WINDOW_S=30 python3 extract_probe_layers.py /workspace/p16pod4b/mf_{CORPUS}_<ARM>.csv "
   f"/workspace/p16pod4b/out/p16_{CORPUS}_<ARM> ad \"Qwen/Qwen2.5-Omni-7B\"   (ARM in orig|paronly|durmatch)",
   f"python3 nested_oof.py /workspace/p16pod4b/out/p16_{CORPUS}_<ARM>_states.npz "
   f"/workspace/p16pod4b/out/p16_{CORPUS}_<ARM> enc",
   f"python3 analyse4b_pod4b.py {CORPUS} out/spec_{CORPUS}.json",
   f"python3 boot.py out/spec_{CORPUS}.json out/boot_{CORPUS}.json",
   f"python3 package4b.py {CORPUS}",
 ],
 "source_absolute_paths": srcs,
 "sha256_first_1MB": {k: sha_1mb(v) for k, v in srcs.items() if os.path.isfile(v)},
 "transformers": subprocess.run(["python3","-c","import transformers;print(transformers.__version__)"],
                                capture_output=True, text=True).stdout.strip(),
 "torch": subprocess.run(["python3","-c","import torch;print(torch.__version__)"],
                         capture_output=True, text=True).stdout.strip(),
 "date": datetime.datetime.now().isoformat(timespec="seconds"),
}
sp = f"{OUT}/{CORPUS}_no_interviewer.sidecar.json"
json.dump(side, open(sp, "w"), indent=1)
print(f"wrote {sp}")
