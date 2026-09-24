"""Re-cut first-30 s clips so the window starts at the patient's first words, not the interviewer's prompt.
Transcribes the first 75 s with faster-whisper (segment timestamps), finds prompt-like segments in the
opening 45 s, starts the window at the end of the last one. Writes *_patient manifests and clips."""
import csv, os, re, subprocess, sys
from faster_whisper import WhisperModel
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
PROMPT = re.compile(r"(tell me|everything (that )?you see|going on in (this|that|the) picture|what do you see|here'?s (the|a) picture|look at (this|that|the) picture|what'?s happening|all (of )?the action|describe)", re.I)
m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
def onset(src):
    segs, _ = m.transcribe(src, language="en", vad_filter=True, clip_timestamps="0,75")
    start = 0.0
    for s in segs:
        if s.start > 45: break
        if PROMPT.search(s.text): start = max(start, s.end)
    return round(min(start, 45.0), 2)
def run(name, manifest, srcdir_fn, outdir):
    rows = list(csv.DictReader(open(manifest)))
    os.makedirs(outdir, exist_ok=True); shifted = 0
    for i, r in enumerate(rows):
        src = srcdir_fn(r)
        st = onset(src); r["start_s"] = st; shifted += st > 0
        wav = f"{outdir}/{r['spk']}.wav"
        if not os.path.exists(wav):
            subprocess.run(["ffmpeg","-loglevel","error","-y","-ss",str(st),"-i",src,"-ac","1","-ar","16000","-t","30",wav], check=True)
        r["segment_path"] = os.path.relpath(wav, D)
        if (i+1) % 25 == 0: print(f"{name} {i+1}/{len(rows)}", flush=True)
    out = manifest.replace(".csv", "_patient.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"{name}: {shifted}/{len(rows)} clips shifted to patient onset, manifest {out}", flush=True)
A = f"{D}/DementiaBank/challenges/ADReSS-M/ADReSS-M-train_x/train"
run("adresso", f"{D}/adresso/adresso_manifest.csv", lambda r: f"{A}/{r['spk']}.mp3", f"{D}/adresso/segments_patient")
B = f"{D}/DementiaBank/challenges/ADReSS-2020"
def src2020(r):
    if r["split"] == "train":
        sub = "cd" if r["label"] == "1" else "cc"
        return f"{B}/ADReSS-IS2020-train_x/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/{sub}/{r['spk']}.wav"
    return f"{B}/ADReSS-IS2020-test_x/ADReSS-IS2020-data/test/Full_wave_enhanced_audio/{r['spk']}.wav"
run("adress2020", f"{D}/adress2020/adress2020_manifest.csv", src2020, f"{D}/adress2020/segments_patient")
print("RECUT DONE", flush=True)
