"""PART 26 Qwen3-Omni E-DAIC: pre-cut the dcaps_proc wavs so the upload is small, without changing the samples the
extractor sees. The extractor does librosa.load(path, sr=16000) then x[:16000*30].
- 241 files are 16 kHz mono PCM_16: librosa does not resample, so the first 480000 frames are cut exactly.
- 34 files are 48 kHz mono PCM_16: librosa resamples (soxr_hq). These are cut to the first 60 s at 48 kHz, and for EVERY
  one of them this script checks that librosa.load(cut, sr=16000)[:480000] equals librosa.load(full, sr=16000)[:480000]
  exactly. A file that fails is copied in full instead.
Also checks that dcaps.csv order, labels and speakers equal the q3o_new zero-shot file, writes the pod manifest
(path,label,speaker) in that order, a sha256 list and a report."""
import csv, hashlib, os, sys, json, shutil
import numpy as np, soundfile as sf, librosa
SRC = "<local data dir>/paper1_local_runs/drive_data/dcaps_proc"
MAN = "<local data dir>/paper1_local_runs/drive_data/dcaps.csv"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_edaic_zeroshot_scores.csv"
OUT = sys.argv[1]
D = os.path.join(OUT, "stage", "dcaps30"); os.makedirs(D, exist_ok=True)
man = list(csv.DictReader(open(MAN))); zs = list(csv.DictReader(open(ZS)))
assert [os.path.basename(r["filepath"]) for r in man] == [r["clip"] for r in zs], "dcaps.csv order differs from zero-shot order"
assert [r["label"] for r in man] == [r["label"] for r in zs], "labels differ"
assert [r["speaker_id"] for r in man] == [r["speaker"] for r in zs], "speakers differ"
rows, shas, info = [], [], []
for r in man:
    b = os.path.basename(r["filepath"]); p = os.path.join(SRC, b); o = os.path.join(D, b)
    i = sf.info(p); assert i.channels == 1 and i.subtype == "PCM_16", (b, i)
    rec = {"clip": b, "samplerate": i.samplerate, "full_frames": i.frames}
    if i.samplerate == 16000:
        x, _ = sf.read(p, frames=480000, dtype="int16"); sf.write(o, x, 16000, subtype="PCM_16")
        x2, _ = sf.read(o, dtype="int16"); assert np.array_equal(x, x2)
        rec.update(mode="cut 480000 frames at 16 kHz", cut_frames=int(len(x)))
    else:
        n = 60 * i.samplerate
        x, _ = sf.read(p, frames=n, dtype="int16"); sf.write(o, x, i.samplerate, subtype="PCM_16")
        a = librosa.load(p, sr=16000)[0][:480000]; c = librosa.load(o, sr=16000)[0][:480000]
        same = a.shape == c.shape and np.array_equal(a, c)
        rec.update(mode=f"cut first 60 s at {i.samplerate} Hz", cut_frames=int(len(x)), librosa_first30s_identical=bool(same),
                   librosa_len=int(len(a)), max_abs_diff=float(np.max(np.abs(a - c))) if a.shape == c.shape else None)
        if not same:
            shutil.copyfile(p, o); rec["mode"] = "FULL FILE (cut was not identical)"
        print(b, rec["mode"], "identical" if same else "DIFF", flush=True)
    h = hashlib.sha256(open(o, "rb").read()).hexdigest(); shas.append(f"{h}  {b}"); rec["sha256"] = h
    rows.append((f"/workspace/data/dcaps30/{b}", r["label"], r["speaker_id"])); info.append(rec)
with open(os.path.join(OUT, "stage", "mf_edaic.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "speaker"]); w.writerows(rows)
open(os.path.join(OUT, "stage", "files.sha256"), "w").write("\n".join(shas) + "\n")
# spot check on 16 kHz files with librosa, the extractor's own load
chk = {}
for b in ["300.wav", "450.wav", "718.wav"]:
    a = librosa.load(os.path.join(SRC, b), sr=16000)[0][:480000]; c = librosa.load(os.path.join(D, b), sr=16000)[0][:480000]
    chk[b] = bool(a.shape == c.shape and np.array_equal(a, c)); print("LIBROSA_CHECK_16k", b, chk[b], flush=True)
json.dump({"n": len(rows), "n_16k": sum(d["samplerate"] == 16000 for d in info), "n_48k": sum(d["samplerate"] == 48000 for d in info),
           "n_48k_identical": sum(bool(d.get("librosa_first30s_identical")) for d in info),
           "n_full_copies": sum(d["mode"].startswith("FULL") for d in info),
           "short_under_30s": [d["clip"] for d in info if d["full_frames"] < 30 * d["samplerate"]],
           "librosa_spot_check_16k": chk, "librosa": librosa.__version__, "soundfile": sf.__version__,
           "zero_shot_file": ZS, "manifest_source": MAN, "clips": info},
          open(os.path.join(OUT, "stage", "cut30_report.json"), "w"), indent=1)
print("CUT30 DONE", len(rows), "full copies:", sum(d["mode"].startswith("FULL") for d in info))
