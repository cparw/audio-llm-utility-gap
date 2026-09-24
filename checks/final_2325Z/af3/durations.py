"""Duration of every Pitt window the part10 AF3 run scored, read from the local copies of the same wavs.
Also checks each local wav's file sha256 against POD4c/work/orig_hashes.csv."""
import csv, hashlib, os, sys
import soundfile as sf
import numpy as np
D = "<local data dir>/paper1_local_runs/clips_local/clips_for_af3/pitt468"
P10 = "<local data dir>/release/overnight2/part10/af3_pitt.csv"
H = {r["clip"]: r for r in csv.DictReader(open("scores/part20/POD4c/work/orig_hashes.csv"))}
rows = list(csv.DictReader(open(P10)))
out = []
bad = 0
for r in rows:
    b = os.path.basename(r["clip_path"]); p = os.path.join(D, b)
    info = sf.info(p)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    ok = sha == H[b]["file_sha256"] and info.frames == int(H[b]["n_samples"])
    bad += not ok
    out.append((b, r["speaker_id"], r["label"], "conflict" if b.startswith("conflict") else "agreement",
                info.samplerate, info.channels, info.frames, info.frames / info.samplerate, ok))
d = np.array([o[7] for o in out])
print(f"n={len(out)} hash_or_length_mismatch={bad} sr={sorted({o[4] for o in out})} ch={sorted({o[5] for o in out})}")
print(f"min {d.min():.3f} median {np.median(d):.3f} max {d.max():.3f} s; n over 30 s = {(d > 30).sum()}")
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "speaker_id", "label", "arm", "sr", "channels", "n_samples", "dur_s", "sha256_matches_orig_hashes"])
    w.writerows(out)
med = np.median(d)
for name, target in (("near 30 s", 30.0), ("near median", med), ("longest", d.max())):
    i = int(np.argmin(np.abs(d - target)))
    print(name, out[i][0], f"{out[i][7]:.3f} s", out[i][3])
