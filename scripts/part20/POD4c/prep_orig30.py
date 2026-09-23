"""Decode the FLAC-transported original 468 Pitt windows to 16 kHz PCM_16 WAV, check each clip's
int16 sample sha256 against orig_hashes.csv (made on the Mac from clips_for_af3/pitt468/*.wav,
which are byte-identical to the cut builder's source segments), build the control scoring manifest."""
import csv, hashlib, os, sys
import soundfile as sf
D = "/workspace/orig30"
H = {r["clip"]: r for r in csv.DictReader(open(f"{D}/orig_hashes.csv"))}
ARM = {os.path.basename(r["segment_path"]): r["set"] for r in csv.DictReader(open("/workspace/refs/pitt_conflict_manifest.csv"))}
cut = {r["orig_clip_id"]: r for r in csv.DictReader(open("/workspace/cut/manifest.csv"))}
os.makedirs(f"{D}/wav", exist_ok=True)
bad = []
with open(f"{D}/score_manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["clip_id", "orig_clip_id", "path", "label", "speaker", "arm"])
    for c in sorted(H):
        x, sr = sf.read(f"{D}/flac/{c[:-4]}.flac", dtype="int16")
        if sr != 16000 or hashlib.sha256(x.tobytes()).hexdigest() != H[c]["int16_samples_sha256"] or len(x) != int(H[c]["n_samples"]):
            bad.append(c); continue
        p = f"{D}/wav/{c}"; sf.write(p, x, 16000, subtype="PCM_16")
        w.writerow([c, c, p, cut[c]["label"], cut[c]["spk"], ARM[c]])
print(f"orig {len(H)} ok {len(H)-len(bad)} bad {len(bad)} {bad[:5]}")
sys.exit(1 if bad or len(H) != 468 else 0)
