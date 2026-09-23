"""PART24 smoke data: 10 synthetic noise clips (two 900 s, eight 20 s), manifest and fold file in the same format
as mf_whole.csv / edaic_groupkfold5_pod.csv. Fold 0 = test (s02 900 s, s03, s04, s05; labels 0,1,0,1)."""
import os, csv, numpy as np, soundfile as sf
D = "/workspace/smoke"; os.makedirs(D, exist_ok=True)
rs = np.random.RandomState(0)
spec = [("s01", 900, 1, 1), ("s02", 900, 0, 0), ("s03", 20, 1, 0), ("s04", 20, 0, 0), ("s05", 20, 1, 0),
        ("s06", 20, 0, 1), ("s07", 20, 1, 1), ("s08", 20, 0, 1), ("s09", 20, 1, 1), ("s10", 20, 0, 1)]
with open(f"{D}/mf.csv", "w", newline="") as a, open(f"{D}/folds.csv", "w", newline="") as b:
    wa, wb = csv.writer(a), csv.writer(b); wa.writerow(["path", "label", "speaker", "pid"]); wb.writerow(["clip_id", "speaker_id", "fold"])
    for pid, sec, lab, fold in spec:
        sf.write(f"{D}/{pid}.wav", (rs.randn(16000 * sec) * 0.05).astype(np.float32), 16000, subtype="PCM_16")
        wa.writerow([f"{D}/{pid}.wav", lab, pid, pid]); wb.writerow([f"{pid}.wav", pid, fold])
print("smoke data ok", len(spec))
