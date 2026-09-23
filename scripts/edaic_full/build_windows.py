"""Rebuild E-DAIC full windows: stitch dcaps_proc-style, then cut windows_full.csv spans.
Stitch rule (empirically confirmed against src_dur_s, 275/275):
  concatenate raw_audio[Start_Time:End_Time] for every transcript row with End_Time > Start_Time,
  in file order, no speaker filter, no duration filter, plain slicing truncated at end-of-audio.
Then cut [start_s, end_s] in STITCHED time at the native sample rate, then resample to 16 kHz mono.
"""
import os, csv, sys, json, numpy as np, soundfile as sf, librosa
from multiprocessing import Pool

D = "/workspace/edaicfull"
os.makedirs(f"{D}/cut", exist_ok=True)

WIN = {r["pid"]: r for r in csv.DictReader(open(f"{D}/windows_full.csv"))}

def one(pid):
    r = WIN[pid]
    out = f"{D}/cut/{pid}.wav"
    try:
        x, sr = sf.read(f"{D}/wav/{pid}_AUDIO.wav", dtype="float32", always_2d=True)
        rows = list(csv.DictReader(open(f"{D}/tr/{pid}_Transcript.csv")))
        segs = []
        for t in rows:
            try:
                a, b = float(t["Start_Time"]), float(t["End_Time"])
            except Exception:
                continue
            if b > a:
                segs.append(x[int(a * sr):int(b * sr)])
        st = np.concatenate(segs, axis=0) if segs else x[:0]
        stitched_s = len(st) / sr
        # cut the window in stitched time at native sr
        s0, s1 = float(r["start_s"]), float(r["end_s"])
        w = st[int(s0 * sr):int(s1 * sr)]
        # downmix to mono then resample to 16 kHz (librosa.load semantics)
        m = w.mean(axis=1) if w.shape[1] > 1 else w[:, 0]
        m16 = librosa.resample(m, orig_sr=sr, target_sr=16000) if sr != 16000 else m
        sf.write(out, m16, 16000, subtype="PCM_16")
        return dict(pid=pid, ok=1, native_sr=sr, ch=x.shape[1], raw_s=round(len(x)/sr, 4),
                    stitched_s=round(stitched_s, 4), src_dur_s=float(r["src_dur_s"]),
                    d_stitch=round(stitched_s - float(r["src_dur_s"]), 4),
                    win_s=round(len(m16)/16000, 4), win_dur_s=float(r["win_dur_s"]),
                    d_win=round(len(m16)/16000 - float(r["win_dur_s"]), 4),
                    capped=int(r["capped"]), sr_expected=int(r["sr"]))
    except Exception as e:
        return dict(pid=pid, ok=0, err=repr(e)[:200])

if __name__ == "__main__":
    pids = list(WIN.keys())
    with Pool(12) as p:
        res = p.map(one, pids, chunksize=1)
    bad = [r for r in res if not r["ok"]]
    good = [r for r in res if r["ok"]]
    with open(f"{D}/build_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(good[0].keys())); w.writeheader()
        for r in sorted(good, key=lambda z: int(z["pid"])): w.writerow(r)
    ds = np.array([r["d_stitch"] for r in good]); dw = np.array([r["d_win"] for r in good])
    print(f"built {len(good)} / {len(pids)}   failures {len(bad)}", flush=True)
    for b in bad: print("FAIL", b, flush=True)
    print(f"stitched vs src_dur_s : max|d| {np.abs(ds).max():.4f}  n within 0.05s {(np.abs(ds)<0.05).sum()}/{len(ds)}")
    print(f"window   vs win_dur_s : max|d| {np.abs(dw).max():.4f}  n within 0.05s {(np.abs(dw)<0.05).sum()}/{len(dw)}")
    srm = sum(1 for r in good if r["native_sr"] != r["sr_expected"])
    print(f"native sr != windows_full.csv sr : {srm}")
    print("worst stitch:", sorted(good, key=lambda z: -abs(z["d_stitch"]))[:3])
    print("worst window:", sorted(good, key=lambda z: -abs(z["d_win"]))[:3])
