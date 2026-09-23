#!/usr/bin/env python3
"""POD6 verifier: own implementation of the five quiet-frame channel features,
written from the definitions (16 kHz mono; 25 ms frames, 10 ms hop; quiet frames
= frame RMS dB at or below the file's 10th percentile; speech frames = above the
median; band 100-7800 Hz; 512-point FFT with a Hann window).
usage: POD6_verify_chan.py LIST.csv(clip,path16k) OUT.csv
"""
import sys, csv
import numpy as np, soundfile as sf

SR, W, H, NF = 16000, 400, 160, 512
FR = np.fft.rfftfreq(NF, 1 / SR); B = (FR >= 100) & (FR <= 7800)


def five(x):
    n = 1 + (len(x) - W) // H
    F = np.stack([x[i * H:i * H + W] for i in range(n)])
    e = np.sqrt(np.mean(F ** 2, axis=1) + 1e-20); d = 20 * np.log10(e)
    q = d <= np.percentile(d, 10); s = d > np.median(d)
    if q.sum() < 2: q = d <= np.percentile(d, 20)
    if s.sum() < 2: s = d >= np.percentile(d, 80)
    P = np.abs(np.fft.rfft(F * np.hanning(W), n=NF, axis=1)) ** 2 + 1e-20
    Pb = P[q][:, B]; fb = FR[B]
    tot = Pb.sum(axis=1) + 1e-20
    cen = (Pb * fb).sum(axis=1) / tot
    cum = np.cumsum(Pb, axis=1) / tot[:, None]
    ro = fb[np.argmax(cum >= 0.85, axis=1)]
    Pm = P[q].mean(axis=0)
    slope = np.polyfit(fb, 10 * np.log10(Pm[B]), 1)[0] * 1000
    return dict(noise_floor=d[q].mean(),
                snr=10 * np.log10((e[s] ** 2).mean() / ((e[q] ** 2).mean() + 1e-20) + 1e-20),
                centroid=cen.mean(), rolloff=ro.mean(), tilt=slope)


if __name__ == "__main__":
    rows = list(csv.DictReader(open(sys.argv[1])))
    with open(sys.argv[2], "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "noise_floor", "snr", "centroid", "rolloff", "tilt"])
        for r in rows:
            x, sr = sf.read(r["path16k"], dtype="float64")
            assert sr == SR
            if x.ndim > 1: x = x.mean(axis=1)
            f = five(x)
            w.writerow([r["clip"]] + [f"{f[k]:.6f}" for k in ("noise_floor", "snr", "centroid", "rolloff", "tilt")])
