#!/usr/bin/env python3
"""4a: channel normalisation, BEFORE/AFTER.

All features are computed on the 16 kHz MONO signal, because that is exactly
what the Qwen2.5-Omni audio encoder receives (extract_probe_layers.py does
librosa.load(sr=16000)). Measuring the channel at the encoder's own bandwidth
is the correct test of "could the probe be reading the recording setup".

QUIET FRAMES, stated explicitly: the signal is framed at 25 ms / 10 ms hop.
Frame energy is RMS. A frame is QUIET if its RMS is at or below the 10th
percentile of that file's frame RMS distribution. SPEECH frames are those above
the file's median RMS.

The five channel features, all from the quiet frames of that file:
  noise_floor  mean RMS over quiet frames, dBFS
  snr          10*log10(mean power over speech frames / mean power over quiet frames), dB
  centroid     mean spectral centroid over quiet frames, Hz
  rolloff      mean 85% spectral roll-off over quiet frames, Hz
  tilt         slope of a linear fit of the quiet-frame mean log-power spectrum
               (dB) against frequency over 100-7800 Hz, in dB per kHz

NORMALISATION
  1. loudness-normalise to -23 LUFS with pyloudnorm
  2. add SHAPED noise so the quiet-frame noise floor AND tilt hit the dataset's
     90th percentile of each (computed AFTER the LUFS step).

SHAPING METHOD, documented: the file's quiet-frame mean power spectrum
P_file(f) is estimated from its quiet frames. The target composite quiet-frame
spectrum is P_tgt(f) = 10**((c + m*f)/10) with m = the dataset's 90th-percentile
tilt (dB/kHz, converted to dB/Hz) and c chosen so that the total power over
100-7800 Hz equals the dataset's 90th-percentile noise floor. The noise that
must be ADDED is P_add(f) = max(0, P_tgt(f) - P_file(f)); noise can only be
added, never subtracted, so any band where the file is already above target is
left alone and counted as residual error. Gaussian white noise is then shaped
by that magnitude response with an FFT filter (frequency-domain multiply by
sqrt(P_add), inverse FFT) and added to the whole signal.
Per file we then RE-MEASURE the quiet-frame noise floor and tilt and report the
residual error against the target.

usage: chan4a.py <manifest.csv> <outdir> <tag>
manifest columns: path,label,speaker,clip
"""
import csv, json, os, sys
import numpy as np, soundfile as sf, pyloudnorm as pyln

MF, OUTD, TAG = sys.argv[1:4]
SR = 16000
NFFT = 512
HOP = int(0.010 * SR)
WIN = int(0.025 * SR)
FLO, FHI = 100.0, 7800.0
TARGET_LUFS = -23.0
rng_global = np.random.default_rng(0)

os.makedirs(f"{OUTD}/norm", exist_ok=True)

def frames(x):
    n = 1 + max(0, (len(x) - WIN) // HOP)
    if n <= 0: return np.zeros((0, WIN))
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]

def spec_frames(F):
    w = np.hanning(WIN)
    S = np.fft.rfft(F * w, n=NFFT, axis=1)
    return (np.abs(S) ** 2) + 1e-20        # power

FREQ = np.fft.rfftfreq(NFFT, 1.0 / SR)
BAND = (FREQ >= FLO) & (FREQ <= FHI)

def tilt_from_pow(P):
    """P: mean power spectrum (len = NFFT/2+1). Returns dB per kHz over the band."""
    d = 10 * np.log10(P[BAND])
    f = FREQ[BAND]
    A = np.vstack([f, np.ones_like(f)]).T
    m, _ = np.linalg.lstsq(A, d, rcond=None)[0]
    return m * 1000.0                      # dB per kHz

def feats(x):
    F = frames(x)
    if len(F) < 5:
        return None
    rms = np.sqrt((F ** 2).mean(axis=1) + 1e-20)
    db = 20 * np.log10(rms)
    q_thr = np.percentile(db, 10)
    s_thr = np.median(db)
    qi = db <= q_thr
    si = db > s_thr
    if qi.sum() < 2: qi = db <= np.percentile(db, 20)
    if si.sum() < 2: si = db >= np.percentile(db, 80)
    P = spec_frames(F)
    Pq = P[qi].mean(axis=0)
    # centroid / rolloff on quiet frames
    Pqf = P[qi]
    tot = Pqf[:, BAND].sum(axis=1) + 1e-20
    cen = (Pqf[:, BAND] * FREQ[BAND]).sum(axis=1) / tot
    cs = np.cumsum(Pqf[:, BAND], axis=1) / tot[:, None]
    ro = FREQ[BAND][np.argmax(cs >= 0.85, axis=1)]
    pw_q = (rms[qi] ** 2).mean()
    pw_s = (rms[si] ** 2).mean()
    return dict(
        noise_floor=float(db[qi].mean()),
        snr=float(10 * np.log10(pw_s / (pw_q + 1e-20) + 1e-20)),
        centroid=float(cen.mean()),
        rolloff=float(ro.mean()),
        tilt=float(tilt_from_pow(Pq)),
    ), Pq

FEATS = ["noise_floor", "snr", "centroid", "rolloff", "tilt"]

rows = list(csv.DictReader(open(MF)))
print(f"{TAG}: {len(rows)} clips", flush=True)

# ---------- pass 1: BEFORE features, and LUFS-normalised quiet spectra ----------
meter = pyln.Meter(SR)
before, Pq_after_lufs, keep, xs_lufs = [], [], [], []
for i, r in enumerate(rows):
    x, sr = sf.read(r["path"], dtype="float64")
    assert sr == SR, (r["path"], sr)
    if x.ndim > 1: x = x.mean(axis=1)
    fb = feats(x)
    if fb is None:
        print(f"  SKIP too short: {r['clip']}", flush=True); continue
    # LUFS normalise
    try:
        lo = meter.integrated_loudness(x)
        y = pyln.normalize.loudness(x, lo, TARGET_LUFS) if np.isfinite(lo) else x.copy()
    except Exception:
        y = x.copy()
    m = np.max(np.abs(y)) 
    if m > 0.999: y = y * (0.999 / m)       # prevent clipping; recorded below
    fl = feats(y)
    before.append((r, fb[0]))
    Pq_after_lufs.append(fl[1]); xs_lufs.append(y); keep.append(r)
    if i % 200 == 0: print(f"  pass1 {i+1}/{len(rows)}", flush=True)

nf_l = np.array([10 * np.log10(np.sum(p[BAND]) + 1e-20) for p in Pq_after_lufs])
tl_l = np.array([tilt_from_pow(p) for p in Pq_after_lufs])
NF_T = float(np.percentile(nf_l, 90))
TL_T = float(np.percentile(tl_l, 90))
print(f"{TAG}: dataset targets after LUFS -- noise floor band-power 90th pct {NF_T:.2f} dB, tilt 90th pct {TL_T:.3f} dB/kHz", flush=True)

# ---------- pass 2: shape noise to the targets ----------
m_hz = TL_T / 1000.0
shape_db = m_hz * FREQ[BAND]
shape_lin = 10 ** (shape_db / 10.0)
c = NF_T - 10 * np.log10(shape_lin.sum())
P_tgt = np.zeros_like(FREQ); P_tgt[BAND] = 10 ** ((c + shape_db) / 10.0)

after, resid = [], []
for k, (r, y, Pq) in enumerate(zip(keep, xs_lufs, Pq_after_lufs)):
    P_add = np.maximum(0.0, P_tgt - Pq)
    frac_band_short = float((P_add[BAND] <= 0).mean())
    rng = np.random.default_rng(1000 + k)
    n = len(y)
    L = int(2 ** np.ceil(np.log2(max(n, NFFT))))
    w = rng.standard_normal(L)
    W = np.fft.rfft(w)
    fL = np.fft.rfftfreq(L, 1.0 / SR)
    mag = np.interp(fL, FREQ, np.sqrt(P_add))
    # white noise frames have E|W|^2 = L; scale so the ADDED frame power spectrum matches P_add
    nz = np.fft.irfft(W * mag, n=L)[:n]
    ref = feats(nz)
    if ref is not None:
        cur = ref[1]
        num = np.sum(P_add[BAND]); den = np.sum(cur[BAND]) + 1e-20
        nz = nz * np.sqrt(num / den)
    z = y + nz
    mm = np.max(np.abs(z))
    if mm > 0.999: z = z * (0.999 / mm)
    fa = feats(z)
    out = f"{OUTD}/norm/{os.path.splitext(r['clip'])[0]}.flac"
    sf.write(out, z.astype(np.float32), SR)
    after.append((r, fa[0], out))
    Pq2 = fa[1]
    resid.append(dict(clip=r["clip"],
                      nf_err=float(10 * np.log10(np.sum(Pq2[BAND]) + 1e-20) - NF_T),
                      tilt_err=float(tilt_from_pow(Pq2) - TL_T),
                      band_already_above_target=frac_band_short))
    if k % 200 == 0: print(f"  pass2 {k+1}/{len(keep)}", flush=True)

nfe = np.array([d["nf_err"] for d in resid]); tle = np.array([d["tilt_err"] for d in resid])
print(f"{TAG}: residual noise-floor error  mean {nfe.mean():+.3f} dB  sd {nfe.std():.3f}  "
      f"p5 {np.percentile(nfe,5):+.3f}  p95 {np.percentile(nfe,95):+.3f}  |err|<1dB {100*(np.abs(nfe)<1).mean():.1f}%", flush=True)
print(f"{TAG}: residual tilt error         mean {tle.mean():+.4f} dB/kHz  sd {tle.std():.4f}  "
      f"p5 {np.percentile(tle,5):+.4f}  p95 {np.percentile(tle,95):+.4f}  |err|<0.1 {100*(np.abs(tle)<0.1).mean():.1f}%", flush=True)

# ---------- write per-file csv + manifest for the AFTER model run ----------
with open(f"{OUTD}/channel_norm_{TAG}.csv", "w", newline="") as f:
    cols = (["clip", "spk", "label"] + [f"{k}_before" for k in FEATS] + [f"{k}_after" for k in FEATS]
            + ["nf_err_db", "tilt_err_db_per_khz"])
    w = csv.writer(f); w.writerow(cols)
    for (r, fb), (_, fa, _), d in zip(before, after, resid):
        w.writerow([r["clip"], r["speaker"], r["label"]]
                   + [f"{fb[k]:.6f}" for k in FEATS] + [f"{fa[k]:.6f}" for k in FEATS]
                   + [f"{d['nf_err']:.4f}", f"{d['tilt_err']:.5f}"])
with open(f"{OUTD}/mf_{TAG}_norm.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "label", "speaker", "clip"])
    for r, fa, out in after: w.writerow([out, r["label"], r["speaker"], r["clip"]])

json.dump(dict(tag=TAG, n=len(after), target_lufs=TARGET_LUFS,
               noise_floor_target_db=NF_T, tilt_target_db_per_khz=TL_T,
               quiet_frame_def="frame RMS at or below the file's 10th percentile; 25 ms window, 10 ms hop",
               speech_frame_def="frame RMS above the file's median",
               band_hz=[FLO, FHI], nfft=NFFT,
               residual_nf_mean=float(nfe.mean()), residual_nf_sd=float(nfe.std()),
               residual_tilt_mean=float(tle.mean()), residual_tilt_sd=float(tle.std())),
          open(f"{OUTD}/channel_norm_{TAG}_shaping.json", "w"), indent=1)
print(f"{TAG}: CHAN DONE -> {OUTD}/channel_norm_{TAG}.csv", flush=True)
