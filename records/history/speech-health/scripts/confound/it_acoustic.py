"""
ITALIAN CONFOUND step 3: acoustic + byte-level forensics of the SILENT portions.

Per file we compute, using the SAME silence rule as silence_extract.py:
  - Welch log-PSD of the silence only, 16 kHz, nperseg=1024 -> 15.625 Hz bins, 513 bins
  - DC offset of the silence (mean sample value), in LSB and as dBFS
  - byte/quantisation forensics on the RAW int16 stream (no resampling, no float):
      n distinct sample values, LSB parity entropy, gcd of the samples (a gcd>1 means the
      stream was scaled from a smaller-range source), fraction of exact zeros, clipping,
      whether the last N samples are exactly zero (encoder padding)
  - full-signal spectral cutoff: highest frequency holding >0.1% of peak band energy,
    which exposes a lossy-codec / resampler brickwall
Everything is written to one npz for the probing stage.
"""
import os, sys, csv, json, warnings, time
import numpy as np
import soundfile as sf
from scipy import signal as sps
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/final"
os.makedirs(OUT, exist_ok=True)

SR = 16000
WIN, HOP = 400, 160
NPERSEG = 1024                      # 15.625 Hz bins at 16 kHz
NBINS = NPERSEG // 2 + 1
FLOOR_MARGIN_DB, DR_MIN_DB = 6.0, 18.0
MIN_RUN_MS, ERODE_MS, MIN_SEG_MS = 150.0, 30.0, 90.0


def frame_db(y):
    n = 1 + max(0, (len(y) - WIN)) // HOP
    if len(y) < WIN:
        return np.zeros(0, np.float32), 0
    idx = np.arange(WIN)[:, None] + HOP * np.arange(n)[None, :]
    fr = y[idx]
    e = np.mean(fr.astype(np.float64) ** 2, axis=0)
    return (10 * np.log10(e + 1e-12)).astype(np.float32), n


def silence_segments(y):
    db, n = frame_db(y)
    if n == 0:
        return [], -1
    nf, l95 = np.percentile(db, 10), np.percentile(db, 95)
    dr = float(l95 - nf)
    if dr < DR_MIN_DB:
        return [], dr
    raw = db < (nf + FLOOR_MARGIN_DB)
    mr, er, ms = int(MIN_RUN_MS / 10), int(ERODE_MS / 10), int(MIN_SEG_MS / 10)
    segs, i = [], 0
    while i < len(raw):
        if not raw[i]:
            i += 1; continue
        j = i
        while j < len(raw) and raw[j]:
            j += 1
        if (j - i) >= mr:
            a, b = i + er, j - er
            if (b - a) >= ms:
                segs.append((a * HOP, (b - 1) * HOP + WIN))
        i = j
    return segs, dr


def load16(path):
    """raw int16 at native rate, plus a 16 kHz float copy."""
    x, sr0 = sf.read(path, dtype="int16", always_2d=True)
    x = x[:, 0]
    yf = x.astype(np.float64) / 32768.0
    if sr0 != SR:
        yf = sps.resample_poly(yf, SR, sr0)
    return x, sr0, yf.astype(np.float32)


def byte_forensics(x):
    d = {}
    d["n_samples_native"] = len(x)
    d["dc_offset_lsb_full"] = float(np.mean(x))
    d["n_distinct"] = float(len(np.unique(x)))
    nz = x[x != 0]
    d["frac_zero"] = float(np.mean(x == 0))
    d["n_clip"] = float(np.sum(np.abs(x) >= 32767))
    if len(nz):
        g = np.gcd.reduce(np.abs(nz).astype(np.int64))
        d["sample_gcd"] = float(g)
    else:
        d["sample_gcd"] = 0.0
    d["lsb_odd_frac"] = float(np.mean((x & 1) != 0))
    # trailing exact-zero run (encoder padding)
    k = 0
    while k < len(x) and x[len(x) - 1 - k] == 0:
        k += 1
    d["trail_zero_samples"] = float(k)
    k = 0
    while k < len(x) and x[k] == 0:
        k += 1
    d["lead_zero_samples"] = float(k)
    return d


def cutoff_hz(yf):
    f, P = sps.welch(yf, SR, nperseg=2048)
    P = P / (P.max() + 1e-30)
    idx = np.where(P > 1e-3)[0]
    return float(f[idx[-1]]) if len(idx) else 0.0


def process(path):
    try:
        x, sr0, yf = load16(path)
    except Exception as e:
        return None
    d = byte_forensics(x)
    d["sr_native"] = float(sr0)
    d["cutoff_hz"] = cutoff_hz(yf)
    d["dur_s"] = len(yf) / SR
    segs, dr = silence_segments(yf)
    d["dyn_range_db"] = dr
    sil = [yf[a:b] for a, b in segs if (b - a) >= NPERSEG]
    d["n_sil_seg"] = float(len(sil))
    d["sil_total_s"] = float(sum(b - a for a, b in segs)) / SR
    if not sil:
        return path, d, None
    psds, wts = [], []
    for s in sil:
        f, P = sps.welch(s.astype(np.float64), SR, nperseg=NPERSEG,
                         noverlap=NPERSEG // 2, detrend=False, scaling="density")
        psds.append(P); wts.append(len(s))
    wts = np.array(wts, float)
    P = np.average(np.array(psds), axis=0, weights=wts)
    logP = 10 * np.log10(P + 1e-20)
    cat = np.concatenate(sil)
    d["dc_offset_lsb_sil"] = float(np.mean(cat) * 32768.0)
    d["sil_rms_dbfs"] = float(10 * np.log10(np.mean(cat ** 2) + 1e-20))
    return path, d, logP.astype(np.float32)


def main():
    rows = list(csv.DictReader(open(f"{OUT}/italian_design.csv")))
    t0 = time.time()
    res = Parallel(n_jobs=16, backend="loky", verbose=2)(
        delayed(process)(r["filepath"]) for r in rows)
    scal_keys = None
    keep, S, PS = [], [], []
    for r, out in zip(rows, res):
        if out is None:
            continue
        p, d, logP = out
        if scal_keys is None:
            scal_keys = sorted(d.keys())
        keep.append(r)
        S.append([d.get(k, np.nan) for k in scal_keys])
        PS.append(logP if logP is not None else np.full(NBINS, np.nan, np.float32))
    S = np.array(S, np.float64); PS = np.array(PS, np.float32)
    f = np.fft.rfftfreq(NPERSEG, 1.0 / SR)
    meta = {k: np.array([r[k] for r in keep]) for k in keep[0].keys()}
    np.savez_compressed(f"{OUT}/italian_silence_psd.npz", S=S, PS=PS, freqs=f,
                        scal_keys=np.array(scal_keys), **meta)
    print(f"processed {len(keep)}/{len(rows)} in {time.time()-t0:.0f}s")
    print(f"PSD bins {PS.shape}  usable-PSD files {int(np.isfinite(PS[:,1]).sum())}")

    lab = meta["label"].astype(int)
    enc = meta["enc"]; task = meta["task"]
    def show(name, vals, mask=None):
        m = np.isfinite(vals) if mask is None else (np.isfinite(vals) & mask)
        print(f"  {name:26s} control {np.median(vals[m & (lab==0)]):>12.4g}   "
              f"patient {np.median(vals[m & (lab==1)]):>12.4g}")
    print("\n### byte/quantisation forensics, MEDIANS")
    for i, k in enumerate(scal_keys):
        show(k, S[:, i])
    print("\n### by encoder toolchain (medians)")
    hdr = ["n", "sample_gcd", "lsb_odd_frac", "n_distinct", "dc_offset_lsb_full",
           "cutoff_hz", "sil_rms_dbfs", "frac_zero", "trail_zero_samples"]
    print(f"{'encoder':34s}" + "".join(f"{h[:12]:>14}" for h in hdr))
    for e in sorted(set(enc)):
        m = enc == e
        vals = [int(m.sum())] + [np.nanmedian(S[m, scal_keys.index(h)]) for h in hdr[1:]]
        print(f"{e[:32]:34s}" + "".join(f"{v:>14.4g}" for v in vals))
    print("\nwrote", f"{OUT}/italian_silence_psd.npz")
    print("JOB_DONE")


if __name__ == "__main__":
    main()
