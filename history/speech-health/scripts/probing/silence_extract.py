"""
SILENCE-ONLY ARTIFACT PROBE  --  stage 1: feature extraction.

Advisor's ask: "If there is any silence in these recordings, just extract the silence
and then do a classification on the silence. Because ... the system essentially detects
the background noise rather than anything in the voice."

This script finds the NON-SPEECH frames of every clip and computes an acoustic
description of those frames only. No speech content is involved by construction.

EXACT SILENCE RULE (stated so it can be checked):
  1. load mono @16 kHz, no normalisation of any kind (absolute level is part of the artifact)
  2. frame: 25 ms window (400 samples), 10 ms hop (160 samples), no centring
  3. frame level in dB = 10*log10(mean(x^2) + 1e-12)
  4. NF = 10th percentile of frame dB in that clip   (the clip's own noise floor)
     L95 = 95th percentile of frame dB in that clip  (the clip's own speech level)
     dynamic range DR = L95 - NF
  4b. if DR < 18 dB the clip has no separable quiet portion at all (e.g. a sustained
     vowel recorded with no leading/trailing pause) -> the clip is DISCARDED as
     "no_dynamic_range". Nothing is invented for such clips.
  5. a frame is NON-SPEECH if its level < NF + 6 dB
     (i.e. within 6 dB of the clip's own floor -- clip-relative, so it is invariant to
     per-file gain / normalisation, which is exactly the artifact we are hunting)
  6. keep only RUNS of >= 150 ms of consecutive non-speech frames -> genuine pauses,
     not inter-word gaps or stop closures
  7. erode 30 ms off each end of every run -> removes speech onset/offset bleed and
     reverberation tails
  8. what survives (>= 90 ms per segment) is the silence of the clip
  9. clips with < 0.3 s of total surviving silence are DISCARDED (counted and reported)

FEATURES (computed on the silence frames ONLY; averaging over the silence frames is
equivalent to concatenating the silence, but avoids the splice discontinuities that
literal concatenation would create):
  noise_floor_db, sil_rms_db, sil_rms_db_p90, sil_db_iqr,
  sil_centroid, sil_rolloff, sil_flatness_db, sil_bandwidth, sil_zcr,
  sil_lowband_frac (energy < 300 Hz, catches mains hum / HVAC),
  sil_hf_frac (energy > 4 kHz, catches hiss / codec), mfcc1..13 means

Also recorded but NOT used in the headline probe (these are pause BEHAVIOUR, which is a
real PD symptom, not a recording artifact): sil_frac, sil_total_s, clip_dur_s, n_segments.

usage: silence_extract.py <dataset>
"""
import os, sys, csv, warnings, time
import numpy as np
import librosa
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/soleymani_checks"

SR = 16000
WIN = 400          # 25 ms
HOP = 160          # 10 ms
NFFT = 512
FLOOR_MARGIN_DB = 6.0   # non-speech if below (NF + FLOOR_MARGIN_DB)
DR_MIN_DB = 18.0        # clip needs this much L95-NF range to contain real silence
MIN_RUN_MS = 150.0
ERODE_MS = 30.0
MIN_SEG_MS = 90.0
MIN_SIL_S = 0.30
CHUNK_FRAMES = 6000   # 60 s of frames per chunk, bounds memory on the 20-min DAIC files

ENV_FEATS = (["noise_floor_db", "sil_rms_db", "sil_rms_db_p90", "sil_db_iqr",
              "sil_centroid", "sil_rolloff", "sil_flatness_db", "sil_bandwidth",
              "sil_zcr", "sil_lowband_frac", "sil_hf_frac"]
             + [f"mfcc{i+1}" for i in range(13)])
PAUSE_FEATS = ["sil_frac", "sil_total_s", "clip_dur_s", "n_segments", "mean_seg_s", "dyn_range_db"]


def frame_db(y):
    """frame level in dB for the whole signal, exact framing used everywhere below."""
    n = 1 + max(0, (len(y) - WIN)) // HOP
    if len(y) < WIN:
        return np.zeros(0, dtype=np.float32), 0
    fr = librosa.util.frame(y[:(n - 1) * HOP + WIN], frame_length=WIN, hop_length=HOP)
    e = np.mean(fr.astype(np.float64) ** 2, axis=0)
    return (10.0 * np.log10(e + 1e-12)).astype(np.float32), n


def silence_mask(db):
    """steps 4-8 of the rule. returns boolean mask over frames + segment list."""
    if db.size == 0:
        return np.zeros(0, bool), [], 0.0
    nf = float(np.percentile(db, 10))
    l95 = float(np.percentile(db, 95))
    if (l95 - nf) < DR_MIN_DB:
        return np.zeros(0, bool), [], l95 - nf
    thr = nf + FLOOR_MARGIN_DB
    raw = db < thr
    min_run = int(round(MIN_RUN_MS / 10.0))     # frames
    erode = int(round(ERODE_MS / 10.0))
    min_seg = int(round(MIN_SEG_MS / 10.0))
    mask = np.zeros_like(raw)
    segs = []
    i, n = 0, len(raw)
    while i < n:
        if not raw[i]:
            i += 1
            continue
        j = i
        while j < n and raw[j]:
            j += 1
        if (j - i) >= min_run:
            a, b = i + erode, j - erode
            if (b - a) >= min_seg:
                mask[a:b] = True
                segs.append((a, b))
        i = j
    return mask, segs, l95 - nf


def spectral_stats(y, mask, n_frames):
    """average the spectral descriptors over the silence frames only, chunked."""
    acc = {k: [] for k in ["cen", "rol", "fla", "bw", "zcr", "low", "hf"]}
    mfcc_acc = []
    mel_fb = None
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    lo_bins = freqs < 300.0
    hi_bins = freqs > 4000.0
    for c0 in range(0, n_frames, CHUNK_FRAMES):
        c1 = min(n_frames, c0 + CHUNK_FRAMES)
        m = mask[c0:c1]
        if not m.any():
            continue
        s0 = c0 * HOP
        s1 = (c1 - 1) * HOP + WIN
        seg = y[s0:s1]
        S = np.abs(librosa.stft(seg, n_fft=NFFT, win_length=WIN, hop_length=HOP,
                                center=False, window="hann"))
        k = min(S.shape[1], len(m))
        S = S[:, :k][:, m[:k]]
        if S.shape[1] == 0:
            continue
        acc["cen"].append(librosa.feature.spectral_centroid(S=S, sr=SR)[0])
        acc["rol"].append(librosa.feature.spectral_rolloff(S=S, sr=SR)[0])
        acc["fla"].append(librosa.feature.spectral_flatness(S=S)[0])
        acc["bw"].append(librosa.feature.spectral_bandwidth(S=S, sr=SR)[0])
        P = S ** 2
        tot = P.sum(axis=0) + 1e-20
        acc["low"].append(P[lo_bins].sum(axis=0) / tot)
        acc["hf"].append(P[hi_bins].sum(axis=0) / tot)
        if mel_fb is None:
            mel_fb = librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=40)
        M = librosa.power_to_db(mel_fb @ P + 1e-12)
        mfcc_acc.append(librosa.feature.mfcc(S=M, n_mfcc=13))
        # zcr on the same frames
        fr = librosa.util.frame(seg[:(k - 1) * HOP + WIN], frame_length=WIN, hop_length=HOP)
        fr = fr[:, :k][:, m[:k]]
        acc["zcr"].append(np.mean(np.abs(np.diff(np.sign(fr), axis=0)) > 0, axis=0))
    if not acc["cen"]:
        return None
    out = {k: float(np.mean(np.concatenate(v))) for k, v in acc.items()}
    mf = np.concatenate(mfcc_acc, axis=1).mean(axis=1)
    return out, mf


def process(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        return None, f"load_fail:{type(e).__name__}"
    if len(y) < WIN * 4:
        return None, "too_short"
    db, n = frame_db(y)
    if n == 0:
        return None, "too_short"
    mask, segs, dr = silence_mask(db)
    sil_s = float(mask.sum()) * HOP / SR
    dur = len(y) / SR
    meta = dict(sil_frac=sil_s / max(dur, 1e-9), sil_total_s=sil_s, clip_dur_s=dur,
                n_segments=float(len(segs)), dyn_range_db=dr,
                mean_seg_s=float(np.mean([(b - a) * HOP / SR for a, b in segs])) if segs else 0.0)
    if len(mask) == 0:
        return None, ("no_dynamic_range", meta)
    if sil_s < MIN_SIL_S:
        return None, ("low_silence", meta)
    st = spectral_stats(y, mask, n)
    if st is None:
        return None, ("low_silence", meta)
    sp, mf = st
    sdb = db[mask]
    vec = [float(np.percentile(sdb, 10)),                                 # noise_floor_db
           float(10 * np.log10(np.mean(10 ** (sdb / 10)) + 1e-12)),       # sil_rms_db
           float(np.percentile(sdb, 90)),
           float(np.percentile(sdb, 75) - np.percentile(sdb, 25)),
           sp["cen"], sp["rol"],
           float(10 * np.log10(sp["fla"] + 1e-12)),
           sp["bw"], sp["zcr"], sp["low"], sp["hf"]] + [float(v) for v in mf]
    return (np.array(vec, dtype=np.float32),
            np.array([meta[k] for k in PAUSE_FEATS], dtype=np.float32)), "ok"


def main():
    ds = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(f"{R}/manifests/{ds}.csv")))
    print(f"[{ds}] {len(rows)} clips in manifest", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=8, backend="loky", verbose=5)(
        delayed(process)(r["filepath"]) for r in rows)
    X, P, lab, grp, task, keep = [], [], [], [], [], []
    disc = []
    for r, (v, status) in zip(rows, res):
        if v is None:
            reason = status[0] if isinstance(status, tuple) else status
            m = status[1] if isinstance(status, tuple) else {}
            disc.append(dict(filepath=r["filepath"], speaker_id=r["speaker_id"],
                             label=r["label"], task_type=r["task_type"], reason=reason,
                             sil_total_s=round(m.get("sil_total_s", -1), 4),
                             clip_dur_s=round(m.get("clip_dur_s", -1), 3),
                             dyn_range_db=round(m.get("dyn_range_db", -1), 2)))
            continue
        X.append(v[0]); P.append(v[1])
        lab.append(int(r["label"])); grp.append(r["speaker_id"]); task.append(r["task_type"])
    X = np.array(X, dtype=np.float32); P = np.array(P, dtype=np.float32)
    np.savez_compressed(f"{OUT}/silence_feats_{ds}.npz", X=X, P=P,
                        y=np.array(lab), g=np.array(grp), task=np.array(task),
                        env_feats=np.array(ENV_FEATS), pause_feats=np.array(PAUSE_FEATS))
    with open(f"{OUT}/silence_discards_{ds}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "speaker_id", "label", "task_type",
                                          "reason", "sil_total_s", "clip_dur_s",
                                          "dyn_range_db"])
        w.writeheader(); w.writerows(disc)
    print(f"[{ds}] kept {len(X)} / {len(rows)}  discarded {len(disc)}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    if len(X):
        print(f"[{ds}] silence fraction: mean {P[:,0].mean():.3f} median "
              f"{np.median(P[:,0]):.3f} | silence sec: median {np.median(P[:,1]):.2f}", flush=True)


if __name__ == "__main__":
    main()
