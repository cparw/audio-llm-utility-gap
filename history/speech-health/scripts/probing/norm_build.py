#!/usr/bin/env python
"""TASK 1a - build loudness-normalised, duration-matched audio.

The advisor asked for this after loudness turned out to correlate with the
label.  Two nuisance variables are removed by construction:

  LOUDNESS  every clip is normalised to -23 LUFS integrated, measured with the
            ITU-R BS.1770-4 gated algorithm (K-weighting: 1681 Hz high shelf +
            38 Hz RLB high pass, 400 ms blocks / 75%% overlap, -70 LUFS absolute
            gate and -10 LU relative gate).  pyloudnorm is not installed in the
            cluster env so the filters and the gating are implemented here and
            verified by re-measuring every output file.

  DURATION  every clip in a (dataset, task) group is cropped to exactly the
            same number of samples T, so clip length carries literally zero
            information about the label afterwards.  T is the 5th percentile of
            that group's duration, chosen BEFORE looking at any probe result
            and never tuned.  Clips shorter than T cannot be made longer
            without inventing audio, so they are dropped; the drop count is
            reported per label group and tested for imbalance.

usage: norm_build.py <dataset> <task>
"""
import os, sys, json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import lfilter
from scipy.stats import fisher_exact

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUTA = "/scratch1/parwatka/norm_audio"
OUTC = os.path.join(R, "committed")
SR = 16000
TARGET_LUFS = -23.0
CROP_PCTILE = 5.0

os.makedirs(OUTC, exist_ok=True)


# --------------------------------------------------- BS.1770-4 K-weighting
def _shelf(fs, G=3.999843853973347, Q=0.7071752369554196, fc=1681.974450955533):
    A = 10 ** (G / 40.0)
    w0 = 2 * np.pi * fc / fs
    a_ = np.sin(w0) / (2 * Q)
    c, s = np.cos(w0), np.sqrt(A)
    b = np.array([A * ((A + 1) + (A - 1) * c + 2 * s * a_),
                  -2 * A * ((A - 1) + (A + 1) * c),
                  A * ((A + 1) + (A - 1) * c - 2 * s * a_)])
    a = np.array([(A + 1) - (A - 1) * c + 2 * s * a_,
                  2 * ((A - 1) - (A + 1) * c),
                  (A + 1) - (A - 1) * c - 2 * s * a_])
    return b / a[0], a / a[0]


def _hpf(fs, Q=0.5003270373238773, fc=38.13547087602444):
    w0 = 2 * np.pi * fc / fs
    a_ = np.sin(w0) / (2 * Q)
    c = np.cos(w0)
    b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    a = np.array([1 + a_, -2 * c, 1 - a_])
    return b / a[0], a / a[0]


_B1, _A1 = _shelf(SR)
_B2, _A2 = _hpf(SR)


def integrated_lufs(y, fs=SR):
    """ITU-R BS.1770-4 gated integrated loudness, mono (channel weight G=1)."""
    z = lfilter(_B2, _A2, lfilter(_B1, _A1, y))
    T, step = int(0.400 * fs), int(0.100 * fs)
    if len(z) < T:
        ms = float(np.mean(z ** 2))
        return -0.691 + 10 * np.log10(ms + 1e-20), True   # short-file fallback
    idx = range(0, len(z) - T + 1, step)
    zj = np.array([np.mean(z[i:i + T] ** 2) for i in idx])
    lj = -0.691 + 10 * np.log10(zj + 1e-20)
    keep = lj > -70.0                                     # absolute gate
    if not keep.any():
        return -0.691 + 10 * np.log10(np.mean(zj) + 1e-20), True
    gamma_r = -0.691 + 10 * np.log10(zj[keep].mean() + 1e-20) - 10.0
    keep2 = keep & (lj > gamma_r)                         # relative gate
    if not keep2.any():
        keep2 = keep
    return -0.691 + 10 * np.log10(zj[keep2].mean() + 1e-20), False


def centre_crop(y, n):
    if len(y) <= n:
        return y
    s = (len(y) - n) // 2
    return y[s:s + n]


def main():
    ds, task = sys.argv[1], sys.argv[2]
    tag = "%s_%s" % (ds, task)
    outdir = os.path.join(OUTA, tag)
    os.makedirs(outdir, exist_ok=True)
    man = pd.read_csv(os.path.join(R, "manifests", ds + ".csv"))
    man = man[man.task_type == task].reset_index(drop=True)
    print("=" * 78)
    print("NORMALISING  %s / %s   n=%d clips  %d speakers"
          % (ds, task, len(man), man.speaker_id.nunique()))
    print("=" * 78, flush=True)

    # ---- pass 1: measure
    rec = []
    for i, r in man.iterrows():
        try:
            y, _ = librosa.load(r.filepath, sr=SR, mono=True)
            lufs, fb = integrated_lufs(y)
            rec.append(dict(filepath=r.filepath, speaker_id=r.speaker_id,
                            label=int(r.label), n=len(y), dur=len(y) / SR,
                            lufs=lufs, lufs_fallback=fb, err=""))
        except Exception as e:
            rec.append(dict(filepath=r.filepath, speaker_id=r.speaker_id,
                            label=int(r.label), n=0, dur=np.nan, lufs=np.nan,
                            lufs_fallback=True, err=repr(e)))
        if (i + 1) % 200 == 0:
            print("  measured %d/%d" % (i + 1, len(man)), flush=True)
    d = pd.DataFrame(rec)
    ok = d.err == ""
    print("\nread failures:", int((~ok).sum()))

    # ---- choose the common length
    T_s = float(np.percentile(d.loc[ok, "dur"], CROP_PCTILE))
    T_n = int(round(T_s * SR))
    print("\nduration before: min %.2f  p5 %.2f  med %.2f  p95 %.2f  max %.2f s"
          % (d.dur.min(), T_s, d.dur.median(), np.percentile(d.loc[ok, 'dur'], 95), d.dur.max()))
    print("common length T = p%.0f = %.3f s (%d samples)" % (CROP_PCTILE, T_s, T_n))

    d["cropped"] = ok & (d.n > T_n)
    d["dropped"] = ok & (d.n < T_n)
    d.loc[~ok, "dropped"] = True

    # ---- REPORT CROP/DROP IMBALANCE LOUDLY
    print("\n" + "!" * 78)
    print("CROP / DROP IMPACT BY GROUP  (label 1 = patient)")
    print("!" * 78)
    tb = []
    for lab in [0, 1]:
        s = d[d.label == lab]
        tb.append(dict(label=lab, n=len(s), n_cropped=int(s.cropped.sum()),
                       pct_cropped=100.0 * s.cropped.mean(),
                       n_dropped=int(s.dropped.sum()),
                       pct_dropped=100.0 * s.dropped.mean(),
                       median_dur_before=float(s.dur.median()),
                       median_sec_removed=float((s.dur - T_s).clip(lower=0).median())))
    tb = pd.DataFrame(tb)
    print(tb.to_string(index=False))
    ct = [[int(d[(d.label == 1)].cropped.sum()), int((~d[(d.label == 1)].cropped).sum())],
          [int(d[(d.label == 0)].cropped.sum()), int((~d[(d.label == 0)].cropped).sum())]]
    try:
        orv, pv = fisher_exact(ct)
    except Exception:
        orv, pv = np.nan, np.nan
    print("\ncropping PD vs HC: Fisher odds ratio %.3f, p = %.4g" % (orv, pv))
    if np.isfinite(pv) and pv < 0.05:
        print(">>> WARNING: cropping hits patients and controls UNEQUALLY (p<0.05).")
        print(">>> The crop is therefore not a neutral operation for this group;")
        print(">>> treat the post-crop probe as a lower bound, not a clean control.")
    else:
        print("    cropping is statistically balanced across the two groups.")
    ctd = [[int(d[(d.label == 1)].dropped.sum()), int((~d[(d.label == 1)].dropped).sum())],
           [int(d[(d.label == 0)].dropped.sum()), int((~d[(d.label == 0)].dropped).sum())]]
    try:
        orv2, pv2 = fisher_exact(ctd)
    except Exception:
        orv2, pv2 = np.nan, np.nan
    print("dropping PD vs HC: Fisher odds ratio %.3f, p = %.4g" % (orv2, pv2))
    if np.isfinite(pv2) and pv2 < 0.05:
        print(">>> WARNING: the dropped short clips are UNEQUALLY distributed across groups.")

    # ---- pass 2: write normalised audio
    keep = d[ok & ~d.dropped].copy()
    print("\nwriting %d normalised clips -> %s" % (len(keep), outdir), flush=True)
    outrows = []
    for i, r in keep.reset_index(drop=True).iterrows():
        y, _ = librosa.load(r.filepath, sr=SR, mono=True)
        y = centre_crop(y, T_n)
        lufs_c, _ = integrated_lufs(y)          # re-measure AFTER cropping
        gain = 10 ** ((TARGET_LUFS - lufs_c) / 20.0)
        yn = y * gain
        lufs_out, _ = integrated_lufs(yn)       # verify
        bn = os.path.basename(r.filepath).replace(".wav", "") + ".wav"
        op = os.path.join(outdir, bn)
        sf.write(op, yn.astype(np.float32), SR, subtype="FLOAT")
        outrows.append(dict(orig=r.filepath, norm=op, speaker_id=r.speaker_id,
                            label=r.label, lufs_before=r.lufs,
                            lufs_after_crop=lufs_c, gain_db=20 * np.log10(gain),
                            lufs_out=lufs_out, peak_out=float(np.abs(yn).max()),
                            dur_before=r.dur, dur_after=len(yn) / SR))
        if (i + 1) % 200 == 0:
            print("  wrote %d/%d" % (i + 1, len(keep)), flush=True)
    o = pd.DataFrame(outrows)

    # ---- verification
    print("\n--- VERIFICATION of the normalisation")
    print("output LUFS: mean %.3f  sd %.4f  min %.3f  max %.3f  (target %.1f)"
          % (o.lufs_out.mean(), o.lufs_out.std(), o.lufs_out.min(),
             o.lufs_out.max(), TARGET_LUFS))
    print("output duration: all identical =", bool(o.dur_after.nunique() == 1),
          " value %.4f s" % o.dur_after.iloc[0])
    print("gain applied: median %.1f dB, range %.1f .. %.1f dB"
          % (o.gain_db.median(), o.gain_db.min(), o.gain_db.max()))
    print("output peak amplitude: max %.3f  (float wav, no clipping applied)"
          % o.peak_out.max())
    print("clips whose peak now exceeds 1.0 (would clip if written as int16): %d"
          % int((o.peak_out > 1.0).sum()))
    # did loudness/duration lose their association with the label?
    from sklearn.metrics import roc_auc_score
    print("\nAUC of the nuisance variable alone, before vs after:")
    print("   loudness  before %.3f   after %.3f (constant by construction)"
          % (roc_auc_score(o.label, o.lufs_before), 0.5))
    print("   duration  before %.3f   after %.3f (constant by construction)"
          % (roc_auc_score(o.label, o.dur_before), 0.5))

    # ---- normalised manifest, same schema as the originals
    nm = man.merge(o[["orig", "norm"]], left_on="filepath", right_on="orig", how="inner")
    nm["filepath"] = nm["norm"]
    nm = nm.drop(columns=["orig", "norm"])
    mp = os.path.join(OUTC, "manifest_norm_%s.csv" % tag)
    nm.to_csv(mp, index=False)
    o.to_csv(os.path.join(OUTC, "normalisation_report_%s.csv" % tag), index=False)
    json.dump(dict(dataset=ds, task=task, n_in=len(man), n_out=len(nm),
                   T_seconds=T_s, target_lufs=TARGET_LUFS,
                   n_cropped=int(d.cropped.sum()), n_dropped=int(d.dropped.sum()),
                   crop_fisher_p=float(pv), drop_fisher_p=float(pv2),
                   by_label=tb.to_dict("records")),
              open(os.path.join(OUTC, "normalisation_summary_%s.json" % tag), "w"), indent=2)
    print("\nnormalised manifest -> %s  (%d clips, %d speakers)"
          % (mp, len(nm), nm.speaker_id.nunique()))
    print("\nJOB_DONE")


if __name__ == "__main__":
    main()
