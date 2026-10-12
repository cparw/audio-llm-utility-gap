#!/usr/bin/env python
"""TASK 1b - re-run the interpretable-feature probe and the recording-property
probe on the loudness-normalised, duration-matched audio.

The question this answers: how much of each result survives once loudness and
duration can no longer help?

Four conditions are run, so that three distinct effects are not confounded:

   A  ORIGINAL audio, full original clip set        <- the published baseline
   B  ORIGINAL audio, restricted to surviving clips <- clip-selection effect
   D  CROP-ONLY audio (duration matched, original loudness)
   C  NORMALISED audio (duration matched AND loudness normalised)

   B - A = cost of dropping the clips that were shorter than the common length
   D - B = effect of duration matching alone (this also removes a lot of audio)
   C - D = effect of killing the loudness cue alone   <- the advisor's question
   C - A = the total, which is what a naive before/after comparison would show

Reporting only C vs A would blame the loudness confound for an AUC drop that is
mostly just having less audio to look at.

usage: norm_probe.py <dataset> <task> [nproc]
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, "/scratch1/parwatka/pd_probing/code_clean")
import pdstats as S
import interp_feats as IF

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = os.path.join(R, "committed")
SR = 16000

# recording-property probe: the same feature list as the original confound audit
RECFEATS = ["duration_s", "rms", "spec_centroid", "spec_rolloff", "zcr", "snr_est"]


def rec_props(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) < 400:
        return None
    Sx = np.abs(librosa.stft(y, n_fft=512))
    fr = librosa.util.frame(y, frame_length=512, hop_length=256)
    e = np.mean(fr ** 2, axis=0) + 1e-12
    return dict(duration_s=len(y) / sr,
                rms=float(np.sqrt(np.mean(y ** 2))),
                spec_centroid=float(np.mean(librosa.feature.spectral_centroid(S=Sx, sr=sr))),
                spec_rolloff=float(np.mean(librosa.feature.spectral_rolloff(S=Sx, sr=sr))),
                zcr=float(np.mean(librosa.feature.zero_crossing_rate(y))),
                snr_est=float(10 * np.log10(np.percentile(e, 90) / np.percentile(e, 10))))


def _w_interp(a):
    i, p, md = a
    try:
        d = IF.clip_features(p, md)
        return i, (d or {"error": "empty"})
    except Exception as e:
        return i, {"error": "%s: %s" % (type(e).__name__, e)}


def _w_rec(a):
    i, p = a
    try:
        d = rec_props(p)
        return i, (d or {"error": "short"})
    except Exception as e:
        return i, {"error": repr(e)}


def build(paths, md, nproc, kind):
    from multiprocessing import Pool
    jobs = [(i, p, md) for i, p in enumerate(paths)] if kind == "interp" \
        else [(i, p) for i, p in enumerate(paths)]
    fn = _w_interp if kind == "interp" else _w_rec
    res = {}
    t0 = time.time()
    with Pool(nproc) as pool:
        for k, (i, d) in enumerate(pool.imap_unordered(fn, jobs, chunksize=4)):
            res[i] = d
            if (k + 1) % 250 == 0:
                print("    %s %d/%d  %.0fs" % (kind, k + 1, len(jobs), time.time() - t0), flush=True)
    return pd.DataFrame([res[i] for i in range(len(jobs))])


def run_probe(F, cols, y, g, name):
    X = F[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).values
    p, _ = S.oof_probe(X, y, g)
    a, b = S.auc_ba(y, p)
    lo, hi = S.spk_bootstrap_auc(y, p, g, n=2000)
    nulls = S.shuffle_null_auc(X, y, g, n=50)
    pv = float((np.sum(nulls >= a) + 1) / (len(nulls) + 1))
    print("   %-46s AUC %.3f [%.3f, %.3f]  bAcc %.3f  shuffle-null %.3f+/-%.3f  p=%.3f"
          % (name, a, lo, hi, b, nulls.mean(), nulls.std(), pv), flush=True)
    return dict(probe=name, n=len(y), speakers=len(set(g)), auc=a, auc_lo=lo,
                auc_hi=hi, balacc=b, shuffle_null=float(nulls.mean()),
                shuffle_null_sd=float(nulls.std()), perm_p=pv)


def main():
    ds, task = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tag = "%s_%s" % (ds, task)
    nm = pd.read_csv(os.path.join(OUT, "manifest_norm_%s.csv" % tag))
    rep = pd.read_csv(os.path.join(OUT, "normalisation_report_%s.csv" % tag))
    orig_man = pd.read_csv(os.path.join(R, "manifests", ds + ".csv"))
    orig_man = orig_man[orig_man.task_type == task].reset_index(drop=True)
    md = IF.MAXDUR.get(ds)

    print("=" * 88)
    print("TASK 1b  post-normalisation probes   %s / %s" % (ds, task))
    print("original clips %d   surviving after duration matching %d   speakers %d"
          % (len(orig_man), len(nm), nm.speaker_id.nunique()))
    print("=" * 88, flush=True)

    surv_orig = rep["orig"].tolist()
    norm_paths = rep["norm"].tolist()
    y_s = rep["label"].astype(int).values
    g_s = rep["speaker_id"].astype(str).values
    cm = pd.read_csv(os.path.join(OUT, "manifest_croponly_%s.csv" % tag))
    crop_paths = cm["filepath"].tolist()
    assert len(crop_paths) == len(norm_paths)

    # ---- interpretable features
    fo = pd.read_csv(os.path.join(R, "soleymani_checks", "feats_%s.csv" % ds))
    fo = fo[fo.task_type == task]
    print("\n[1] interpretable features on ORIGINAL audio: reusing %s" % ("feats_%s.csv" % ds))
    fo_full = fo.reset_index(drop=True)
    fo_sub = fo.set_index("filepath").reindex(surv_orig).reset_index()
    print("[2] interpretable features on CROP-ONLY audio (%d clips)" % len(crop_paths), flush=True)
    fd_ = build(crop_paths, md, nproc, "interp")
    print("[3] interpretable features on NORMALISED audio (%d clips)" % len(norm_paths), flush=True)
    fn_ = build(norm_paths, md, nproc, "interp")
    print("    errors:", int(fn_["error"].notna().sum()) if "error" in fn_ else 0)

    # ---- recording properties
    print("[4] recording properties on ORIGINAL audio (%d clips)" % len(surv_orig), flush=True)
    ro_sub = build(surv_orig, None, nproc, "rec")
    print("[5] recording properties on CROP-ONLY audio (%d clips)" % len(crop_paths), flush=True)
    rd_ = build(crop_paths, None, nproc, "rec")
    print("[6] recording properties on NORMALISED audio (%d clips)" % len(norm_paths), flush=True)
    rn_ = build(norm_paths, None, nproc, "rec")
    print("[7] recording properties on ORIGINAL audio, FULL set (%d clips)" % len(orig_man), flush=True)
    ro_full = build(orig_man.filepath.tolist(), None, nproc, "rec")

    icols = [c for c in IF.FEATS if c in fn_.columns and c in fo_full.columns]
    print("\ninterpretable features used: %d" % len(icols))

    rows = []
    print("\n--- INTERPRETABLE-FEATURE PROBE")
    rows.append(run_probe(fo_full, icols, fo_full.label.astype(int).values,
                          fo_full.speaker_id.astype(str).values,
                          "A interp | ORIGINAL audio, full set"))
    rows.append(run_probe(fo_sub, icols, y_s, g_s,
                          "B interp | ORIGINAL audio, surviving clips"))
    rows.append(run_probe(fd_, icols, y_s, g_s,
                          "D interp | CROP-ONLY audio"))
    rows.append(run_probe(fn_, icols, y_s, g_s,
                          "C interp | NORMALISED audio"))

    print("\n--- RECORDING-PROPERTY PROBE (no speech content at all)")
    rows.append(run_probe(ro_full, RECFEATS, orig_man.label.astype(int).values,
                          orig_man.speaker_id.astype(str).values,
                          "A recprop | ORIGINAL audio, full set"))
    rows.append(run_probe(ro_sub, RECFEATS, y_s, g_s,
                          "B recprop | ORIGINAL audio, surviving clips"))
    rows.append(run_probe(rd_, RECFEATS, y_s, g_s,
                          "D recprop | CROP-ONLY audio"))
    rows.append(run_probe(rn_, RECFEATS, y_s, g_s,
                          "C recprop | NORMALISED audio"))

    # loudness / duration alone, before and after
    print("\n--- the two nuisance variables on their own")
    for nm_, col in [("duration", "duration_s"), ("loudness (rms)", "rms")]:
        for lbl, F in [("ORIGINAL", ro_sub), ("CROP-ONLY", rd_), ("NORMALISED", rn_)]:
            v = pd.to_numeric(F[col], errors="coerce").values
            if np.nanstd(v) < 1e-12:
                print("   %-14s %-11s constant across all clips -> AUC undefined (0.500 by construction)"
                      % (nm_, lbl))
                continue
            from sklearn.metrics import roc_auc_score
            print("   %-14s %-11s clip-level AUC %.3f" % (nm_, lbl, roc_auc_score(y_s, v)))

    rf = pd.DataFrame(rows)
    rf["dataset"] = ds
    rf["task"] = task
    rf.to_csv(os.path.join(OUT, "norm_probe_%s.csv" % tag), index=False)
    fn_.assign(filepath=norm_paths, label=y_s, speaker_id=g_s).to_csv(
        os.path.join(OUT, "feats_norm_%s.csv" % tag), index=False)

    print("\n" + "=" * 88)
    print("SURVIVAL SUMMARY  %s / %s" % (ds, task))
    print("=" * 88)
    for kind in ["interp", "recprop"]:
        a = rf[rf.probe.str.contains("A " + kind)].auc.iloc[0]
        b = rf[rf.probe.str.contains("B " + kind)].auc.iloc[0]
        dd = rf[rf.probe.str.contains("D " + kind)].auc.iloc[0]
        c = rf[rf.probe.str.contains("C " + kind)].auc.iloc[0]
        print("%-9s  A orig/full %.3f -> B orig/matched %.3f -> D crop-only %.3f -> C normalised %.3f"
              % (kind, a, b, dd, c))
        print("           clip-selection %+.3f | duration-match %+.3f | loudness-norm %+.3f | TOTAL %+.3f"
              % (b - a, dd - b, c - dd, c - a))
    print("\nJOB_DONE")


if __name__ == "__main__":
    main()
