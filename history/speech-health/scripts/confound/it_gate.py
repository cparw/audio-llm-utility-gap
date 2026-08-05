"""
ITALIAN CONFOUND step 7: is the low patient noise floor GAIN or GATING?

  GAIN  -> speech and silence are both lowered by the same number of dB; loudness
           normalisation would fix it. (Prior work says loudness normalisation did NOT fix it,
           so this should NOT be what we see.)
  GATE / noise reduction -> silence is pushed down far more than speech, i.e. the
           speech-to-floor distance grows. Loudness normalisation cannot fix that, because it
           is a ratio, not a level.

Also counts hard digital-zero runs inside the silence: a gate that closes fully writes exact
zeros, which no acoustic recording chain ever produces.
"""
import csv, json, warnings
import numpy as np
import soundfile as sf
from scipy import signal as sps
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/final"
SR, WIN, HOP = 16000, 400, 160


def proc(path):
    try:
        xi, sr0 = sf.read(path, dtype="int16", always_2d=True)
        xi = xi[:, 0]
    except Exception:
        return None
    y = xi.astype(np.float64) / 32768.0
    if sr0 != SR:
        y = sps.resample_poly(y, SR, sr0)
    n = 1 + max(0, (len(y) - WIN)) // HOP
    if n < 10:
        return None
    idx = np.arange(WIN)[:, None] + HOP * np.arange(n)[None, :]
    fr = y[idx]
    db = 10 * np.log10(np.mean(fr ** 2, axis=0) + 1e-12)
    nf, l95 = np.percentile(db, 10), np.percentile(db, 95)
    speech = db > (l95 - 10.0)
    quiet = db < (nf + 6.0)
    # hard digital-zero runs on the NATIVE int16 stream
    z = (xi == 0).astype(np.int8)
    dz = np.diff(np.r_[0, z, 0])
    starts, ends = np.where(dz == 1)[0], np.where(dz == -1)[0]
    runs = ends - starts
    thr = int(0.005 * sr0)          # 5 ms of consecutive exact zeros
    return dict(
        speech_rms_dbfs=float(10 * np.log10(np.mean(fr[:, speech] ** 2) + 1e-14)) if speech.any() else np.nan,
        quiet_rms_dbfs=float(10 * np.log10(np.mean(fr[:, quiet] ** 2) + 1e-14)) if quiet.any() else np.nan,
        p99_dbfs=float(np.percentile(db, 99)),
        floor_dbfs=float(nf),
        speech_minus_floor_db=float(l95 - nf),
        n_zero_runs_5ms=float((runs >= thr).sum()),
        longest_zero_run_ms=float(runs.max() / sr0 * 1000) if len(runs) else 0.0,
        frac_zero=float(np.mean(z)),
        frac_in_zero_runs_5ms=float(runs[runs >= thr].sum() / len(xi)) if len(runs) else 0.0,
    )


def main():
    rows = list(csv.DictReader(open(f"{OUT}/italian_design.csv")))
    res = Parallel(n_jobs=16, verbose=1)(delayed(proc)(r["filepath"]) for r in rows)
    keep = [(r, d) for r, d in zip(rows, res) if d is not None]
    keys = sorted(keep[0][1].keys())
    S = np.array([[d[k] for k in keys] for _, d in keep])
    lab = np.array([int(r["label"]) for r, _ in keep])
    enc = np.array([r["enc"] for r, _ in keep])
    grp = np.array([r["group"] for r, _ in keep])
    date = np.array([r["date"] for r, _ in keep])
    with open(f"{OUT}/italian_gate_stats.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filepath", "label", "enc", "group", "date"] + keys)
        for (r, d), s in zip(keep, S):
            w.writerow([r["filepath"], r["label"], r["enc"], r["group"], r["date"]] + list(s))

    def med(m, k):
        return np.nanmedian(S[m, keys.index(k)])

    print(f"n={len(keep)}\n")
    order = ["speech_rms_dbfs", "quiet_rms_dbfs", "speech_minus_floor_db", "floor_dbfs",
             "p99_dbfs", "frac_zero", "frac_in_zero_runs_5ms", "n_zero_runs_5ms",
             "longest_zero_run_ms"]
    print("### GAIN vs GATE, medians by LABEL")
    print(f"  {'stat':26s}{'control':>13}{'patient':>13}{'PD - ctrl':>13}")
    for k in order:
        c, p = med(lab == 0, k), med(lab == 1, k)
        print(f"  {k:26s}{c:>13.4g}{p:>13.4g}{p-c:>13.4g}")

    print("\n### by GROUP DIRECTORY")
    print(f"  {'group':34s}{'n':>6}" + "".join(f"{k[:11]:>13}" for k in order[:5]))
    for gname in sorted(set(grp)):
        m = grp == gname
        print(f"  {gname[:32]:34s}{m.sum():>6}" + "".join(f"{med(m,k):>13.4g}" for k in order[:5]))

    print("\n### by ENCODER TOOLCHAIN")
    print(f"  {'encoder':34s}{'n':>6}" + "".join(f"{k[:11]:>13}" for k in order))
    for e in sorted(set(enc)):
        m = enc == e
        print(f"  {e[:32]:34s}{m.sum():>6}" + "".join(f"{med(m,k):>13.4g}" for k in order))

    print("\n### by RECORDING SESSION (date)")
    print(f"  {'date':12s}{'lab':>5}{'n':>6}" + "".join(f"{k[:11]:>13}" for k in order[:5]))
    for dt in sorted(set(date)):
        m = date == dt
        print(f"  {dt:12s}{lab[m][0]:>5}{m.sum():>6}" + "".join(f"{med(m,k):>13.4g}" for k in order[:5]))

    # how separable is each single statistic, speaker level
    from sklearn.metrics import roc_auc_score
    spk = np.array([r["spk"] for r, _ in keep])
    us = np.unique(spk)
    sl = np.array([lab[spk == s][0] for s in us])
    print("\n### speaker-level AUC of each single statistic (n=%d speakers)" % len(us))
    for k in order:
        v = np.array([np.nanmedian(S[spk == s, keys.index(k)]) for s in us])
        ok = np.isfinite(v)
        try:
            a = roc_auc_score(sl[ok], v[ok])
        except Exception:
            a = np.nan
        print(f"  {k:26s} AUC={a:.3f}   (|AUC-0.5|={abs(a-0.5):.3f})")
    print("\nwrote", f"{OUT}/italian_gate_stats.csv")
    print("JOB_DONE")


if __name__ == "__main__":
    main()
