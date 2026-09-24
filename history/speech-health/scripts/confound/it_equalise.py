"""
ITALIAN CONFOUND step 6: can the artifact be removed at all?

Intervention, in the AUDIO domain, not the feature domain:
  for every clip, measure its own silence PSD, synthesise additive noise whose PSD is
  exactly (corpus-wide target PSD - own PSD), clipped at zero, and add it. Afterwards
  every clip's background floor equals the same target spectrum, so the noise-floor
  level AND shape are identical across groups by construction.
  target = per-bin 97.5th percentile over the corpus + 3 dB (so no clip needs subtraction).

Then re-run, on the SAME clips, before and after:
  (a) the silence-only probe (the artifact)      -> should collapse if the fix works
  (b) a speech-only probe (MFCC over SPEECH frames) -> shows what the fix costs
Speaker-disjoint throughout, speaker-level shuffle null.
"""
import os, csv, json, warnings, time
import numpy as np
import soundfile as sf
from scipy import signal as sps
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/final"
SR, WIN, HOP, NPERSEG = 16000, 400, 160, 1024
NB = NPERSEG // 2 + 1


def load(path):
    x, sr0 = sf.read(path, dtype="float64", always_2d=True)
    x = x[:, 0]
    if sr0 != SR:
        x = sps.resample_poly(x, SR, sr0)
    return x


def segments(y):
    n = 1 + max(0, (len(y) - WIN)) // HOP
    if n == 0:
        return [], []
    idx = np.arange(WIN)[:, None] + HOP * np.arange(n)[None, :]
    db = 10 * np.log10(np.mean(y[idx] ** 2, axis=0) + 1e-12)
    nf, l95 = np.percentile(db, 10), np.percentile(db, 95)
    if (l95 - nf) < 18.0:
        return [], []
    raw = db < (nf + 6.0)
    sil, i = [], 0
    while i < len(raw):
        if not raw[i]:
            i += 1; continue
        j = i
        while j < len(raw) and raw[j]:
            j += 1
        if (j - i) >= 15:
            a, b = i + 3, j - 3
            if (b - a) >= 9:
                sil.append((a * HOP, (b - 1) * HOP + WIN))
        i = j
    sp = np.where(db > (l95 - 10.0))[0]           # speech frames: top 10 dB of the clip
    return sil, sp


def psd_of(chunks):
    ps, ws = [], []
    for s in chunks:
        if len(s) < NPERSEG:
            continue
        f, P = sps.welch(s, SR, nperseg=NPERSEG, noverlap=NPERSEG // 2, detrend=False)
        ps.append(P); ws.append(len(s))
    if not ps:
        return None
    return np.average(np.array(ps), axis=0, weights=np.array(ws, float))


def feats(y, sil, sp):
    """(silence env features, speech mfcc features) for one clip."""
    out_s = out_p = None
    ch = [y[a:b] for a, b in sil]
    P = psd_of(ch)
    if P is not None:
        lp = 10 * np.log10(P + 1e-20)
        cat = np.concatenate(ch)
        out_s = np.concatenate([lp, [10 * np.log10(np.mean(cat ** 2) + 1e-20)]])
    if len(sp) > 5:
        n = 1 + max(0, (len(y) - WIN)) // HOP
        idx = np.arange(WIN)[:, None] + HOP * np.arange(n)[None, :]
        fr = y[idx][:, sp]
        w = np.hanning(WIN)[:, None]
        S = np.abs(np.fft.rfft(fr * w, n=512, axis=0)) ** 2
        fb = mel_fb()
        M = 10 * np.log10(fb @ S + 1e-12)
        from scipy.fftpack import dct
        C = dct(M, type=2, axis=0, norm="ortho")[:13]
        out_p = np.concatenate([C.mean(1), C.std(1)])
    return out_s, out_p


_FB = None
def mel_fb():
    global _FB
    if _FB is None:
        import librosa
        _FB = librosa.filters.mel(sr=SR, n_fft=512, n_mels=40)
    return _FB


def pass1(path):
    try:
        y = load(path)
    except Exception:
        return None
    sil, sp = segments(y)
    if not sil:
        return None
    P = psd_of([y[a:b] for a, b in sil])
    return P


def pass2(path, target):
    """add shaped noise so the silence PSD becomes `target`, then recompute features."""
    try:
        y = load(path)
    except Exception:
        return None
    sil, sp = segments(y)
    if not sil:
        return None
    f0, p0 = feats(y, sil, sp)
    P = psd_of([y[a:b] for a, b in sil])
    if P is None:
        return None
    need = np.maximum(target - P, 0.0)
    rng = np.random.RandomState(abs(hash(path)) % 2**31)
    w = rng.randn(len(y))
    Wf = np.fft.rfft(w)
    fbins = np.fft.rfftfreq(len(y), 1.0 / SR)
    src = np.fft.rfftfreq(NPERSEG, 1.0 / SR)
    # welch PSD of unit-variance white noise is 1/(SR/2) per Hz -> scale accordingly
    gain = np.sqrt(np.interp(fbins, src, need) * (SR / 2.0))
    noise = np.fft.irfft(Wf * gain, n=len(y))
    yn = y + noise
    f1, p1 = feats(yn, *segments(yn)) if segments(yn)[0] else (None, None)
    return f0, p0, f1, p1


def oof(X, y, g, ns=5):
    ns = min(ns, len(np.unique(g)))
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(ns).split(X, y, g):
        assert not (set(g[tr]) & set(g[te]))
        if len(np.unique(y[tr])) < 2:
            continue
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        c.fit(X[tr], y[tr]); p[te] = c.predict_proba(X[te])[:, 1]
    return p


def report(tag, X, y, g, nsh=100):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    p = oof(X, y, g); m = ~np.isnan(p)
    auc = roc_auc_score(y[m], p[m])
    d = {}
    for gi, yi, pi in zip(g[m], y[m], p[m]):
        d.setdefault(gi, [yi, []])[1].append(pi)
    ys = np.array([v[0] for v in d.values()]); ps = np.array([np.mean(v[1]) for v in d.values()])
    sa = roc_auc_score(ys, ps)
    rng = np.random.RandomState(4); spk = np.unique(g)
    lab = np.array([y[g == s][0] for s in spk]); nul = []
    for _ in range(nsh):
        mp = dict(zip(spk, rng.permutation(lab)))
        yy = np.array([mp[s] for s in g]); pp = oof(X, yy, g); mm = ~np.isnan(pp)
        if len(np.unique(yy[mm])) > 1:
            nul.append(roc_auc_score(yy[mm], pp[mm]))
    nul = np.array(nul)
    pv = (np.sum(nul >= auc) + 1) / (len(nul) + 1)
    # speaker bootstrap
    rng2 = np.random.RandomState(6); idx = {s: np.where(g[m] == s)[0] for s in np.unique(g[m])}
    sp_ = list(idx); a = []
    for _ in range(2000):
        sel = np.concatenate([idx[sp_[k]] for k in rng2.choice(len(sp_), len(sp_), True)])
        if len(np.unique(y[m][sel])) > 1:
            a.append(roc_auc_score(y[m][sel], p[m][sel]))
    lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
    print(f"  {tag:56s} n={len(y):4d} spk={len(np.unique(g)):3d}  AUC={auc:.3f} [{lo:.3f},{hi:.3f}]"
          f"  spkAUC={sa:.3f}  null={nul.mean():.3f}(p95 {np.percentile(nul,95):.3f})  p={pv:.3f}")
    return dict(tag=tag, n=int(len(y)), n_spk=int(len(np.unique(g))), auc=round(float(auc), 4),
                ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4), spk_auc=round(float(sa), 4),
                shuffle_mean=round(float(nul.mean()), 4), perm_p=round(float(pv), 4))


def main():
    rows = list(csv.DictReader(open(f"{OUT}/italian_design.csv")))
    t0 = time.time()
    P1 = Parallel(n_jobs=16, verbose=1)(delayed(pass1)(r["filepath"]) for r in rows)
    good = [p for p in P1 if p is not None]
    A = np.array(good)
    target = np.percentile(A, 97.5, axis=0) * 10 ** (3.0 / 10.0)
    print(f"pass1 {len(good)}/{len(rows)} clips, target floor median "
          f"{np.median(10*np.log10(target)):.1f} dB/Hz  ({time.time()-t0:.0f}s)")
    P2 = Parallel(n_jobs=16, verbose=1)(delayed(pass2)(r["filepath"], target) for r in rows)

    keep, S0, SP0, S1, SP1 = [], [], [], [], []
    for r, o in zip(rows, P2):
        if o is None:
            continue
        f0, p0, f1, p1 = o
        if f0 is None or f1 is None or p0 is None or p1 is None:
            continue
        keep.append(r); S0.append(f0); SP0.append(p0); S1.append(f1); SP1.append(p1)
    S0, SP0, S1, SP1 = map(np.array, (S0, SP0, S1, SP1))
    y = np.array([int(r["label"]) for r in keep])
    g = np.array([r["spk"] for r in keep])
    task = np.array([r["task"] for r in keep])
    enc = np.array([r["enc"] for r in keep]); sr = np.array([r["sr"] for r in keep])
    print(f"pass2 usable both before and after: {len(y)} clips, "
          f"{len(np.unique(g))} speakers  ({time.time()-t0:.0f}s)")

    res = []
    print("\n=== BEFORE vs AFTER noise-floor equalisation (same clips throughout) ===")
    res.append(report("SILENCE probe  BEFORE", S0, y, g))
    res.append(report("SILENCE probe  AFTER  (floors equalised)", S1, y, g))
    res.append(report("SPEECH-frame MFCC probe  BEFORE", SP0, y, g))
    res.append(report("SPEECH-frame MFCC probe  AFTER", SP1, y, g))
    m = (enc == "bare_fmt_data") & (sr == "16000")
    if m.sum() > 30 and len(np.unique(y[m])) > 1:
        print("  -- matched container (bare fmt|data, 16 kHz) --")
        res.append(report("SILENCE  matched container  BEFORE", S0[m], y[m], g[m]))
        res.append(report("SILENCE  matched container  AFTER", S1[m], y[m], g[m]))
        res.append(report("SPEECH   matched container  BEFORE", SP0[m], y[m], g[m]))
        res.append(report("SPEECH   matched container  AFTER", SP1[m], y[m], g[m]))
    mr = task == "read"
    if mr.sum() > 30:
        print("  -- task=read --")
        res.append(report("SILENCE  read  BEFORE", S0[mr], y[mr], g[mr]))
        res.append(report("SILENCE  read  AFTER", S1[mr], y[mr], g[mr]))
        res.append(report("SPEECH   read  BEFORE", SP0[mr], y[mr], g[mr]))
        res.append(report("SPEECH   read  AFTER", SP1[mr], y[mr], g[mr]))

    with open(f"{OUT}/italian_equalisation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    np.savez_compressed(f"{OUT}/italian_equalise_feats.npz", S0=S0, S1=S1, SP0=SP0, SP1=SP1,
                        y=y, g=g, task=task, enc=enc, sr=sr, target=target)
    print(f"\nwrote {OUT}/italian_equalisation.csv")
    print("JOB_DONE")


if __name__ == "__main__":
    main()
