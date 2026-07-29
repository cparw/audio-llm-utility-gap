"""
Does Maxim's preprocessing explain why he never saw the near-perfect AUC?

His pipeline (from /project2/msoleyma_946/msiniukov/soundrepr/constructdataset.py):
    strip_silence_simple(waveform, sr, mode=0/1)     # remove silence
    'segment_duration': 3, 'segment_step': 1          # fixed 3s segments, 1s hop

Our audit found the Italian artifact is driven mainly by DURATION
(PD 70.9s vs HC 50.3s) and SNR (51.6 vs 35.0). Both of those are destroyed by
silence-stripping + fixed-length segmentation: every segment becomes exactly 3s,
and the silence that drives the SNR estimate is gone.

So: recompute the ARTIFACT FLOOR (recording-properties-only AUC, no speech content)
under (a) raw clips, and (b) Maxim-style preprocessing. If the floor collapses,
that is the explanation.
"""
import csv, os, json, argparse, warnings
import numpy as np, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R = "/project2/msoleyma_946/speech_health/results_chaitanya"
FE = ["duration_s", "rms", "spec_centroid", "spec_rolloff", "zcr", "snr_est"]

def feats(y, sr=16000):
    if len(y) < 400: return None
    S = np.abs(librosa.stft(y, n_fft=512))
    fr = librosa.util.frame(y, frame_length=512, hop_length=256)
    e = np.mean(fr ** 2, axis=0) + 1e-12
    return [len(y)/sr, float(np.sqrt(np.mean(y**2))),
            float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))),
            float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))),
            float(np.mean(librosa.feature.zero_crossing_rate(y))),
            float(10*np.log10(np.percentile(e,90)/np.percentile(e,10)))]

def strip_silence(y, top_db=30):
    iv = librosa.effects.split(y, top_db=top_db)
    return np.concatenate([y[a:b] for a, b in iv]) if len(iv) else y

def segment(y, sr=16000, dur=3.0, hop=1.0):
    n, h = int(dur*sr), int(hop*sr)
    return [y[i:i+n] for i in range(0, max(1, len(y)-n+1), h)] if len(y) >= n else []

def probe(X, y, g):
    ns = min(5, len(set(g)))
    if ns < 2 or len(set(y)) < 2: return float("nan")
    a = []
    for tr, te in GroupKFold(ns).split(X, y, g):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2: continue
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        c.fit(X[tr], y[tr])
        try: a.append(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1]))
        except Exception: pass
    return float(np.mean(a)) if a else float("nan")

def boot(X, y, g, n=1000, seed=0):
    rng = np.random.RandomState(seed); u = np.unique(g)
    idx = {s: np.where(g == s)[0] for s in u}; out = []
    for _ in range(n):
        p = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[s] for s in p])
        if len(set(y[ii])) < 2: continue
        out.append(probe(X[ii], y[ii], g[ii]))
    out = np.array([v for v in out if v == v])
    return (round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)) if len(out) else (None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="italian,neurovoz,pcgita")
    ap.add_argument("--task", default="read")
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    rep = {}
    print("=" * 100)
    print("ARTIFACT FLOOR: raw clips  vs  Maxim-style (silence stripped + fixed 3s segments)")
    print("=" * 100)
    for ds in a.datasets.split(","):
        mp = f"{R}/manifests/{ds}.csv"
        if not os.path.exists(mp): continue
        rows = [r for r in csv.DictReader(open(mp)) if r["task_type"] == a.task]
        Xr, yr, gr, Xs, ys, gs = [], [], [], [], [], []
        dur_raw = {0: [], 1: []}; dur_seg = {0: [], 1: []}
        for r in rows:
            try:
                y0, sr = librosa.load(r["filepath"], sr=16000, mono=True)
                lab = int(r["label"]); spk = r["speaker_id"]
                f = feats(y0)
                if f: Xr.append(f); yr.append(lab); gr.append(spk); dur_raw[lab].append(len(y0)/sr)
                ys_ = strip_silence(y0)
                for seg in segment(ys_):
                    fs = feats(seg)
                    if fs: Xs.append(fs); ys.append(lab); gs.append(spk); dur_seg[lab].append(len(seg)/sr)
            except Exception:
                continue
        if len(set(yr)) < 2 or len(set(ys)) < 2:
            print(f"[skip] {ds}"); continue
        Xr, yr, gr = np.array(Xr), np.array(yr), np.array(gr)
        Xs, ys, gs = np.array(Xs), np.array(ys), np.array(gs)
        ar, as_ = probe(Xr, yr, gr), probe(Xs, ys, gs)
        cr, cs = boot(Xr, yr, gr, a.boot), boot(Xs, ys, gs, a.boot)
        rep[ds] = {"raw": {"auc": round(ar, 3), "ci": cr, "n": int(len(yr))},
                   "maxim_style": {"auc": round(as_, 3), "ci": cs, "n": int(len(ys))},
                   "mean_dur_raw_PD": round(float(np.mean(dur_raw[1])), 1),
                   "mean_dur_raw_HC": round(float(np.mean(dur_raw[0])), 1)}
        print(f"\n{ds} ({a.task})")
        print(f"   raw clips        n={len(yr):5d}  artifact floor {ar:.3f}  CI {cr}   "
              f"(dur PD {np.mean(dur_raw[1]):.1f}s vs HC {np.mean(dur_raw[0]):.1f}s)")
        print(f"   Maxim-style      n={len(ys):5d}  artifact floor {as_:.3f}  CI {cs}   (all segments 3.0s)")
        print(f"   >> floor drops by {ar-as_:+.3f}")
    os.makedirs(f"{R}/audit", exist_ok=True)
    json.dump(rep, open(f"{R}/audit/maxim_repro_{a.task}.json", "w"), indent=2)
    print(f"\nwrote {R}/audit/maxim_repro_{a.task}.json")

if __name__ == "__main__":
    main()
