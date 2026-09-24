"""
Soleymani's two remaining manipulations, applied to the probe that reaches 0.995:

  "add some noise and change the result... at some point it's not changing the
   result, there is something else about the data that's causing it"
  "or shuffle them, like copy from one person to another, does it change the result"

Test 1: add noise at several SNRs, recompute recording features, re-probe.
        If the AUC does NOT degrade, the confound is structural in the data.
Test 2: shuffle audio across speakers (permute the label-to-audio assignment at the
        SPEAKER level). If the pipeline is sound this must fall to chance.
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

def add_noise(y, snr_db, rng):
    p = np.mean(y**2) + 1e-12
    n = rng.randn(len(y)).astype(np.float32)
    n *= np.sqrt(p / (10**(snr_db/10.0)) / (np.mean(n**2)+1e-12))
    return (y + n).astype(np.float32)

def probe(X, y, g):
    ns = min(5, len(set(g)))
    if ns < 2 or len(set(y)) < 2: return float("nan")
    a=[]
    for tr, te in GroupKFold(ns).split(X, y, g):
        if len(set(y[tr]))<2 or len(set(y[te]))<2: continue
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        c.fit(X[tr], y[tr])
        try: a.append(roc_auc_score(y[te], c.predict_proba(X[te])[:,1]))
        except Exception: pass
    return float(np.mean(a)) if a else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="italian,neurovoz")
    ap.add_argument("--task", default="read")
    a = ap.parse_args()
    out = {}
    for ds in a.datasets.split(","):
        rows = [r for r in csv.DictReader(open(f"{R}/manifests/{ds}.csv")) if r["task_type"] == a.task]
        rng = np.random.RandomState(0)
        base, Y, G = [], [], []
        noisy = {s: [] for s in (20, 10, 0)}
        for r in rows:
            try:
                y0, sr = librosa.load(r["filepath"], sr=16000, mono=True)
                f = feats(y0)
                if f is None: continue
                base.append(f); Y.append(int(r["label"])); G.append(r["speaker_id"])
                for s in noisy:
                    fn = feats(add_noise(y0, s, rng))
                    noisy[s].append(fn if fn else f)
            except Exception:
                continue
        if len(set(Y)) < 2: continue
        X = np.array(base); Y = np.array(Y); G = np.array(G)
        res = {"clean": round(probe(X, Y, G), 3)}
        for s in sorted(noisy, reverse=True):
            res[f"noise_{s}dB"] = round(probe(np.array(noisy[s]), Y, G), 3)
        # speaker-level label shuffle (audio "copied" between people)
        sp = np.unique(G)
        lab_by = {s: Y[G == s][0] for s in sp}
        aucs = []
        for seed in range(20):
            rr = np.random.RandomState(seed)
            perm = rr.permutation(sp)
            mapping = {s: lab_by[p] for s, p in zip(sp, perm)}
            Ys = np.array([mapping[s] for s in G])
            if len(set(Ys)) < 2: continue
            aucs.append(probe(X, Ys, G))
        res["speaker_shuffled"] = round(float(np.nanmean(aucs)), 3)
        out[ds] = res
        print(f"\n=== {ds} ({a.task}, n={len(Y)}, {len(sp)} speakers) ===")
        print(f"   clean                {res['clean']}")
        for s in sorted(noisy, reverse=True):
            print(f"   + noise at {s:>2} dB SNR   {res[f'noise_{s}dB']}")
        print(f"   audio shuffled across speakers (20 perms)  {res['speaker_shuffled']}  (chance = 0.5)")
    os.makedirs(f"{R}/audit", exist_ok=True)
    json.dump(out, open(f"{R}/audit/noise_shuffle_{a.task}.json", "w"), indent=2)
    print(f"\nwrote {R}/audit/noise_shuffle_{a.task}.json")

if __name__ == "__main__":
    main()
