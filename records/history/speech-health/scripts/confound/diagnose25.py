"""
25 test cases to pin down EXACTLY what causes the Italian artifact,
and whether any preprocessing removes it.

Part A (9 cases): preprocessing variants. Does any of them kill the artifact?
Part B (16 cases): feature diagnostics. WHICH recording property carries it?
   - each single feature alone (6)
   - leave-one-feature-out (6)
   - grouped subsets (4)

Everything speaker-disjoint, with speaker-clustered bootstrap CIs.
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

def strip_sil(y, top_db=30):
    iv = librosa.effects.split(y, top_db=top_db)
    return np.concatenate([y[a:b] for a, b in iv]) if len(iv) else y

def seg(y, sr, dur, hop=1.0):
    n, h = int(dur*sr), int(hop*sr)
    return [y[i:i+n] for i in range(0, max(1, len(y)-n+1), h)] if len(y) >= n else []

def rmsnorm(y, target=0.03):
    r = np.sqrt(np.mean(y**2)) + 1e-9
    return (y * (target/r)).astype(np.float32)

def probe(X, y, g):
    ns = min(5, len(set(g)))
    if ns < 2 or len(set(y)) < 2 or X.shape[1] == 0: return float("nan")
    a = []
    for tr, te in GroupKFold(ns).split(X, y, g):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2: continue
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        c.fit(X[tr], y[tr])
        try: a.append(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1]))
        except Exception: pass
    return float(np.mean(a)) if a else float("nan")

def boot(X, y, g, n=300, seed=0):
    rng = np.random.RandomState(seed); u = np.unique(g)
    idx = {s: np.where(g == s)[0] for s in u}; o = []
    for _ in range(n):
        p = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[s] for s in p])
        if len(set(y[ii])) < 2: continue
        v = probe(X[ii], y[ii], g[ii])
        if v == v: o.append(v)
    return (round(float(np.percentile(o,2.5)),3), round(float(np.percentile(o,97.5)),3)) if o else (None,None)

def build(rows, mode):
    X, Y, G = [], [], []
    for r in rows:
        try:
            y0, sr = librosa.load(r["filepath"], sr=16000, mono=True)
            lab, spk = int(r["label"]), r["speaker_id"]
            chunks = None
            if mode == "raw": chunks = [y0]
            elif mode == "rmsnorm": chunks = [rmsnorm(y0)]
            elif mode == "silence": chunks = [strip_sil(y0)]
            elif mode.startswith("seg"): chunks = seg(y0, sr, float(mode[3:]))
            elif mode.startswith("sil_seg"):
                chunks = seg(strip_sil(y0), sr, float(mode.split("_")[-1]))
            elif mode.startswith("silrms_seg"):
                chunks = [rmsnorm(c) for c in seg(strip_sil(y0), sr, float(mode.split("_")[-1]))]
            for c in (chunks or []):
                f = feats(c)
                if f: X.append(f); Y.append(lab); G.append(spk)
        except Exception:
            continue
    return np.array(X), np.array(Y), np.array(G)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="italian"); ap.add_argument("--task", default="read")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(f"{R}/manifests/{a.dataset}.csv")) if r["task_type"] == a.task]
    res = []; n = 0
    print("=" * 104)
    print(f"25 TEST CASES on {a.dataset} {a.task}: what causes the artifact, and does anything remove it?")
    print("=" * 104)
    print("\n--- PART A: preprocessing variants (all 6 recording features) ---")
    cache = {}
    for mode in ["raw", "rmsnorm", "silence", "seg1.0", "seg3.0", "seg5.0", "seg10.0",
                 "sil_seg_3.0", "silrms_seg_3.0"]:
        X, Y, G = build(rows, mode)
        if len(Y) < 20: print(f"  [skip] {mode}"); continue
        cache[mode] = (X, Y, G)
        auc = probe(X, Y, G); ci = boot(X, Y, G, a.boot); n += 1
        res.append(dict(case=n, part="A", name=mode, n=int(len(Y)), auc=round(auc,3), ci=ci))
        print(f"  {n:2d}. {mode:16s} n={len(Y):6d}  AUC {auc:.3f}  CI {ci}")

    print("\n--- PART B: which feature carries it? (on raw clips) ---")
    X, Y, G = cache["raw"]
    for i, f in enumerate(FE):
        auc = probe(X[:, [i]], Y, G); ci = boot(X[:, [i]], Y, G, a.boot); n += 1
        res.append(dict(case=n, part="B-single", name=f, n=int(len(Y)), auc=round(auc,3), ci=ci))
        print(f"  {n:2d}. ONLY {f:14s} AUC {auc:.3f}  CI {ci}")
    for i, f in enumerate(FE):
        k = [j for j in range(len(FE)) if j != i]
        auc = probe(X[:, k], Y, G); ci = boot(X[:, k], Y, G, a.boot); n += 1
        res.append(dict(case=n, part="B-loo", name=f"drop_{f}", n=int(len(Y)), auc=round(auc,3), ci=ci))
        print(f"  {n:2d}. DROP {f:14s} AUC {auc:.3f}  CI {ci}")
    groups = {"spectral_only": [2,3,4], "level_only": [1,5], "duration_only": [0],
              "no_duration_no_snr": [1,2,3,4]}
    for gname, k in groups.items():
        auc = probe(X[:, k], Y, G); ci = boot(X[:, k], Y, G, a.boot); n += 1
        res.append(dict(case=n, part="B-group", name=gname, n=int(len(Y)), auc=round(auc,3), ci=ci))
        print(f"  {n:2d}. {gname:20s} AUC {auc:.3f}  CI {ci}")

    os.makedirs(f"{R}/audit", exist_ok=True)
    json.dump(res, open(f"{R}/audit/diagnose25_{a.dataset}_{a.task}.json", "w"), indent=2)
    print(f"\n{n} test cases. wrote {R}/audit/diagnose25_{a.dataset}_{a.task}.json")

if __name__ == "__main__":
    main()
