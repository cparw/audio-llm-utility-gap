"""
Rigorous replacement for the naive "artifact floor gap".

Three problems with the first version:
  (1) the encoder AUC was the MAX over 33 layers, selected on the same data, so it was
      optimistically biased; the floor was a single fixed feature set. Unfair comparison.
  (2) subtracting two separately-computed AUCs is not a valid statistical test.
  (3) no confidence intervals anywhere.

This fixes all three:
  - HONEST encoder AUC via NESTED CV: the layer is chosen on inner folds of the training
    speakers only, never on the test fold.
  - Same outer folds for every model, so predictions are paired.
  - The real question asked properly: does the encoder add anything BEYOND recording
    properties? -> AUC(recording+encoder) minus AUC(recording), with a speaker-clustered
    paired bootstrap CI on the increment.
"""
import os, csv, json, argparse, warnings
import numpy as np, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
FEATS = ["duration_s", "rms", "spec_centroid", "spec_rolloff", "zcr", "snr_est"]

def rec_feats(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if len(y) < 400: return None
    S = np.abs(librosa.stft(y, n_fft=512))
    fr = librosa.util.frame(y, frame_length=512, hop_length=256)
    e = np.mean(fr ** 2, axis=0) + 1e-12
    return [len(y)/sr, float(np.sqrt(np.mean(y**2))),
            float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))),
            float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))),
            float(np.mean(librosa.feature.zero_crossing_rate(y))),
            float(10*np.log10(np.percentile(e,90)/np.percentile(e,10)))]

def cached_rec(ds, task, rows):
    cp = f"{R}/audit/recfeats_{ds}_{task}.npz"
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        return {str(k): v for k, v in zip(z["paths"], z["X"])}
    d = {}
    for r in rows:
        f = rec_feats(r["filepath"])
        if f is not None: d[r["filepath"]] = np.array(f, float)
    os.makedirs(f"{R}/audit", exist_ok=True)
    np.savez(cp, paths=np.array(list(d.keys())), X=np.array(list(d.values())))
    return d

def fit_auc(Xtr, ytr, Xte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]

def paired_boot(y, spk, preds, n=2000, seed=0):
    """CI on AUC and on pairwise increments over 'rec'."""
    rng = np.random.RandomState(seed); u = np.unique(spk)
    idx = {s: np.where(spk == s)[0] for s in u}
    acc = {k: [] for k in preds}; inc = {k: [] for k in preds if k != "rec"}
    for _ in range(n):
        p = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[s] for s in p]); yy = y[ii]
        if len(set(yy)) < 2: continue
        base = roc_auc_score(yy, preds["rec"][ii])
        for k in preds:
            a = roc_auc_score(yy, preds[k][ii]); acc[k].append(a)
            if k != "rec": inc[k].append(a - base)
    out = {}
    for k in preds:
        d = np.array(acc[k])
        out[k] = {"auc_lo": round(float(np.percentile(d, 2.5)), 3),
                  "auc_hi": round(float(np.percentile(d, 97.5)), 3)}
    for k, d in inc.items():
        d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
        out[k].update(inc_mean=round(float(np.mean(d)), 3), inc_lo=round(float(lo), 3),
                      inc_hi=round(float(hi), 3), inc_significant=bool(lo > 0))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="italian:read,kcl:read,neurovoz:read,pcgita:read,italian:vowel,neurovoz:vowel,pcgita:vowel")
    a = ap.parse_args()
    report = {}
    print("=" * 118)
    print("NESTED test: does the encoder add anything BEYOND recording properties?")
    print("layer chosen on inner folds only (no peeking). paired speaker bootstrap, 2000x")
    print("=" * 118)
    for pair in a.pairs.split(","):
        ds, task = pair.split(":")
        mp = f"{R}/manifests/{ds}.csv"
        fp = f"{R}/features/qwen2/{ds}/encoder_features.npz"
        if not (os.path.exists(mp) and os.path.exists(fp)):
            print(f"[skip] {pair}"); continue
        rows = [r for r in csv.DictReader(open(mp)) if r["task_type"] == task]
        rec = cached_rec(ds, task, rows)
        z = np.load(fp, allow_pickle=True)
        paths = z["path"] if "path" in z else z["filepath"]
        sel = np.array([i for i, p in enumerate(paths)
                        if z["task"][i] == task and str(p) in rec])
        if len(sel) < 30: print(f"[skip] {pair}: {len(sel)} clips"); continue
        y = z["label"][sel].astype(int); spk = z["speaker"][sel]
        Xr = np.array([rec[str(paths[i])] for i in sel])
        nL = int(z["n_layers"])
        L = [z[f"layer_{i:02d}"][sel] for i in range(nL)]

        oof = {k: np.zeros(len(y)) for k in ("rec", "enc", "both")}
        ns = min(5, len(set(spk)))
        for tr, te in GroupKFold(ns).split(Xr, y, spk):
            # inner layer selection on TRAIN speakers only
            best, bs = 0, -1
            itr_spk = spk[tr]
            ins = min(3, len(set(itr_spk)))
            for li in range(nL):
                sc = []
                for i2, (a2, b2) in enumerate(GroupKFold(ins).split(L[li][tr], y[tr], itr_spk)):
                    if len(set(y[tr][a2])) < 2 or len(set(y[tr][b2])) < 2: continue
                    try: sc.append(roc_auc_score(y[tr][b2], fit_auc(L[li][tr][a2], y[tr][a2], L[li][tr][b2])))
                    except Exception: pass
                m = np.mean(sc) if sc else 0
                if m > bs: bs, best = m, li
            oof["rec"][te] = fit_auc(Xr[tr], y[tr], Xr[te])
            oof["enc"][te] = fit_auc(L[best][tr], y[tr], L[best][te])
            oof["both"][te] = fit_auc(np.hstack([Xr, L[best]])[tr], y[tr], np.hstack([Xr, L[best]])[te])
        pt = {k: round(roc_auc_score(y, v), 3) for k, v in oof.items()}
        ci = paired_boot(y, spk, oof)
        report[pair] = {"n": int(len(y)), "spk": int(len(set(spk))), "point": pt, "ci": ci}
        b = ci["both"]; e = ci["enc"]
        print(f"\n{ds} {task}  ({len(y)} clips, {len(set(spk))} speakers)")
        print(f"   recording only        AUC {pt['rec']:.3f}  [{ci['rec']['auc_lo']}, {ci['rec']['auc_hi']}]")
        print(f"   encoder only (honest) AUC {pt['enc']:.3f}  [{e['auc_lo']}, {e['auc_hi']}]")
        print(f"   recording + encoder   AUC {pt['both']:.3f}  [{b['auc_lo']}, {b['auc_hi']}]")
        print(f"   >> encoder ADDS {b['inc_mean']:+.3f} over recording  CI [{b['inc_lo']}, {b['inc_hi']}]  "
              f"{'SIGNIFICANT' if b['inc_significant'] else 'NOT SIGNIFICANT'}")
    json.dump(report, open(f"{R}/audit/nested_confound_test.json", "w"), indent=2)
    print(f"\nwrote {R}/audit/nested_confound_test.json")

if __name__ == "__main__":
    main()
