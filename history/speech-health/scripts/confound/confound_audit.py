"""
Dataset confound audit (Soleymani, Jul 17: audit datasets for confounding factors).

Question: how much of the "disease" label is predictable from RECORDING PROPERTIES
alone, with no speech content at all? If a probe on duration/loudness/sample-rate/
spectral stats reaches high AUC, then the near-perfect within-dataset AUC is an
artifact of how the two groups were recorded, not the disease.

Outputs per dataset:
  - per-class means of each recording property (so you can see WHICH one differs)
  - speaker-disjoint AUC of a probe using ONLY those properties
  - a label-shuffle control (should sit at ~0.5, proves the pipeline is sane)
"""
import os, csv, argparse, json, warnings
import numpy as np, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

FEATS = ["duration_s", "rms", "orig_sr", "spec_centroid", "spec_rolloff", "zcr", "snr_est"]

def feats_for(path, orig_sr):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if len(y) < 400: return None
    rms = float(np.sqrt(np.mean(y ** 2)))
    S = np.abs(librosa.stft(y, n_fft=512))
    cen = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    rol = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    # crude SNR: energy of loud frames vs quiet frames
    fr = librosa.util.frame(y, frame_length=512, hop_length=256)
    e = np.mean(fr ** 2, axis=0) + 1e-12
    hi, lo = np.percentile(e, 90), np.percentile(e, 10)
    snr = float(10 * np.log10(hi / lo))
    return dict(duration_s=len(y) / sr, rms=rms, orig_sr=float(orig_sr or 16000),
                spec_centroid=cen, spec_rolloff=rol, zcr=zcr, snr_est=snr)

def probe(X, y, g, seed=0):
    ns = min(5, len(set(g)))
    if ns < 2 or len(set(y)) < 2: return float("nan")
    aucs = []
    for tr, te in GroupKFold(ns).split(X, y, g):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2: continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        try: aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        except Exception: pass
    return float(np.mean(aucs)) if aucs else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="italian,kcl,neurovoz,pcgita")
    ap.add_argument("--tasks", default="read")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    R = "/project2/msoleyma_946/speech_health/results_chaitanya"
    keep = set(a.tasks.split(","))
    report = {}

    for ds in a.datasets.split(","):
        mp = f"{R}/manifests/{ds}.csv"
        if not os.path.exists(mp):
            print(f"[skip] no manifest {ds}"); continue
        rows = [r for r in csv.DictReader(open(mp)) if r["task_type"] in keep]
        if a.limit: rows = rows[:a.limit]
        X, y, g, M = [], [], [], []
        for r in rows:
            try:
                f = feats_for(r["filepath"], r.get("orig_sr"))
                if f is None: continue
                X.append([f[k] for k in FEATS]); y.append(int(r["label"]))
                g.append(r["speaker_id"])
                # metadata: age + sex (Soleymani: check participant metadata vs disease labels)
                try: age = float(r.get("age") or "nan")
                except Exception: age = float("nan")
                sx = (r.get("sex") or "").strip().lower()
                sex = 1.0 if sx.startswith("m") else (0.0 if sx.startswith("f") else float("nan"))
                M.append([age, sex])
            except Exception:
                continue
        if len(y) < 10:
            print(f"[skip] {ds}: too few clips"); continue
        X = np.array(X); y = np.array(y); g = np.array(g); M = np.array(M, dtype=float)

        auc = probe(X, y, g)
        yshuf = y.copy(); np.random.RandomState(0).shuffle(yshuf)
        auc_shuf = probe(X, yshuf, g)
        # metadata-only probe (age/sex), median-imputed
        auc_meta = float("nan"); auc_comb = float("nan")
        if M.size and not np.all(np.isnan(M)):
            Mi = M.copy()
            for j in range(Mi.shape[1]):
                col = Mi[:, j]; med = np.nanmedian(col)
                if np.isnan(med): med = 0.0
                col[np.isnan(col)] = med; Mi[:, j] = col
            auc_meta = probe(Mi, y, g)
            auc_comb = probe(np.hstack([X, Mi]), y, g)
        age_pd = float(np.nanmean(M[y == 1, 0])) if M.size else float("nan")
        age_hc = float(np.nanmean(M[y == 0, 0])) if M.size else float("nan")
        male_pd = float(np.nanmean(M[y == 1, 1])) if M.size else float("nan")
        male_hc = float(np.nanmean(M[y == 0, 1])) if M.size else float("nan")

        means = {}
        for i, k in enumerate(FEATS):
            means[k] = {"PD": round(float(np.mean(X[y == 1, i])), 3),
                        "HC": round(float(np.mean(X[y == 0, i])), 3)}
        report[ds] = {"n_clips": int(len(y)), "n_speakers": int(len(set(g))),
                      "n_PD": int((y == 1).sum()), "n_HC": int((y == 0).sum()),
                      "AUC_from_recording_properties_only": round(auc, 3),
                      "AUC_from_metadata_only_age_sex": round(auc_meta, 3),
                      "AUC_recording_plus_metadata": round(auc_comb, 3),
                      "AUC_label_shuffled_control": round(auc_shuf, 3),
                      "mean_age_PD": round(age_pd, 1), "mean_age_HC": round(age_hc, 1),
                      "frac_male_PD": round(male_pd, 2), "frac_male_HC": round(male_hc, 2),
                      "per_class_means": means}
        print(f"\n=== {ds} ({len(y)} clips, {len(set(g))} speakers, PD {int((y==1).sum())} / HC {int((y==0).sum())}) ===")
        print(f"  AUC from RECORDING PROPERTIES ONLY : {auc:.3f}")
        print(f"  AUC from METADATA ONLY (age,sex)   : {auc_meta:.3f}")
        print(f"  AUC recording + metadata           : {auc_comb:.3f}")
        print(f"  AUC labels shuffled (sanity)       : {auc_shuf:.3f}")
        print(f"  mean age  PD {age_pd:.1f}  vs HC {age_hc:.1f}   |  frac male PD {male_pd:.2f} vs HC {male_hc:.2f}")
        for k in FEATS:
            print(f"    {k:14s} PD {means[k]['PD']:>10}   HC {means[k]['HC']:>10}")

    os.makedirs(f"{R}/audit", exist_ok=True)
    json.dump(report, open(f"{R}/audit/confound_audit_{a.tasks.replace(',','_')}.json", "w"), indent=2)
    print(f"\nwrote {R}/audit/confound_audit_{a.tasks.replace(',','_')}.json")

if __name__ == "__main__":
    main()
