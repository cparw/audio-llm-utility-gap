"""
SILENCE-ONLY ARTIFACT PROBE  --  stage 2: probing + statistics.

Reads silence_feats_<ds>.npz written by silence_extract.py and asks the advisor's
question directly: can you tell patients from controls using ONLY the background
noise in the pauses?

Protocol (matches the group's statistics rules):
  - StandardScaler + LogisticRegression(class_weight='balanced', max_iter=5000)
  - GroupKFold(5) on speaker_id, speaker overlap between train/test ASSERTED to be zero
  - out-of-fold predictions pooled -> one OOF AUC (clip level) and one speaker-level AUC
    (mean OOF probability per speaker)
  - label-shuffle control: labels permuted at the SPEAKER level (a speaker keeps one
    label, the labels are shuffled among speakers), 50 repeats, whole CV rerun each time
  - speaker-clustered bootstrap 95% CI: resample speakers with replacement from the OOF
    predictions, 4000 reps
  - run per dataset and, where a dataset has several task types, per task type

Two feature sets are reported separately and the distinction matters:
  ENV   = the recording environment of the silence (noise floor, spectrum, MFCCs).
          This is the headline. No speech content is involved by construction, so any
          AUC above chance here is a RECORDING ARTIFACT.
  PAUSE = how much silence there is (fraction, total, segment count, duration).
          This is pause behaviour, which is a genuine PD/depression symptom, so it is
          reported for context only and is NOT part of the artifact claim.
"""
import os, json, csv, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/soleymani_checks"
DATASETS = ["italian", "kcl", "neurovoz", "pcgita", "dcaps"]
N_SHUF = 50
N_BOOT = 4000
RNG = np.random.RandomState(0)

# recording-property probes on FULL audio, for comparison (given by the advisor)
FULL_AUDIO_REF = {"italian": 0.995, "neurovoz": 0.80, "kcl": 0.74, "pcgita": 0.68,
                  "dcaps": 0.45}


def oof_predict(X, y, g, n_splits=5):
    """pooled out-of-fold probabilities with speaker-disjoint folds. asserts disjointness."""
    ns = min(n_splits, len(np.unique(g)))
    if ns < 2 or len(np.unique(y)) < 2:
        return None, 0
    oof = np.full(len(y), np.nan)
    overlaps = 0
    for tr, te in GroupKFold(ns).split(X, y, g):
        if set(g[tr]) & set(g[te]):
            overlaps += 1
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    assert overlaps == 0, "SPEAKER LEAK between train and test folds"
    return oof, ns


def safe_auc(y, p):
    m = ~np.isnan(p)
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], p[m]))


def speaker_level(y, p, g):
    """average OOF probability per speaker, then AUC over speakers."""
    m = ~np.isnan(p)
    if m.sum() < 4:
        return float("nan"), 0
    spk = {}
    for gi, yi, pi in zip(g[m], y[m], p[m]):
        spk.setdefault(gi, [yi, []])[1].append(pi)
    ys = np.array([v[0] for v in spk.values()])
    ps = np.array([np.mean(v[1]) for v in spk.values()])
    if len(np.unique(ys)) < 2:
        return float("nan"), len(ys)
    return float(roc_auc_score(ys, ps)), len(ys)


def boot_ci(y, p, g, n=N_BOOT, rng=None):
    """speaker-clustered bootstrap: resample SPEAKERS with replacement."""
    rng = rng or np.random.RandomState(1)
    m = ~np.isnan(p)
    y, p, g = y[m], p[m], g[m]
    spk = np.unique(g)
    idx = {s: np.where(g == s)[0] for s in spk}
    aucs = []
    for _ in range(n):
        pick = rng.choice(len(spk), len(spk), replace=True)
        sel = np.concatenate([idx[spk[k]] for k in pick])
        if len(np.unique(y[sel])) < 2:
            continue
        aucs.append(roc_auc_score(y[sel], p[sel]))
    if len(aucs) < 50:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def speaker_shuffle_labels(y, g, rng):
    """permute labels among speakers, keeping one label per speaker."""
    spk = np.unique(g)
    lab = np.array([y[g == s][0] for s in spk])
    perm = rng.permutation(lab)
    mp = dict(zip(spk, perm))
    return np.array([mp[s] for s in g])


def run_block(X, y, g, tag, rng):
    if len(y) < 20 or len(np.unique(y)) < 2 or len(np.unique(g)) < 4:
        return dict(tag=tag, n_clips=int(len(y)), n_speakers=int(len(np.unique(g))),
                    note="too few clips/speakers, skipped")
    oof, ns = oof_predict(X, y, g)
    if oof is None:
        return dict(tag=tag, note="no folds")
    auc = safe_auc(y, oof)
    auc_spk, nspk_used = speaker_level(y, oof, g)
    lo, hi = boot_ci(y, oof, g, rng=np.random.RandomState(abs(hash(tag)) % 2**31))
    shuf = []
    for i in range(N_SHUF):
        ys = speaker_shuffle_labels(y, g, rng)
        o2, _ = oof_predict(X, ys, g)
        if o2 is not None:
            a = safe_auc(ys, o2)
            if a == a:
                shuf.append(a)
    shuf = np.array(shuf) if shuf else np.array([np.nan])
    # one-sided permutation p: how often does a shuffled run beat the real one
    pval = float((np.sum(shuf >= auc) + 1) / (len(shuf) + 1))
    spk_lab = {s: y[g == s][0] for s in np.unique(g)}
    return dict(tag=tag, n_clips=int(len(y)), n_speakers=int(len(np.unique(g))),
                n_folds=int(ns),
                n_pos_clips=int((y == 1).sum()), n_neg_clips=int((y == 0).sum()),
                n_pos_spk=int(sum(v == 1 for v in spk_lab.values())),
                n_neg_spk=int(sum(v == 0 for v in spk_lab.values())),
                auc_clip=round(auc, 4), auc_clip_ci=[round(lo, 4), round(hi, 4)],
                auc_speaker=round(auc_spk, 4), n_speakers_scored=int(nspk_used),
                shuffle_mean=round(float(np.nanmean(shuf)), 4),
                shuffle_p95=round(float(np.nanpercentile(shuf, 95)), 4),
                perm_p=round(pval, 4))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(7)
    report, table = {}, []

    for ds in DATASETS:
        fp = f"{OUT}/silence_feats_{ds}.npz"
        if not os.path.exists(fp):
            print(f"[skip] no features for {ds}")
            continue
        d = np.load(fp, allow_pickle=True)
        X, P, y, g, task = d["X"], d["P"], d["y"], d["g"], d["task"]
        env_names = list(d["env_feats"])
        pause_names = list(d["pause_feats"])
        if len(y) == 0:
            print(f"[{ds}] ZERO clips survived silence extraction")
            report[ds] = {"note": "no clips survived"}
            continue
        # guard against non-finite features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- discard accounting, by group ----
        disc = list(csv.DictReader(open(f"{OUT}/silence_discards_{ds}.csv")))
        n_manifest = len(disc) + len(y)
        dpos = sum(1 for r in disc if r["label"] == "1")
        dneg = len(disc) - dpos
        tot_pos = int((y == 1).sum()) + dpos
        tot_neg = int((y == 0).sum()) + dneg
        rate_pos = dpos / tot_pos if tot_pos else float("nan")
        rate_neg = dneg / tot_neg if tot_neg else float("nan")
        reasons = {}
        for r in disc:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

        si = pause_names.index("sil_frac")
        st = pause_names.index("sil_total_s")

        def dist(mask):
            if mask.sum() == 0:
                return {}
            return {"n": int(mask.sum()),
                    "sil_frac_median": round(float(np.median(P[mask, si])), 4),
                    "sil_frac_mean": round(float(np.mean(P[mask, si])), 4),
                    "sil_sec_median": round(float(np.median(P[mask, st])), 3),
                    "sil_sec_p25": round(float(np.percentile(P[mask, st], 25)), 3),
                    "sil_sec_p75": round(float(np.percentile(P[mask, st], 75)), 3)}

        ds_rep = {
            "n_in_manifest": n_manifest, "n_kept": int(len(y)), "n_discarded": len(disc),
            "discard_reasons": reasons,
            "discard_rate_patient": round(rate_pos, 4),
            "discard_rate_control": round(rate_neg, 4),
            "discard_bias_flag": bool(abs(rate_pos - rate_neg) > 0.10),
            "silence_distribution_patient": dist(y == 1),
            "silence_distribution_control": dist(y == 0),
            "blocks": {},
        }

        print("\n" + "=" * 96)
        print(f"### {ds.upper()}   kept {len(y)}/{n_manifest} clips "
              f"({len(np.unique(g))} speakers)   discarded {len(disc)} {reasons}")
        print(f"    discard rate  patients {rate_pos:.3f}   controls {rate_neg:.3f}"
              + ("    <<< GROUP-BIASED DISCARD, read the numbers with care"
                 if ds_rep["discard_bias_flag"] else ""))
        for k in ("patient", "control"):
            print(f"    silence {k:8s}: {ds_rep['silence_distribution_'+k]}")

        blocks = [("ALL", np.ones(len(y), bool))]
        tt = [t for t in np.unique(task)]
        if len(tt) > 1:
            blocks += [(f"task={t}", task == t) for t in tt]

        for name, m in blocks:
            for fs, XX, fnames in (("ENV", X, env_names), ("PAUSE", P, pause_names)):
                res = run_block(XX[m], y[m], g[m], f"{ds}|{name}|{fs}", rng)
                ds_rep["blocks"][f"{name}|{fs}"] = res
                if "auc_clip" in res:
                    print(f"    {name:16s} {fs:5s}  n={res['n_clips']:5d} "
                          f"spk={res['n_speakers']:3d}  AUC={res['auc_clip']:.3f} "
                          f"[{res['auc_clip_ci'][0]:.3f},{res['auc_clip_ci'][1]:.3f}]  "
                          f"spkAUC={res['auc_speaker']:.3f}  "
                          f"shuffle={res['shuffle_mean']:.3f} (p95 {res['shuffle_p95']:.3f}) "
                          f"perm_p={res['perm_p']:.3f}")
                    if fs == "ENV":
                        table.append(dict(dataset=ds, block=name,
                                          n_clips=res["n_clips"], n_speakers=res["n_speakers"],
                                          silence_only_AUC=res["auc_clip"],
                                          ci_lo=res["auc_clip_ci"][0], ci_hi=res["auc_clip_ci"][1],
                                          speaker_AUC=res["auc_speaker"],
                                          shuffle_mean=res["shuffle_mean"],
                                          perm_p=res["perm_p"],
                                          full_audio_recprop_AUC=FULL_AUDIO_REF.get(ds, ""),
                                          pct_clips_kept=round(len(y) / n_manifest, 3)))
                else:
                    print(f"    {name:16s} {fs:5s}  {res.get('note')}")

        # which silence features carry the signal (fit on all data, for description only)
        try:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, class_weight="balanced"))
            clf.fit(X, y)
            co = clf[-1].coef_[0]
            order = np.argsort(-np.abs(co))[:6]
            ds_rep["top_silence_features_full_fit"] = [
                {"feature": env_names[i], "coef": round(float(co[i]), 3),
                 "mean_patient": round(float(X[y == 1, i].mean()), 3),
                 "mean_control": round(float(X[y == 0, i].mean()), 3)} for i in order]
            print("    top silence features (descriptive, full-data fit):")
            for e in ds_rep["top_silence_features_full_fit"]:
                print(f"      {e['feature']:18s} coef {e['coef']:>7}  "
                      f"patient {e['mean_patient']:>10}  control {e['mean_control']:>10}")
        except Exception as e:
            print("    [feature-importance failed]", e)

        report[ds] = ds_rep

    json.dump(report, open(f"{OUT}/silence_probe_report.json", "w"), indent=2)
    hdr = ["dataset", "block", "n_clips", "n_speakers", "pct_clips_kept",
           "silence_only_AUC", "ci_lo", "ci_hi", "speaker_AUC", "shuffle_mean",
           "perm_p", "full_audio_recprop_AUC"]
    with open(f"{OUT}/silence_only_auc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(table)

    print("\n" + "=" * 96)
    print("HEADLINE: AUC from BACKGROUND NOISE IN THE PAUSES ONLY (no speech content)")
    print("=" * 96)
    print("".join(f"{h:>13}" for h in ["dataset", "block", "n_clips", "sil_AUC",
                                       "95% CI", "spk_AUC", "shuffle", "fullaudio"]))
    for r in table:
        print("".join(f"{v:>13}" for v in [
            r["dataset"], r["block"][:12], r["n_clips"], f"{r['silence_only_AUC']:.3f}",
            f"{r['ci_lo']:.2f}-{r['ci_hi']:.2f}", f"{r['speaker_AUC']:.3f}",
            f"{r['shuffle_mean']:.3f}", str(r["full_audio_recprop_AUC"])]))
    print(f"\nwrote {OUT}/silence_probe_report.json")
    print(f"wrote {OUT}/silence_only_auc.csv")
    print("\nVERDICT RULE: if the silence-only CI excludes 0.5 the dataset's group difference")
    print("is at least partly a recording artifact, because no speech was used to get it.")


if __name__ == "__main__":
    main()
