"""
ITALIAN CONFOUND step 8 -- the decisive test.

The corpus has 10 recording sessions (dates parsed out of the filenames). Every session
contains exactly one group. So "which session is this clip from" DETERMINES the label.

If the separation is a session fingerprint rather than a disease effect, then a probe
trained on some sessions and tested on ENTIRELY UNSEEN sessions must collapse, because the
held-out sessions' fingerprints were never in training. This is leave-one-control-session-
and-one-patient-session-out: 5 x 4 = 20 folds, each fold's test set contains two sessions
the model has never seen.

Run for:
  - the silence-only PSD  (the artifact)
  - the qwen2 encoder features (the reported result)
Also reports which clips survive the silence rule per session, because that selection is
itself session-biased.
"""
import os, csv, json, warnings, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/final"


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=4000, class_weight="balanced"))


def leave_session_pair_out(X, y, sess, verbose_name=""):
    """test folds are (one control session, one patient session) never seen in training."""
    cs = sorted({s for s, l in zip(sess, y) if l == 0})
    ps = sorted({s for s, l in zip(sess, y) if l == 1})
    aucs, det = [], []
    for c, p in itertools.product(cs, ps):
        te = (sess == c) | (sess == p)
        tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = clf(); m.fit(X[tr], y[tr])
        pr = m.predict_proba(X[te])[:, 1]
        a = roc_auc_score(y[te], pr)
        aucs.append(a)
        det.append(dict(held_out_control_session=c, held_out_patient_session=p,
                        n_test=int(te.sum()), auc=round(float(a), 4)))
    return np.array(aucs), det


def spk_disjoint_auc(X, y, g):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(5, len(np.unique(g)))).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = clf(); m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    k = ~np.isnan(p)
    return float(roc_auc_score(y[k], p[k]))


def main():
    D = list(csv.DictReader(open(f"{OUT}/italian_design.csv")))
    dmap = {r["filepath"]: r for r in D}

    # ---------- selection bias of the silence rule, per session ----------
    sil = np.load(f"{OUT}/italian_silence_psd.npz", allow_pickle=True)
    fp = sil["filepath"].astype(str)
    sk = list(sil["scal_keys"]); S = sil["S"]; PS = sil["PS"]
    nseg = S[:, sk.index("n_sil_seg")]
    usable = np.isfinite(PS).all(1)
    G = np.load(f"{OUT}/italian_gate_stats.csv", allow_pickle=True) if False else None
    gate = {r["filepath"]: r for r in csv.DictReader(open(f"{OUT}/italian_gate_stats.csv"))}

    print("### the silence rule keeps very different fractions of each session")
    print(f"  {'session':12s}{'label':>7}{'n_clips':>9}{'kept':>7}{'kept %':>9}"
          f"{'floor dBFS':>12}{'speech-floor dB':>17}{'encoders'}")
    persess = {}
    for dt in sorted({r["date"] for r in D}):
        m = np.array([dmap[p]["date"] == dt for p in fp])
        n_all = sum(1 for r in D if r["date"] == dt)
        kept = int((m & usable).sum())
        fl = np.median([float(gate[r["filepath"]]["quiet_rms_dbfs"]) for r in D if r["date"] == dt])
        sf = np.median([float(gate[r["filepath"]]["speech_minus_floor_db"]) for r in D if r["date"] == dt])
        lab = [r["label"] for r in D if r["date"] == dt][0]
        encs = sorted({r["enc"] for r in D if r["date"] == dt})
        print(f"  {dt:12s}{lab:>7}{n_all:>9}{kept:>7}{100*kept/n_all:>8.1f}%"
              f"{fl:>12.1f}{sf:>17.1f}  {[e[:18] for e in encs]}")
        persess[dt] = dict(label=int(lab), n=n_all, kept=kept, floor_dbfs=round(float(fl), 2),
                           speech_minus_floor_db=round(float(sf), 2), encoders=encs)
    json.dump(persess, open(f"{OUT}/italian_sessions.json", "w"), indent=2)

    rows = []
    # ---------- silence PSD: speaker-disjoint vs session-disjoint ----------
    ok = usable
    Xs = PS[ok]
    ys = np.array([int(dmap[p]["label"]) for p in fp[ok]])
    gs = np.array([dmap[p]["spk"] for p in fp[ok]])
    ss = np.array([dmap[p]["date"] for p in fp[ok]])
    print("\n" + "=" * 100)
    print("### SILENCE-ONLY PSD probe")
    a_spk = spk_disjoint_auc(Xs, ys, gs)
    a_ses, det = leave_session_pair_out(Xs, ys, ss)
    print(f"  speaker-disjoint (standard protocol)          AUC = {a_spk:.3f}   n={len(ys)}")
    print(f"  SESSION-disjoint (leave 1 ctrl + 1 PD session out, {len(a_ses)} folds)")
    print(f"      mean AUC = {a_ses.mean():.3f}   median {np.median(a_ses):.3f}   "
          f"range {a_ses.min():.3f}-{a_ses.max():.3f}   n folds at/below chance: "
          f"{int((a_ses <= 0.5).sum())}/{len(a_ses)}")
    rows.append(dict(features="silence PSD", n_clips=len(ys),
                     speaker_disjoint_auc=round(a_spk, 4),
                     session_disjoint_mean_auc=round(float(a_ses.mean()), 4),
                     session_disjoint_median=round(float(np.median(a_ses)), 4),
                     session_disjoint_min=round(float(a_ses.min()), 4),
                     session_disjoint_max=round(float(a_ses.max()), 4),
                     n_folds=len(a_ses), n_folds_le_chance=int((a_ses <= 0.5).sum())))
    for d in det:
        print(f"      hold out  ctrl {d['held_out_control_session']}  +  PD "
              f"{d['held_out_patient_session']}   n={d['n_test']:4d}   AUC={d['auc']:.3f}")

    # ---------- qwen2 encoder features ----------
    F = np.load(f"{R}/features/qwen2/italian/encoder_features.npz", allow_pickle=True)
    paths = F["path"].astype(str); yf = F["label"].astype(int); tf = F["task"].astype(str)
    gf = F["speaker"].astype(str)
    sf_ = np.array([dmap[p]["date"] for p in paths])
    nl = int(np.array(F["n_layers"]).ravel()[0])
    print("\n" + "=" * 100)
    print("### qwen2 ENCODER features (the reported italian result)")
    for tname in ["read", "vowel", "ddk"]:
        m = tf == tname
        if m.sum() < 30:
            continue
        best_spk, best_ses, bl = -1, None, None
        percur = []
        for li in range(nl):
            X = F[f"layer_{li:02d}"][m]
            a1 = spk_disjoint_auc(X, yf[m], gf[m])
            a2, _ = leave_session_pair_out(X, yf[m], sf_[m])
            percur.append((li, a1, float(a2.mean())))
            if a1 > best_spk:
                best_spk, best_ses, bl = a1, a2, li
        print(f"  task={tname}  n={m.sum()}   best layer by speaker-disjoint AUC = {bl}")
        print(f"     speaker-disjoint AUC  = {best_spk:.3f}")
        print(f"     SESSION-disjoint AUC  = {best_ses.mean():.3f}  "
              f"(median {np.median(best_ses):.3f}, range {best_ses.min():.3f}-{best_ses.max():.3f}, "
              f"{int((best_ses<=0.5).sum())}/{len(best_ses)} folds at/below chance)")
        # averaged over all layers, to avoid peak-of-N
        arr = np.array(percur)
        print(f"     mean over all {nl} layers:  speaker-disjoint {arr[:,1].mean():.3f}   "
              f"session-disjoint {arr[:,2].mean():.3f}")
        rows.append(dict(features=f"qwen2 encoder task={tname} (best layer {bl})",
                         n_clips=int(m.sum()), speaker_disjoint_auc=round(float(best_spk), 4),
                         session_disjoint_mean_auc=round(float(best_ses.mean()), 4),
                         session_disjoint_median=round(float(np.median(best_ses)), 4),
                         session_disjoint_min=round(float(best_ses.min()), 4),
                         session_disjoint_max=round(float(best_ses.max()), 4),
                         n_folds=len(best_ses), n_folds_le_chance=int((best_ses <= 0.5).sum())))
        rows.append(dict(features=f"qwen2 encoder task={tname} (mean over all {nl} layers)",
                         n_clips=int(m.sum()), speaker_disjoint_auc=round(float(arr[:,1].mean()), 4),
                         session_disjoint_mean_auc=round(float(arr[:,2].mean()), 4),
                         session_disjoint_median="", session_disjoint_min="",
                         session_disjoint_max="", n_folds=len(best_ses), n_folds_le_chance=""))

    with open(f"{OUT}/italian_session_disjoint.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}/italian_session_disjoint.csv and italian_sessions.json")
    print("JOB_DONE")


if __name__ == "__main__":
    main()
