"""
ITALIAN CONFOUND step 4: where does the separation live, what causes it, can it be removed.

Consumes italian_silence_psd.npz. Everything speaker-disjoint (GroupKFold on speaker_id),
speaker-level label shuffle, speaker-clustered bootstrap CI.
"""
import os, json, csv, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")
R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/final"
N_SHUF, N_BOOT = 200, 2000


def oof(X, y, g, ns=5):
    ns = min(ns, len(np.unique(g)))
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(ns).split(X, y, g):
        assert not (set(g[tr]) & set(g[te])), "SPEAKER LEAK"
        if len(np.unique(y[tr])) < 2:
            continue
        c = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000, class_weight="balanced"))
        c.fit(X[tr], y[tr]); p[te] = c.predict_proba(X[te])[:, 1]
    return p


def spk_auc(y, p, g):
    m = ~np.isnan(p)
    d = {}
    for gi, yi, pi in zip(g[m], y[m], p[m]):
        d.setdefault(gi, [yi, []])[1].append(pi)
    ys = np.array([v[0] for v in d.values()]); ps = np.array([np.mean(v[1]) for v in d.values()])
    if len(np.unique(ys)) < 2:
        return float("nan")
    return float(roc_auc_score(ys, ps))


def boot(y, p, g, n=N_BOOT, seed=1):
    rng = np.random.RandomState(seed); m = ~np.isnan(p)
    y, p, g = y[m], p[m], g[m]
    spk = np.unique(g); idx = {s: np.where(g == s)[0] for s in spk}
    a = []
    for _ in range(n):
        pick = rng.choice(len(spk), len(spk), True)
        sel = np.concatenate([idx[spk[k]] for k in pick])
        if len(np.unique(y[sel])) > 1:
            a.append(roc_auc_score(y[sel], p[sel]))
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) > 50 else (np.nan, np.nan)


def shuf_null(X, y, g, n=N_SHUF, seed=3):
    rng = np.random.RandomState(seed); spk = np.unique(g)
    lab = np.array([y[g == s][0] for s in spk]); a = []
    for _ in range(n):
        mp = dict(zip(spk, rng.permutation(lab)))
        ys = np.array([mp[s] for s in g])
        p = oof(X, ys, g)
        m = ~np.isnan(p)
        if len(np.unique(ys[m])) > 1:
            a.append(roc_auc_score(ys[m], p[m]))
    return np.array(a)


def run(tag, X, y, g, nshuf=N_SHUF, verbose=True):
    if len(np.unique(y)) < 2 or len(np.unique(g)) < 4:
        return dict(tag=tag, note="degenerate", n=int(len(y)))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    p = oof(X, y, g)
    m = ~np.isnan(p)
    auc = float(roc_auc_score(y[m], p[m]))
    sa = spk_auc(y, p, g)
    lo, hi = boot(y, p, g, seed=abs(hash(tag)) % 2**31)
    nul = shuf_null(X, y, g, nshuf) if nshuf else np.array([np.nan])
    pv = float((np.sum(nul >= auc) + 1) / (len(nul) + 1))
    r = dict(tag=tag, n_clips=int(len(y)), n_spk=int(len(np.unique(g))),
             n_pos_spk=int(len({s for s, l in zip(g, y) if l == 1})),
             n_neg_spk=int(len({s for s, l in zip(g, y) if l == 0})),
             auc=round(auc, 4), ci=[round(lo, 4), round(hi, 4)],
             spk_auc=round(sa, 4), shuf_mean=round(float(np.nanmean(nul)), 4),
             shuf_p95=round(float(np.nanpercentile(nul, 95)), 4), perm_p=round(pv, 4))
    if verbose:
        print(f"  {tag[:48]:50s} n={r['n_clips']:4d} spk={r['n_spk']:3d}({r['n_neg_spk']}/{r['n_pos_spk']})"
              f"  AUC={r['auc']:.3f} [{lo:.3f},{hi:.3f}]  spkAUC={r['spk_auc']:.3f}"
              f"  null={r['shuf_mean']:.3f}(p95 {r['shuf_p95']:.3f}) p={r['perm_p']:.3f}")
    return r


def main():
    d = np.load(f"{OUT}/italian_silence_psd.npz", allow_pickle=True)
    PS, S, F = d["PS"], d["S"], d["freqs"]
    keys = list(d["scal_keys"])
    y = d["label"].astype(int); g = d["spk"].astype(str); task = d["task"].astype(str)
    enc = d["enc"].astype(str); date = d["date"].astype(str); sr = d["sr"].astype(str)
    grp = d["group"].astype(str)
    ok = np.isfinite(PS).all(1)
    print(f"files {len(y)}  with usable silence PSD {ok.sum()}")
    res = {}
    tbl = []

    # ---------- 0. audio-free baseline: the RIFF chunk graph alone ----------
    print("\n=== 0. NO AUDIO AT ALL: probe using only the container/toolchain fingerprint ===")
    encs = sorted(set(enc)); srs = sorted(set(sr))
    Xc = np.array([[1.0 * (e == k) for k in encs] + [1.0 * (s == k) for k in srs]
                   for e, s in zip(enc, sr)])
    res["container_only"] = run("container(encoder+sr) ALL tasks", Xc, y, g)
    for t in ["read", "vowel", "ddk"]:
        m = task == t
        res[f"container_{t}"] = run(f"container(encoder+sr) task={t}", Xc[m], y[m], g[m])

    # ---------- 1. per-frequency separation curve ----------
    print("\n=== 1. per-frequency speaker-level AUC of the silence log-PSD ===")
    o = ok
    spk = np.unique(g[o])
    slab = np.array([y[o][g[o] == s][0] for s in spk])
    M = np.array([PS[o][g[o] == s].mean(0) for s in spk])
    fauc = np.array([roc_auc_score(slab, M[:, i]) for i in range(M.shape[1])])
    np.savetxt(f"{OUT}/italian_freq_auc.csv",
               np.c_[F, fauc, M[slab == 0].mean(0), M[slab == 1].mean(0)],
               delimiter=",", header="freq_hz,speaker_auc,ctrl_mean_dB,pd_mean_dB", comments="")
    print(f"  {'freq Hz':>9} {'spkAUC':>8} {'ctrl dB':>9} {'PD dB':>9} {'PD-ctrl':>9}")
    for i in list(range(0, 16)) + list(range(16, len(F), 16)):
        print(f"  {F[i]:>9.1f} {fauc[i]:>8.3f} {M[slab==0].mean(0)[i]:>9.2f} "
              f"{M[slab==1].mean(0)[i]:>9.2f} {M[slab==1].mean(0)[i]-M[slab==0].mean(0)[i]:>9.2f}")

    # band probes
    print("\n=== 2. band-limited silence probes (speaker-disjoint) ===")
    bands = [("DC..20Hz", 0, 20), ("20..100", 20, 100), ("100..300", 100, 300),
             ("300..1k", 300, 1000), ("1k..3.4k", 1000, 3400), ("3.4k..8k", 3400, 8000),
             ("FULL 0..8k", 0, 8000), ("TELEPHONE 300..3.4k", 300, 3400)]
    for name, lo, hi in bands:
        bm = (F >= lo) & (F <= hi)
        if bm.sum() == 0:
            continue
        res[f"band_{name}"] = run(f"band {name} ({bm.sum()} bins)", PS[o][:, bm], y[o], g[o], nshuf=50)

    # ---------- 3. is it session identity? ----------
    print("\n=== 3. does the SILENCE identify the recording SESSION (10 days)? ===")
    dates = np.unique(date[o]); dmap = {v: i for i, v in enumerate(dates)}
    yd = np.array([dmap[v] for v in date[o]])
    accs = []
    for tr, te in GroupKFold(5).split(PS[o], yd, g[o]):
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        c.fit(PS[o][tr], yd[tr]); accs.append(accuracy_score(yd[te], c.predict(PS[o][te])))
    base = max(np.bincount(yd)) / len(yd)
    print(f"  10-way SESSION accuracy from silence PSD, speaker-disjoint: "
          f"{np.mean(accs):.4f}  (majority baseline {base:.4f})")
    res["session_id"] = dict(acc=round(float(np.mean(accs)), 4), majority=round(float(base), 4),
                             n_sessions=int(len(dates)))
    # encoder identity from silence
    encs_u = np.unique(enc[o]); emap = {v: i for i, v in enumerate(encs_u)}
    ye = np.array([emap[v] for v in enc[o]])
    ae = []
    for tr, te in GroupKFold(5).split(PS[o], ye, g[o]):
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        c.fit(PS[o][tr], ye[tr]); ae.append(accuracy_score(ye[te], c.predict(PS[o][te])))
    print(f"  {len(encs_u)}-way ENCODER accuracy from silence PSD: {np.mean(ae):.4f} "
          f"(majority {max(np.bincount(ye))/len(ye):.4f})")
    res["encoder_id"] = dict(acc=round(float(np.mean(ae)), 4))

    # ---------- 4. matched-toolchain / matched-rate subsets ----------
    print("\n=== 4. MATCHED subsets: does the effect survive when the recording chain matches? ===")
    subsets = [
        ("ALL", o),
        ("16 kHz only", o & (sr == "16000")),
        ("bare fmt|data only", o & (enc == "bare_fmt_data")),
        ("bare fmt|data AND 16 kHz", o & (enc == "bare_fmt_data") & (sr == "16000")),
        ("bare+16k, elderly HC vs PD (age-matched)",
         o & (enc == "bare_fmt_data") & (sr == "16000") &
         (np.char.find(grp, "15 Young") < 0)),
    ]
    for name, m in subsets:
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            print(f"  {name:50s} n={m.sum()} skipped")
            continue
        res[f"subset_{name}"] = run(f"silence PSD | {name}", PS[m], y[m], g[m], nshuf=50)
        tbl.append(dict(analysis="matched_subset", condition=name,
                        **{k: res[f"subset_{name}"].get(k) for k in
                           ("n_clips", "n_spk", "auc", "spk_auc", "shuf_mean", "perm_p")}))

    # ---------- 5. removability battery ----------
    print("\n=== 5. REMOVABILITY: normalisations applied to the silence PSD ===")
    m = o & (enc == "bare_fmt_data") & (sr == "16000")
    for base_name, mm in [("ALL files", o), ("matched chain (bare+16k)", m)]:
        P = PS[mm]
        norms = {
            "raw": P,
            "per-file mean removed (gain-invariant)": P - P.mean(1, keepdims=True),
            "per-file mean+linear tilt removed": None,
            "per-file z-scored (shape only)": (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9),
            "high-pass: drop <300 Hz": P[:, F >= 300],
            "telephone band 300-3400 only": P[:, (F >= 300) & (F <= 3400)],
            "tel band + per-file mean removed": None,
        }
        x = np.arange(P.shape[1], dtype=float); x = (x - x.mean()) / x.std()
        A = np.c_[np.ones_like(x), x]
        coef = np.linalg.lstsq(A, P.T, rcond=None)[0]
        norms["per-file mean+linear tilt removed"] = P - (A @ coef).T
        tb = (F >= 300) & (F <= 3400)
        Pt = P[:, tb]
        norms["tel band + per-file mean removed"] = Pt - Pt.mean(1, keepdims=True)
        print(f"  -- {base_name} (n={mm.sum()}) --")
        for nm, XX in norms.items():
            r = run(f"[{base_name}] {nm}", XX, y[mm], g[mm], nshuf=50)
            res[f"norm_{base_name}_{nm}"] = r
            tbl.append(dict(analysis="removability", condition=f"{base_name} | {nm}",
                            **{k: r.get(k) for k in ("n_clips", "n_spk", "auc", "spk_auc",
                                                     "shuf_mean", "perm_p")}))

    # ---------- 6. scalar forensics by group ----------
    print("\n=== 6. scalar forensics, median by label and by encoder ===")
    show = ["dc_offset_lsb_full", "dc_offset_lsb_sil", "sil_rms_dbfs", "n_distinct",
            "sample_gcd", "lsb_odd_frac", "frac_zero", "cutoff_hz", "n_clip",
            "lead_zero_samples", "trail_zero_samples", "dyn_range_db"]
    print(f"  {'stat':26s}{'control':>14}{'patient':>14}")
    for k in show:
        if k not in keys:
            continue
        i = keys.index(k); v = S[:, i]
        print(f"  {k:26s}{np.nanmedian(v[y==0]):>14.5g}{np.nanmedian(v[y==1]):>14.5g}")
    print(f"\n  {'encoder':34s}" + "".join(f"{k[:12]:>14}" for k in show[:8]))
    for e in sorted(set(enc)):
        mm = enc == e
        print(f"  {e[:32]:34s}" + "".join(
            f"{np.nanmedian(S[mm, keys.index(k)]):>14.5g}" if k in keys else f"{'':>14}"
            for k in show[:8]))

    # ---------- 7. single-scalar probes: name the mechanism ----------
    print("\n=== 7. SINGLE-SCALAR probes (one number per clip, speaker-disjoint) ===")
    singles = ["sil_rms_dbfs", "dc_offset_lsb_sil", "dc_offset_lsb_full", "n_distinct",
               "cutoff_hz", "frac_zero", "lsb_odd_frac", "trail_zero_samples",
               "lead_zero_samples", "dyn_range_db"]
    for k in singles:
        if k not in keys:
            continue
        v = S[:, keys.index(k)][o].reshape(-1, 1)
        r = run(f"SCALAR {k}", v, y[o], g[o], nshuf=50)
        res[f"scalar_{k}"] = r
        tbl.append(dict(analysis="single_scalar", condition=k,
                        **{kk: r.get(kk) for kk in ("n_clips","n_spk","auc","spk_auc","shuf_mean","perm_p")}))
    # same on the matched chain
    print("  -- restricted to matched chain (bare fmt|data, 16 kHz native) --")
    mch = o & (enc == "bare_fmt_data") & (sr == "16000")
    for k in ["sil_rms_dbfs", "dc_offset_lsb_sil", "n_distinct"]:
        if k not in keys:
            continue
        v = S[:, keys.index(k)][mch].reshape(-1, 1)
        r = run(f"SCALAR {k} | matched chain", v, y[mch], g[mch], nshuf=50)
        res[f"scalar_matched_{k}"] = r
        tbl.append(dict(analysis="single_scalar_matched", condition=k,
                        **{kk: r.get(kk) for kk in ("n_clips","n_spk","auc","spk_auc","shuf_mean","perm_p")}))
    # distribution of the noise floor level per session
    print("\n  silence RMS dBFS per recording session:")
    i = keys.index("sil_rms_dbfs")
    print(f"    {'date':12s}{'label':>7}{'n':>6}{'median dBFS':>14}{'p10':>9}{'p90':>9}{'encoders'}")
    for dt in sorted(set(date[o])):
        m2 = o & (date == dt)
        v = S[m2, i]
        print(f"    {dt:12s}{y[m2][0]:>7}{m2.sum():>6}{np.nanmedian(v):>14.2f}"
              f"{np.nanpercentile(v,10):>9.2f}{np.nanpercentile(v,90):>9.2f}  "
              f"{sorted(set(enc[m2]))}")

    json.dump(res, open(f"{OUT}/italian_confound_probes.json", "w"), indent=2, default=str)
    if tbl:
        with open(f"{OUT}/italian_confound_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tbl[0].keys())); w.writeheader(); w.writerows(tbl)
    print(f"\nwrote {OUT}/italian_confound_probes.json, italian_confound_table.csv, italian_freq_auc.csv")
    print("JOB_DONE")


if __name__ == "__main__":
    main()
