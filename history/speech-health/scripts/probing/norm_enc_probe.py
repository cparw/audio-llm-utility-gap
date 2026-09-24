#!/usr/bin/env python
"""TASK 1c - re-run the ENCODER probe on the normalised audio.

Compares the Qwen2-Audio encoder probe across the same conditions as the
interpretable probe:

   A  ORIGINAL encoder features, full original clip set   (the published number)
   B  ORIGINAL encoder features, surviving clips only     (clip-selection effect)
   D  CROP-ONLY encoder features                          (duration matched only)
   C  NORMALISED encoder features                         (duration + loudness)

Also runs a front-end invariance check: Qwen2-Audio uses a Whisper-style
feature extractor, whose log-mel step rescales each clip by its own maximum.
If that is so, a pure gain change cannot reach the encoder at all, and the
loudness normalisation is expected to be a no-op for this model.  That is a
claim about the code path, so it is tested rather than asserted.

usage: norm_enc_probe.py <dataset> <task>
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch1/parwatka/pd_probing/code_clean")
import pdstats as S

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = os.path.join(R, "committed")


def load(npz):
    d = np.load(npz, allow_pickle=True)
    return d


def probe_all_layers(d, idx, y, g, name, rows, bylayer_out):
    L = int(d["n_layers"])
    per = []
    best = None
    for li in range(L):
        X = d["layer_%02d" % li][idx]
        p, _ = S.oof_probe(X, y, g)
        a, b = S.auc_ba(y, p)
        per.append((li, a, b))
        if best is None or (np.isfinite(a) and a > best[1]):
            best = (li, a, b, p)
    li, a, b, p = best
    lo, hi = S.spk_bootstrap_auc(y, p, g, n=2000)
    nulls = S.shuffle_null_auc(d["layer_%02d" % li][idx], y, g, n=50)
    pv = float((np.sum(nulls >= a) + 1) / (len(nulls) + 1))
    print("   %-44s peak L%-2d AUC %.3f [%.3f, %.3f] bAcc %.3f  null %.3f  p=%.3f"
          % (name, li, a, lo, hi, b, nulls.mean(), pv), flush=True)
    pd.DataFrame(per, columns=["layer", "auc", "balacc"]).to_csv(bylayer_out, index=False)
    rows.append(dict(condition=name, n=len(idx), speakers=len(set(g)),
                     peak_layer=li, auc=a, auc_lo=lo, auc_hi=hi, balacc=b,
                     shuffle_null=float(nulls.mean()), perm_p=pv,
                     mean_auc_all_layers=float(np.nanmean([x[1] for x in per]))))
    return per


def frontend_invariance_check():
    print("\n--- front-end gain-invariance check (does a gain change reach the encoder?)")
    try:
        import torch, librosa
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        fe = proc.feature_extractor
        man = pd.read_csv(os.path.join(R, "manifests", "neurovoz.csv")).head(5)
        deltas = []
        for p in man.filepath:
            y, _ = librosa.load(p, sr=fe.sampling_rate, mono=True)
            f1 = fe(y, sampling_rate=fe.sampling_rate, return_tensors="np")["input_features"]
            f2 = fe(y * (10 ** (20 / 20.0)), sampling_rate=fe.sampling_rate,
                    return_tensors="np")["input_features"]
            deltas.append(float(np.abs(f1 - f2).max()))
        print("    max abs difference in the encoder INPUT features after a +20 dB gain:")
        print("    per-clip: %s" % ", ".join("%.6f" % d for d in deltas))
        if max(deltas) < 1e-4:
            print("    -> the Whisper-style log-mel front-end rescales each clip by its own")
            print("       maximum, so a pure gain change is ERASED BEFORE THE ENCODER SEES IT.")
            print("       Loudness normalisation is therefore a no-op for this model, and the")
            print("       original encoder result was never able to use absolute loudness.")
        else:
            print("    -> the front-end does pass gain through; loudness normalisation is")
            print("       a real change for this model.")
        return dict(max_delta=float(max(deltas)), per_clip=deltas)
    except Exception as e:
        print("    check could not run:", repr(e))
        return dict(error=repr(e))


def main():
    ds, task = sys.argv[1], sys.argv[2]
    tag = "%s_%s" % (ds, task)
    rows = []
    print("=" * 92)
    print("TASK 1c  ENCODER probe on normalised audio   %s / %s" % (ds, task))
    print("=" * 92, flush=True)

    # ---- original features
    do = load(os.path.join(R, "features", "qwen2", ds, "encoder_features.npz"))
    tsk = do["task"].astype(str)
    po = np.array([str(x) for x in do["path"]])
    mask_full = tsk == task
    idx_full = np.where(mask_full)[0]

    rep = pd.read_csv(os.path.join(OUT, "normalisation_report_%s.csv" % tag))
    surv = set(rep["orig"].tolist())
    idx_sub = np.array([i for i in idx_full if po[i] in surv])
    print("original npz: %d clips for task '%s'; %d of them survived duration matching"
          % (len(idx_full), task, len(idx_sub)))

    lab = do["label"].astype(int)
    spk = do["speaker"].astype(str)
    print("\n--- ENCODER PROBE")
    probe_all_layers(do, idx_full, lab[idx_full], spk[idx_full],
                     "A ORIGINAL audio, full set", rows,
                     os.path.join(OUT, "encnorm_%s_A_bylayer.csv" % tag))
    probe_all_layers(do, idx_sub, lab[idx_sub], spk[idx_sub],
                     "B ORIGINAL audio, surviving clips", rows,
                     os.path.join(OUT, "encnorm_%s_B_bylayer.csv" % tag))

    for key, label in [("croponly", "D CROP-ONLY audio"), ("norm", "C NORMALISED audio")]:
        sub = "qwen2_norm" if key == "norm" else "qwen2_croponly"
        f = os.path.join(R, "features", sub, tag, "encoder_features.npz")
        if not os.path.exists(f):
            print("   %-44s MISSING (%s not extracted)" % (label, f))
            continue
        dn = load(f)
        idx = np.arange(len(dn["label"]))
        probe_all_layers(dn, idx, dn["label"].astype(int), dn["speaker"].astype(str),
                         label, rows,
                         os.path.join(OUT, "encnorm_%s_%s_bylayer.csv" % (tag, key[0].upper())))

    rf = pd.DataFrame(rows)
    rf["dataset"] = ds
    rf["task"] = task
    rf.to_csv(os.path.join(OUT, "encoder_norm_probe_%s.csv" % tag), index=False)
    print("\n" + "=" * 92)
    print("ENCODER SURVIVAL SUMMARY  %s / %s" % (ds, task))
    print("=" * 92)
    print(rf[["condition", "n", "speakers", "peak_layer", "auc", "auc_lo", "auc_hi",
              "balacc", "shuffle_null"]].to_string(index=False))
    g = {r["condition"][0]: r["auc"] for _, r in rf.iterrows()}
    if "A" in g and "B" in g:
        print("\nclip-selection effect  B-A = %+.3f" % (g["B"] - g["A"]))
    if "B" in g and "D" in g:
        print("duration-match effect  D-B = %+.3f" % (g["D"] - g["B"]))
    if "D" in g and "C" in g:
        print("loudness-norm effect   C-D = %+.3f" % (g["C"] - g["D"]))
    if "A" in g and "C" in g:
        print("TOTAL                  C-A = %+.3f" % (g["C"] - g["A"]))

    # the front-end gain-invariance question is answered by frontend_check.py,
    # which is run as its own job; do not recompute it here.
    print("\nJOB_DONE")


if __name__ == "__main__":
    main()
