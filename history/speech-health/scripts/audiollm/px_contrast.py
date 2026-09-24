"""Between-model contrasts on the paradox set.

All four models scored exactly the same 297 clips, so the models can be compared with a
PAIRED speaker-clustered bootstrap: resample the 74 speakers once per draw and recompute
every model's statistic on that same resample.

Statistics contrasted:
  gap        = pooled AUC(lexical direction) - pooled AUC(clinical label)   [threshold-free]
  auc_confl  = AUC against the clinical label on the 195 conflict segments
"""
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
P, F = f"{R}/committed/paradox", f"{R}/final"
MODELS = [("qwen2audio", "qwen2-audio"), ("omni", "qwen2.5-omni"),
          ("af3", "audio-flamingo-3"), ("kimi", "kimi-audio")]
NB = 4000

M = pd.read_csv(f"{P}/manifest.csv")
D = M[["filepath", "set", "pair_id", "speaker_id", "label", "words_say_depressed"]].copy()
for short, _ in MODELS:
    S = pd.read_csv(f"{F}/px_scores_{short}.csv")
    S = S[S.task_type == "paradox_segment"][["filepath", "p_main", "p_voice"]]
    S = S.rename(columns={"p_main": f"{short}_main", "p_voice": f"{short}_voice"})
    D = D.merge(S, on="filepath", how="left")
assert len(D) == 390 and D.notna().all().all()

spk = D.speaker_id.values
uspk = np.unique(spk)
idx_by = {s: np.where(spk == s)[0] for s in uspk}
yl, yw = D.label.values, D.words_say_depressed.values
conf = (D.set == "conflict").values


def auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def stats(col, idx=None):
    p = D[col].values if idx is None else D[col].values[idx]
    a, b, c = (yl, yw, conf) if idx is None else (yl[idx], yw[idx], conf[idx])
    return dict(gap=auc(b, p) - auc(a, p), auc_confl=auc(a[c], p[c]),
                auc_agree=auc(a[~c], p[~c]), auc_label=auc(a, p), auc_lex=auc(b, p))


for pr in ["main", "voice"]:
    cols = {nice: f"{s}_{pr}" for s, nice in MODELS}
    obs = {n: stats(c) for n, c in cols.items()}

    rng = np.random.default_rng(7)
    draws = {n: {k: [] for k in ["gap", "auc_confl"]} for n in cols}
    for _ in range(NB):
        s = rng.choice(uspk, len(uspk), replace=True)
        idx = np.concatenate([idx_by[x] for x in s])
        for n, c in cols.items():
            st = stats(c, idx)
            draws[n]["gap"].append(st["gap"])
            draws[n]["auc_confl"].append(st["auc_confl"])
    for n in cols:
        for k in draws[n]:
            draws[n][k] = np.array(draws[n][k], dtype=float)

    print("\n" + "=" * 96)
    print("PROMPT = %s   paired speaker-clustered bootstrap, %d draws, 74 speakers" % (pr, NB))
    print("=" * 96)
    print("\n%-18s %26s %26s" % ("model", "gap = AUC_lex - AUC_label", "conflict-set AUC vs label"))
    for n in cols:
        g, a = draws[n]["gap"], draws[n]["auc_confl"]
        print("%-18s  %+.3f [%+.3f,%+.3f]  p(gap<=0)=%.4f  %.3f [%.3f,%.3f] p(auc>=.5)=%.4f"
              % (n, obs[n]["gap"], np.nanpercentile(g, 2.5), np.nanpercentile(g, 97.5),
                 float(np.nanmean(g <= 0)),
                 obs[n]["auc_confl"], np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5),
                 float(np.nanmean(a >= 0.5))))

    print("\n-- pairwise contrast of the gap (model A minus model B), paired draws --")
    names = list(cols)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = draws[names[i]]["gap"] - draws[names[j]]["gap"]
            o = obs[names[i]]["gap"] - obs[names[j]]["gap"]
            star = "*" if (np.nanpercentile(d, 2.5) > 0 or np.nanpercentile(d, 97.5) < 0) else " "
            print("  %-17s - %-17s  %+.3f [%+.3f,%+.3f] %s"
                  % (names[i], names[j], o, np.nanpercentile(d, 2.5),
                     np.nanpercentile(d, 97.5), star))

    print("\n-- pairwise contrast of conflict-set AUC (model A minus model B) --")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = draws[names[i]]["auc_confl"] - draws[names[j]]["auc_confl"]
            o = obs[names[i]]["auc_confl"] - obs[names[j]]["auc_confl"]
            star = "*" if (np.nanpercentile(d, 2.5) > 0 or np.nanpercentile(d, 97.5) < 0) else " "
            print("  %-17s - %-17s  %+.3f [%+.3f,%+.3f] %s"
                  % (names[i], names[j], o, np.nanpercentile(d, 2.5),
                     np.nanpercentile(d, 97.5), star))

    print("\n-- three-reader average (qwen2-audio, omni, af3) vs kimi --")
    g3 = np.nanmean([draws[n]["gap"] for n in
                     ["qwen2-audio", "qwen2.5-omni", "audio-flamingo-3"]], axis=0)
    d = g3 - draws["kimi-audio"]["gap"]
    o3 = np.mean([obs[n]["gap"] for n in ["qwen2-audio", "qwen2.5-omni", "audio-flamingo-3"]])
    print("  gap(mean of 3) - gap(kimi) = %+.3f [%+.3f,%+.3f]"
          % (o3 - obs["kimi-audio"]["gap"], np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)))

    json.dump({n: obs[n] for n in cols},
              open(f"{F}/px_contrast_{pr}.json", "w"), indent=2, default=float)

print("\nJOB_DONE")
