"""
The literal PCLM question: does conditioning the layer weighting ON THE PROMPT help?

Experiment 1 pools a single task, so its "mixer" is just a learned static
weighting. Here the prompt actually varies: PC-GITA has 5 elicitation tasks and
Neurovoz 3, and each task is a different instruction to the speaker, i.e. the
closest thing this data has to a prompt. We pool all tasks of a corpus and
compare a mixer with ONE weight vector against a mixer with one weight vector
PER TASK, classifier shared, everything else identical.

  single_nested   best single layer, picked by inner speaker-disjoint CV
  mean_layers     unweighted mean of the 33 layers
  mixer_static    one softmax over layers for the whole corpus
  mixer_taskcond  one softmax over layers per task (prompt-conditioned)

GroupKFold by speaker, speaker-level shuffle control, speaker-clustered CIs.
Minibatch training so the larger pooled sets stay affordable.
"""
import os, json, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/committed"
NL = 33


def load_all(ds):
    z = np.load(f"{R}/features/qwen2/{ds}/encoder_features.npz", allow_pickle=True)
    assert int(z["n_layers"]) == NL
    X = np.stack([z[f"layer_{i:02d}"][()] for i in range(NL)], 1).astype(np.float32)
    return X, z["label"][()].astype(int), z["speaker"][()].astype(str), z["task"][()].astype(str)


def speaker_labels(y, spk):
    us = np.unique(spk)
    lab = {}
    for s in us:
        v = np.unique(y[spk == s])
        assert len(v) == 1, "speaker %s mixed labels %s" % (s, v)
        lab[s] = int(v[0])
    return us, np.array([lab[s] for s in us])


def shuffle_speaker_labels(y, spk, rng):
    us, ul = speaker_labels(y, spk)
    m = dict(zip(us, rng.permutation(ul)))
    return np.array([m[s] for s in spk], dtype=int)


class Mixer(nn.Module):
    def __init__(self, D, L=NL, mode="static", n_task=1):
        super().__init__()
        self.mode = mode
        if mode == "static":
            self.w = nn.Parameter(torch.zeros(L))
        else:
            self.emb = nn.Embedding(n_task, L)
            nn.init.zeros_(self.emb.weight)
        self.cls = nn.Linear(D, 1)

    def attn(self, x, t):
        if self.mode == "static":
            return torch.softmax(self.w, 0).unsqueeze(0).expand(x.shape[0], -1)
        return torch.softmax(self.emb(t), 1)

    def forward(self, x, t):
        a = self.attn(x, t)
        return self.cls(torch.einsum("bld,bl->bd", x, a)).squeeze(-1), a


def _T(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def train_mixer(Xtr, ytr, ttr, gtr, Xte, tte, mode, n_task, epochs=60,
                patience=10, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    ng = min(4, len(np.unique(gtr)))
    a_i, b_i = next(iter(GroupKFold(ng).split(Xtr, ytr, gtr)))
    tr_i, va_i = a_i, b_i
    if len(np.unique(ytr[tr_i])) < 2 or len(np.unique(ytr[va_i])) < 2:
        tr_i = va_i = np.arange(len(ytr))

    m = Mixer(Xtr.shape[2], mode=mode, n_task=n_task)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    npos = max(1, int((ytr[tr_i] == 1).sum()))
    pw = torch.tensor([float((ytr[tr_i] == 0).sum()) / npos], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    xa = _T(Xtr[tr_i]); ya = _T(ytr[tr_i].astype(np.float32)); ta = _T(ttr[tr_i])
    xv = _T(Xtr[va_i]); tv = _T(ttr[va_i]); yv = ytr[va_i]
    n = len(tr_i)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lo, _ = m(xa[j], ta[j])
            lossf(lo, ya[j]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            lv, _ = m(xv, tv)
        try:
            auc = roc_auc_score(yv, lv.numpy())
        except ValueError:
            auc = 0.5
        if auc > best + 1e-5:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    out = []
    with torch.no_grad():
        xt, tt = _T(Xte), _T(tte)
        for i in range(0, len(Xte), 1024):
            lo, _ = m(xt[i:i + 1024], tt[i:i + 1024])
            out.append(lo.numpy())
        A = m.attn(xt[:1], tt[:1]) if mode == "static" else torch.softmax(m.emb.weight, 1)
    return np.concatenate(out), A.detach().numpy()


def logreg(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    lr.fit(sc.transform(Xtr), ytr)
    return lr.decision_function(sc.transform(Xte))


def run_cv(X, y, spk, tid, n_task, seed=0, do_mixers=True, epochs=60, n_folds=5):
    n = len(y)
    keys = ["single_nested", "mean_layers"] + (["mixer_static", "mixer_taskcond"] if do_mixers else [])
    oof = {k: np.full(n, np.nan) for k in keys}
    picks, A_static, A_task = [], [], []
    for tr, te in GroupKFold(min(n_folds, len(np.unique(spk)))).split(X, y, spk):
        assert len(set(spk[tr]) & set(spk[te])) == 0, "SPEAKER LEAK"
        mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        gin = min(4, len(np.unique(spk[tr])))
        inner = np.zeros(NL)
        for itr, iva in GroupKFold(gin).split(Xtr, y[tr], spk[tr]):
            if len(np.unique(y[tr][iva])) < 2 or len(np.unique(y[tr][itr])) < 2:
                continue
            for L in range(NL):
                inner[L] += roc_auc_score(y[tr][iva], logreg(Xtr[itr][:, L], y[tr][itr], Xtr[iva][:, L]))
        pick = int(np.argmax(inner)); picks.append(pick)
        oof["single_nested"][te] = logreg(Xtr[:, pick], y[tr], Xte[:, pick])
        oof["mean_layers"][te] = logreg(Xtr.mean(1), y[tr], Xte.mean(1))
        if do_mixers:
            s, a = train_mixer(Xtr, y[tr], tid[tr], spk[tr], Xte, tid[te],
                               "static", n_task, epochs=epochs, seed=seed)
            oof["mixer_static"][te] = s; A_static.append(a)
            s, a = train_mixer(Xtr, y[tr], tid[tr], spk[tr], Xte, tid[te],
                               "taskcond", n_task, epochs=epochs, seed=seed)
            oof["mixer_taskcond"][te] = s; A_task.append(a)
    info = dict(picks=picks,
                attn_static=np.mean(A_static, 0)[0].tolist() if A_static else None,
                attn_taskcond=np.mean(A_task, 0).tolist() if A_task else None)
    return oof, info


def boot(y, spk, scores, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    us = np.unique(spk)
    idx = {s: np.where(spk == s)[0] for s in us}
    draws = {k: [] for k in scores}
    for _ in range(n):
        pick = rng.choice(len(us), len(us), replace=True)
        ii = np.concatenate([idx[us[j]] for j in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        for k in scores:
            draws[k].append(roc_auc_score(y[ii], scores[k][ii]))
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}
    return ci, draws


def paired(draws, a, b):
    d = np.array(draws[a]) - np.array(draws[b])
    return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), frac_gt0=float((d > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--n_shuffle", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    X, y, spk, task = load_all(args.ds)
    tasks = sorted(np.unique(task))
    tmap = {t: i for i, t in enumerate(tasks)}
    tid = np.array([tmap[t] for t in task], dtype=np.int64)
    us, ul = speaker_labels(y, spk)
    print("=" * 74)
    print("PROMPT-CONDITIONED LAYER MIXER  %s   %d clips  %d speakers  tasks=%s"
          % (args.ds, len(y), len(us), tasks))
    print("=" * 74, flush=True)

    oof, info = run_cv(X, y, spk, tid, len(tasks), seed=0)
    res = {}
    for k, s in oof.items():
        assert not np.isnan(s).any(), k
        res[k] = dict(auc=round(float(roc_auc_score(y, s)), 4))
    ci, draws = boot(y, spk, oof, n=2000, seed=1)
    for k in res:
        res[k]["ci95"] = [round(ci[k][0], 4), round(ci[k][1], 4)]
    per_task = {k: {t: round(float(roc_auc_score(y[task == t], oof[k][task == t])), 4)
                    for t in tasks} for k in oof}

    null = {k: [] for k in oof}
    for i in range(args.n_shuffle):
        yp = shuffle_speaker_labels(y, spk, np.random.default_rng(2000 + i))
        o, _ = run_cv(X, yp, spk, tid, len(tasks), seed=i, do_mixers=True, epochs=30)
        for k, s in o.items():
            null[k].append(float(roc_auc_score(yp, s)))
        print("  shuffle %d/%d  (%.0fs)" % (i + 1, args.n_shuffle, time.time() - t0), flush=True)
    for k in res:
        v = np.array(null[k])
        res[k]["null_mean"] = round(float(v.mean()), 4)
        res[k]["null_n"] = int(len(v))
        res[k]["perm_p"] = round((float((v >= res[k]["auc"]).sum()) + 1) / (len(v) + 1), 4)

    d = {"taskcond_minus_static": paired(draws, "mixer_taskcond", "mixer_static"),
         "taskcond_minus_single_nested": paired(draws, "mixer_taskcond", "single_nested"),
         "static_minus_single_nested": paired(draws, "mixer_static", "single_nested")}

    out = dict(dataset=args.ds, tasks=tasks, n_clips=int(len(y)), n_speakers=int(len(us)),
               results=res, per_task_auc=per_task, deltas=d,
               nested_layer_picks=info["picks"],
               attn_static=[round(v, 5) for v in info["attn_static"]],
               attn_taskcond={t: [round(v, 5) for v in info["attn_taskcond"][tmap[t]]] for t in tasks},
               runtime_s=round(time.time() - t0, 1))
    json.dump(out, open("%s/pclm_taskcond_%s.json" % (OUT, args.ds), "w"), indent=2)

    print("")
    print("%-18s%7s%18s%9s%9s" % ("method", "AUC", "95% CI", "shuffle", "perm p"))
    for k in ["single_nested", "mean_layers", "mixer_static", "mixer_taskcond"]:
        r = res[k]
        print("%-18s%7.3f  [%.3f,%.3f]%9.3f%9.3f"
              % (k, r["auc"], r["ci95"][0], r["ci95"][1], r["null_mean"], r["perm_p"]))
    print("")
    print("per-task AUC (pooled model, scored within task)")
    print("%-18s" % "method" + "".join("%12s" % t for t in tasks))
    for k in ["single_nested", "mean_layers", "mixer_static", "mixer_taskcond"]:
        print("%-18s" % k + "".join("%12.3f" % per_task[k][t] for t in tasks))
    print("")
    for k, v in d.items():
        print("  %-32s%+.4f  [%+.4f,%+.4f]  P(>0)=%.3f" % (k, v["mean"], v["lo"], v["hi"], v["frac_gt0"]))
    print("")
    a = np.array(info["attn_static"])
    print("static attention peak layer %d (w=%.3f)" % (int(a.argmax()), a.max()))
    print("per-task attention peaks (prompt-conditioned):")
    for t in tasks:
        v = np.array(info["attn_taskcond"][tmap[t]])
        print("   %-12s peak %2d  w=%.3f   mass 0-3 %.3f  4-15 %.3f  16-32 %.3f"
              % (t, int(v.argmax()), v.max(), v[:4].sum(), v[4:16].sum(), v[16:].sum()))
    print("")
    print("wrote %s/pclm_taskcond_%s.json  (%.0fs)" % (OUT, args.ds, time.time() - t0))


if __name__ == "__main__":
    main()
