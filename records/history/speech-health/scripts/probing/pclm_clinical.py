"""
PCLM on clinical speech: does a learned weighting over Qwen2-Audio encoder
layers beat picking one layer honestly (nested inside training folds)?

Methods, all inside the SAME outer GroupKFold by speaker:
  single_nested   best layer picked by inner speaker-disjoint CV on train only
  single_oracle   best layer picked on the test OOF, reported ONLY as a peeking
                  upper bound so the cost of honest selection is visible
  mean_layers     unweighted mean over the 33 layers
  concat_layers   all 33 layers concatenated (42240 dim)
  mixer_static    learned free softmax over layers, one weight vector
  mixer_inputcond learned softmax whose logits depend on the clip content

Speaker-level label shuffle control and speaker-clustered bootstrap CIs for all.
"""
import os, sys, json, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/committed"
NL = 33


# ----------------------------------------------------------------- data
def load(ds, task):
    z = np.load(f"{R}/features/qwen2/{ds}/encoder_features.npz", allow_pickle=True)
    keep = z["task"] == task if task != "ALL" else np.ones(len(z["label"]), bool)
    assert int(z["n_layers"]) == NL, z["n_layers"]
    X = np.stack([z[f"layer_{i:02d}"][keep] for i in range(NL)], 1).astype(np.float32)
    return X, z["label"][keep].astype(int), z["speaker"][keep].astype(str), z["task"][keep].astype(str)


def speaker_labels(y, spk):
    """One label per speaker; assert the corpus is speaker-pure."""
    us = np.unique(spk)
    lab = {}
    for s in us:
        v = np.unique(y[spk == s])
        assert len(v) == 1, "speaker %s has mixed labels %s" % (s, v)
        lab[s] = int(v[0])
    return us, np.array([lab[s] for s in us])


def shuffle_speaker_labels(y, spk, rng):
    us, ul = speaker_labels(y, spk)
    perm = rng.permutation(ul)
    m = dict(zip(us, perm))
    return np.array([m[s] for s in spk], dtype=int)


# ----------------------------------------------------------------- models
class Mixer(nn.Module):
    def __init__(self, D, L=NL, mode="static", n_task=1):
        super().__init__()
        self.mode = mode
        if mode == "static":
            self.w = nn.Parameter(torch.zeros(L))
        elif mode == "inputcond":
            self.proj = nn.Linear(D, 1)
        elif mode == "taskcond":
            self.emb = nn.Embedding(n_task, L)
            nn.init.zeros_(self.emb.weight)
        else:
            raise ValueError(mode)
        self.cls = nn.Linear(D, 1)

    def attn(self, x, t=None):
        if self.mode == "static":
            return torch.softmax(self.w, 0).unsqueeze(0).expand(x.shape[0], -1)
        if self.mode == "inputcond":
            return torch.softmax(self.proj(x).squeeze(-1), 1)
        return torch.softmax(self.emb(t), 1)

    def forward(self, x, t=None):
        a = self.attn(x, t)
        mixed = torch.einsum("bld,bl->bd", x, a)
        return self.cls(mixed).squeeze(-1), a


def _T(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def train_mixer(Xtr, ytr, gtr, Xte, mode, ttr=None, tte=None, n_task=1,
                epochs=400, patience=50, lr=1e-3, seed=0):
    """Inner speaker-disjoint val split for early stopping. Never sees test y."""
    torch.manual_seed(seed)
    ng = min(4, len(np.unique(gtr)))
    tr_i, va_i = next(iter(GroupKFold(ng).split(Xtr, ytr, gtr)))
    if len(np.unique(ytr[tr_i])) < 2 or len(np.unique(ytr[va_i])) < 2:
        tr_i = np.arange(len(ytr))
        va_i = np.arange(len(ytr))

    m = Mixer(Xtr.shape[2], mode=mode, n_task=n_task)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    npos = max(1, int((ytr[tr_i] == 1).sum()))
    pw = torch.tensor([float((ytr[tr_i] == 0).sum()) / npos], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    xa, ya = _T(Xtr[tr_i]), _T(ytr[tr_i].astype(np.float32))
    xv, yv = _T(Xtr[va_i]), ytr[va_i]
    ta = _T(ttr[tr_i]) if ttr is not None else None
    tv = _T(ttr[va_i]) if ttr is not None else None

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        m.train()
        opt.zero_grad()
        lo, _ = m(xa, ta)
        lossf(lo, ya).backward()
        opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
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
                if bad > max(1, patience // 5):
                    break
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        lo, _ = m(_T(Xte), _T(tte) if tte is not None else None)
        _, a_all = m(_T(Xtr), _T(ttr) if ttr is not None else None)
    return lo.numpy(), a_all.mean(0).numpy()


def logreg(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    lr.fit(sc.transform(Xtr), ytr)
    return lr.decision_function(sc.transform(Xte))


# ----------------------------------------------------------------- main CV
def run_cv(X, y, spk, seed=0, do_mixers=True, epochs=400, patience=50, n_folds=5):
    n = len(y)
    keys = ["single_nested", "mean_layers", "concat_layers"]
    if do_mixers:
        keys += ["mixer_static", "mixer_inputcond"]
    oof = {k: np.full(n, np.nan) for k in keys}
    oof_perlayer = np.full((n, NL), np.nan)
    picks, attn_static, attn_input = [], [], []

    gkf = GroupKFold(min(n_folds, len(np.unique(spk))))
    for tr, te in gkf.split(X, y, spk):
        assert len(set(spk[tr]) & set(spk[te])) == 0, "SPEAKER LEAK"
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-6
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd

        for L in range(NL):
            oof_perlayer[te, L] = logreg(Xtr[:, L], y[tr], Xte[:, L])

        # nested layer selection, inner CV on training speakers only
        gin = min(4, len(np.unique(spk[tr])))
        inner = np.zeros(NL)
        for itr, iva in GroupKFold(gin).split(Xtr, y[tr], spk[tr]):
            if len(np.unique(y[tr][iva])) < 2 or len(np.unique(y[tr][itr])) < 2:
                continue
            for L in range(NL):
                s = logreg(Xtr[itr][:, L], y[tr][itr], Xtr[iva][:, L])
                inner[L] += roc_auc_score(y[tr][iva], s)
        pick = int(np.argmax(inner))
        picks.append(pick)
        oof["single_nested"][te] = oof_perlayer[te, pick]

        oof["mean_layers"][te] = logreg(Xtr.mean(1), y[tr], Xte.mean(1))
        oof["concat_layers"][te] = logreg(Xtr.reshape(len(tr), -1), y[tr],
                                          Xte.reshape(len(te), -1))
        if do_mixers:
            s, a = train_mixer(Xtr, y[tr], spk[tr], Xte, "static",
                               epochs=epochs, patience=patience, seed=seed)
            oof["mixer_static"][te] = s
            attn_static.append(a)
            s, a = train_mixer(Xtr, y[tr], spk[tr], Xte, "inputcond",
                               epochs=epochs, patience=patience, seed=seed)
            oof["mixer_inputcond"][te] = s
            attn_input.append(a)

    aucs = [roc_auc_score(y, oof_perlayer[:, L]) for L in range(NL)]
    oracle_L = int(np.argmax(aucs))
    oof["single_oracle"] = oof_perlayer[:, oracle_L]
    info = dict(picks=picks, oracle_layer=oracle_L, per_layer_auc=aucs,
                attn_static=np.mean(attn_static, 0).tolist() if attn_static else None,
                attn_inputcond=np.mean(attn_input, 0).tolist() if attn_input else None)
    return oof, info


# ----------------------------------------------------------------- metrics
def metrics(y, s):
    auc = roc_auc_score(y, s)
    return auc, balanced_accuracy_score(y, (s > np.median(s)).astype(int))


def boot(y, spk, scores, n=2000, seed=0):
    """Speaker-clustered bootstrap, shared resamples so deltas are paired."""
    rng = np.random.default_rng(seed)
    us = np.unique(spk)
    idx = {s: np.where(spk == s)[0] for s in us}
    keys = list(scores)
    draws = {k: [] for k in keys}
    for _ in range(n):
        pick = rng.choice(len(us), len(us), replace=True)
        ii = np.concatenate([idx[us[j]] for j in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        for k in keys:
            draws[k].append(roc_auc_score(y[ii], scores[k][ii]))
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
          for k, v in draws.items()}
    return ci, draws


def paired_delta(draws, a, b):
    d = np.array(draws[a]) - np.array(draws[b])
    return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), frac_gt0=float((d > 0).mean()))


# ----------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--n_shuffle", type=int, default=50)
    ap.add_argument("--n_shuffle_mixer", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = "%s_%s" % (args.ds, args.task)
    t0 = time.time()

    X, y, spk, _ = load(args.ds, args.task)
    us, ul = speaker_labels(y, spk)
    print("=" * 74)
    print("PCLM  %s:%s   %d clips  %d speakers  (%d patient / %d control speakers)"
          % (args.ds, args.task, len(y), len(us), ul.sum(), (ul == 0).sum()))
    print("=" * 74, flush=True)

    oof, info = run_cv(X, y, spk, seed=0)
    res = {}
    for k, s in oof.items():
        assert not np.isnan(s).any(), k
        a, b = metrics(y, s)
        res[k] = dict(auc=round(float(a), 4), balacc=round(float(b), 4))

    ci, draws = boot(y, spk, oof, n=2000, seed=1)
    for k in res:
        res[k]["ci95"] = [round(ci[k][0], 4), round(ci[k][1], 4)]

    spk_res = {}
    if len(us) < len(y):
        for k, s in oof.items():
            ms = np.array([s[spk == u].mean() for u in us])
            spk_res[k] = round(float(roc_auc_score(ul, ms)), 4)

    # ---- speaker-level label shuffle control
    null = {k: [] for k in oof}
    for i in range(args.n_shuffle):
        rng = np.random.default_rng(1000 + i)
        yp = shuffle_speaker_labels(y, spk, rng)
        if len(np.unique(yp)) < 2:
            continue
        dm = i < args.n_shuffle_mixer
        o, _ = run_cv(X, yp, spk, seed=i, do_mixers=dm, epochs=200, patience=30)
        for k, s in o.items():
            null[k].append(float(roc_auc_score(yp, s)))
        if i % 5 == 0:
            print("  shuffle %d/%d  (%.0fs)" % (i + 1, args.n_shuffle, time.time() - t0), flush=True)
    for k in res:
        v = np.array(null[k])
        res[k]["null_mean"] = round(float(v.mean()), 4)
        res[k]["null_p95"] = round(float(np.percentile(v, 95)), 4)
        res[k]["null_n"] = int(len(v))
        res[k]["perm_p"] = round((float((v >= res[k]["auc"]).sum()) + 1) / (len(v) + 1), 4)

    deltas = {}
    for b in ["single_nested", "mean_layers", "concat_layers"]:
        for a in ["mixer_static", "mixer_inputcond"]:
            deltas["%s_minus_%s" % (a, b)] = paired_delta(draws, a, b)
    deltas["single_oracle_minus_single_nested"] = paired_delta(draws, "single_oracle", "single_nested")

    out = dict(dataset=args.ds, task=args.task, n_clips=int(len(y)),
               n_speakers=int(len(us)), n_patient_spk=int(ul.sum()),
               results=res, speaker_level_auc=spk_res, deltas=deltas,
               nested_layer_picks=info["picks"], oracle_layer=info["oracle_layer"],
               per_layer_auc=[round(v, 4) for v in info["per_layer_auc"]],
               attn_static=[round(v, 5) for v in info["attn_static"]],
               attn_inputcond=[round(v, 5) for v in info["attn_inputcond"]],
               runtime_s=round(time.time() - t0, 1))
    json.dump(out, open("%s/pclm_%s.json" % (OUT, tag), "w"), indent=2)

    print("")
    print("%s:%s  (%d clips, %d speakers)" % (args.ds, args.task, len(y), len(us)))
    hdr = "%-18s%7s%18s%9s%9s%8s" % ("method", "AUC", "95% CI", "shuffle", "perm p", "balacc")
    print(hdr)
    for k in ["single_nested", "single_oracle", "mean_layers", "concat_layers",
              "mixer_static", "mixer_inputcond"]:
        r = res[k]
        print("%-18s%7.3f  [%.3f,%.3f]%9.3f%9.3f%8.3f"
              % (k, r["auc"], r["ci95"][0], r["ci95"][1], r["null_mean"],
                 r["perm_p"], r["balacc"]))
    if spk_res:
        print("")
        print("speaker-level AUC: " + "  ".join("%s=%.3f" % (k, v) for k, v in spk_res.items()))
    print("")
    print("nested layer picks per fold: %s   oracle layer: %d" % (info["picks"], info["oracle_layer"]))
    a = np.array(info["attn_static"])
    print("static attention:    peak layer %d (w=%.3f), mass 0-3 %.3f, 4-15 %.3f, 16-32 %.3f"
          % (int(a.argmax()), a.max(), a[:4].sum(), a[4:16].sum(), a[16:].sum()))
    a2 = np.array(info["attn_inputcond"])
    print("inputcond attention: peak layer %d (w=%.3f), mass 0-3 %.3f  (uniform = %.3f/layer)"
          % (int(a2.argmax()), a2.max(), a2[:4].sum(), 1.0 / NL))
    print("")
    print("paired speaker-clustered bootstrap deltas (positive = first method better):")
    for k, v in deltas.items():
        print("  %-44s%+.4f  [%+.4f,%+.4f]  P(>0)=%.3f"
              % (k, v["mean"], v["lo"], v["hi"], v["frac_gt0"]))
    print("")
    print("wrote %s/pclm_%s.json   (%.0fs)" % (OUT, tag, time.time() - t0))


if __name__ == "__main__":
    main()
