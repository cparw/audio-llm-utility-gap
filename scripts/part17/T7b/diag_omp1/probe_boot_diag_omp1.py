"""E-DAIC full window: nested layer-selected probe with per-clip OOF predictions, + paired bootstrap.
Nesting: outer GroupKFold(5) by speaker, inner GroupKFold(4) picks the layer, 5 repeated splits.
E-DAIC full is ONE CLIP PER SPEAKER (275 clips, 275 speakers) so speaker folds == clip folds.
Probe: StandardScaler(train only) + LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0).
"""
import os, csv, json, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

D = "/workspace/diag_omp1"
OUT = f"{D}/out"
REPEATS = 5

def auc_rank(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); sv = s[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j+1] == sv[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))

# ---------- load ----------
sc = list(csv.DictReader(open(f"{OUT}/full_zeroshot_scores.csv")))
clips = [r["clip"] for r in sc]
y   = np.array([int(r["label"]) for r in sc])
spk = np.array([r["speaker"] for r in sc])
pz  = np.array([float(r["p_yes"]) for r in sc])
mass= np.array([float(r["mass"]) for r in sc])
fit = np.array([int(r["fit_in_context"]) for r in sc])
print(f"n clips {len(clips)}  n speakers {len(set(spk))}  positives {int(y.sum())}", flush=True)
assert len(set(spk)) == len(clips), "expected one clip per speaker"

S = {k: [] for k in ("enc","proj","llm","ans")}
for c in clips:
    z = np.load(f"{OUT}/shards/{c}.npz")
    for k in S: S[k].append(z[k])
for k in S:
    S[k] = np.stack(S[k])
    print(f"  {k}: {S[k].shape}", flush=True)

def probe(Xl, yl, tr, te):
    # yl MUST be the label vector aligned to Xl's rows. Passing it explicitly (rather than
    # closing over the global y) is what keeps the INNER loop's labels aligned to the
    # outer-train subset it is indexing into.
    ss = StandardScaler().fit(Xl[tr])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(ss.transform(Xl[tr]), yl[tr])
    return lr.predict_proba(ss.transform(Xl[te]))[:, 1]

def run_stage(key):
    X = S[key]; nL = X.shape[1]
    per_repeat, oof_all, picks = [], [], []
    for rep in range(REPEATS):
        oof = np.zeros(len(y))
        outer = GroupKFold(n_splits=5, shuffle=True, random_state=rep)
        for tr, te in outer.split(X, y, groups=spk):
            if nL == 1:
                best = 0
            else:
                inner = GroupKFold(n_splits=4, shuffle=True, random_state=rep)
                isp = list(inner.split(X[tr], y[tr], groups=spk[tr]))
                ytr = y[tr]; Xtr = X[tr]
                def score_layer(L):
                    o = np.zeros(len(tr))
                    for itr, ite in isp:
                        o[ite] = probe(Xtr[:, L, :], ytr, itr, ite)
                    return auc_rank(ytr, o)
                sc_l = Parallel(n_jobs=16, prefer="processes")(delayed(score_layer)(L) for L in range(nL))
                best = int(np.argmax(sc_l))
            picks.append(best)
            oof[te] = probe(X[:, best, :], y, tr, te)
        a = auc_rank(y, oof)
        per_repeat.append(round(a, 4)); oof_all.append(oof)
        print(f"  {key} repeat {rep}: AUC {a:.4f}  layers {picks[-5:]}", flush=True)
    oof_mean = np.mean(oof_all, axis=0)
    return dict(key=key, per_repeat=per_repeat, mean_of_repeat_aucs=round(float(np.mean(per_repeat)), 4),
                auc_of_mean_oof=round(auc_rank(y, oof_mean), 4), n_points=int(nL),
                layers_picked=picks, oof_mean=oof_mean, oof_repeats=np.array(oof_all))

res = {}
for k in ("enc","proj","llm","ans"):
    print(f"--- stage {k} ---", flush=True)
    res[k] = run_stage(k)

# ---------- per-clip csv ----------
with open(f"{OUT}/edaic_full_perclip.csv","w",newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["pid","label","p_yes_answer","answer_mass","oof_enc","oof_proj","oof_llm","oof_ans","fit_in_context"])
    for i,c in enumerate(clips):
        w.writerow([c.replace(".wav",""), y[i], pz[i], mass[i],
                    res["enc"]["oof_mean"][i], res["proj"]["oof_mean"][i],
                    res["llm"]["oof_mean"][i], res["ans"]["oof_mean"][i], fit[i]])
np.savez_compressed(f"{OUT}/edaic_full_oof_repeats.npz",
                    **{f"{k}_oof_repeats": res[k]["oof_repeats"] for k in res},
                    label=y, speaker=spk, p_yes=pz, clip=np.array(clips))

# ---------- bootstrap ----------
NB = 2000
uspk = np.array(sorted(set(spk)))
idx_of = {s:i for i,s in enumerate(spk)}
def boot(vecA, vecB=None):
    """CI for AUC(vecA), and if vecB given, for AUC(vecA)-AUC(vecB). One speaker draw per replicate."""
    rng = np.random.default_rng(0)
    va, vd = [], []
    for _ in range(NB):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.array([idx_of[uspk[p]] for p in pick])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        a = auc_rank(yy, vecA[ii]); va.append(a)
        if vecB is not None: vd.append(a - auc_rank(yy, vecB[ii]))
    q = lambda v: (float(np.percentile(v,2.5)), float(np.percentile(v,97.5)))
    out = dict(n_draws=NB, usable=len(va), auc_lo=q(va)[0], auc_hi=q(va)[1])
    if vecB is not None:
        out.update(diff_lo=q(vd)[0], diff_hi=q(vd)[1], diff_point=float(np.mean(vd)))
    return out

rows = []
def add(cid, what, value, lo, hi, extra=None):
    r = dict(id=cid, what=what, value=round(value,4), lo=round(lo,4), hi=round(hi,4),
             n=len(y), n_spk=len(uspk), excludes_zero="")
    if extra: r.update(extra)
    rows.append(r); print(f"{cid:34s} {what:46s} {value:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    return r

print("\n=== SINGLE-ARM AUCs with 2000-draw speaker bootstrap ===", flush=True)
b = boot(pz); add("edaic_full_zeroshot_answer","Qwen2.5-Omni zero-shot answer AUC", auc_rank(y,pz), b["auc_lo"], b["auc_hi"])
singles = {"enc":"encoder probe","proj":"projector probe","llm":"LM probe","ans":"answer-state probe"}
for k,lab in singles.items():
    v = res[k]["oof_mean"]; b = boot(v)
    add(f"edaic_full_probe_{k}", f"{lab} AUC (per-clip OOF averaged over repeats)", auc_rank(y,v), b["auc_lo"], b["auc_hi"])

print("\n=== PAIRED probe - answer, 2000-draw speaker bootstrap ===", flush=True)
gaps = []
for k,lab in [("llm","LM probe"),("enc","encoder probe"),("ans","answer-state probe"),("proj","projector probe")]:
    v = res[k]["oof_mean"]; b = boot(v, pz)
    ez = "yes" if (b["diff_lo"]>0 or b["diff_hi"]<0) else "no"
    r = add(f"edaic_full_gap_{k}_minus_answer", f"{lab} minus zero-shot answer (paired)",
            auc_rank(y,v)-auc_rank(y,pz), b["diff_lo"], b["diff_hi"])
    r["excludes_zero"] = ez; r["usable_draws"] = b["usable"]
    print(f"{'':34s} {'excludes zero:':46s} {ez}   usable draws {b['usable']}/{NB}", flush=True)
    gaps.append(r)

with open(f"{OUT}/edaic_full_gap.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id","what","value","lo","hi","n","n_spk","excludes_zero","usable_draws"],
                       extrasaction="ignore")
    w.writeheader()
    for r in rows: w.writerow({**{"usable_draws":NB}, **r})

summ = {k: {kk: vv for kk, vv in res[k].items() if kk not in ("oof_mean","oof_repeats","layers_picked")} for k in res}
for k in res: summ[k]["layers_picked"] = [int(x) for x in res[k]["layers_picked"]]
json.dump(summ, open(f"{OUT}/edaic_full_nested_repeats.json","w"), indent=1)
print("\n=== NESTED PROBE, BOTH CONVENTIONS ===", flush=True)
for k in ("enc","proj","llm","ans"):
    print(f"{k:5s} mean-of-{REPEATS}-repeat-AUCs {summ[k]['mean_of_repeat_aucs']:.4f}   "
          f"AUC-of-mean-OOF {summ[k]['auc_of_mean_oof']:.4f}   per_repeat {summ[k]['per_repeat']}", flush=True)
print("DONE", flush=True)
