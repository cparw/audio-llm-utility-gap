"""PART B (d): severity regression, SPEAKER level, nested layer+alpha selection.
usage: partB_sev.py <pcgita|neurovoz|dcaps|all>"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = R + "/final"; os.makedirs(OUT, exist_ok=True)
NL = 33; ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5]
NBOOT = 2000; NPERM = 50
WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"
SUF = "" if WHICH == "all" else "_" + WHICH

L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 2 * np.cov(a, b)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-12)

def load(ds):
    z = np.load(R + "/features/qwen2/%s/encoder_features.npz" % ds, allow_pickle=True)
    d = pd.DataFrame(dict(path=z["path"].astype(str), speaker=z["speaker"].astype(str),
                          label=z["label"].astype(int), task=z["task"].astype(str)))
    X = np.stack([z["layer_%02d" % i] for i in range(NL)]).astype(np.float32)
    return d, X

def spk_mat(X, d, mask):
    sp = d.speaker.values[mask]
    order = list(pd.unique(sp))
    idx = {s: np.where(sp == s)[0] for s in order}
    Xm = X[:, mask]
    return np.stack([np.stack([Xm[l][idx[s]].mean(0) for s in order]) for l in range(NL)]), order

def nested_pred(Xall, y, seed=0):
    n = len(y); pred = np.zeros(n); sel = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
        best, bl, ba = -9, 0, ALPHAS[0]
        for l in range(NL):
            for al in ALPHAS:
                ip = np.zeros(len(tr))
                for itr, ite in KFold(3, shuffle=True, random_state=seed + 1).split(tr):
                    A, B = tr[itr], tr[ite]
                    sc = StandardScaler().fit(Xall[l][A])
                    ip[ite] = Ridge(alpha=al).fit(sc.transform(Xall[l][A]), y[A]).predict(
                        sc.transform(Xall[l][B]))
                s = ccc(y[tr], ip)
                if s > best:
                    best, bl, ba = s, l, al
        sel.append(bl)
        sc = StandardScaler().fit(Xall[bl][tr])
        pred[te] = Ridge(alpha=ba).fit(sc.transform(Xall[bl][tr]), y[tr]).predict(
            sc.transform(Xall[bl][te]))
    return pred, sel

def flat_pred(X, y, seed=0, alpha=1e3):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(np.arange(len(y))):
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return pred

def boot(y, p, fn, nboot=NBOOT, seed=5):
    rng = np.random.default_rng(seed); o = []
    for _ in range(nboot):
        b = rng.choice(len(y), len(y), replace=True)
        if np.std(y[b]) < 1e-9:
            continue
        o.append(fn(y[b], p[b]))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))

rows = []
def cell(name, ds, task, target, Xall, y, extra=None):
    n = len(y)
    if n < 25 or np.std(y) < 1e-9:
        P("skip %s (n=%d)" % (name, n)); return
    pred, sel = nested_pred(Xall, y)
    c = ccc(y, pred); r = float(np.corrcoef(y, pred)[0, 1])
    clo, chi = boot(y, pred, ccc)
    rlo, rhi = boot(y, pred, lambda a, b: np.corrcoef(a, b)[0, 1])
    lay = np.array([ccc(y, flat_pred(Xall[l], y)) for l in range(NL)])
    peakL = int(np.argmax(lay)); cpeak = float(lay[peakL])
    rng = np.random.default_rng(3)
    nul = np.array([ccc(yp, nested_pred(Xall, yp)[0]) for yp in
                    (rng.permutation(y) for _ in range(NPERM))])
    pv = (1 + int((nul >= c).sum())) / (1 + len(nul))
    d = dict(theme="d_severity", cell=name, dataset=ds, task=task, target=target,
             n_speakers=n, stat="CCC (nested)", value=round(c, 4), ci_lo=round(clo, 4),
             ci_hi=round(chi, 4), ci_includes_zero=bool(clo <= 0 <= chi),
             pearson_r=round(r, 4), r_ci_lo=round(rlo, 4), r_ci_hi=round(rhi, 4),
             r_ci_includes_zero=bool(rlo <= 0 <= rhi),
             ccc_peak_of_33=round(cpeak, 4), peak_layer=peakL, inflation=round(cpeak - c, 4),
             layers_selected=",".join(map(str, sel)),
             perm_null_mean=round(float(nul.mean()), 4), perm_null_sd=round(float(nul.std()), 4),
             perm_p=pv, n_perm=NPERM)
    if extra:
        d.update(extra)
    rows.append(d)
    pd.DataFrame(rows).to_csv(OUT + "/tableD_severity%s.csv" % SUF, index=False)
    P("%-46s n=%3d  CCC %.3f [%.3f,%.3f]  r=%.3f  peak-of-33 %.3f (%+.3f)  perm p=%.3f"
      % (name, n, c, clo, chi, r, cpeak, cpeak - c, pv))

P("=" * 100); P("(d) SEVERITY REGRESSION, speaker level, nested layer+alpha  [%s]" % WHICH); P("=" * 100)

if WHICH in ("all", "pcgita"):
    xl = pd.ExcelFile("/project2/msoleyma_946/speech_health/spanish_dataset/"
                      "Copia de PCGITA_metadata.xlsx").parse("PD+HC")
    xl["sid"] = xl["RECODING ORIGINAL NAME"].astype(str).str.strip()
    d, X = load("pcgita")
    PC = pd.read_csv(OUT + "/pcgita_corrected.csv")
    d["keep"] = d.path.map(dict(zip(PC.filepath, PC.keep))).fillna(False).astype(bool)
    for task in ["read", "vowel", "ddk", "words", "monologue"]:
        m = ((d.task == task) & d.keep).values
        if m.sum() == 0:
            continue
        Xs, sp = spk_mat(X, d, m)
        md = xl.set_index("sid").reindex(sp)
        age = pd.to_numeric(md["AGE"], errors="coerce").values
        for tgt, col in [("UPDRS total", "UPDRS"), ("UPDRS speech item", "UPDRS-speech"),
                         ("Hoehn-Yahr", "H/Y"), ("years since diagnosis", "time after diagnosis")]:
            y = pd.to_numeric(md[col], errors="coerce").values
            ok = np.isfinite(y)
            if ok.sum() < 25:
                continue
            aa, yy = age[ok], y[ok]; k = np.isfinite(aa)
            ar = float(np.corrcoef(aa[k], yy[k])[0, 1]) if k.sum() > 5 else float("nan")
            cell("pcgita/%s/%s" % (task, tgt), "pcgita", task, tgt, Xs[:, ok], yy,
                 dict(age_only_pearson_r=round(ar, 3) if ar == ar else None,
                      note="patients only"))
    del X

if WHICH in ("all", "neurovoz"):
    nvm = pd.read_csv("/project2/msoleyma_946/speech_health/spanish_neurovoz/zenodo_upload/"
                      "metadata/data_pd.csv").drop_duplicates("ID")
    nvm["sid"] = nvm.ID.astype(str)
    d, X = load("neurovoz")
    NC = pd.read_csv(OUT + "/neurovoz_corrected.csv")
    d["taskc"] = d.path.map(dict(zip(NC.filepath, NC.task_type_corrected))).fillna(d.task)
    for task in ["read", "vowel", "ddk", "sentence_repeat", "spontaneous"]:
        m = (d.taskc == task).values
        if m.sum() == 0:
            continue
        Xs, sp = spk_mat(X, d, m)
        md = nvm.set_index("sid").reindex(sp)
        age = pd.to_numeric(md["Age"], errors="coerce").values
        for tgt, col in [("UPDRS total", "UPDRS scale"), ("Hoehn-Yahr", "H-Y Stadium"),
                         ("years since diagnosis", "Time Disease (years)")]:
            y = pd.to_numeric(md[col], errors="coerce").values
            ok = np.isfinite(y)
            if ok.sum() < 25:
                continue
            aa, yy = age[ok], y[ok]; k = np.isfinite(aa)
            ar = float(np.corrcoef(aa[k], yy[k])[0, 1]) if k.sum() > 5 else float("nan")
            cell("neurovoz/%s/%s" % (task, tgt), "neurovoz", task, tgt, Xs[:, ok], yy,
                 dict(age_only_pearson_r=round(ar, 3) if ar == ar else None,
                      note="patients only"))
    del X

if WHICH in ("all", "dcaps"):
    lab = pd.read_csv("/project2/msoleyma_946/speech_health/DCAPS_challenge/labels/"
                      "detailed_lables.csv")
    lab["sid"] = lab.Participant.astype(str)
    d, X = load("dcaps")
    Xs, sp = spk_mat(X, d, (d.task == "interview").values)
    md = lab.set_index("sid").reindex(sp)
    for tgt, col in [("PHQ-8 total", "Depression_severity"),
                     ("PCL-C PTSD severity", "PTSD_severity")]:
        y = pd.to_numeric(md[col], errors="coerce").values
        ok = np.isfinite(y)
        ag = pd.to_numeric(md["age"], errors="coerce").values[ok]; k = np.isfinite(ag)
        ar = float(np.corrcoef(ag[k], y[ok][k])[0, 1]) if k.sum() > 5 else float("nan")
        cell("dcaps/interview/%s" % tgt, "dcaps", "interview", tgt, Xs[:, ok], y[ok],
             dict(age_only_pearson_r=round(ar, 3) if ar == ar else None,
                  note="all speakers, severity defined for everyone"))

T = pd.DataFrame(rows)
T.to_csv(OUT + "/tableD_severity%s.csv" % SUF, index=False)
P(""); P(T.to_string(index=False))
open(OUT + "/partB_d_log%s.txt" % SUF, "w").write("\n".join(L) + "\n")
print("JOB_DONE", flush=True)
