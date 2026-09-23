"""p16twins VERIFIER: M5 (Qwen2-Audio readout direction) refit on the Mac, where M5 ran (sklearn 1.7.2).
Random floor: probe direction w = unit(mean over GroupKFold(5) folds of coef/scale); 2000 standard normal vectors of
dim 4096 drawn as one (2000, 4096) block from a fresh default_rng(0), each normalised; floor = mean |w.v|, CI = 2.5/97.5.
Also refits the whole pipeline (direction d from the cached lm_head shard) to cross-check the per-clip columns."""
import os, json, glob, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn
t0 = time.time()
z = np.load("<local data dir>/paper1_local_runs/probe2/pitt_states.npz", allow_pickle=True)
X = np.asarray(z["ans"][:, -1, :], dtype=np.float64); y = np.asarray(z["label"]).astype(int)
spk = np.asarray(z["spk"]).astype(str); name = np.asarray(z["name"]).astype(str)
man = pd.read_csv("<local data dir>/DementiaBank/pitt_conflict_manifest.csv", dtype={"spk": str})
setof = {os.path.basename(p): s for p, s in zip(man.segment_path, man.set)}
arm = np.array([setof[n] for n in name])
print("X", X.shape, "arms", {a: int((arm == a).sum()) for a in ("conflict", "agreement")}, flush=True)

snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/*"))[0]
import struct
_f = open(snap + "/model-00005-of-00005.safetensors", "rb"); _h = struct.unpack("<Q", _f.read(8))[0]
_hdr = json.loads(_f.read(_h)); _e = _hdr["language_model.lm_head.weight"]; _a, _b = _e["data_offsets"]
assert _e["dtype"] == "BF16", _e["dtype"]
_u = np.memmap(snap + "/model-00005-of-00005.safetensors", dtype="<u2", mode="r", offset=8 + _h + _a, shape=tuple(_e["shape"]))
W = (np.asarray(_u).astype(np.uint32) << 16).view(np.float32)
from tokenizers import Tokenizer
tk = Tokenizer.from_file(snap + "/tokenizer.json")
def ids(words):
    s = set()
    for w in words:
        for f in (w, " " + w):
            e = tk.encode(f, add_special_tokens=False).ids
            if len(e) == 1: s.add(e[0])
    return sorted(s)
YES_tok, NO_tok = ids(["Yes", "yes", "YES"]), ids(["No", "no", "NO"])
_j = json.load(open("<local data dir>/release/edaic_rerun/part16/M5_readout_direction_q2a.json"))
YES, NO = [int(i) for i in _j["yes_ids"]], [int(i) for i in _j["no_ids"]]   # token ids are an input definition, taken from M5
print("raw tokenizers ids", YES_tok, NO_tok, "M5 ids", YES, NO, flush=True)
print("W", W.shape, W.dtype, "YES", YES, "NO", NO, flush=True)
# log-softmax normaliser in chunks
lse = None
for a in range(0, W.shape[0], 20000):
    b = X @ W[a:a + 20000].astype(np.float64).T
    mx = b.max(1)
    c = mx + np.log(np.exp(b - mx[:, None]).sum(1))
    lse = c if lse is None else np.logaddexp(lse, c)
PY = np.exp(X @ W[YES].astype(np.float64).T - lse[:, None]); PN = np.exp(X @ W[NO].astype(np.float64).T - lse[:, None])
def direction(m):
    a = PY[m].mean(0); a = a / a.sum(); b = PN[m].mean(0); b = b / b.sum()
    d = W[YES].astype(np.float64).T @ a - W[NO].astype(np.float64).T @ b
    return d / np.linalg.norm(d)

def probe(Xm, ym, sm, d):
    oof = np.zeros(len(ym)); wo = np.zeros(len(ym)); dirs = []
    for tr, te in GroupKFold(n_splits=5).split(Xm, ym, groups=sm):
        sc = StandardScaler().fit(Xm[tr]); Xt = sc.transform(Xm[tr]); Xe = sc.transform(Xm[te])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xt, ym[tr])
        oof[te] = m.predict_proba(Xe)[:, 1]; c = m.coef_.ravel(); dirs.append(c / sc.scale_)
        cp = c - (c @ d) / (d @ d) * d; pt = Xt @ cp
        wo[te] = (Xe @ cp - pt.mean()) / (pt.std() + 1e-12)
    w = np.mean(dirs, 0); w = w / np.linalg.norm(w)
    return oof, wo, w

V = np.random.default_rng(0).normal(size=(2000, X.shape[1])); V = V / np.linalg.norm(V, axis=1, keepdims=True)
res = {}
per = pd.read_csv("<local data dir>/release/edaic_rerun/part16/M5_readout_direction_q2a.csv")
assert (per["clip"].to_numpy(str) == name).all()
for tag, m in (("all", np.ones(len(y), bool)), ("conflict_refit", arm == "conflict"), ("agreement_refit", arm == "agreement")):
    d = direction(m)
    oof, wo, w = probe(X[m], y[m], spk[m], d)
    r = np.abs(V @ w)
    col = "z_probe_oof_without_d_all" if tag == "all" else "z_probe_oof_without_d_armrefit"
    colp = "p_probe_oof_all" if tag == "all" else "p_probe_oof_armrefit"
    res[tag] = dict(randfloor=float(r.mean()), randfloor_ci=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                    cos=float(w @ d),
                    max_abs_diff_vs_csv_wo=float(np.abs(wo - per[col].to_numpy(float)[m]).max()),
                    max_abs_diff_vs_csv_probe=float(np.abs(oof - per[colp].to_numpy(float)[m]).max()))
    print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[tag].items()}, flush=True)
res["env"] = dict(sklearn=sklearn.__version__, numpy=np.__version__, seconds=round(time.time() - t0, 1), yes=YES, no=NO, yes_rawtok=YES_tok, no_rawtok=NO_tok, nan_in_lse=bool(np.isnan(lse).any()))
json.dump(res, open("<local data dir>/scratch/vp16/m5_mac_results.json", "w"), indent=1)
print("done", round(time.time() - t0, 1))
