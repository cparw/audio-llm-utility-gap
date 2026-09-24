"""PART 25 track C (r2) supplement: Kimi-Audio ENCODER probe from the PART 26 new extraction.
NOT a master_lookup row. master_lookup rows 17..178 have no Kimi-Audio encoder probe, and the release Method says the Kimi
encoder side is skipped. PART 26 added new hooks on the Whisper-large-v3 encoder inside Kimi-Audio (the continuous-feature
path), 32 layers, mean pooled, first 30 s, then a five-repeat nested probe on a Linux CPU pod. Whether the paper uses these
cells is the authors' decision; they are reported separately and never mixed into the master-row counts.
For each Kimi-Audio dataset whose PART 26 per-clip file AND sidecar exist (landed), own code:
  - the five per-repeat probe AUCs (sklearn roc_auc_score) from the per-clip file, compared with the PART 26 sidecar at 4 dp;
  - the answer = master_lookup Kimi-Audio "zero shot answer" row, recomputed from its per-clip csv;
  - paired speaker bootstrap exactly as in C_r2_recheck.py (2000 draws, fresh default_rng(0), np.unique of the probe-file
    speaker ids, idx = rng.choice(n_spk, size=n_spk, replace=True), mean of five per-repeat AUCs minus answer AUC per draw);
  - compared with the PART 26 sidecar interval.
Writes r2/kimi_enc/{perclip,sidecars,draws}/ and r2/kimi_enc/C_r2_kimi_enc_cells.json.
usage: /usr/local/bin/python3 C_r2_kimi_enc_supplement.py"""
import os, sys, re, csv, json, glob, hashlib, datetime, warnings
import numpy as np, pandas as pd, sklearn, scipy
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
P26 = "<local data dir>/release_from_mac/scores/part26"
OUT = "<local data dir>/release_from_mac/scores/part25/C/r2/kimi_enc"
ML = "<local data dir>/release/master/master_lookup.csv"
for d in ("perclip", "sidecars", "draws"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
NB = 2000
DSETS = [("pcgita", "PC-GITA", "PD"), ("neurovoz", "NeuroVoz", "PD"), ("kcl", "MDVR-KCL", "PD"), ("edaic", "E-DAIC", "MDD"),
         ("pitt", "Pitt", "AD"), ("adresso", "ADReSSo", "AD"), ("adress2020", "ADReSS-2020", "AD")]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def r4(x): return float(f"{x:.4f}")

with open(ML, newline="") as f:
    rd = csv.reader(f); header = next(rd); data = list(rd)
H = {h: i for i, h in enumerate(header)}
def master(ds, stream):
    hit = [(k, r) for k, r in enumerate(data, start=1) if 17 <= k <= 178 and r[H["model"]] == "Kimi-Audio" and r[H["dataset"]] == ds and r[H["stream"]] == stream]
    return hit[0] if hit else (None, None)

def find_lists(obj, path=""):
    """every list of five numbers in the sidecar whose key path mentions per_repeat"""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items(): out += find_lists(v, f"{path}.{k}")
    elif isinstance(obj, list) and len(obj) == 5 and all(isinstance(x, (int, float)) for x in obj) and "per_repeat" in path:
        out.append((path, obj))
    return out
def find_ci(obj, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items(): out += find_ci(v, f"{path}.{k}")
    elif isinstance(obj, list) and len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj) and re.search(r"gap(_ci|\.ci)", path) and "secondary" not in path and "sensitivity" not in path:
        out.append((path, obj))
    return out

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
cells = []
for ds, dname, cond in DSETS:
    folder = f"{P26}/Kimi-Audio_{dname}"
    pcs = sorted(glob.glob(f"{folder}/*perclip.csv") + glob.glob(f"{folder}/per_clip/*perclip.csv"))
    scs = []
    if len(pcs) == 1:
        dd = os.path.dirname(pcs[0]); stem0 = os.path.basename(pcs[0])[:-4]
        scs = [x for x in [f"{dd}/{stem0}.sidecar.json"] if os.path.exists(x)] or sorted(x for x in glob.glob(f"{dd}/*sidecar.json") if "backup" not in x)
    rec = dict(model="Kimi-Audio", dataset=dname, ds_key=ds, condition=cond, created_utc=now, script=os.path.abspath(__file__),
               status=None, not_master_lookup="PART 26 new extraction; not a master_lookup row")
    if len(pcs) != 1 or len(scs) < 1:
        rec["status"] = f"NOT LANDED at {now}: per-clip files {len(pcs)}, sidecars {len(scs)} in {folder}"
        cells.append(rec); print(dname, rec["status"]); continue
    pf = pcs[0]; sc = scs[0]
    P = pd.read_csv(pf, dtype={"clip": str, "speaker": str})
    cols = [c for c in P.columns if re.fullmatch(r"p_(probe_)?seed[0-4]", c)]
    if len(cols) != 5:
        rec["status"] = f"per-clip file has {len(cols)} per-repeat columns: {cols}"; cells.append(rec); print(dname, rec["status"]); continue
    S26 = json.load(open(sc))
    lists = find_lists(S26); cis = find_ci(S26)
    ai, arow = master(ds, "zero shot answer")
    src = arow[H["source"]]; af = src if src.endswith(".csv") else src.replace("_zeroshot.json", "_zeroshot_scores.csv")
    A = pd.read_csv(af, dtype={"clip": str, "speaker": str}).set_index("clip")
    y = P["label"].astype(int).values
    per = [float(roc_auc_score(y, P[c].astype(float).values)) for c in cols]
    keep = P["clip"].isin(A.index).values
    J = P.loc[keep].reset_index(drop=True); y = J["label"].astype(int).values
    z = A.loc[J["clip"], "p_yes"].astype(float).values; la = A.loc[J["clip"], "label"].astype(int).values
    sp = J["speaker"].astype(str).values
    Smat = np.stack([J[c].astype(float).values for c in cols], 1)
    probe = float(np.mean([roc_auc_score(y, Smat[:, k]) for k in range(5)])); ans = float(roc_auc_score(y, z))
    ids = np.unique(sp); where = {s: np.flatnonzero(sp == s) for s in ids}
    rng = np.random.default_rng(0); d = np.full(NB, np.nan); a = np.full(NB, np.nan); b = np.full(NB, np.nan)
    for t in range(NB):
        ii = np.concatenate([where[ids[i]] for i in rng.choice(len(ids), size=len(ids), replace=True)])
        if y[ii].min() == y[ii].max(): continue
        a[t] = np.mean([roc_auc_score(y[ii], Smat[ii, k]) for k in range(5)]); b[t] = roc_auc_score(y[ii], z[ii]); d[t] = a[t] - b[t]
    ok = ~np.isnan(d); lo, hi = [float(v) for v in np.percentile(d[ok], [2.5, 97.5])]
    stem = f"C_r2_kimi_{ds}_enc_part26"
    np.savez(f"{OUT}/draws/{stem}_draws.npz", diff=d, probe_mean5=a, answer=b, point_diff=probe - ans, seed=0, n_boot=NB, n_spk=len(ids), speaker_ids=ids)
    o = pd.DataFrame({"clip": J["clip"], "speaker": sp, "label": y})
    for k in range(5): o[f"p_enc_probe_rep{k}"] = Smat[:, k]
    o["p_yes_answer"] = z; o.to_csv(f"{OUT}/perclip/{stem}_perclip.csv", index=False, float_format="%.10g")
    rec.update(status="landed", part26_perclip=pf, part26_perclip_sha256=sha256(pf), part26_sidecar=sc,
               part26_per_repeat_lists=lists, part26_gap_ci=cis,
               per_repeat_recomputed=per, per_repeat_recomputed_4dp=[r4(v) for v in per],
               per_repeat_equal_part26_4dp=any([r4(v) for v in L] == [r4(v) for v in per] for _, L in lists),
               probe_mean5=probe, answer_row=ai, answer_master=float(arow[H["auc"]]), answer_file=af, answer_file_sha256=sha256(af),
               answer_recomputed=ans, answer_verified=r4(ans) == r4(float(arow[H["auc"]])),
               n=int(len(J)), n_lost=int((~keep).sum()), label_disagreements=int((la != y).sum()), n_spk=int(len(ids)), n_pos=int(y.sum()),
               diff_point=probe - ans, diff_lo=lo, diff_hi=hi, boot_usable=int(ok.sum()), excludes_zero=bool(lo > 0 or hi < 0),
               part26_interval_agrees_4dp=any(r4(c[0]) == r4(lo) and r4(c[1]) == r4(hi) for _, c in cis),
               master_lm_probe=float(master(ds, "LM probe")[1][H["auc"]]),
               env=dict(python=sys.version.split()[0], numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
               perclip_file=f"{OUT}/perclip/{stem}_perclip.csv", draws_file=f"{OUT}/draws/{stem}_draws.npz", method=__doc__)
    json.dump(rec, open(f"{OUT}/sidecars/{stem}.sidecar.json", "w"), indent=1, default=str)
    cells.append(rec)
    print(f"{dname:11s} enc(P26) {probe:.4f} per-rep eq P26 {rec['per_repeat_equal_part26_4dp']} ans {ans:.4f}/{rec['answer_master']} "
          f"diff {probe-ans:+.4f} [{lo:+.4f}, {hi:+.4f}] P26 ci {cis} agree {rec['part26_interval_agrees_4dp']} n={len(J)} spk={len(ids)}", flush=True)
json.dump(cells, open(f"{OUT}/C_r2_kimi_enc_cells.json", "w"), indent=1, default=str)
print("KIMI ENC SUPPLEMENT DONE", sum(c["status"] == "landed" for c in cells), "landed of 7")
