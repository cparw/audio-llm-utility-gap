"""PART 17 task 4: record decisions and new values in master/master_lookup.csv.

Every value written here is recomputed from its per-clip file (rank formula, ties averaged)
and checked against the expected value before anything is written. The only exceptions are
the two E-DAIC mean-of-five rows: their per-repeat per-clip OOF file was never copied to this
Mac, so those two values are the mean of the per_repeat AUCs listed in the json, and the row
text says so.

Edits are made line by line on the pre-Part-17 backup: untouched rows stay byte identical.
Nothing is written under omni_final.
"""
import os, io, csv, json, hashlib, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata

R = "<local data dir>/release/"
GD = "<local data dir>/paper work/paper1_local_runs/"
LOOK = R + "master/master_lookup.csv"
BAK = R + "master/master_lookup.csv.bak_pre_part17"
SUP = R + "master/SUPERSEDED.csv"
P17 = R + "edaic_rerun/part17/"
FRAG = P17 + "rows/T4.tsv"
SIDE = P17 + "T4_lookup.json"
DISC = R + "edaic_rerun/DISCREPANCIES.md"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
STALE = "<local data dir>/paper1_local_runs/"
DRAWS = 2000
DATE = "2026-09-23"
CMD = "/usr/local/bin/python3 " + P17 + "T4_lookup.py"

for p in (FRAG, SIDE, SUP):
    assert not os.path.exists(p), "refusing to overwrite existing " + p

# ---------------------------------------------------------------- helpers
def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def ntie(s):
    _, c = np.unique(np.asarray(s, float), return_counts=True); return int((c > 1).sum())

def sha1mb(p):
    with open(p, "rb") as f: return hashlib.sha256(f.read(1 << 20)).hexdigest()

def boot(stat, spk):
    """fresh default_rng(0) per cell, speakers resampled with replacement, stat(rows) inside each draw."""
    spk = np.asarray(spk).astype(str)
    uniq, inv = np.unique(spk, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(0); K = len(uniq); v = []
    for _ in range(DRAWS):
        rows = np.concatenate([groups[i] for i in rng.integers(0, K, K)])
        x = stat(rows)
        if np.isfinite(x): v.append(x)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

SOURCES = {}
def src(p):
    assert os.path.exists(p), "missing " + p
    SOURCES[p] = sha1mb(p); return p

def check(name, got, want):
    ok = round(got, 4) == round(want, 4)
    print(f"  check {name:<46} recomputed {got:.6f}  expected {want:.4f}  {'OK' if ok else 'MISMATCH'}")
    if not ok: raise SystemExit("value did not reproduce: " + name)

# ---------------------------------------------------------------- Pitt arms spine
pm = pd.read_csv(src(MAN), dtype={"spk": str, "session": str})
pm["key"] = pm.segment_path.map(os.path.basename)
pm["spkkey"] = pm.grp.astype(str) + pm.spk.astype(str)
spine = pm[["key", "set", "spkkey", "label"]].copy()
assert len(spine) == 468 and (spine["set"] == "conflict").sum() == 146 and spine.key.is_unique

def pitt_join(path, keycol, pcol):
    d = pd.read_csv(path); d["key"] = d[keycol].map(os.path.basename)
    assert d.key.is_unique and len(d) == 468
    m = spine.merge(d[["key", pcol, "label"]].rename(columns={pcol: "p", "label": "lab_f"}), on="key", how="left")
    assert m.p.notna().all() and (m.label == m.lab_f).all()
    return m

RES = {}      # id -> dict(value, lo, hi, n, n_spk, file, what, draws)
def rec(i, what, value, lo, hi, n, nspk, f, draws=None):
    RES[i] = dict(what=what, value=value, lo=lo, hi=hi, n=n, n_spk=nspk, file=f, draws=draws)

def pitt_cells(tag, label, path, keycol, pcol, expect):
    m = pitt_join(src(path), keycol, pcol)
    y = m.label.values; p = m.p.values; spk = m.spkkey.values; arm = m["set"].values
    for cell, mask in (("pooled", np.ones(len(m), bool)), ("conflict", arm == "conflict"), ("agreement", arm == "agreement")):
        yy, pp, ss = y[mask], p[mask], spk[mask]
        v = auc(yy, pp)
        if expect.get(cell) is not None: check(f"{tag} {cell}", v, expect[cell])
        lo, hi, dr = boot(lambda r: auc(yy[r], pp[r]), ss)
        rec(f"T4_{tag}_{cell}", f"{label} Pitt {cell} zero-shot answer AUC", v, lo, hi, int(mask.sum()), len(set(ss)), path, dr)
    return m

print("recomputing every value from its per-clip file")

# 1. Qwen2.5-Omni Pitt encoder probe, mean of five per-repeat AUCs
NPZ = src(R + "overnight2/part10/pitt_enc_nested5_oof.npz")
z = np.load(NPZ, allow_pickle=True); y = z["label"].astype(int); oof = z["oof"]; zspk = z["spk"].astype(str)
per = [auc(y, oof[i]) for i in range(oof.shape[0])]
check("o25 pitt encoder mean of five", float(np.mean(per)), 0.7706)
check("o25 pitt encoder averaged-prob (NOT used)", auc(y, oof.mean(0)), 0.7917)
lo, hi, dr = boot(lambda r: float(np.mean([auc(y[r], oof[i][r]) for i in range(5)])), zspk)
rec("T4_pitt_o25_enc_mean5", "Qwen2.5-Omni Pitt encoder probe, mean of five per-repeat AUCs", float(np.mean(per)), lo, hi, len(y), len(set(zspk)), NPZ, dr)
lo, hi, dr = boot(lambda r: auc(y[r], oof.mean(0)[r]), zspk)
rec("T4_pitt_o25_enc_avgprob_NOTUSED", "Qwen2.5-Omni Pitt encoder probe, AUC of averaged probabilities (NOT used)", auc(y, oof.mean(0)), lo, hi, len(y), len(set(zspk)), NPZ, dr)
old_single = src(R + "omni_final/omni_pitt_enc_nested_oof.csv")
ds = pd.read_csv(old_single); check("o25 pitt encoder single split (SUPERSEDED)", auc(ds.label, ds.p_probe), 0.7969)

# 2-3. E-DAIC 300 s window, Qwen2.5-Omni, POD2 rebuild
EPC = src(R + "edaic_rerun/part16/EDAICFULL/edaic_full_perclip.csv")
EJS = src(R + "edaic_rerun/part16/EDAICFULL/edaic_full_nested_repeats.json")
e = pd.read_csv(EPC); assert len(e) == 275 and e.pid.is_unique
ey = e.label.values; espk = e.pid.astype(str).values
per_repeat_cols = [c for c in e.columns if "rep" in c.lower()]
for col, tag, want, lab in (("oof_llm", "llm", 0.7088, "LM"), ("oof_enc", "enc", 0.5916, "encoder")):
    v = auc(ey, e[col]); check(f"edaic {lab} probe averaged-prob", v, want)
    s = e[col].values
    lo, hi, dr = boot(lambda r: auc(ey[r], s[r]), espk)
    rec(f"T4_edaic_o25_{tag}_avgprob", f"Qwen2.5-Omni E-DAIC 300 s window {lab} probe, AUC of per-clip OOF averaged over 5 repeats", v, lo, hi, 275, 275, EPC, dr)
    a = e.p_yes_answer.values
    d0 = auc(ey, s) - auc(ey, a)
    lo, hi, dr = boot(lambda r: auc(ey[r], s[r]) - auc(ey[r], a[r]), espk)
    rec(f"T4_edaic_o25_{tag}_minus_answer_avgprob", f"Qwen2.5-Omni E-DAIC 300 s window {lab} probe minus zero-shot answer, paired, averaged-prob convention", d0, lo, hi, 275, 275, EPC, dr)
ej = json.load(open(EJS))
for k, tag, want, lab in (("llm", "llm", 0.7001, "LM"), ("enc", "enc", 0.5884, "encoder")):
    listed = ej[k]["per_repeat"]; v = float(np.mean(listed))
    check(f"edaic {lab} mean of five (json per_repeat, NOT per-clip)", v, want)
    assert round(v, 4) == ej[k]["mean_of_repeat_aucs"]
    rec(f"T4_edaic_o25_{tag}_mean5_jsononly", f"Qwen2.5-Omni E-DAIC 300 s window {lab} probe, mean of five per-repeat AUCs (json per_repeat list, per-clip repeats not on Mac)", v, None, None, 275, 275, EJS)
print("  E-DAIC per-repeat per-clip OOF columns in edaic_full_perclip.csv:", per_repeat_cols or "NONE (only the 5-repeat averaged oof_* columns)")

# 4. Qwen3-Omni encoder probes, POD3
def q3o_enc(tag, path, want_mean, lab):
    d = pd.read_csv(src(path)); cols = [f"p_probe_rep{i}" for i in range(5)]
    yy = d.label.values; ss = d.speaker.astype(str).values; P = d[cols].values
    per = [auc(yy, P[:, i]) for i in range(5)]
    check(f"q3o {tag} encoder mean of five", float(np.mean(per)), want_mean)
    lo, hi, dr = boot(lambda r: float(np.mean([auc(yy[r], P[r, i]) for i in range(5)])), ss)
    rec(f"T4_{tag}_q3o_enc_mean5", f"Qwen3-Omni {lab} encoder probe, mean of five per-repeat AUCs", float(np.mean(per)), lo, hi, len(d), len(set(ss)), path, dr)
    av = auc(yy, P.mean(1))
    lo, hi, dr = boot(lambda r: auc(yy[r], P[r].mean(1)), ss)
    rec(f"T4_{tag}_q3o_enc_avgprob_NOTUSED", f"Qwen3-Omni {lab} encoder probe, AUC of averaged probabilities (NOT used)", av, lo, hi, len(d), len(set(ss)), path, dr)
    return per, av
q3_pg = R + "edaic_rerun/part16/POD3/q3o_pcgita_encoder_nested_oof.csv"
q3_pt = R + "edaic_rerun/part16/POD3/q3o_pitt_encoder_nested_oof.csv"
pg_per, pg_av = q3o_enc("pcgita", q3_pg, 0.8969, "PC-GITA")
pt_per, pt_av = q3o_enc("pitt", q3_pt, 0.8122, "Pitt")
for mirror in (R + "edaic_rerun/part16/POD3/out/q3o_pcgita_encoder_nested_oof.csv", R + "edaic_rerun/part16/POD3/out/q3o_pitt_encoder_nested_oof.csv"):
    assert sha1mb(mirror) == SOURCES[q3_pg if "pcgita" in mirror else q3_pt]
pj = json.load(open(src(R + "edaic_rerun/part16/pod_sync/q3o_pitt_nested_repeats.json")))
assert pj["enc"]["mean"] == 0.8122 and [round(v, 4) for v in pt_per] == pj["enc"]["per_repeat"]
oldpg = json.load(open(src(R + "overnight2/q3o_new/q3o_pcgita_nested_repeats.json")))["enc"]
oldpt = json.load(open(src(R + "overnight2/final/q3o_pitt_nested_repeats.json")))["enc"]
print(f"  q3o pcgita old json enc mean {oldpg['mean']} per_repeat {oldpg['per_repeat']}  POD3 per_repeat {[round(v,4) for v in pg_per]}")
print(f"  q3o pitt   old json enc mean {oldpt['mean']} per_repeat {oldpt['per_repeat']}")

# 5. Kimi-Audio Pitt canonical + superseded copy
KIMI = R + "overnight2/part3/kimi_pitt_zeroshot_scores.csv"
KIMI_OLD = R + "overnight/kimi/kimi_pitt_zeroshot_scores.csv"
pitt_cells("pitt_kimi", "Kimi-Audio", KIMI, "clip", "p_yes", {"pooled": 0.7180, "conflict": 0.7197, "agreement": 0.7227})
pitt_cells("pitt_kimi_SUPERSEDED_overnight", "Kimi-Audio SUPERSEDED overnight/kimi copy", KIMI_OLD, "clip", "p_yes", {"pooled": 0.6964, "conflict": 0.7078, "agreement": 0.7014})

# 6. AF2 Pitt canonical + superseded
AF2 = GD + "af2_results/af2_pitt468.csv"
AF2_OLD = GD + "af2_pitt_audio.csv"
m_af2 = pitt_cells("pitt_af2", "Audio Flamingo 2", AF2, "clip_path", "p_yes", {"pooled": 0.5724, "conflict": 0.6310, "agreement": 0.5544})
af2_ag = m_af2[m_af2["set"] == "agreement"]
assert round(auc(af2_ag.label, af2_ag.p), 4) != 0.5536 and ntie(af2_ag.p) == 0
pitt_cells("pitt_af2_SUPERSEDED_audio", "Audio Flamingo 2 SUPERSEDED af2_pitt_audio", AF2_OLD, "path", "p_yes", {"pooled": 0.5273, "conflict": 0.6691, "agreement": 0.4895})

# 7. Qwen3-Omni Pitt by arm, full-fit answer-state probe restricted to the arm
QN = src(R + "edaic_rerun/part16/pod_sync/q3o_perp_cols.npz")
g = np.load(QN, allow_pickle=True)
gd = pd.DataFrame({"key": g["name"].astype(str), "oof": g["oof"], "lab": g["label"].astype(int)})
gm = spine.merge(gd, on="key", how="left"); assert gm.oof.notna().all() and (gm.label == gm.lab).all()
for cell, want in (("conflict", 0.6394), ("agreement", 0.9032)):
    mk = (gm["set"] == cell).values; yy = gm.label.values[mk]; pp = gm.oof.values[mk]; ss = gm.spkkey.values[mk]
    v = auc(yy, pp); check(f"q3o pitt {cell} full-fit answer-state probe", v, want)
    lo, hi, dr = boot(lambda r: auc(yy[r], pp[r]), ss)
    rec(f"T4_pitt_q3o_{cell}_fullfit", f"Qwen3-Omni Pitt {cell} arm answer-state probe, full-fit probe restricted to the arm", v, lo, hi, int(mk.sum()), len(set(ss)), QN, dr)
BYARM = src(R + "edaic_rerun/part16/POD3/out/readout_direction_q3o_by_arm.csv")
ba = pd.read_csv(BYARM).set_index("arm")
assert ba.loc["conflict", "auc_probe"] == 0.6145 and ba.loc["agreement", "auc_probe"] == 0.9349

# 8. LoRA and LoRA plus projector
LORA = R + "edaic_rerun/part16/POD2/lora_pitt_oof.csv"
BOTH = R + "omni_final/abl2_pitt_both_oof.csv"
for tag, path, kc, want, lab in (("lora", LORA, "name", 0.8250, "LoRA fine tune"), ("lora_plus_proj", BOTH, "path", 0.7696, "LoRA plus projector fine tune")):
    m = pitt_join(src(path), kc, "p_yes")
    yy = m.label.values; pp = m.p.values; ss = m.spkkey.values
    v = auc(yy, pp); check(f"o25 pitt {lab}", v, want)
    lo, hi, dr = boot(lambda r: auc(yy[r], pp[r]), ss)
    rec(f"T4_pitt_o25_{tag}", f"Qwen2.5-Omni Pitt {lab} out-of-fold AUC ({ntie(pp)} tied values, ties averaged)", v, lo, hi, 468, len(set(ss)), path, dr)
bj = json.load(open(src(R + "omni_final/abl_pitt_both.json"))); b2 = json.load(open(src(R + "omni_final/abl2_pitt_both.json")))
assert bj["auc_oof"] == 0.7872 and b2["auc_oof"] == 0.7696
src(R + "overnight2/part3/kimi_pitt_zeroshot.json"); src(R + "edaic_rerun/variants/o25_full_nested_repeats.json")
src(R + "edaic_rerun/part16/POD2/lora_pitt_oof.sidecar.json")

# ---------------------------------------------------------------- lookup edit, line level
raw = open(BAK, "rb").read()
assert sha1mb(BAK) == sha1mb(LOOK) and open(LOOK, "rb").read() == raw, "live lookup changed since the backup; stop"
lines = raw.decode("utf-8").split("\r\n"); assert lines[-1] == ""; lines = lines[:-1]
rows = [next(csv.reader([l])) for l in lines]
def ser(r):
    b = io.StringIO(); csv.writer(b, lineterminator="\r\n").writerow(r); return b.getvalue()[:-2]
assert all(ser(r) == l for r, l in zip(rows, lines)), "csv dialect does not round trip"
hdr = rows[0]; assert hdr == ["model", "dataset", "stream", "auc", "n", "estimator", "source"]
n_before = len(rows) - 1
stale_before = sum(STALE in l for l in lines[1:])

def fmt(v): return repr(round(float(v), 4))
def find(model, dsn, stream, old_auc):
    hit = [i for i, r in enumerate(rows) if i > 0 and r[:3] == [model, dsn, stream]]
    assert len(hit) == 1, (model, dsn, stream, hit)
    assert round(float(rows[hit[0]][3]), 4) == round(old_auc, 4), (model, dsn, stream, rows[hit[0]][3])
    return hit[0]

V = {k: v["value"] for k, v in RES.items()}
CHANGES = []
def update(model, dsn, stream, old_auc, new_auc, estimator, source, what):
    i = find(model, dsn, stream, old_auc); before = list(rows[i])
    rows[i] = [model, dsn, stream, fmt(new_auc), rows[i][4], estimator, source if source is not None else rows[i][6]]
    CHANGES.append(("updated", i, before, list(rows[i]), what))
def add(model, dsn, stream, v, n, estimator, source, what):
    assert not any(r[:3] == [model, dsn, stream] for r in rows[1:]), (model, dsn, stream)
    rows.append([model, dsn, stream, fmt(v), str(n), estimator, source])
    CHANGES.append(("new", len(rows) - 1, None, list(rows[-1]), what))

W300 = ("300 s window: the first 300 s of each E-DAIC full window (Qwen2_5OmniProcessor truncates audio at 300 s; "
        "217 of 275 windows truncated), not the full interview")

update("Qwen2.5-Omni", "pitt", "encoder probe", 0.7706, V["T4_pitt_o25_enc_mean5"],
       "mean of five per-repeat AUCs (five split nested, 5 repeats, part10 re-run); 0.7917 is the AUC of the per-clip "
       "probabilities averaged over the five repeats and is NOT used (authors' decision 2026-09-23)", None, "T4_pitt_o25_enc_mean5")
update("Qwen2.5-Omni", "edaic_full", "LM probe", 0.6853, V["T4_edaic_o25_llm_avgprob"],
       W300 + "; five split nested, 5 repeats, AUC of the per-clip OOF probability AVERAGED over the 5 repeats "
       "(the mean of the 5 per-repeat AUCs from this same run is 0.7001, separate row); POD2 rebuild 2026-09-23, "
       "E-DAIC 275 speakers [SUPERSEDES variants/o25_full_nested_repeats.json 0.6853, mean of 5 repeats, original run, "
       "no per-clip file]", EPC, "T4_edaic_o25_llm_avgprob")
update("Qwen2.5-Omni", "edaic_full", "encoder probe", 0.5930, V["T4_edaic_o25_enc_avgprob"],
       W300 + "; five split nested, 5 repeats, AUC of the per-clip OOF probability AVERAGED over the 5 repeats "
       "(the mean of the 5 per-repeat AUCs from this same run is 0.5884, separate row); POD2 rebuild 2026-09-23, "
       "E-DAIC 275 speakers [SUPERSEDES variants/o25_full_nested_repeats.json 0.5930, mean of 5 repeats, original run, "
       "no per-clip file]", EPC, "T4_edaic_o25_enc_avgprob")
update("Qwen3-Omni-30B-A3B", "pcgita", "encoder probe", 0.8969, V["T4_pcgita_q3o_enc_mean5"],
       "mean of five per-repeat AUCs (five split nested, layer chosen inside the training folds), POD3 re-extraction; "
       "the AUC of the averaged probabilities is 0.9109 and is NOT used; same convention as the Qwen3-Omni Pitt encoder row "
       "[SUPERSEDES source overnight2/q3o_new/q3o_pcgita_nested_repeats.json: same mean 0.8969, different per-repeat AUCs, "
       "no per-clip file]", q3_pg, "T4_pcgita_q3o_enc_mean5")
update("Qwen3-Omni-30B-A3B", "pitt", "encoder probe", 0.8289, V["T4_pitt_q3o_enc_mean5"],
       "mean of five per-repeat AUCs (five split nested, layer chosen inside the training folds), POD3 re-extraction; "
       "the AUC of the averaged probabilities is 0.8354 and is NOT used; per-repeat AUCs also in "
       "part16/pod_sync/q3o_pitt_nested_repeats.json [SUPERSEDES overnight2/final/q3o_pitt_nested_repeats.json 0.8289, "
       "no per-clip file]", q3_pt, "T4_pitt_q3o_enc_mean5")
update("Kimi-Audio", "pitt", "zero shot answer", 0.7180, V["T4_pitt_kimi_pooled"],
       "zero shot, 30 s window; canonical Kimi-Audio Pitt file (authors' decision 2026-09-23); conflict arm 0.7197 and "
       "agreement arm 0.7227 from the same file (rows pitt_conflict / pitt_agreement) [SUPERSEDES "
       "overnight/kimi/kimi_pitt_zeroshot_scores.csv: pooled 0.6964, conflict 0.7078, agreement 0.7014; source was "
       "overnight2/part3/kimi_pitt_zeroshot.json, same run, now the per-clip csv]", KIMI, "T4_pitt_kimi_pooled")
update("Audio Flamingo 2", "pitt", "zero shot answer", 0.5724, V["T4_pitt_af2_pooled"],
       "zero shot; canonical AF2 Pitt file (authors' decision 2026-09-23); conflict arm 0.6310 and agreement arm 0.5544 "
       "from the same file (rows pitt_conflict / pitt_agreement) [SUPERSEDES af2_pitt_audio.csv: pooled 0.5273, "
       "conflict 0.6691, agreement 0.4895]", AF2, "T4_pitt_af2_pooled")

JSONONLY = ("NOT recomputed from per-clip scores: the per-repeat OOF file edaic_full_oof_repeats.npz was written on the "
            "pod and is not on this Mac, so this is the mean of the five per_repeat AUCs listed in the json")
add("Qwen2.5-Omni", "edaic_full", "LM probe (mean of 5 per-repeat AUCs)", V["T4_edaic_o25_llm_mean5_jsononly"], 275,
    W300 + "; five split nested, MEAN OF FIVE PER-REPEAT AUCs, the convention decided for Pitt (0.7706); same POD2 run "
    "as the 0.7088 LM probe row; " + JSONONLY + " (0.6918, 0.7062, 0.6893, 0.7055, 0.7078)", EJS, "T4_edaic_o25_llm_mean5_jsononly")
add("Qwen2.5-Omni", "edaic_full", "encoder probe (mean of 5 per-repeat AUCs)", V["T4_edaic_o25_enc_mean5_jsononly"], 275,
    W300 + "; five split nested, MEAN OF FIVE PER-REPEAT AUCs, the convention decided for Pitt (0.7706); same POD2 run "
    "as the 0.5916 encoder probe row; " + JSONONLY + " (0.5964, 0.5826, 0.5962, 0.5608, 0.606)", EJS, "T4_edaic_o25_enc_mean5_jsononly")
ARMTXT = "arm from DementiaBank/pitt_conflict_manifest.csv set column, AUC restricted to the arm"
add("Kimi-Audio", "pitt_conflict", "zero shot answer", V["T4_pitt_kimi_conflict"], 146,
    "zero shot, 30 s window, Pitt conflict arm only (146 clips, 100 speakers; " + ARMTXT + "); canonical file "
    "(authors' decision 2026-09-23) [SUPERSEDES overnight/kimi/kimi_pitt_zeroshot_scores.csv conflict 0.7078]", KIMI, "T4_pitt_kimi_conflict")
add("Kimi-Audio", "pitt_agreement", "zero shot answer", V["T4_pitt_kimi_agreement"], 322,
    "zero shot, 30 s window, Pitt agreement arm only (322 clips, 175 speakers; " + ARMTXT + "); canonical file "
    "(authors' decision 2026-09-23) [SUPERSEDES overnight/kimi/kimi_pitt_zeroshot_scores.csv agreement 0.7014]", KIMI, "T4_pitt_kimi_agreement")
add("Audio Flamingo 2", "pitt_conflict", "zero shot answer", V["T4_pitt_af2_conflict"], 146,
    "zero shot, Pitt conflict arm only (146 clips, 100 speakers; " + ARMTXT + "); canonical file (authors' decision "
    "2026-09-23) [SUPERSEDES af2_pitt_audio.csv conflict 0.6691]", AF2, "T4_pitt_af2_conflict")
add("Audio Flamingo 2", "pitt_agreement", "zero shot answer", V["T4_pitt_af2_agreement"], 322,
    "zero shot, Pitt agreement arm only (322 clips, 175 speakers; " + ARMTXT + "); canonical file (authors' decision "
    "2026-09-23); the decision text typed 0.5536, which no AF2 file produces (0 tied scores, so no tie convention "
    "gives it); 0.5544 is what this file gives [SUPERSEDES af2_pitt_audio.csv agreement 0.4895]", AF2, "T4_pitt_af2_agreement")
FULLFIT = ("full-fit probe restricted to the arm, same as Qwen2.5-Omni Part H: logistic probe on the final LM stage at "
           "the first answer position, out of fold over all 468 Pitt clips (GroupKFold 5 by speaker, not nested), AUC "
           "computed on the arm's clips")
add("Qwen3-Omni-30B-A3B", "pitt_conflict", "answer-state probe", V["T4_pitt_q3o_conflict_fullfit"], 146,
    FULLFIT + " (146 clips, 100 speakers); 0.6145 is the probe refit within the arm (POD3) and is NOT used "
    "(authors' decision 2026-09-23); per-clip column oof", QN, "T4_pitt_q3o_conflict_fullfit")
add("Qwen3-Omni-30B-A3B", "pitt_agreement", "answer-state probe", V["T4_pitt_q3o_agreement_fullfit"], 322,
    FULLFIT + " (322 clips, 175 speakers); 0.9349 is the probe refit within the arm (POD3), the estimator not used "
    "under the same decision; per-clip column oof", QN, "T4_pitt_q3o_agreement_fullfit")
add("Qwen2.5-Omni", "pitt", "LoRA fine tune", V["T4_pitt_o25_lora"], 468,
    "LoRA rank 8 (alpha 16, dropout 0.05) on q/k/v/o of all 28 thinker LM layers, audio tower and projector frozen, "
    "3 epochs, lr 1e-4, 5 fold GroupKFold by speaker, out of fold AUC on p_yes, 30 s window, POD2 fresh run "
    "2026-09-23 (Part 16); 125 tied p_yes values, ties averaged", LORA, "T4_pitt_o25_lora")
add("Qwen2.5-Omni", "pitt", "LoRA plus projector fine tune", V["T4_pitt_o25_lora_plus_proj"], 468,
    "LoRA rank 8 plus projector trained together (9,637,376 trainable params), 3 epochs, lr 1e-4, 5 folds, out of "
    "fold AUC, 30 s window, run 2026-09-18; recomputed from the per-clip csv (114 tied p_yes values, ties averaged); "
    "abl_pitt_both.json claims 0.7872 but no per-clip file pairs with it; abl2_pitt_both.json, which pairs with this "
    "csv, says 0.7696; the csv value is recorded", BOTH, "T4_pitt_o25_lora_plus_proj")

new_lines = [ser(r) for r in rows]
changed_idx = {c[1] for c in CHANGES}
for i, l in enumerate(lines):
    if i not in changed_idx: assert new_lines[i] == l
out = ("\r\n".join(new_lines) + "\r\n").encode("utf-8")
with open(LOOK, "wb") as f: f.write(out)
n_after = len(rows) - 1
stale_after = sum(STALE in l for l in new_lines[1:])
chk = pd.read_csv(LOOK)
assert len(chk) == n_after and not chk.duplicated(["model", "dataset", "stream"]).any()
for kind, i, b, a, rid in CHANGES:
    assert round(float(a[3]), 4) == round(RES[rid]["value"], 4)
    if a[6].startswith("/"): assert os.path.exists(a[6].split(" [")[0]), a[6]
    assert STALE not in a[6] and STALE not in a[5]

# ---------------------------------------------------------------- SUPERSEDED registry
SUPROWS = [
 (KIMI_OLD, KIMI, "Kimi-Audio Pitt: older copy with different per-clip scores (pooled 0.6964, conflict 0.7078, agreement "
  "0.7014); canonical is the part3 file (pooled 0.7180, conflict 0.7197, agreement 0.7227), authors' decision"),
 (AF2_OLD, AF2, "Audio Flamingo 2 Pitt: pooled 0.5273, conflict 0.6691, agreement 0.4895; canonical is af2_pitt468.csv "
  "(pooled 0.5724, conflict 0.6310, agreement 0.5544), authors' decision; old path <local data dir>/paper1_local_runs/af2_pitt_audio.csv"),
 (old_single, NPZ, "as the Qwen2.5-Omni Pitt encoder probe source: single-split nested OOF 0.7969; the paper value is the "
  "mean of five per-repeat AUCs 0.7706 (the averaged-probability AUC of the same five repeats, 0.7917, is not used either)"),
 (R + "edaic_rerun/variants/o25_full_nested_repeats.json", EPC, "as the source of the Qwen2.5-Omni edaic_full LM probe "
  "(0.6853) and encoder probe (0.5930) only, both mean of 5 repeats from the original run; replaced by the POD2 rebuild "
  "averaged-probability AUCs 0.7088 / 0.5916 (mean-of-five from the rebuild: 0.7001 / 0.5884); the json remains the "
  "source of the edaic_full answer-state (0.7542) and projector (0.5293) rows"),
 (BYARM, QN, "Qwen3-Omni Pitt by arm: conflict 0.6145 / agreement 0.9349 = answer-state probe refit within the arm; "
  "the decided estimator is the full-fit probe restricted to the arm, conflict 0.6394 / agreement 0.9032"),
 (R + "overnight2/final/q3o_pitt_nested_repeats.json", q3_pt, "as the source of the Qwen3-Omni Pitt encoder probe only "
  "(0.8289, no per-clip file); replaced by the POD3 re-extraction 0.8122, mean of five per-repeat AUCs; the json remains "
  "the source of the Qwen3-Omni Pitt LM, answer-state and projector rows"),
 (R + "overnight2/q3o_new/q3o_pcgita_nested_repeats.json", q3_pg, "as the source of the Qwen3-Omni PC-GITA encoder probe "
  "only: same mean 0.8969 but no per-clip file; POD3 per-clip file gives 0.8969 (mean of five per-repeat AUCs)"),
 (R + "omni_final/abl_pitt_both.json", BOTH, "claims LoRA plus projector auc_oof 0.7872 with no per-clip file; the per-clip "
  "csv recomputes to 0.7696 and pairs with abl2_pitt_both.json (0.7696); the csv value is recorded"),
]
with open(SUP, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "superseded_by", "reason", "decided_on"])
    for a, b, c in SUPROWS: w.writerow([a, b, c, DATE])

# ---------------------------------------------------------------- fragment + sidecar
def f4(x): return "" if x is None else f"{x:.4f}"
os.makedirs(P17 + "rows", exist_ok=True)
with open(FRAG, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["id", "what", "value", "lo", "hi", "n", "n_spk", "file"])
    for k, r in RES.items(): w.writerow([k, r["what"], f4(r["value"]), f4(r["lo"]), f4(r["hi"]), r["n"], r["n_spk"], r["file"]])
    w.writerow(["T4_lookup_rows_before", "master_lookup.csv data rows before Part 17 T4", n_before, "", "", "", "", BAK])
    w.writerow(["T4_lookup_rows_after", "master_lookup.csv data rows after Part 17 T4", n_after, "", "", "", "", LOOK])
    w.writerow(["T4_lookup_stale_prefix_rows", "master_lookup.csv rows whose source still starts <local data dir>/paper1_local_runs/ (after T4)", stale_after, "", "", "", "", LOOK])

side = dict(
    task="PART 17 task 4: record decisions and new values in master/master_lookup.csv", date=DATE,
    exact_command=CMD, seed="numpy.random.default_rng(0), fresh per cell", bootstrap_draws=DRAWS,
    bootstrap_unit="speaker, resampled with replacement within the cell's own rows; mean-of-five cells recompute all five per-repeat AUCs inside each draw; paired cells recompute both AUCs inside the same draw",
    percentiles=[2.5, 97.5], auc_method="rank formula, scipy rankdata average ties, recomputed from per-clip scores",
    pitt_speakers="manifest grp+spk (228 on the 468 clips; 100 conflict, 175 agreement)",
    source_sha256_first1mb=SOURCES,
    lookup=dict(file=LOOK, backup=BAK, rows_before=n_before, rows_after=n_after,
                stale_prefix_rows_before=stale_before, stale_prefix_rows_after=stale_after,
                sha256_before=hashlib.sha256(raw).hexdigest(), sha256_after=hashlib.sha256(out).hexdigest(),
                untouched_rows_byte_identical=True,
                changes=[dict(kind=k, line=i + 1, before=b, after=a, recomputed_id=rid) for k, i, b, a, rid in CHANGES]),
    superseded_registry=SUP,
    edaic_per_repeat_perclip="edaic_full_perclip.csv carries only oof_* columns averaged over the 5 repeats; the per-repeat OOF (edaic_full_oof_repeats.npz, written by probe_boot.py on the pod) is not on this Mac, G-Drive or any scratch folder. The mean-of-five values 0.7001 / 0.5884 cannot be recomputed from per-clip scores and no mean-of-five paired probe-minus-answer interval can be computed.",
    values={k: dict(v, n=int(v["n"]), n_spk=int(v["n_spk"])) for k, v in RES.items()},
)
json.dump(side, open(SIDE, "w"), indent=1, default=float)

# ---------------------------------------------------------------- print
print(f"\nmaster_lookup.csv rows before {n_before}, after {n_after}; stale-prefix rows before {stale_before}, after {stale_after}")
print("\nmodel | dataset | stream | value | n | estimator | source | new/updated")
for kind, i, b, a, rid in CHANGES:
    print(" | ".join([a[0], a[1], a[2], f"{RES[rid]['value']:.4f}", a[4], a[5], a[6], kind]))
print("\nintervals (2000 draws, speakers, default_rng(0) per cell):")
for k, r in RES.items():
    print(f"  {k:<46} {r['value']:.4f} [{f4(r['lo'])}, {f4(r['hi'])}] n={r['n']} spk={r['n_spk']}")
