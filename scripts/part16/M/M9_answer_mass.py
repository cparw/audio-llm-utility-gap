#!/usr/bin/env python3
"""
PART 16 / M9 -- Answer mass sanity check.
Median P(Yes)+P(No) per model per dataset, from the existing per-clip score files.
Writes: part16/M9_answer_mass.csv, part16/M9_inventory.csv, part16/sidecars/M9.json, part16/rows/M9.tsv
NEVER writes under omni_final.
"""
import os, sys, json, hashlib, csv as _csv
import numpy as np, pandas as pd

REL   = "<local data dir>/Desktop/release"
OUT   = os.path.join(REL, "edaic_rerun", "part16")
ROWS  = os.path.join(OUT, "rows")
SIDE  = os.path.join(OUT, "sidecars")
DISC  = os.path.join(REL, "edaic_rerun", "DISCREPANCIES.md")
HDRTSV= "<local data dir>/scratch/all_csv_headers.tsv"
MASTER= os.path.join(REL, "master", "master_lookup.csv")
os.makedirs(ROWS, exist_ok=True); os.makedirs(SIDE, exist_ok=True)

disc_lines = []
def disc(sev, msg):
    disc_lines.append(f"- **{sev}** [M9 answer mass] {msg}")

def sha1mb(p):
    try:
        with open(p,'rb') as f: return hashlib.sha256(f.read(1024*1024)).hexdigest()
    except Exception as e: return "ERR:"+str(e)

# ---------------------------------------------------------------- 1. inventory
SCORE_COLS = ["p_yes","p_probe","p","p_egemaps","p_text","p_audio","p_dep","p_after"]
MASS_CANDS = ["answer_mass","mass","denom","p_sum","psum"]

def classify_model(p):
    b = os.path.basename(p).lower(); low = p.lower()
    if b.startswith("af2") or "/af2" in low or "af2_" in b: return "Audio Flamingo 2"
    if b.startswith("af3") or "/af3" in low or "af3_" in b: return "Audio Flamingo 3"
    if b.startswith("kimi") or "/kimi" in low: return "Kimi-Audio"
    if b.startswith("q3o") or "q3osft" in b: return "Qwen3-Omni-30B-A3B"
    if b.startswith("q2a") or b.startswith("qwen2audio") or "/probe2/" in low or "/probe_results/" in low or "/sft/qwen2audio" in low: return "Qwen2-Audio"
    if b.startswith("o25") or b.startswith("omni") or b.startswith("smoke_omni") or "/omni_final/" in low or "/omni_results/" in low or "/omni_probe/" in low: return "Qwen2.5-Omni"
    if b.startswith("egemaps"): return "eGeMAPS baseline"
    return "unattributed"

DSKEYS = ["adress2020","adresso237","adresso236","adresso","pcgita1100","pcgita40","pcgita","neurovoz1270","neurovoz",
          "kcl37","kcl30b","kcl30","kcl","edaic_conflict","edaicfull","edaic30","edaic275","edaic","pitt468","pitt_all","pitt",
          "conflict","smoke"]
def classify_dataset(p):
    b = os.path.basename(p).lower()
    for k in DSKEYS:
        if k in b: return k
    return "unknown"

inv = []
hdrs = []
if os.path.exists(HDRTSV):
    for line in open(HDRTSV, errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2: hdrs.append((parts[0], parts[1]))
else:
    print("!! header tsv missing", HDRTSV); sys.exit(1)
# pick up anything created after the scan
for extra_root in [os.path.join(REL,"edaic_rerun","part16")]:
    for dp,dn,fn in os.walk(extra_root):
        for f in fn:
            if f.lower().endswith(".csv"):
                pp = os.path.join(dp,f)
                if pp not in dict(hdrs):
                    try: hdrs.append((pp, open(pp,errors="replace").readline().strip()))
                    except Exception: pass

for p, h in hdrs:
    cols = [c.strip().strip('"').lower() for c in h.split(",")]
    sc = [c for c in SCORE_COLS if c in cols]
    if not sc: continue
    mc = [c for c in MASS_CANDS if c in cols]
    recon = ("p_yes_raw" in cols and "p_no_raw" in cols)
    inv.append(dict(file=p, model=classify_model(p), dataset=classify_dataset(p),
                    score_cols=";".join(sc), mass_col=(mc[0] if mc else ("p_yes_raw+p_no_raw" if recon else "")),
                    has_mass=bool(mc or recon), header=h))
inv_df = pd.DataFrame(inv).sort_values(["model","dataset","file"])
inv_path = os.path.join(OUT,"M9_inventory.csv")
inv_df.to_csv(inv_path, index=False)
print(f"[inventory] per-clip score files found: {len(inv_df)}   with answer mass: {int(inv_df.has_mass.sum())}   -> {inv_path}")

# ---------------------------------------------------------------- 2. canonical grid from master_lookup
m = pd.read_csv(MASTER)
ANSWER_STREAMS = {"zero shot answer","transcript only","projector fine tune"}

def resolve_perclip(src):
    """master_lookup 'source' may be a json; find the sibling per-clip csv."""
    if src.endswith(".csv") and os.path.exists(src): return src, "as listed"
    if src.endswith(".json"):
        stem = src[:-5]
        cands = [stem+"_scores.csv", stem+".csv", stem.replace("_zeroshot","")+"_zeroshot_scores.csv"]
        if stem.endswith("_sft"): cands.insert(0, stem[:-4]+"_oof.csv")
        for cand in cands:
            if os.path.exists(cand): return cand, "sibling of the listed json"
    if os.path.exists(src) and src.endswith(".csv"): return src, "as listed"
    return None, "NOT FOUND"

def mass_column(df):
    cols = {c.lower(): c for c in df.columns}
    for k in MASS_CANDS:
        if k in cols: return cols[k], None
    if "p_yes_raw" in cols and "p_no_raw" in cols: return None, ("p_yes_raw","p_no_raw")
    return None, None

def spk_column(df):
    for k in ("speaker","speaker_id","spk","pid"):
        for c in df.columns:
            if c.lower()==k: return c
    return None

def boot_median(vals, spk, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    if spk is None:
        spk = np.arange(len(vals))
    spk = np.asarray(spk)
    uniq = np.unique(spk)
    idx = {s: np.where(spk==s)[0] for s in uniq}
    out, usable = [], 0
    for _ in range(draws):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[s] for s in pick])
        if len(sel)==0: continue
        v = vals[sel]; v = v[np.isfinite(v)]
        if len(v)==0: continue
        out.append(np.median(v)); usable += 1
    if not out: return float("nan"), float("nan"), 0
    out = np.array(out)
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5)), usable

rows, sources = [], {}
for _, r in m.iterrows():
    model, ds, stream = r["model"], r["dataset"], r["stream"]
    src = str(r["source"])
    if stream not in ANSWER_STREAMS:
        # a probe score is a logistic-regression probability, not a Yes/No answer
        # distribution, so it has no answer mass. Not a discrepancy.
        rows.append(dict(model=model, dataset=ds, stream=stream, master_auc=r["auc"], master_n=r["n"],
                         file=src, resolved="probe stream - no per-clip answer distribution",
                         n=int(r["n"]), n_speakers=0, median_mass=np.nan, p10_mass=np.nan,
                         **{"share_below_0.01":np.nan,"share_below_0.15":np.nan},
                         mass_column_used="n/a - probe score, not an answer distribution",
                         lo=np.nan, hi=np.nan, boot_usable=0))
        continue
    f, how = resolve_perclip(src)
    base = dict(model=model, dataset=ds, stream=stream, master_auc=r["auc"], master_n=r["n"],
                file=(f or src), resolved=how)
    if f is None:
        disc("MEDIUM", f"{model} / {ds} / {stream}: no per-clip file resolved from master_lookup source `{src}`.")
        rows.append({**base, "n":0,"n_speakers":0,"median_mass":np.nan,"p10_mass":np.nan,
                     "share_below_0.01":np.nan,"share_below_0.15":np.nan,
                     "mass_column_used":"source file not found","lo":np.nan,"hi":np.nan,"boot_usable":0})
        continue
    try:
        df = pd.read_csv(f)
    except Exception as e:
        disc("MEDIUM", f"{model} / {ds} / {stream}: could not read `{f}` ({e}).")
        rows.append({**base,"n":0,"n_speakers":0,"median_mass":np.nan,"p10_mass":np.nan,
                     "share_below_0.01":np.nan,"share_below_0.15":np.nan,
                     "mass_column_used":"unreadable","lo":np.nan,"hi":np.nan,"boot_usable":0}); continue
    sources[f] = sha1mb(f)
    sc = spk_column(df)
    nspk = int(df[sc].nunique()) if sc else 0
    mcol, recon = mass_column(df)
    if mcol is None and recon is None:
        note = ("n/a - probe score, not an answer distribution"
                if stream not in ANSWER_STREAMS else "no mass recorded")
        if stream in ANSWER_STREAMS:
            disc("LOW", f"{model} / {ds} / {stream}: answer-probability file `{f}` carries no mass column; cell reported as 'no mass recorded'.")
        rows.append({**base,"n":len(df),"n_speakers":nspk,"median_mass":np.nan,"p10_mass":np.nan,
                     "share_below_0.01":np.nan,"share_below_0.15":np.nan,
                     "mass_column_used":note,"lo":np.nan,"hi":np.nan,"boot_usable":0}); continue
    if mcol is not None:
        mass = pd.to_numeric(df[mcol], errors="coerce").to_numpy(float); used = mcol
    else:
        mass = (pd.to_numeric(df[recon[0]],errors="coerce")+pd.to_numeric(df[recon[1]],errors="coerce")).to_numpy(float)
        used = "p_yes_raw + p_no_raw (reconstructed)"
    good = np.isfinite(mass)
    if good.sum() < len(mass):
        disc("LOW", f"{model} / {ds} / {stream}: {int((~good).sum())} of {len(mass)} mass values non-finite in `{f}`; dropped from the summary.")
    mv = mass[good]
    spkv = df[sc].to_numpy()[good] if sc else None
    lo, hi, usable = boot_median(mv, spkv, 2000, 0)
    rows.append({**base,"n":int(len(mv)),"n_speakers":nspk,
                 "median_mass":float(np.median(mv)),"p10_mass":float(np.percentile(mv,10)),
                 "share_below_0.01":float((mv<0.01).mean()),"share_below_0.15":float((mv<0.15).mean()),
                 "mass_column_used":used,"lo":lo,"hi":hi,"boot_usable":usable})

g = pd.DataFrame(rows)
cols = ["model","dataset","stream","n","n_speakers","median_mass","p10_mass","share_below_0.01",
        "share_below_0.15","mass_column_used","lo","hi","boot_usable","master_auc","resolved","file"]
g = g[cols].sort_values(["model","stream","dataset"])
csv_path = os.path.join(OUT,"M9_answer_mass.csv")
g.to_csv(csv_path, index=False)

# ---------------------------------------------------------------- 3. print the grid
models   = ["Qwen2.5-Omni","Qwen2-Audio","Qwen3-Omni-30B-A3B","Kimi-Audio","Audio Flamingo 2","Audio Flamingo 3"]
datasets = sorted(m.dataset.unique())
def grid(stream, title):
    print("\n"+"="*150); print(title); print("="*150)
    w = max(len(d) for d in datasets)+2
    print("model".ljust(22)+"".join(d.ljust(w) for d in datasets))
    for mo in models:
        line = mo.ljust(22)
        for d in datasets:
            sub = g[(g.model==mo)&(g.dataset==d)&(g.stream==stream)]
            if len(sub)==0: cell="  ."
            else:
                v = sub.iloc[0]["median_mass"]
                cell = "  -" if not np.isfinite(v) else f"OK {v:.4f}"
            line += cell.ljust(w)
        print(line)
    print("\nlegend:  OK <median mass>  = cell present and mass recorded     '-' = cell present, no mass recorded     '.' = gap, no such cell in the release")

grid("zero shot answer", "GRID A -- zero shot ANSWER (audio prompt): median answer mass P(Yes)+P(No) per model x dataset")
grid("transcript only", "GRID B -- transcript only (text prompt): median answer mass per model x dataset")

print("\n"+"="*150); print("GRID C -- every master_lookup cell, all streams, mass availability"); print("="*150)
piv = g.pivot_table(index=["model","stream"], columns="dataset", values="median_mass", aggfunc="first", dropna=False)
allcells = g.pivot_table(index=["model","stream"], columns="dataset", values="n", aggfunc="first", dropna=False)
w = max(len(d) for d in datasets)+2
print("model / stream".ljust(46)+"".join(d.ljust(w) for d in datasets))
for (mo,st) in allcells.index:
    line = f"{mo} / {st}"[:44].ljust(46)
    for d in datasets:
        n = allcells.loc[(mo,st), d] if d in allcells.columns else np.nan
        v = piv.loc[(mo,st), d] if d in piv.columns else np.nan
        if not (isinstance(n,(int,float)) and np.isfinite(n)): cell = "  ."
        elif isinstance(v,(int,float)) and np.isfinite(v):      cell = f"OK {v:.4f}"
        else:                                                   cell = "  -"
        line += cell.ljust(w)
    print(line)

# ---------------------------------------------------------------- 4. minimum
mm = g[np.isfinite(g.median_mass)]
print("\n"+"="*150)
print(f"CELLS WITH MASS RECORDED: {len(mm)} of {len(g)} master_lookup cells")
print(f"MINIMUM median answer mass across the whole grid: {mm.median_mass.min():.4f}")
worst = mm.sort_values("median_mass").head(12)
print("\n12 lowest cells:")
for _, r in worst.iterrows():
    print(f"  {r.median_mass:.4f}  [{r.lo:.4f}, {r.hi:.4f}]  p10={r.p10_mass:.4f}  <0.01={r['share_below_0.01']:.4f}  <0.15={r['share_below_0.15']:.4f}  n={int(r.n)} n_spk={int(r.n_speakers)}  {r.model} / {r.dataset} / {r.stream}  col={r.mass_column_used}")
print("\n12 highest cells:")
for _, r in mm.sort_values("median_mass").tail(12).iterrows():
    print(f"  {r.median_mass:.4f}  n={int(r.n)}  {r.model} / {r.dataset} / {r.stream}")

low = mm[mm.median_mass < 0.15]
print("\n"+"="*150)
print(f"CELLS WITH MEDIAN MASS BELOW 0.15: {len(low)}")
for _, r in low.iterrows():
    print(f"  {r.model} / {r.dataset} / {r.stream}  median={r.median_mass:.4f}  n={int(r.n)}  file={r.file}")
    disc("HIGH", f"{r.model} / {r.dataset} / {r.stream}: median answer mass {r.median_mass:.4f} < 0.15 in `{r.file}`. Published AUC depending on this cell: master_lookup {r.model}/{r.dataset}/{r.stream} = {r.master_auc}.")

# ------------------------------------------------- 4b. sweep EVERY mass file on disk
print("\n"+"="*150)
print("SWEEP -- every per-clip file on disk that records an answer mass (not just the published cells)")
print("="*150)
sweep = []
for _, r in inv_df[inv_df.has_mass].iterrows():
    f = r["file"]
    try: df = pd.read_csv(f)
    except Exception as e:
        disc("LOW", f"sweep: could not read mass-bearing file `{f}` ({e}).")
        continue
    mcol, recon = mass_column(df)
    if mcol is not None:
        mass = pd.to_numeric(df[mcol], errors="coerce").to_numpy(float); used = mcol
    elif recon is not None:
        mass = (pd.to_numeric(df[recon[0]],errors="coerce")+pd.to_numeric(df[recon[1]],errors="coerce")).to_numpy(float)
        used = "p_yes_raw + p_no_raw (reconstructed)"
    else:
        continue
    mv = mass[np.isfinite(mass)]
    if len(mv)==0:
        disc("MEDIUM", f"sweep: `{f}` has a mass column (`{used}`) but no finite values."); continue
    sc = spk_column(df)
    sweep.append(dict(model=r["model"], dataset=r["dataset"], n=int(len(mv)),
                      n_speakers=int(df[sc].nunique()) if sc else 0,
                      median_mass=float(np.median(mv)), p10_mass=float(np.percentile(mv,10)),
                      min_mass=float(mv.min()),
                      **{"share_below_0.01":float((mv<0.01).mean()),"share_below_0.15":float((mv<0.15).mean())},
                      mass_column_used=used, file=f))
sw = pd.DataFrame(sweep).sort_values("median_mass")
sw_path = os.path.join(OUT,"M9_all_files_mass.csv"); sw.to_csv(sw_path, index=False)
print(f"files swept: {len(sw)}")
print(f"MINIMUM median answer mass across EVERY mass-bearing file on disk: {sw.median_mass.min():.4f}")
print("\n15 lowest files:")
for _, r in sw.head(15).iterrows():
    print(f"  {r.median_mass:.4f}  p10={r.p10_mass:.4f}  min={r.min_mass:.4f}  <0.01={r['share_below_0.01']:.4f}  <0.15={r['share_below_0.15']:.4f}  n={int(r.n)}  {r.model} / {r.dataset}  {r.file}")
swlow = sw[sw.median_mass < 0.15]
print(f"\nfiles on disk whose median mass is below 0.15: {len(swlow)}")
for _, r in swlow.iterrows():
    print(f"  {r.median_mass:.4f}  n={int(r.n)}  {r.file}")
    disc("HIGH", f"sweep: `{r.file}` ({r.model} / {r.dataset}) has median answer mass {r.median_mass:.4f} < 0.15.")

# ---------------------------------------------------------------- 5. outputs
tsv = os.path.join(ROWS,"M9.tsv")
with open(tsv,"w") as fh:
    fh.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    for _, r in g.iterrows():
        if not np.isfinite(r.median_mass): continue
        rid = f"M9.{r.model.replace(' ','')}.{r.dataset}.{r.stream.replace(' ','_')}"
        fh.write(f"{rid}\tmedian answer mass P(Yes)+P(No)\t{r.median_mass:.4f}\t{r.lo:.4f}\t{r.hi:.4f}\t{int(r.n)}\t{int(r.n_speakers)}\t{r.file}\n")
    b = mm.sort_values("median_mass").iloc[0]
    fh.write(f"M9.MINIMUM\tminimum median answer mass across the published grid ({b.model} / {b.dataset} / {b.stream})\t{b.median_mass:.4f}\t{b.lo:.4f}\t{b.hi:.4f}\t{int(b.n)}\t{int(b.n_speakers)}\t{b.file}\n")
    if len(sw):
        c = sw.iloc[0]
        fh.write(f"M9.SWEEPMIN\tminimum median answer mass across every mass-bearing file on disk ({c.model} / {c.dataset})\t{c.median_mass:.4f}\t\t\t{int(c.n)}\t{int(c.n_speakers)}\t{c.file}\n")
        fh.write(f"M9.CELLS\tmaster_lookup cells with an answer mass recorded\t{len(mm)}\t\t\t{len(g)}\t\t{csv_path}\n")
        fh.write(f"M9.FILES\tper-clip score files on disk with an answer mass recorded\t{int(inv_df.has_mass.sum())}\t\t\t{int(len(inv_df))}\t\t{inv_path}\n")

side = dict(
    task="PART16 M9 answer mass",
    definition="answer mass = P(Yes)+P(No) at the first answer position, bare and leading-space token variants summed; p_yes = P(Yes)/mass",
    master_lookup=MASTER, master_lookup_sha256_first1mb=sha1mb(MASTER),
    header_scan_tsv=HDRTSV, n_csv_scanned=len(hdrs), n_perclip_score_files=int(len(inv_df)),
    n_perclip_files_with_mass=int(inv_df.has_mass.sum()),
    n_master_cells=int(len(g)), n_cells_with_mass=int(len(mm)),
    seed=0, bootstrap_draws=2000, bootstrap_unit="speaker", percentiles=[2.5,97.5],
    folds="not applicable - M9 computes no cross-validated quantity, so no fold file is read or recomputed",
    model_id="none - M9 reads existing score files only, no model is run",
    prompt="varies per source file; the prompt column of each source csv is preserved in that file",
    shell_command="/usr/local/bin/python3 scripts/part16/M/M9_answer_mass.py",
    outputs=[csv_path, inv_path, sw_path, tsv],
    source_files_sha256_first1mb=sources,
)
sp = os.path.join(SIDE,"M9.json")
json.dump(side, open(sp,"w"), indent=2)

if disc_lines:
    with open(DISC,"a") as fh:
        fh.write("\n\n## PART 16 / M9 answer mass (%s)\n" % pd.Timestamp.now().isoformat(timespec="seconds"))
        for l in sorted(set(disc_lines)): fh.write(l+"\n")

print("\n"+"="*150)
print("WROTE")
print("  "+csv_path); print("  "+inv_path); print("  "+sw_path); print("  "+tsv); print("  "+sp)
print("  discrepancy lines appended: %d -> %s" % (len(set(disc_lines)), DISC))
