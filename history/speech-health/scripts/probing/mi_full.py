#!/usr/bin/env python
"""TASK 2 - mutual information between every participant attribute and the
diagnosis label, for all five datasets.

Advisor, 17 July: "look at the mutual information between different attributes
of people and the labels, to see if you're picking up something else".

Everything is computed at the SPEAKER level (one row per person), because these
are properties of people, not of clips.  Clip-level MI would just be speaker MI
multiplied by however many clips that speaker contributed.

MI estimator: plug-in discrete MI in bits.  Continuous attributes (age, UPDRS,
PHQ-8 ...) are quantile-binned into 4 bins.  The plug-in estimator is biased
upward when bins are many and n is small, so every value is reported against a
label-permutation null (1000 permutations) and a permutation p-value.  Read the
null column, not the raw MI.

max possible MI = H(label), also reported, so MI can be read as a fraction of
the total label information.
"""
import os, sys, json, re, unicodedata
import numpy as np
import pandas as pd

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
SH = "/project2/msoleyma_946/speech_health"
OUT = os.path.join(R, "committed")
os.makedirs(OUT, exist_ok=True)
NPERM = 1000
NBIN = 4
RS = np.random.RandomState(20260804)

# attributes that are definitionally downstream of the diagnosis - a control
# cannot have a UPDRS score, so MI with the label is guaranteed and meaningless
# as evidence of confounding.  Flagged, not hidden.
DEFINED_BY_LABEL = {
    "severity_updrs", "updrs", "updrs_speech", "hy", "time_disease",
    "hypophonic", "vocal_tremor", "dysphagia", "sialorrhoea",
    "cephalic_tremor", "mandibular_tremor", "medication", "medication_status",
    "phq8_total", "depression_severity", "diagnosis",
}
PHQ_ITEM = re.compile(r"^phq8_\d+$")


def disc(v, nbin=NBIN):
    """Return integer codes + n_levels.  NaN -> its own level so missingness is
    visible rather than silently dropped."""
    v = pd.Series(v)
    if v.dtype.kind in "OSUb" or v.nunique(dropna=True) <= nbin:
        codes = v.astype("object").where(v.notna(), "__NA__").astype(str)
        return pd.factorize(codes)[0]
    num = pd.to_numeric(v, errors="coerce")
    q = pd.qcut(num, nbin, labels=False, duplicates="drop")
    q = pd.Series(q).where(num.notna(), -1)
    return pd.factorize(q.astype(str))[0]


def mi_bits(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    ct = pd.crosstab(a, b).values.astype(float)
    p = ct / n
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = p * np.log2(p / (px * py))
    return float(np.nansum(t))


def entropy_bits(b):
    p = pd.Series(b).value_counts(normalize=True).values
    return float(-(p * np.log2(p)).sum())


def mi_with_null(attr, label):
    ok = pd.notna(pd.Series(label))
    a = disc(pd.Series(attr)[ok.values])
    y = np.asarray(pd.Series(label)[ok.values])
    n = len(y)
    n_obs = int(pd.notna(pd.Series(attr)[ok.values]).sum())
    if n_obs == 0 or len(set(a)) < 2:
        return dict(n=n, n_observed=n_obs, mi_bits=np.nan, null_mean=np.nan,
                    null_p95=np.nan, p_perm=np.nan, mi_frac_of_H=np.nan,
                    n_levels=len(set(a)))
    m = mi_bits(a, y)
    nulls = np.empty(NPERM)
    for i in range(NPERM):
        nulls[i] = mi_bits(a, RS.permutation(y))
    H = entropy_bits(y)
    return dict(n=n, n_observed=n_obs, mi_bits=m, null_mean=float(nulls.mean()),
                null_p95=float(np.percentile(nulls, 95)),
                p_perm=float((np.sum(nulls >= m) + 1) / (NPERM + 1)),
                mi_frac_of_H=float(m / H) if H > 0 else np.nan,
                n_levels=len(set(a)))


# ------------------------------------------------------ build attribute tables
def manifest(ds):
    return pd.read_csv(os.path.join(R, "manifests", ds + ".csv"))


tables = {}
prov = {}

# ---- PC-GITA : xlsx has AGE SEX UPDRS UPDRS-speech H/Y time-after-diagnosis
m = manifest("pcgita")
spk = m.drop_duplicates("speaker_id")[["speaker_id", "label"]].copy()
x = pd.read_excel(os.path.join(SH, "spanish_dataset", "Copia de PCGITA_metadata.xlsx"),
                  sheet_name="PD+HC")
x.columns = [str(c).strip() for c in x.columns]
x["id"] = x["RECODING ORIGINAL NAME"].astype(str).str.strip()
assert x["id"].is_unique, "pcgita metadata ids not unique"
# EXACT id join.  AVPEPUDEA#### = patient, AVPEPUDEAC#### = control; never strip the C.
j = spk.merge(x, left_on="speaker_id", right_on="id", how="left")
t = pd.DataFrame(dict(speaker_id=j.speaker_id, label=j.label,
                      age=pd.to_numeric(j["AGE"], errors="coerce"),
                      sex=j["SEX"].astype(str).str.strip().str.upper().str[:1],
                      updrs=pd.to_numeric(j["UPDRS"], errors="coerce"),
                      updrs_speech=pd.to_numeric(j["UPDRS-speech"], errors="coerce"),
                      hy=pd.to_numeric(j["H/Y"], errors="coerce"),
                      time_disease=pd.to_numeric(j["time after diagnosis"], errors="coerce")))
t.loc[t.sex.isin(["N", "NAN", ""]), "sex"] = np.nan
tables["pcgita"] = t
prov["pcgita"] = "Copia de PCGITA_metadata.xlsx sheet PD+HC, exact-id join"
print("[pcgita] speakers %d, matched to metadata %d, unmatched %d"
      % (len(t), int(j["id"].notna().sum()), int(j["id"].isna().sum())))

# ---- Neurovoz : per-speaker clinical metadata
mv = manifest("neurovoz")
mv["base"] = mv.filepath.map(os.path.basename)
mv["grp"] = mv.base.str.split("_").str[0].str.upper()
mv["sid_int"] = pd.to_numeric(
    mv.base.str.replace(".wav", "", regex=False).str.split("_").str[-1], errors="coerce")
sp = mv.drop_duplicates("speaker_id")[["speaker_id", "label", "grp", "sid_int"]]
fr = []
for f, g in [("data_hc.csv", "HC"), ("data_pd.csv", "PD")]:
    dd = pd.read_csv(os.path.join(SH, "spanish_neurovoz", "zenodo_upload", "metadata", f))
    dd.columns = [c.strip() for c in dd.columns]
    dd["grp"] = g
    fr.append(dd)
md = pd.concat(fr, ignore_index=True)
md["sid_int"] = pd.to_numeric(md["ID"], errors="coerce")
md = md.drop_duplicates(subset=["grp", "sid_int"])
j = sp.merge(md, on=["grp", "sid_int"], how="left")
t = pd.DataFrame(dict(speaker_id=j.speaker_id, label=j.label,
                      age=pd.to_numeric(j["Age"], errors="coerce"),
                      sex=pd.to_numeric(j["Sex"], errors="coerce").map({1.0: "M", 0.0: "F"})))
for name, col in [("updrs", "UPDRS scale"), ("hy", "H-Y Stadium"),
                  ("time_disease", "Time Disease (years)"),
                  ("hypophonic", "Hypophonic voice"), ("vocal_tremor", "Vocal tremor"),
                  ("cephalic_tremor", "Cephalic tremor"),
                  ("mandibular_tremor", "Mandibular tremor"),
                  ("dysphagia", "Dysphagia"), ("sialorrhoea", "Sialorrhoea"),
                  ("medication_status", "Medication status"),
                  ("labour_situation", "Labour situation")]:
    if col in j.columns:
        t[name] = pd.to_numeric(j[col], errors="coerce") if name not in (
            "medication_status", "labour_situation") else j[col].astype(str)
tables["neurovoz"] = t
prov["neurovoz"] = "zenodo_upload/metadata/data_hc.csv + data_pd.csv, join on (group, zero-padded last filename field)"
print("[neurovoz] speakers %d, age present %d" % (len(t), int(t.age.notna().sum())))

# ---- DAIC / dcaps : PHQ-8 items, PTSD, gender, age, split
mm = manifest("dcaps")
sp = mm.drop_duplicates("speaker_id")[["speaker_id", "label"]].copy()
sp["pid"] = pd.to_numeric(sp.speaker_id, errors="coerce")
dd = pd.read_csv(os.path.join(SH, "DCAPS_challenge", "labels", "detailed_lables.csv"))
dd.columns = [c.strip() for c in dd.columns]
dd["pid"] = pd.to_numeric(dd["Participant"], errors="coerce")
j = sp.merge(dd, on="pid", how="left")
t = pd.DataFrame(dict(speaker_id=j.speaker_id, label=j.label,
                      age=pd.to_numeric(j["age"], errors="coerce"),
                      sex=j["gender"].astype(str).str.strip().str.lower().map(
                          {"male": "M", "female": "F", "1": "M", "0": "F"}),
                      split=j["split"].astype(str),
                      depression_severity=pd.to_numeric(j["Depression_severity"], errors="coerce"),
                      ptsd_label=pd.to_numeric(j["PTSD_label"], errors="coerce"),
                      ptsd_severity=pd.to_numeric(j["PTSD_severity"], errors="coerce")))
for i in range(1, 9):
    c = "PHQ8_%d" % i
    if c in j.columns:
        t["phq8_%d" % i] = pd.to_numeric(j[c], errors="coerce")
pcl = [c for c in j.columns if c.upper().startswith("PCL-C_")]
if pcl:
    t["pclc_total"] = j[pcl].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
tables["dcaps"] = t
prov["dcaps"] = "DCAPS_challenge/labels/detailed_lables.csv, join on Participant id"
print("[dcaps] speakers %d, age present %d" % (len(t), int(t.age.notna().sum())))

# ---- Italian and KCL : take whatever the prebuilt demographics table has
demo = pd.read_csv(os.path.join(R, "soleymani_checks", "demographics_speakers.csv"))
for ds in ["italian", "kcl"]:
    dsub = demo[demo.dataset == ds].copy()
    man = manifest(ds)
    spk = man.drop_duplicates("speaker_id")[["speaker_id", "label"]]
    if len(dsub):
        t = spk.merge(dsub.drop(columns=["label"], errors="ignore"),
                      on="speaker_id", how="left")
    else:
        t = spk.copy()
    # italian: the corpus is organised as young_hc / elderly_hc / pd, so the
    # recruitment subgroup is itself an attribute worth testing
    if "group" in man.columns:
        gm = man.drop_duplicates("speaker_id").set_index("speaker_id")["group"]
        t["subgroup_recruitment"] = t.speaker_id.map(gm)
    if "duration_sec" in man.columns:
        dur = man.groupby("speaker_id")["duration_sec"].mean()
        t["mean_clip_duration_s"] = t.speaker_id.map(dur)
        t["n_clips"] = t.speaker_id.map(man.groupby("speaker_id").size())
    keepc = [c for c in ["speaker_id", "label", "age", "sex", "subgroup_recruitment",
                         "mean_clip_duration_s", "n_clips"] if c in t.columns]
    tables[ds] = t[keepc]
    prov[ds] = "soleymani_checks/demographics_speakers.csv + manifest-derived"
    print("[%s] speakers %d, age present %d" % (ds, len(t),
          int(pd.to_numeric(t.get("age"), errors="coerce").notna().sum()) if "age" in t else 0))

# recording-load attributes for every dataset (n clips / mean duration per speaker)
for ds in ["pcgita", "neurovoz", "dcaps"]:
    man = manifest(ds)
    t = tables[ds]
    t["n_clips"] = t.speaker_id.map(man.groupby("speaker_id").size())
    fp = os.path.join(R, "soleymani_checks", "feats_%s.csv" % ds)
    if os.path.exists(fp):
        fe = pd.read_csv(fp)
        fe["speaker_id"] = fe["speaker_id"].astype(str)
        t["mean_clip_duration_s"] = t.speaker_id.astype(str).map(
            fe.groupby("speaker_id")["duration_s"].mean())
    tables[ds] = t

# ---------------------------------------------------------------- compute MI
rows = []
for ds, t in tables.items():
    y = pd.to_numeric(t["label"], errors="coerce")
    H = entropy_bits(y.dropna())
    for col in t.columns:
        if col in ("speaker_id", "label"):
            continue
        r = mi_with_null(t[col], y)
        base = col.lower()
        flag = ""
        if base in DEFINED_BY_LABEL or PHQ_ITEM.match(base):
            flag = "DEFINED BY THE LABEL - a control cannot score on this; MI is tautological"
        elif base in ("n_clips", "mean_clip_duration_s"):
            flag = "recording-protocol property, not a person attribute"
        elif base == "subgroup_recruitment":
            flag = "recruitment stratum; encodes the label by construction in this corpus"
        elif base == "split":
            flag = "dataset split assignment, not a person attribute"
        sig = (np.isfinite(r["p_perm"]) and r["p_perm"] < 0.05)
        rows.append(dict(dataset=ds, attribute=col, label_entropy_bits=H,
                         significant=bool(sig), note=flag, **r))

res = pd.DataFrame(rows)
res = res[["dataset", "attribute", "n", "n_observed", "n_levels", "mi_bits",
           "null_mean", "null_p95", "p_perm", "mi_frac_of_H",
           "label_entropy_bits", "significant", "note"]]
res.to_csv(os.path.join(OUT, "mutual_information_all.csv"), index=False)
json.dump(prov, open(os.path.join(OUT, "mutual_information_sources.json"), "w"), indent=2)

pd.set_option("display.width", 200)
print("\n" + "=" * 100)
print("TASK 2  MUTUAL INFORMATION between participant attributes and the diagnosis label")
print("speaker level, bits, plug-in estimator, %d-perm null, %d quantile bins" % (NPERM, NBIN))
print("=" * 100)
for ds in ["italian", "kcl", "neurovoz", "pcgita", "dcaps"]:
    sub = res[res.dataset == ds].sort_values("mi_bits", ascending=False)
    if not len(sub):
        continue
    print("\n### %s   (H(label) = %.3f bits = the most any attribute could carry)"
          % (ds.upper(), sub.label_entropy_bits.iloc[0]))
    for _, r in sub.iterrows():
        if not np.isfinite(r.mi_bits):
            print("   %-24s  n/a (no usable values, n_observed=%d)" % (r.attribute, r.n_observed))
            continue
        star = "*" if r.significant else " "
        print("   %s %-24s MI %.3f  null %.3f (p95 %.3f)  p=%.4f  = %4.1f%% of H  n_obs=%d  %s"
              % (star, r.attribute, r.mi_bits, r.null_mean, r.null_p95, r.p_perm,
                 100 * r.mi_frac_of_H, r.n_observed, r.note))

print("\n" + "-" * 100)
print("ATTRIBUTES CARRYING REAL INFORMATION ABOUT THE LABEL (p<0.05, excluding")
print("variables that are definitionally downstream of the diagnosis):")
print("-" * 100)
gen = res[res.significant & (res.note == "")].sort_values("mi_bits", ascending=False)
if len(gen):
    for _, r in gen.iterrows():
        print("   %-9s %-24s MI %.3f bits = %.1f%% of the label information (p=%.4f)"
              % (r.dataset, r.attribute, r.mi_bits, 100 * r.mi_frac_of_H, r.p_perm))
else:
    print("   none")
print("\nwrote:", os.path.join(OUT, "mutual_information_all.csv"))
print("\nJOB_DONE")
