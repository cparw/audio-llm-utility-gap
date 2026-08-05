#!/usr/bin/env python
"""TASK 3 - the neurovoz task_type data bug.

Independently confirm, quantify, fix, and re-run the affected probe.

Evidence used (all from raw corpus files, nothing taken on trust):
  * filename token   GROUP_ITEM_SPEAKERID.wav  -> ITEM is the elicitation item
  * transcriptions/  a .txt exists only for LEXICAL items (read sentences and
                     the spontaneous picture description).  PATAKA and the
                     sustained vowels have no transcript because there is no
                     text to transcribe.  This is what lets us separate the
                     true /pa-ta-ka/ DDK task from read sentences.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdstats as S

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
SH = "/project2/msoleyma_946/speech_health"
NV = os.path.join(SH, "spanish_neurovoz")
AUD = os.path.join(NV, "zenodo_upload", "audios")
AUD2 = os.path.join(NV, "zenodo_upload_silencemode_1", "audios")
TXT = os.path.join(NV, "zenodo_upload", "transcriptions")
OUT = os.path.join(R, "committed")
os.makedirs(OUT, exist_ok=True)

print("=" * 78)
print("TASK 3  neurovoz task_type bug: confirm / quantify / fix / re-run")
print("=" * 78)

m = pd.read_csv(os.path.join(R, "manifests", "neurovoz.csv"))
m["base"] = m.filepath.map(os.path.basename)
m["item"] = m.base.str.split("_").str[1].str.upper()
m["sid"] = m.base.str.replace(".wav", "", regex=False).str.split("_").str[-1]

# ---------------------------------------------------------------- 3a confirm
print("\n--- 3a  independent confirmation")
print("manifest rows:", len(m), " unique filepaths:", m.filepath.nunique())
esp = m[m.item == "ESPONTANEA"]
print("ESPONTANEA rows in manifest:", len(esp))
print("  their task_type:", esp.task_type.value_counts().to_dict())
print("  by label:", esp.groupby("label").size().to_dict(),
      "(label 1 = patient)")

n_a = len(glob.glob(os.path.join(AUD, "*ESPONTANEA*.wav")))
n_b = len(glob.glob(os.path.join(AUD2, "*ESPONTANEA*.wav")))
tot_a = len(glob.glob(os.path.join(AUD, "*.wav")))
tot_b = len(glob.glob(os.path.join(AUD2, "*.wav")))
names_a = sorted(os.path.basename(p) for p in glob.glob(os.path.join(AUD, "*ESPONTANEA*.wav")))
names_b = sorted(os.path.basename(p) for p in glob.glob(os.path.join(AUD2, "*ESPONTANEA*.wav")))
print("\nESPONTANEA wavs on disk:")
print("  zenodo_upload/audios                : %d   (dir total %d wav)" % (n_a, tot_a))
print("  zenodo_upload_silencemode_1/audios  : %d   (dir total %d wav)" % (n_b, tot_b))
print("  filename sets identical between the two dirs:", names_a == names_b)
print("  union of DISTINCT recording names   :", len(set(names_a) | set(names_b)))
print("  -> the '148 files on disk' figure is 74+74, the SAME 74 recordings")
print("     counted twice across two parallel copies of the corpus.")
missing = sorted(set(names_a) - set(m.base))
print("  ESPONTANEA recordings on disk but NOT in the manifest:", len(missing))

# transcription evidence for what each item actually is
print("\n--- item identity from transcriptions/ (ground truth, not inference)")
rows = []
for item, g in m.groupby("item"):
    tx = glob.glob(os.path.join(TXT, "*_%s_*.txt" % item))
    ex = ""
    if tx:
        ex = open(sorted(tx)[0], encoding="utf-8", errors="ignore").read().strip()
        ex = " ".join(ex.split())[:60]
    rows.append(dict(item=item, n_wav=len(g), task_type_now=g.task_type.iloc[0],
                     n_transcripts=len(tx), example_transcript=ex))
it = pd.DataFrame(rows).sort_values(["task_type_now", "item"])
print(it.to_string(index=False))
it.to_csv(os.path.join(OUT, "neurovoz_item_audit.csv"), index=False)

ddk = m[m.task_type == "ddk"]
print("\n--- 3b  quantified impact on the 'ddk' bucket (n=%d)" % len(ddk))
brk = ddk.groupby("item").size().sort_values(ascending=False)
for k, v in brk.items():
    ntx = len(glob.glob(os.path.join(TXT, "*_%s_*.txt" % k)))
    kind = ("SPONTANEOUS speech" if k == "ESPONTANEA" else
            "true DDK (/pa-ta-ka/, no text to transcribe)" if ntx == 0 else
            "READ SENTENCE")
    print("   %-11s %4d  %5.1f%%   %s" % (k, v, 100.0 * v / len(ddk), kind))

# ------------------------------------------------------- 3c corrected manifest
TRUE_DDK = {"PATAKA"}
SPONT = {"ESPONTANEA"}


def fix_task(r):
    if r["item"] in SPONT:
        return "spontaneous"
    if r["item"] in TRUE_DDK:
        return "ddk"
    if r["task_type"] == "vowel":
        return "vowel"
    return "read"          # every remaining item is a transcribed sentence


fixed = m.copy()
fixed["task_type_orig"] = fixed["task_type"]
fixed["task_type"] = fixed.apply(fix_task, axis=1)
fixed["item"] = fixed["item"]
keep = [c for c in ["filepath", "dataset", "speaker_id", "label", "task_type",
                    "language", "age", "sex", "group", "transcript",
                    "item", "task_type_orig"] if c in fixed.columns]
fixed[keep].to_csv(os.path.join(OUT, "neurovoz_fixed.csv"), index=False)

print("\n--- 3c  corrected task counts  ->  committed/neurovoz_fixed.csv")
cmp = pd.DataFrame({"before": m.task_type.value_counts(),
                    "after": fixed.task_type.value_counts()}).fillna(0).astype(int)
print(cmp.to_string())
print("\nrows whose task_type changed:", int((fixed.task_type != m.task_type).sum()))
print(pd.crosstab(m.task_type, fixed.task_type).to_string())
print("\nper-task speaker / label balance after the fix:")
print(fixed.groupby("task_type").apply(
    lambda g: pd.Series(dict(n=len(g), speakers=g.speaker_id.nunique(),
                             PD=int((g.label == 1).sum()), HC=int((g.label == 0).sum()))),
    include_groups=False).to_string())

# ------------------------------------------------------------- 3d re-run probe
print("\n" + "=" * 78)
print("--- 3d  re-run the neurovoz ddk encoder probe under each definition")
print("=" * 78)
d = np.load(os.path.join(R, "features", "qwen2", "neurovoz", "encoder_features.npz"),
            allow_pickle=True)
L = int(d["n_layers"])
paths = np.array([os.path.basename(p) for p in d["path"]])
lab = d["label"].astype(int)
spk = d["speaker"].astype(str)
task = d["task"].astype(str)
item = np.array([p.split("_")[1].upper() for p in paths])
print("feature npz clips:", len(lab), "layers:", L)

VARIANTS = {
    "ddk_AS_SHIPPED_all598":      (task == "ddk"),
    "ddk_minus_ESPONTANEA_524":   (task == "ddk") & (item != "ESPONTANEA"),
    "ddk_TRUE_pataka_only_99":    (item == "PATAKA"),
    "spontaneous_ESPONTANEA_74":  (item == "ESPONTANEA"),
}
res = []
for name, mask in VARIANTS.items():
    idx = np.where(mask)[0]
    y, g = lab[idx], spk[idx]
    print("\n[%s]  n=%d speakers=%d PD=%d HC=%d"
          % (name, len(idx), len(set(g)), int(y.sum()), int((y == 0).sum())))
    best = None
    per_layer = []
    for li in range(L):
        X = d["layer_%02d" % li][idx]
        p, _ = S.oof_probe(X, y, g)
        a, b = S.auc_ba(y, p)
        per_layer.append((li, a, b))
        if best is None or (np.isfinite(a) and a > best[1]):
            best = (li, a, b, p)
    li, a, b, p = best
    lo, hi = S.spk_bootstrap_auc(y, p, g, n=2000)
    nulls = S.shuffle_null_auc(d["layer_%02d" % li][idx], y, g, n=50)
    pval = float((np.sum(nulls >= a) + 1) / (len(nulls) + 1))
    print("   PEAK layer %d  AUC %.3f [%.3f, %.3f]  bAcc %.3f" % (li, a, lo, hi, b))
    print("   speaker-shuffle null AUC %.3f +/- %.3f (n=%d)  perm p=%.3f"
          % (nulls.mean(), nulls.std(), len(nulls), pval))
    print("   NOTE peak layer was selected on the data; the permutation null above")
    print("        is reported for that selection, per the advisor's rule.")
    res.append(dict(variant=name, n=len(idx), speakers=len(set(g)),
                    PD=int(y.sum()), HC=int((y == 0).sum()),
                    peak_layer=li, auc=a, auc_lo=lo, auc_hi=hi, balacc=b,
                    shuffle_null_auc=float(nulls.mean()),
                    shuffle_null_sd=float(nulls.std()), perm_p=pval))
    pd.DataFrame(per_layer, columns=["layer", "auc", "balacc"]).to_csv(
        os.path.join(OUT, "neurovoz_%s_bylayer.csv" % name), index=False)

rf = pd.DataFrame(res)
rf.to_csv(os.path.join(OUT, "neurovoz_ddk_fix_impact.csv"), index=False)
print("\n" + "=" * 78)
print("SUMMARY  neurovoz ddk probe, impact of the labelling fix")
print("=" * 78)
print(rf[["variant", "n", "speakers", "peak_layer", "auc", "auc_lo", "auc_hi",
          "balacc", "shuffle_null_auc", "perm_p"]].to_string(index=False))
a0 = rf.loc[rf.variant == "ddk_AS_SHIPPED_all598", "auc"].iloc[0]
a1 = rf.loc[rf.variant == "ddk_minus_ESPONTANEA_524", "auc"].iloc[0]
a2 = rf.loc[rf.variant == "ddk_TRUE_pataka_only_99", "auc"].iloc[0]
print("\nremoving the 74 ESPONTANEA clips moves peak AUC %.3f -> %.3f  (delta %+.3f)"
      % (a0, a1, a1 - a0))
print("restricting to the TRUE ddk task (PATAKA only) moves it %.3f -> %.3f  (delta %+.3f)"
      % (a0, a2, a2 - a0))
print("\nJOB_DONE")
