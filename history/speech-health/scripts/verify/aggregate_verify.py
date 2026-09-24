import os, csv, json, glob, subprocess
import numpy as np
R="/project2/msoleyma_946/speech_health/results_chaitanya"

def peak_layer(csvpath):
    try:
        rows=list(csv.DictReader(open(csvpath)))
        keys=rows[0].keys()
        col=next((c for c in ("auc_mean","auc","roc_auc","balacc_mean","balanced_accuracy") if c in keys), None)
        if col is None: return "?", "no-auc-col"
        best=max(rows,key=lambda r: float(r.get(col,"nan") or "nan"))
        return best.get("layer","?"), round(float(best[col]),3), col
    except Exception as e:
        return "?", f"ERR {repr(e)[:40]}", "?"

print("="*78)
print("PROBING PEAK LAYER PER DATASET/TASK (speaker-disjoint GroupKFold, AUC)")
print("="*78)
for f in sorted(glob.glob(f"{R}/probing/*.csv")):
    b=os.path.basename(f)
    if b.startswith("localize_") or b.startswith("cross_"): continue
    L,A,col=peak_layer(f)
    print(f"  {b:42s} peak layer {str(L):>3}  {col}={A}")

print()
print("="*78)
print("CROSS-DATASET MATRIX (train rows x test cols), read task")
print("="*78)
cm=f"{R}/crossdataset/cross_matrix_read.csv"
if os.path.exists(cm):
    for line in open(cm): print("  "+line.rstrip())
else:
    print("  MISSING cross_matrix_read.csv")

print()
print("="*78)
print("LOCALIZATION: acoustic AUC survival through the model (audio-token positions)")
print("="*78)
for f in sorted(glob.glob(f"{R}/probing/localize_*.csv")):
    try:
        rows=list(csv.DictReader(open(f)))
        col="auc" if "auc" in rows[0] else list(rows[0].keys())[-1]
        vals=[float(r[col]) for r in rows if r.get(col) not in (None,"","nan")]
        stages=[r.get("stage",r.get("layer","?")) for r in rows]
        print(f"  {os.path.basename(f):32s} stages {len(rows):>3}  AUC min {min(vals):.3f}  max {max(vals):.3f}  final {vals[-1]:.3f}")
    except Exception as e:
        print(f"  {os.path.basename(f)} ERR {repr(e)[:50]}")

print()
print("="*78)
print("RELIANCE / BEHAVIORAL (full model, audio held fixed, text varied)")
print("="*78)
hdr=("tag","model","cond","n","spk","behavDiag","behavSymp","acNeut","acTrue","lexSens","lexFrac","flip","sickTrueAud","sickShufAud","mass")
print("  "+" ".join(f"{h:>11}" for h in hdr))
rel_rows=[]
for f in sorted(glob.glob(f"{R}/reliance/*_summary.json")):
    s=json.load(open(f))
    def g(k):
        v=s.get(k,"");
        return "" if v=="" else (f"{v:.3f}" if isinstance(v,float) else str(v))
    row=(s.get("tag",""),s.get("model","Qwen2-Audio"),s.get("cond",""),s.get("n",""),s.get("speakers",""),
         g("behavioral_diag_AUC"),g("behavioral_symptom_AUC"),g("acoustic_neutral_AUC"),g("acoustic_true_AUC"),
         g("mean_lexical_sensitivity"),g("lexical_fraction"),g("flip_rate"),
         g("sick_score_true_audio"),g("sick_score_shuffled_audio"),g("mean_answer_mass"))
    rel_rows.append(row)
    print("  "+" ".join(f"{str(x):>11}" for x in row))

print()
print("="*78)
print("SLURM CROSS-CHECK: my jobs today (sacct) vs result-file mtimes")
print("="*78)
try:
    out=subprocess.run(["sacct","-u","parwatka","--starttime","today","-n",
        "-o","JobID%14,JobName%20,State%14,Elapsed"],capture_output=True,text=True,timeout=30).stdout
    for line in out.splitlines():
        if ".ext" in line or ".bat" in line: continue
        print("  "+line.rstrip())
except Exception as e:
    print("  sacct unavailable", repr(e)[:60])

print()
print("Result files and modification times:")
for sub in ["reliance","probing","crossdataset"]:
    for f in sorted(glob.glob(f"{R}/{sub}/*")):
        st=os.stat(f)
        print(f"  {st.st_mtime:.0f}  {os.path.basename(f)}")
