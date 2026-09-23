"""PART20 POD6 job C: collect (not rerun) the landed, Mac-verified Pitt rows with their source files."""
import sys, os, shutil, json, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p20_common import sidecar, ROWS
OUT = sys.argv[1]; R = "<local data dir>/release"
M8 = f"{R}/edaic_rerun/part16/M8_conflict_cells.csv"; T6 = [f"{R}/edaic_rerun/part17/T6_pitt_text_sep20_arms.{e}" for e in ("csv", "json", "py")] + [f"{R}/edaic_rerun/part17/rows/T6.tsv", f"{R}/edaic_rerun/part17/T6_run.log"]
m8 = pd.read_csv(M8); b = m8[m8.set == "B_pitt"].copy(); b.to_csv(f"{OUT}/C_M8_B_pitt_rows.csv", index=False)
for f in T6: shutil.copy2(f, f"{OUT}/C_{os.path.basename(f)}")
srcs = sorted(set(b.source_file)) + [M8] + T6
sidecar(f"{OUT}/C_collected.sidecar.json", result="COLLECTED, NOT RERUN: Pitt conflict-minus-agreement six models (M8 B_pitt rows, verified 55/55 on the Mac) and Pitt transcript by arm on the Sep 20 run (T6)",
        m8_rows=int(len(b)), models=sorted(b.model.unique()), sources=srcs, command="/usr/local/bin/python3 " + " ".join(sys.argv))
with open(ROWS, "a") as fh:
    for _, r in b.iterrows():
        tag = ("interval excludes zero" if (r.lo > 0 or r.hi < 0) else "interval includes zero") if r.arm == "conflict_minus_agreement" else ("above 0.5" if r.lo > 0.5 else ("below 0.5" if r.hi < 0.5 else "interval includes 0.5"))
        fh.write(f"P20.POD6.C.collected.M8.{r.model.replace(' ','_')}.{r.source_tag}.{r.arm}\tCOLLECTED not rerun: Pitt {r.model} zero-shot answer AUC {r.arm} ({r.source_tag})\t{r.auc:.4f}\t{r.lo:.4f}\t{r.hi:.4f}\t{r.n}\t{r.n_spk}\t{r.source_file}\t{r.auc:.2f} [{r.lo:.2f}, {r.hi:.2f}]\t{tag}\n")
        print(f"PASTE COLLECTED M8 {r.model} {r.source_tag} {r.arm} | {r.auc:.4f} [{r.lo:.4f}, {r.hi:.4f}] | n={r.n} n_speakers={r.n_spk} | {r.source_file} | two-dp {r.auc:.2f} [{r.lo:.2f}, {r.hi:.2f}] {tag}")
    t6 = pd.read_csv(f"{R}/edaic_rerun/part17/rows/T6.tsv", sep="\t")
    for _, r in t6.iterrows():
        if pd.isna(r.lo):
            fh.write(f"P20.POD6.C.collected.{r['id']}\tCOLLECTED not rerun: {r.what}\t{r.value:.4f}\t\t\t{r.n}\t{r.n_spk}\t{r.file}\t{r.value:.2f}\tno interval\n"); continue
        d = ("minus" in r["id"]) or ("gap" in r["id"])
        tag = ("interval excludes zero" if (r.lo > 0 or r.hi < 0) else "interval includes zero") if d else ("above 0.5" if r.lo > 0.5 else ("below 0.5" if r.hi < 0.5 else "interval includes 0.5"))
        fh.write(f"P20.POD6.C.collected.{r['id']}\tCOLLECTED not rerun: {r.what}\t{r.value:.4f}\t{r.lo:.4f}\t{r.hi:.4f}\t{int(r.n)}\t{int(r.n_spk)}\t{r.file}\t{r.value:.2f} [{r.lo:.2f}, {r.hi:.2f}]\t{tag}\n")
        print(f"PASTE COLLECTED T6 {r['id']} | {r.value:.4f} [{r.lo:.4f}, {r.hi:.4f}] | n={int(r.n)} n_speakers={int(r.n_spk)} | {r.file} | two-dp {r.value:.2f} [{r.lo:.2f}, {r.hi:.2f}] {tag}")
open(f"{OUT}/C_collected.RESULT", "w").close()
