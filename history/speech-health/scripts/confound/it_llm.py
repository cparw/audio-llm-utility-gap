"""ITALIAN CONFOUND step 9: do the audio-LLM per-clip scores track the recording setup?"""
import os, csv, json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/final"
G = {r["filepath"]: r for r in csv.DictReader(open(f"{OUT}/italian_gate_stats.csv"))}
D = {r["filepath"]: r for r in csv.DictReader(open(f"{OUT}/italian_design.csv"))}
SD = "/scratch1/parwatka/scores"

print("### per-clip audio-LLM p(yes) vs the RECORDING-SETUP variables")
print("    'noise floor' = quiet-frame RMS in dBFS; 'SNR' = speech level minus floor, dB")
print(f"  {'score file':40s}{'n':>5}{'AUC model':>11}{'AUC floor':>11}{'AUC SNR':>10}"
      f"{'rho(p,floor)':>14}{'rho(p,SNR)':>12}{'rho | ctrl':>12}{'rho | PD':>10}")
rows = []
for fn in sorted(os.listdir(SD)):
    if not fn.startswith("italian") or not fn.endswith("perclip.csv"):
        continue
    p, fl, sn, lab = [], [], [], []
    for r in csv.DictReader(open(f"{SD}/{fn}")):
        g = G.get(r["filepath"])
        if g is None:
            continue
        try:
            v = float(r["p_yes"])
        except Exception:
            continue
        p.append(v); fl.append(float(g["quiet_rms_dbfs"]))
        sn.append(float(g["speech_minus_floor_db"])); lab.append(int(r["label"]))
    if len(p) < 20:
        continue
    p, fl, sn, lab = map(np.array, (p, fl, sn, lab))
    a_m = roc_auc_score(lab, p)
    a_f = roc_auc_score(lab, fl); a_f = max(a_f, 1 - a_f)
    a_s = roc_auc_score(lab, sn); a_s = max(a_s, 1 - a_s)
    rf = spearmanr(p, fl)[0]; rs = spearmanr(p, sn)[0]
    rc = spearmanr(p[lab == 0], fl[lab == 0])[0]
    rp = spearmanr(p[lab == 1], fl[lab == 1])[0]
    name = fn.replace("_perclip.csv", "")
    print(f"  {name:40s}{len(p):>5}{a_m:>11.3f}{a_f:>11.3f}{a_s:>10.3f}"
          f"{rf:>14.3f}{rs:>12.3f}{rc:>12.3f}{rp:>10.3f}")
    rows.append(dict(score_file=name, n=len(p), auc_model=round(a_m, 4),
                     auc_noise_floor_alone=round(a_f, 4), auc_snr_alone=round(a_s, 4),
                     rho_score_vs_floor=round(float(rf), 4), rho_score_vs_snr=round(float(rs), 4),
                     rho_within_control=round(float(rc), 4), rho_within_pd=round(float(rp), 4)))
with open(f"{OUT}/italian_llm_vs_recording.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# how well do the two recording-setup scalars alone separate the groups, speaker level
print("\n### recording-setup scalars alone, SPEAKER-level AUC over all 831 clips / 65 speakers")
spk = {}
for fp, g in G.items():
    d = D[fp]
    spk.setdefault(d["spk"], dict(lab=int(d["label"]), fl=[], sn=[]))
    spk[d["spk"]]["fl"].append(float(g["quiet_rms_dbfs"]))
    spk[d["spk"]]["sn"].append(float(g["speech_minus_floor_db"]))
y = np.array([v["lab"] for v in spk.values()])
for k, nm in [("fl", "noise floor dBFS"), ("sn", "speech-to-floor SNR dB")]:
    v = np.array([np.median(vv[k]) for vv in spk.values()])
    a = roc_auc_score(y, v)
    print(f"  {nm:26s} speaker AUC = {a:.3f}   (flipped {max(a,1-a):.3f})   "
          f"median control {np.median(v[y==0]):.1f}  median PD {np.median(v[y==1]):.1f}")
print(f"\nwrote {OUT}/italian_llm_vs_recording.csv")
