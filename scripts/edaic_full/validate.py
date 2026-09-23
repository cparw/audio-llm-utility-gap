import csv, numpy as np, json, sys
def auc_rank(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); sv = s[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j+1] == sv[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))

new = {r["clip"]: r for r in csv.DictReader(open("/workspace/edaicfull/out/full_zeroshot_scores.csv"))}
ref = list(csv.DictReader(open("/workspace/edaicfull/ref_zeroshot.csv")))
clips = [r["clip"] for r in ref]
miss = [c for c in clips if c not in new]
print("n ref", len(ref), "n new", len(new), "missing", len(miss))
if miss: print("MISSING:", miss[:10]); sys.exit(1)
y  = np.array([int(r["label"]) for r in ref])
pr = np.array([float(r["p_yes"]) for r in ref])
pn = np.array([float(new[c]["p_yes"]) for c in clips])
mn = np.array([float(new[c]["mass"]) for c in clips])
mr = np.array([float(r["mass"]) for r in ref])
at = np.array([int(new[c]["n_audio_tok"]) for c in clips])
sq = np.array([int(new[c]["seq_len"]) for c in clips])
ft = np.array([int(new[c]["fit_in_context"]) for c in clips])
du = np.array([float(new[c]["dur_s"]) for c in clips])
print()
print("AUC reference file (recomputed, rank formula) : %.4f" % auc_rank(y, pr))
print("AUC published in o25_full_zeroshot.json       : 0.8285")
print("AUC THIS REBUILD (recomputed, rank formula)   : %.4f" % auc_rank(y, pn))
print()
print("per-clip p_yes  pearson r : %.6f" % np.corrcoef(pr, pn)[0,1])
print("per-clip p_yes  spearman  : %.6f" % np.corrcoef(np.argsort(np.argsort(pr)), np.argsort(np.argsort(pn)))[0,1])
print("per-clip p_yes  max |diff|: %.6f" % np.abs(pr-pn).max())
print("per-clip p_yes  mean|diff|: %.6f" % np.abs(pr-pn).mean())
print("n clips with |diff| > 0.01: %d / %d" % ((np.abs(pr-pn)>0.01).sum(), len(pr)))
print("answer_mass  median new %.4f  ref %.4f  max|diff| %.6f" % (np.median(mn), np.median(mr), np.abs(mn-mr).max()))
print()
print("=== CONTEXT / TRUNCATION ===")
print("audio tokens: min %d  median %d  max %d" % (at.min(), np.median(at), at.max()))
print("seq len     : min %d  median %d  max %d" % (sq.min(), np.median(sq), sq.max()))
print("fit_in_context (seq <= 32768): %d / %d" % (ft.sum(), len(ft)))
cap = at >= 7500
print("clips at the 7500-audio-token cap (=300 s of audio): %d / %d" % (cap.sum(), len(cap)))
print("clips whose window is longer than 300 s           : %d / %d" % ((du>300.5).sum(), len(du)))
print("audio seconds discarded by the 300 s cap: total %.0f s, median over capped %.0f s, max %.0f s"
      % (np.clip(du-300,0,None).sum(), np.median(np.clip(du-300,0,None)[du>300.5]), np.clip(du-300,0,None).max()))
worst = np.argsort(-np.abs(pr-pn))[:5]
print("\nlargest per-clip disagreements:")
for i in worst: print("  %s ref %.5f new %.5f d %.5f dur %.0fs tok %d" % (clips[i], pr[i], pn[i], pn[i]-pr[i], du[i], at[i]))
