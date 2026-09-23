"""M7 Pitt correction: use the part10 5-repeat encoder OOF (the source master_lookup names),
with the paper's aggregation (mean of the 5 per-repeat AUCs), not the superseded single split."""
import numpy as np, pandas as pd, json, hashlib, os
from scipy.stats import rankdata

NPZ = 'scores/part10/pitt_enc_nested5_oof.npz'
ZS  = '<local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv'
OUT = '<local data dir>/Desktop/release/edaic_rerun/part16'

def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

z = np.load(NPZ, allow_pickle=True)
oof, y, spk, name = z['oof'], z['label'].astype(int), z['spk'], z['name']

zs = pd.read_csv(ZS)
ncol = [c for c in zs.columns if c.lower() in ('name','clip','id','file','basename')][0]
pcol = [c for c in zs.columns if 'p_yes' in c.lower()][0]
zs['_k'] = zs[ncol].astype(str).map(lambda s: os.path.basename(str(s)))
key = pd.Series([os.path.basename(str(s)) for s in name])
m = key.map(dict(zip(zs['_k'], zs[pcol])))
assert m.notna().all(), f"unmatched {int(m.isna().sum())}"
zsv = m.values.astype(float)

per_rep = [auc(y, oof[i]) for i in range(oof.shape[0])]
probe_mean = float(np.mean(per_rep))
probe_avgprob = auc(y, oof.mean(0))
zs_auc = auc(y, zsv)

# paired speaker bootstrap: one speaker draw per replicate, both quantities inside it
usp = np.unique(spk); idx = {s: np.where(spk == s)[0] for s in usp}
rng = np.random.default_rng(0)
bp, bz, bd = [], [], []
for _ in range(2000):
    pick = rng.choice(usp, size=len(usp), replace=True)
    ii = np.concatenate([idx[s] for s in pick])
    yy = y[ii]
    if yy.sum() == 0 or yy.sum() == len(yy): continue
    p = float(np.mean([auc(yy, oof[r][ii]) for r in range(oof.shape[0])]))
    q = auc(yy, zsv[ii])
    bp.append(p); bz.append(q); bd.append(p - q)

def ci(v): return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
lp, hp = ci(bp); lz, hz = ci(bz); ld, hd = ci(bd)

pd.DataFrame({'name': name, 'spk': spk, 'label': y,
              'probe_oof_mean5': oof.mean(0), 'zeroshot_p_yes': zsv}).to_csv(
    f'{OUT}/M7_perclip/pitt_FIXED.csv', index=False)

print(f"per-repeat AUCs        {[round(v,4) for v in per_rep]}")
print(f"probe (mean of 5 AUCs) {probe_mean:.4f} [{lp:.4f}, {hp:.4f}]   <- matches master_lookup 0.7706")
print(f"probe (AUC of mean p)  {probe_avgprob:.4f}   (different aggregation, for reference)")
print(f"zero-shot              {zs_auc:.4f} [{lz:.4f}, {hz:.4f}]")
print(f"PAIRED gap             {probe_mean - zs_auc:+.4f} [{ld:.4f}, {hd:.4f}]  usable draws {len(bd)}/2000")
print(f"excludes zero          {ld > 0 or hd < 0}")

json.dump({
 'task':'M7 Pitt correction',
 'why':'M7 used omni_final/omni_pitt_enc_nested_oof.csv (single split, 0.7969), which master_lookup.csv marks SUPERSEDED. The paper cites 0.7706 from the part10 5-repeat rerun.',
 'source_files':{'probe':NPZ,'zeroshot':ZS},
 'sha256_first1mb':{f: hashlib.sha256(open(f,'rb').read(1<<20)).hexdigest() for f in (NPZ,ZS)},
 'n':int(len(y)),'n_speakers':int(len(usp)),'seed':0,'n_boot':2000,'usable_draws':len(bd),
 'model_id':'Qwen/Qwen2.5-Omni-7B',
 'aggregation':'mean of the 5 per-repeat AUCs, recomputed inside every bootstrap draw',
 'per_repeat_auc':[round(v,4) for v in per_rep],
 'probe_mean_of_5_aucs':round(probe_mean,4),'probe_auc_of_mean_prob':round(probe_avgprob,4),
 'zeroshot':round(zs_auc,4),'paired_gap':round(probe_mean-zs_auc,4),'gap_ci':[round(ld,4),round(hd,4)],
 'command':'/usr/local/bin/python3 scripts/part16/M/M7_pitt_fix.py',
}, open(f'{OUT}/M7_pitt_FIXED.json','w'), indent=1)

with open(f'{OUT}/rows/M7fix.tsv','w') as f:
    f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    f.write(f"M7fix\tQwen2.5-Omni Pitt encoder probe, 5-repeat (CORRECTED source)\t{probe_mean:.4f}\t{lp:.4f}\t{hp:.4f}\t468\t228\t{NPZ}\n")
    f.write(f"M7fix\tQwen2.5-Omni Pitt paired probe minus zero-shot (CORRECTED)\t{probe_mean-zs_auc:+.4f}\t{ld:.4f}\t{hd:.4f}\t468\t228\t{OUT}/M7_perclip/pitt_FIXED.csv\n")
print("wrote M7_pitt_FIXED.json, M7_perclip/pitt_FIXED.csv, rows/M7fix.tsv")
