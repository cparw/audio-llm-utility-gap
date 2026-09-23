#!/usr/bin/env python3
"""M10 - Interval for the MDVR-KCL transcript answer (~0.76). Qwen2.5-Omni."""
import os, json, hashlib, subprocess
import numpy as np, pandas as pd
import scipy.stats as st

OUT = "<local data dir>/Desktop/release/edaic_rerun/part16"
DISC = "<local data dir>/Desktop/release/edaic_rerun/DISCREPANCIES.md"
SEED = 0
NBOOT = 2000

F_TEXT = "<local data dir>/Desktop/release/overnight2/text_new/o25_kcl_text.csv"
F_TEXT_J = "<local data dir>/Desktop/release/overnight2/text_new/o25_kcl_text.json"
F_AUD = "<local data dir>/Desktop/release/omni_final/omni_kcl_zeroshot_scores.csv"
F_AUD_J = "<local data dir>/Desktop/release/omni_final/omni_kcl_zeroshot.json"
F_ENC = "<local data dir>/Desktop/release/omni_final/omni_kcl_enc_nested_oof.csv"
F_REP = "<local data dir>/Desktop/release/omni_final/omni_kcl_nested_repeats.json"
F_MASTER = "lookup/master_lookup.csv"
SRC = [F_TEXT, F_TEXT_J, F_AUD, F_AUD_J, F_ENC, F_REP, F_MASTER]

def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def sha1mb(p):
    with open(p, 'rb') as f: return hashlib.sha256(f.read(1024*1024)).hexdigest()

WRITE_DISC = os.environ.get("M10_WRITE_DISC") == "1"
def disc(sev, msg):
    if not WRITE_DISC:
        return
    with open(DISC, 'a') as f:
        f.write(f"\n- **[{sev}] M10 (part16)** {msg}\n")

# ---------- load ----------
t = pd.read_csv(F_TEXT).rename(columns={'id':'clip','speaker_id':'speaker','p_yes':'p_text'})
a = pd.read_csv(F_AUD).rename(columns={'p_yes':'p_audio'})
e = pd.read_csv(F_ENC).rename(columns={'p_probe':'p_enc'})

df = (t[['clip','speaker','label','p_text']]
      .merge(a[['clip','speaker','label','p_audio']], on=['clip','speaker','label'], how='inner')
      .merge(e[['clip','speaker','label','p_enc']], on=['clip','speaker','label'], how='inner')
      .sort_values('clip').reset_index(drop=True))
assert len(df) == len(t) == 37, f"merge lost rows: {len(df)} vs {len(t)}"

n = len(df); spk = df['speaker'].values; y = df['label'].values.astype(int)
uspk = np.unique(spk); n_spk = len(uspk)
clips_per_spk = df.groupby('speaker').size()
one_row_per_speaker = bool(clips_per_spk.max() == 1)

# ---------- point AUCs ----------
A = {'transcript': auc(y, df['p_text'].values),
     'audio':      auc(y, df['p_audio'].values),
     'encoder':    auc(y, df['p_enc'].values)}
A['transcript_minus_audio']   = A['transcript'] - A['audio']
A['transcript_minus_encoder'] = A['transcript'] - A['encoder']

# ---------- tie diagnosis + independent sklearn cross-check ----------
from sklearn.metrics import roc_auc_score
P = df['p_text'].values[y == 1]; N = df['p_text'].values[y == 0]
n_pairs = len(P) * len(N)
wins = int(sum((pp > nn) for pp in P for nn in N))
ties = int(sum((pp == nn) for pp in P for nn in N))
auc_ties_as_loss = wins / n_pairs
auc_ties_half    = (wins + 0.5 * ties) / n_pairs
auc_sklearn      = float(roc_auc_score(y, df['p_text'].values))

# speaker-level check (identical when one clip per speaker)
sp_df = df.groupby('speaker').agg(label=('label','max'), p_text=('p_text','mean')).reset_index()
auc_spk_level = auc(sp_df['label'].values, sp_df['p_text'].values)

# ---------- published values from master_lookup ----------
m = pd.read_csv(F_MASTER)
mk = m[(m['dataset'].str.lower() == 'kcl')]
def pub(model, stream):
    r = mk[(mk['model'] == model) & (mk['stream'] == stream)]
    return (float(r['auc'].iloc[0]), str(r['source'].iloc[0])) if len(r) else (float('nan'), '')
pub_text, pub_text_src   = pub('Qwen2.5-Omni', 'transcript only')
pub_aud,  pub_aud_src    = pub('Qwen2.5-Omni', 'zero shot answer')
pub_enc,  pub_enc_src    = pub('Qwen2.5-Omni', 'encoder probe')

# ---------- bootstrap: speakers with replacement, paired ----------
spk_idx = {s: np.where(spk == s)[0] for s in uspk}
rng = np.random.default_rng(SEED)
keys = ['transcript','audio','encoder','transcript_minus_audio','transcript_minus_encoder']
boot = {k: [] for k in keys}
usable = 0
for _ in range(NBOOT):
    draw = rng.choice(uspk, size=n_spk, replace=True)          # ONE speaker list per replicate
    idx = np.concatenate([spk_idx[s] for s in draw])
    yy = y[idx]
    if yy.sum() == 0 or yy.sum() == len(yy):
        continue
    usable += 1
    vt = auc(yy, df['p_text'].values[idx])
    va = auc(yy, df['p_audio'].values[idx])
    ve = auc(yy, df['p_enc'].values[idx])
    boot['transcript'].append(vt); boot['audio'].append(va); boot['encoder'].append(ve)
    boot['transcript_minus_audio'].append(vt - va)
    boot['transcript_minus_encoder'].append(vt - ve)

CI = {k: (float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))) for k in keys}
p_diff_gt0 = float(np.mean(np.array(boot['transcript_minus_audio']) > 0))

# ---------- discrepancy checks ----------
if round(A['transcript'], 4) != round(pub_text, 4):
    disc("HIGH", f"KCL transcript AUC recomputed {A['transcript']:.4f} from {F_TEXT} does not match published {pub_text:.4f} in master_lookup.csv / {pub_text_src}")
if round(A['audio'], 4) != round(pub_aud, 4):
    disc("HIGH", f"KCL Omni audio zero-shot AUC recomputed {A['audio']:.4f} from {F_AUD} does not match published {pub_aud:.4f}")
if round(A['encoder'], 4) != round(pub_enc, 4):
    disc("MEDIUM", f"KCL Omni encoder probe: recomputed {A['encoder']:.4f} from the single saved OOF file {F_ENC}; master_lookup publishes {pub_enc:.4f} which is the mean of 5 nested repeats in {F_REP} (per_repeat {json.load(open(F_REP))['enc']['per_repeat']}). The OOF csv holds ONE repeat, so the two are not the same estimator. Not an error, but the published encoder number cannot be reproduced from the per-clip file alone.")
disc("LOW", f"No saved fold assignment file was found anywhere under <local data dir>/Desktop/release or <local data dir>/paper1_local_runs (searched *fold*.csv/json/npz), and the KCL *_nested_oof.csv files carry no 'fold' column. M10 did NOT need folds: it recomputes AUC from already-out-of-fold saved scores. No folds were recomputed.")

# ---------- outputs ----------
csv_path = f"{OUT}/M10_kcl_transcript.csv"
out = df[['clip','speaker','label','p_text','p_audio','p_enc']].copy()
out.to_csv(csv_path, index=False)

cmd = "/usr/local/bin/python3 scripts/part16/M/M10_kcl_transcript.py"
side = {
  "task": "M10", "title": "Interval for the MDVR-KCL transcript answer",
  "source_files": {p: {"sha256_first_1mb": sha1mb(p)} for p in SRC},
  "per_clip_csv": csv_path,
  "n": int(n), "n_speakers": int(n_spk), "n_positive": int(y.sum()), "n_negative": int(n - y.sum()),
  "one_row_per_speaker": one_row_per_speaker,
  "max_clips_per_speaker": int(clips_per_spk.max()),
  "clip_level_equals_speaker_level_auc": bool(round(A['transcript'],4) == round(auc_spk_level,4)),
  "auc_speaker_level_transcript": round(auc_spk_level, 4),
  "seed": SEED, "n_bootstrap": NBOOT, "bootstrap_usable_draws": int(usable),
  "bootstrap_unit": "speaker, with replacement, paired (one speaker list per replicate)",
  "model_id": json.load(open(F_TEXT_J))["model"],
  "prompt_transcript": json.load(open(F_TEXT_J))["prompt"],
  "prompt_audio": json.load(open(F_AUD_J))["prompt"],
  "encoder_probe_note": "p_enc is the saved nested out-of-fold probe score (one repeat). master_lookup publishes the mean of 5 repeats.",
  "folds": "not required; scores are already out-of-fold. No saved fold file found on disk.",
  "published_master_lookup": {"transcript": pub_text, "audio": pub_aud, "encoder": pub_enc},
  "recomputed_rank_formula": {k: round(v,4) for k,v in A.items()},
  "ci_2p5_97p5": {k: [round(CI[k][0],4), round(CI[k][1],4)] for k in keys},
  "p_boot_transcript_gt_audio": round(p_diff_gt0, 4),
  "tie_diagnosis_transcript": {
     "n_pairs": n_pairs, "strict_wins": wins, "tied_pairs": ties,
     "auc_ties_as_loss": round(auc_ties_as_loss, 6),
     "auc_ties_half_credit_rank_formula": round(auc_ties_half, 6),
     "auc_sklearn_roc_auc_score": round(auc_sklearn, 6),
     "explains_published_0_7619": bool(round(auc_ties_as_loss, 4) == round(pub_text, 4)),
     "note": "The published 0.7619 counts the single tied pair as a loss (strict >). The PART16 rank formula gives ties half credit -> 0.7634. sklearn agrees with the rank formula."
  },
  "shell_command": cmd,
}
side_path = f"{OUT}/M10_kcl_transcript.sidecar.json"
json.dump(side, open(side_path, 'w'), indent=1)

rows_path = f"{OUT}/rows/M10.tsv"
with open(rows_path, 'w') as f:
    f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    lab = {'transcript':'MDVR-KCL Qwen2.5-Omni transcript-only answer AUC',
           'audio':'MDVR-KCL Qwen2.5-Omni audio zero-shot answer AUC',
           'encoder':'MDVR-KCL Qwen2.5-Omni encoder probe AUC (single saved OOF repeat)',
           'transcript_minus_audio':'MDVR-KCL paired transcript minus audio answer AUC',
           'transcript_minus_encoder':'MDVR-KCL paired transcript minus encoder probe AUC'}
    src = {'transcript':F_TEXT,'audio':F_AUD,'encoder':F_ENC,
           'transcript_minus_audio':csv_path,'transcript_minus_encoder':csv_path}
    for k in keys:
        f.write(f"M10\t{lab[k]}\t{A[k]:.4f}\t{CI[k][0]:.4f}\t{CI[k][1]:.4f}\t{n}\t{n_spk}\t{src[k]}\n")

# ---------- print ----------
print("="*100)
print("M10  MDVR-KCL transcript answer, interval")
print("="*100)
print("\nEvery kcl row in master_lookup.csv:")
print(mk[['model','stream','auc','n','estimator','source']].to_string(index=False))
print(f"\nPicked: Qwen2.5-Omni / 'transcript only' / {pub_text} -> per-clip file {F_TEXT}")
print(f"\nn = {n} clips, n_speakers = {n_spk}, positives = {int(y.sum())}, negatives = {int(n-y.sum())}")
print(f"one row per speaker = {one_row_per_speaker} (max clips per speaker = {int(clips_per_spk.max())})")
print(f"clip-level transcript AUC = {A['transcript']:.4f} ; speaker-level transcript AUC = {auc_spk_level:.4f} -> identical = {round(A['transcript'],4)==round(auc_spk_level,4)}")
print(f"\nbootstrap: {NBOOT} draws, rng default_rng({SEED}), speakers with replacement, usable = {usable}/{NBOOT}")
print(f"published transcript {pub_text:.4f} vs recomputed {A['transcript']:.4f} -> match to 4dp = {round(A['transcript'],4)==round(pub_text,4)}")
print(f"published audio      {pub_aud:.4f} vs recomputed {A['audio']:.4f} -> match to 4dp = {round(A['audio'],4)==round(pub_aud,4)}")
print(f"published encoder    {pub_enc:.4f} (mean of 5 repeats) vs recomputed {A['encoder']:.4f} (single saved OOF) -> match = {round(A['encoder'],4)==round(pub_enc,4)}")
print(f"\nTIE DIAGNOSIS (transcript): {n_pairs} pairs = 16 pos x 21 neg; strict wins {wins}; tied pairs {ties}")
print(f"  ties as loss        {auc_ties_as_loss:.6f} -> {auc_ties_as_loss:.4f}  (this is the published 0.7619)")
print(f"  ties half (rank fm) {auc_ties_half:.6f} -> {auc_ties_half:.4f}  (this is the PART16 value 0.7634)")
print(f"  sklearn roc_auc     {auc_sklearn:.6f}  -> agrees with rank formula: {abs(auc_sklearn-A['transcript'])<1e-9}")

print("\nPASTE LINES")
for k in keys:
    print(f"MDVR-KCL {k}: AUC {A[k]:.4f} [{CI[k][0]:.4f}, {CI[k][1]:.4f}]  n=37  n_speakers=37  {src[k]}")
print(f"\nP(bootstrap transcript > audio) = {p_diff_gt0:.4f}")
print(f"\nwrote {csv_path}\nwrote {side_path}\nwrote {rows_path}")
