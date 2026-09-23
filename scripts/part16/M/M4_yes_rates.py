#!/usr/bin/env python3
"""M4 - Yes rate per model on the Part 14 483 pairs, both arms."""
import os, json, hashlib, sys
import numpy as np, pandas as pd

R    = '<local data dir>/Desktop/release/edaic_rerun/'
OUT  = R + 'part16/'
ROWS = OUT + 'rows/'
DISC = R + 'DISCREPANCIES.md'
MANIFEST = R + 'part14_manifest_new.csv'
SHELL_CMD = '/usr/local/bin/python3 ' + OUT + 'M4_yes_rates.py'

MODELS = [
    ('Qwen2.5-Omni',       R+'part14/p14_o25_zeroshot_scores.csv',  'clip',      'p_yes', 'mass'),
    ('Qwen2-Audio',        R+'part14/p14_q2a_zeroshot_scores.csv',  'clip',      'p_yes', 'mass'),
    ('Qwen3-Omni-30B-A3B', R+'part14/p14_q3o_zeroshot_scores.csv',  'clip',      'p_yes', 'mass'),
    ('AudioFlamingo2',     R+'part14/p14_af2.csv',                  'clip_path', 'p_yes', 'answer_mass'),
    ('AudioFlamingo3',     R+'part14/p14_af3.csv',                  'clip_path', 'p_yes', 'answer_mass'),
    ('Kimi-Audio',         R+'part14/p14_kimi_zeroshot_scores.csv', 'clip',      'p_yes', 'mass'),
]
ARMS = ['conflict', 'agreement']
NBOOT = 2000

def sha1mb(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()

def disc(sev, txt):
    with open(DISC, 'a') as f:
        f.write(f"\n- [{sev}] PART16 M4: {txt}\n")

# ---------------- load manifest ----------------
m = pd.read_csv(MANIFEST)
m['key'] = m['set'].astype(str) + '_' + m.speaker_id.astype(str) + '_' + m.seg_uid.astype(str)
n_clip = len(m); n_spk_all = m.speaker_id.nunique(); n_pairs = m.pair_id.nunique()
print(f"[manifest] {MANIFEST}")
print(f"[manifest] rows={n_clip}  pairs={n_pairs}  speakers={n_spk_all}  "
      f"conflict={(m['set']=='conflict').sum()}  agreement={(m['set']=='agreement').sum()}")
assert n_clip == 966 and n_pairs == 483 and n_spk_all == 138

n_uni_agree = m.loc[m['set']=='agreement','seg_uid'].nunique()
if n_uni_agree != 483:
    disc('LOW', f"agreement arm has {n_uni_agree} unique seg_uids across 483 pair rows "
                f"(agreement segments are reused across pairs of the same speaker). All 483 rows kept, "
                f"so a reused clip contributes its score once per pair. Conflict arm has 483 unique seg_uids.")

# ---------------- load models ----------------
data, srcinfo, missing = {}, {}, []
for name, path, clipcol, pcol, mcol in MODELS:
    if not os.path.exists(path):
        missing.append((name, path)); continue
    d = pd.read_csv(path)
    base = d[clipcol].astype(str) if clipcol == 'clip' else d[clipcol].astype(str).map(os.path.basename)
    key = base.str.replace('.wav', '', regex=False)
    if not (key.values == m.key.values).all():
        disc('HIGH', f"{name}: row order does not match manifest; joined on key instead.")
        d = d.assign(_k=key.values).set_index('_k').loc[m.key.values].reset_index()
        base = d['_k']
    if not (d['label'].values == m.label.values).all():
        disc('HIGH', f"{name}: label column disagrees with manifest label.")
    nn = int(d[pcol].isna().sum() + d[mcol].isna().sum())
    if nn:
        disc('MEDIUM', f"{name}: {nn} NaN values in p_yes/mass.")
    data[name] = pd.DataFrame({'arm': m['set'].values, 'spk': m.speaker_id.values,
                               'p_yes': d[pcol].astype(float).values,
                               'mass': d[mcol].astype(float).values})
    srcinfo[name] = {'path': path, 'n_rows': int(len(d)), 'sha256_first_1mb': sha1mb(path),
                     'p_yes_col': pcol, 'mass_col': mcol,
                     'prompt': str(d['prompt'].iloc[0]) if 'prompt' in d.columns else None}
    if srcinfo[name]['prompt'] is None:
        js = os.path.splitext(path)[0] + '.json'
        if os.path.exists(js):
            j = json.load(open(js))
            srcinfo[name]['prompt'] = j.get('prompt')
            srcinfo[name]['prompt_source'] = js
            if 'prompt_provenance' in j:
                srcinfo[name]['prompt_provenance'] = j['prompt_provenance']
    print(f"[model] {name:20s} n_rows={len(d)}  {path}")

for name, path in missing:
    print(f"[model] {name:20s} MISSING  expected {path}")
    disc('HIGH', f"{name}: no Part 14 per-clip file at {path}.")

# ---------------- bootstrap ----------------
def boot_rate(df, seed=0):
    """speaker bootstrap on the Yes rate"""
    rng = np.random.default_rng(seed)
    spks = df.spk.unique()
    idx = {s: df.index[df.spk == s].to_numpy() for s in spks}
    yes = (df.p_yes.values > 0.5).astype(float)
    pos = {s: np.searchsorted(df.index.to_numpy(), idx[s]) for s in spks}
    out, usable = [], 0
    for _ in range(NBOOT):
        draw = rng.choice(spks, size=len(spks), replace=True)
        sel = np.concatenate([pos[s] for s in draw])
        if sel.size == 0:
            continue
        out.append(yes[sel].mean()); usable += 1
    a = np.asarray(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), usable

def boot_paired_diff(df, seed=0):
    """one speaker draw per replicate, both arms recomputed inside that draw"""
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True)
    yes = (d.p_yes.values > 0.5).astype(float)
    spks = np.sort(d.spk.unique())
    pos = {}
    for s in spks:
        pos[s] = {a: np.where((d.spk.values == s) & (d.arm.values == a))[0] for a in ARMS}
    out, usable = [], 0
    for _ in range(NBOOT):
        draw = rng.choice(spks, size=len(spks), replace=True)
        sc = np.concatenate([pos[s]['conflict']  for s in draw]) if len(draw) else np.array([], int)
        sa = np.concatenate([pos[s]['agreement'] for s in draw]) if len(draw) else np.array([], int)
        if sc.size == 0 or sa.size == 0:
            continue
        out.append(yes[sc].mean() - yes[sa].mean()); usable += 1
    a = np.asarray(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), usable

# ---------------- compute ----------------
rows, tsv, diffs, flags = [], [], [], []
for name, _, _, _, _ in MODELS:
    if name not in data:
        continue
    df = data[name]
    for arm in ARMS:
        sub = df[df.arm == arm].reset_index(drop=True)
        n = len(sub); nsp = sub.spk.nunique()
        yes_count = int((sub.p_yes.values > 0.5).sum())
        rate = yes_count / n
        lo, hi, us = boot_rate(sub, seed=0)
        med = float(np.median(sub['mass'].values))
        low_mass = med < 0.15
        if low_mass:
            flags.append((name, arm, med))
            disc('HIGH', f"{name} / {arm}: median answer mass {med:.4f} < 0.15 - the model was "
                         f"not really answering Yes/No at the first answer position.")
        spread = float(np.std(sub.p_yes.values))
        if rate > 0.95 or rate < 0.05:
            disc('MEDIUM', f"{name} / {arm}: Yes rate {rate:.4f} is near-constant "
                           f"(p_yes sd={spread:.4f}); the model answers the same word on almost every "
                           f"clip, so its arm contrast carries little information.")
        rows.append(dict(model=name, arm=arm, n=n, n_spk=nsp, yes_count=yes_count,
                         yes_rate=round(rate, 4), lo=round(lo, 4), hi=round(hi, 4),
                         median_answer_mass=round(med, 4), boot_usable=us,
                         low_mass_flag=int(low_mass), p_yes_sd=round(spread, 4)))
        tsv.append(f"M4\tyes_rate_{name}_{arm}\t{rate:.4f}\t{lo:.4f}\t{hi:.4f}\t{n}\t{nsp}\t{OUT}M4_yes_rates.csv")
        tsv.append(f"M4\tmedian_answer_mass_{name}_{arm}\t{med:.4f}\t\t\t{n}\t{nsp}\t{OUT}M4_yes_rates.csv")
    c = df[df.arm == 'conflict']; a = df[df.arm == 'agreement']
    dr = (c.p_yes.values > 0.5).mean() - (a.p_yes.values > 0.5).mean()
    dlo, dhi, dus = boot_paired_diff(df, seed=0)
    diffs.append(dict(model=name, diff=round(float(dr), 4), lo=round(dlo, 4), hi=round(dhi, 4),
                      n=len(df), n_spk=df.spk.nunique(), boot_usable=dus))
    tsv.append(f"M4\tyes_rate_diff_conflict_minus_agreement_{name}\t{dr:.4f}\t{dlo:.4f}\t{dhi:.4f}"
               f"\t{len(df)}\t{df.spk.nunique()}\t{OUT}M4_yes_rates.csv")

out_csv = OUT + 'M4_yes_rates.csv'
pd.DataFrame(rows).to_csv(out_csv, index=False)
diff_csv = OUT + 'M4_yes_rate_diffs.csv'
pd.DataFrame(diffs).to_csv(diff_csv, index=False)
with open(ROWS + 'M4.tsv', 'w') as f:
    f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    f.write("\n".join(tsv) + "\n")

side = dict(task='M4', description='Yes rate per model on the Part 14 483 pairs, both arms',
            manifest=dict(path=MANIFEST, sha256_first_1mb=sha1mb(MANIFEST), n_clips=n_clip,
                          n_pairs=n_pairs, n_speakers=n_spk_all,
                          n_conflict=int((m['set']=='conflict').sum()),
                          n_agreement=int((m['set']=='agreement').sum()),
                          n_unique_agreement_seg_uid=int(n_uni_agree)),
            sources=srcinfo, models_missing=[{'model': n, 'expected': p} for n, p in missing],
            yes_rule='p_yes > 0.5 where p_yes = P(Yes)/(P(Yes)+P(No)) at the first answer position',
            answer_mass='P(Yes)+P(No) at the first answer position; median reported; flagged if < 0.15',
            bootstrap=dict(n_draws=NBOOT, rng='numpy.random.default_rng(0) reseeded per cell',
                           unit='speaker, with replacement', percentiles=[2.5, 97.5],
                           paired='one speaker draw per replicate, both arms recomputed inside it'),
            seed=0, model_id=None, prompt_note='per-model prompt string under sources',
            auc_note='no AUC in this task; rule 5 not applicable',
            folds_note='no cross-validation in this task; rule 7 not applicable',
            shell_command=SHELL_CMD,
            outputs=[out_csv, diff_csv, ROWS + 'M4.tsv', OUT + 'M4_yes_rates.json'])
with open(OUT + 'M4_yes_rates.json', 'w') as f:
    json.dump(side, f, indent=2)

# ---------------- print ----------------
print("\n=== M4 YES RATE (p_yes > 0.5), Part 14 483 pairs, both arms ===")
for r in rows:
    print(f"M4 | {r['model']:20s} | {r['arm']:9s} | yes_rate={r['yes_rate']:.4f} "
          f"[{r['lo']:.4f}, {r['hi']:.4f}] | yes={r['yes_count']}/{r['n']} | n={r['n']} "
          f"n_speakers={r['n_spk']} | median_answer_mass={r['median_answer_mass']:.4f} "
          f"| boot_usable={r['boot_usable']}/2000 | {out_csv}")
print("\n=== M4 PAIRED YES-RATE DIFFERENCE (conflict - agreement) ===")
for d in diffs:
    print(f"M4 | {d['model']:20s} | diff={d['diff']:+.4f} [{d['lo']:.4f}, {d['hi']:.4f}] "
          f"| n={d['n']} n_speakers={d['n_spk']} | boot_usable={d['boot_usable']}/2000 | {diff_csv}")
print("\nLOW-MASS FLAGS (median P(Yes)+P(No) < 0.15):",
      "NONE" if not flags else "; ".join(f"{a}/{b}={c:.4f}" for a, b, c in flags))
print("MISSING MODELS:", "NONE" if not missing else "; ".join(n for n, _ in missing))
