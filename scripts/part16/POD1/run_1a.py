import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_auc import auc_rank, boot_auc, boot_paired

SYNC = "<local data dir>/Desktop/release/edaic_rerun/part16/pod_sync"
P14  = "<local data dir>/Desktop/release/edaic_rerun/part14"
MAN  = "manifests/part14_manifest_new.csv"
OUT  = "<local data dir>/Desktop/release/edaic_rerun/part16/POD1"
os.makedirs(OUT, exist_ok=True)

man = pd.read_csv(MAN)
print("manifest rows", len(man), "| arms", man['set'].value_counts().to_dict(),
      "| speakers", man['speaker_id'].nunique())

# arm lookup keyed by seg_uid
arm_by_seg = dict(zip(man['seg_uid'], man['set']))
lab_by_seg = dict(zip(man['seg_uid'], man['label']))
spk_by_seg = dict(zip(man['seg_uid'], man['speaker_id']))

rows_out = []
summary = {}

man['key'] = man['set'] + '_' + man['speaker_id'].astype(str) + '_' + man['seg_uid']

for tag, textf, audiof in [
    ("o25", f"{SYNC}/p14_o25_text.csv",  f"{P14}/p14_o25_zeroshot_scores.csv"),
    ("q2a", f"{SYNC}/p14_q2a_text.csv",  f"{P14}/p14_q2a_zeroshot_scores.csv"),
]:
    t = pd.read_csv(textf)
    a = pd.read_csv(audiof)
    a['key'] = a['clip'].str.replace('.wav', '', regex=False)
    print(f"\n=== {tag} === text rows {len(t)} audio rows {len(a)}")
    # POSITIONAL join: verified identical row order (see DISCREPANCIES.md)
    assert len(t) == len(man) == len(a)
    assert (man['key'].values == t['id'].values).all(), "text order differs from manifest"
    assert (man['key'].values == a['key'].values).all(), "audio order differs from manifest"
    assert (man['label'].values == t['label'].values).all()
    assert (man['label'].values == a['label'].values).all()
    assert (man['speaker_id'].values == t['speaker_id'].values).all()
    assert (man['speaker_id'].values == a['speaker'].values).all()

    m = pd.DataFrame({
        'key': man['key'].values, 'seg': man['seg_uid'].values,
        'speaker_id': man['speaker_id'].values, 'label': man['label'].values,
        'arm': man['set'].values,
        'p_yes_text': t['p_yes'].values, 'mass_text': t['answer_mass'].values,
        'p_yes_audio': a['p_yes'].values, 'mass_audio': a['mass'].values,
    })
    print(f"  merged n={len(m)}  arms: {m['arm'].value_counts().to_dict()}  speakers={m['speaker_id'].nunique()}")

    m.to_csv(f"{OUT}/p14_{tag}_1a_joined.csv", index=False)

    # pooled AUC (text) as reproduction check of the already-reported 1a number
    pooled = auc_rank(m['label'], m['p_yes_text'])
    print(f"  pooled text AUC (rank formula) = {pooled:.4f}")

    # per-arm AUCs, PAIRED speaker bootstrap
    res = {}
    for arm in ('conflict','agreement'):
        sub = m[m['arm'] == arm]
        res[arm] = dict(auc=auc_rank(sub['label'], sub['p_yes_text']),
                        n=len(sub), n_spk=int(sub['speaker_id'].nunique()))
    ci = boot_paired(m['label'].values, m['p_yes_text'].values,
                     m['speaker_id'].values, m['arm'].values, 2000, 0)
    for arm in ('conflict','agreement'):
        res[arm]['lo'], res[arm]['hi'] = ci[arm]
    print(f"  usable paired draws = {ci['usable']}/2000")

    # Pearson r text vs audio
    x = m['p_yes_text'].values.astype(float); y = m['p_yes_audio'].values.astype(float)
    r = float(np.corrcoef(x, y)[0,1])
    # same-decision share at 0.5
    same = float(((x > 0.5) == (y > 0.5)).mean())

    print(f"  CONFLICT  AUC {res['conflict']['auc']:.4f} [{res['conflict']['lo']:.4f},{res['conflict']['hi']:.4f}] n={res['conflict']['n']} spk={res['conflict']['n_spk']}")
    print(f"  AGREEMENT AUC {res['agreement']['auc']:.4f} [{res['agreement']['lo']:.4f},{res['agreement']['hi']:.4f}] n={res['agreement']['n']} spk={res['agreement']['n_spk']}")
    print(f"  PEARSON r(text,audio) = {r:.4f}")
    print(f"  SAME-DECISION share   = {same:.4f}")

    summary[tag] = dict(pooled_text_auc=round(pooled,6),
                        conflict=res['conflict'], agreement=res['agreement'],
                        pearson_r=round(r,6), same_decision=round(same,6),
                        usable_draws=ci['usable'], n=len(m),
                        n_spk=int(m['speaker_id'].nunique()))

    f = f"{OUT}/p14_{tag}_1a_joined.csv"
    rows_out += [
        ("1a_"+tag+"_text_conflict_auc","text-prompt conflict-arm AUC, "+tag, res['conflict']['auc'], res['conflict']['lo'], res['conflict']['hi'], res['conflict']['n'], res['conflict']['n_spk'], f),
        ("1a_"+tag+"_text_agreement_auc","text-prompt agreement-arm AUC, "+tag, res['agreement']['auc'], res['agreement']['lo'], res['agreement']['hi'], res['agreement']['n'], res['agreement']['n_spk'], f),
        ("1a_"+tag+"_pearson_text_audio","Pearson r transcript p_yes vs audio p_yes, "+tag, r, "", "", len(m), m['speaker_id'].nunique(), f),
        ("1a_"+tag+"_same_decision","share text decision == audio decision at 0.5, "+tag, same, "", "", len(m), m['speaker_id'].nunique(), f),
    ]

json.dump(summary, open(f"{OUT}/p14_1a_completion.json","w"), indent=1)

with open("reports/part16/rows/POD1.tsv","a") as fh:
    for r_ in rows_out:
        vals = [r_[0], r_[1]] + [f"{v:.4f}" if isinstance(v,float) else str(v) for v in r_[2:5]] + [str(r_[5]), str(r_[6]), r_[7]]
        fh.write("\t".join(vals) + "\n")
print("\nrows appended:", len(rows_out))
