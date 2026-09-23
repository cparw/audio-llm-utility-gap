"""PART20 POD6 job B step 1: NeuroVoz age and sex matched PD / healthy pairs.
Rules: same sex, |age difference| <= 5 years, one-to-one, greedy, seed 0.
Greedy order: PD speakers sorted by number of eligible healthy partners (fewest first), ties broken by
numpy.random.default_rng(0).permutation of the PD ids; each PD speaker takes the unmatched eligible healthy
speaker with the smallest |age difference|, ties broken by a second permutation from the same generator.
The greedy pair count is compared with the maximum bipartite matching size (scipy) so we know it is the largest.
Only speakers in the 1270-clip paper set (omni_final/omni_neurovoz_zeroshot_scores.csv) are eligible.
usage: nv_pairs.py OUTDIR"""
import sys, json, hashlib, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
OUT = sys.argv[1]
MD = "<local data dir>/paper work/paper1_local_runs/drive_data/spanish_neurovoz/zenodo_upload/metadata"
ZS = "<local data dir>/release/omni_final/omni_neurovoz_zeroshot_scores.csv"
pdm = pd.read_csv(f"{MD}/data_pd.csv"); hc = pd.read_csv(f"{MD}/data_hc.csv"); z = pd.read_csv(ZS)
spk_in = set(z.speaker.astype(str))
def per_spk(d, grp):
    g = d.groupby("ID").agg(sex=("Sex", lambda s: sorted(set(s.dropna()))), age=("Age", lambda s: sorted(set(s.dropna()))))
    g = g.reset_index(); g["ID"] = g.ID.astype(str); g["group"] = grp
    return g
P = per_spk(pdm, "PD"); H = per_spk(hc, "HC")
excl = []
for g in (P, H):
    for _, r in g.iterrows():
        if r.ID not in spk_in: excl.append((r.ID, r.group, "not in the 1270-clip paper set"))
        elif len(r.sex) != 1 or len(r.age) != 1: excl.append((r.ID, r.group, f"sex {r.sex} age {r.age}: missing or ambiguous"))
ex_ids = {e[0] for e in excl}
P = P[~P.ID.isin(ex_ids)].copy(); H = H[~H.ID.isin(ex_ids)].copy()
for g in (P, H): g["sex"] = g.sex.str[0].astype(int); g["age"] = g.age.str[0].astype(float)
P = P.reset_index(drop=True); H = H.reset_index(drop=True)
E = np.array([[(P.sex[i] == H.sex[j]) and abs(P.age[i] - H.age[j]) <= 5 for j in range(len(H))] for i in range(len(P))])
mm = maximum_bipartite_matching(csr_matrix(E.astype(int)), perm_type="column")
max_pairs = int((mm >= 0).sum())
rng = np.random.default_rng(0)
tieP = {p: k for k, p in enumerate(rng.permutation(P.ID.values))}
tieH = {h: k for k, h in enumerate(rng.permutation(H.ID.values))}
order = sorted(range(len(P)), key=lambda i: (int(E[i].sum()), tieP[P.ID[i]]))
used, pairs = set(), []
for i in order:
    c = [j for j in range(len(H)) if E[i, j] and j not in used]
    if not c: continue
    j = min(c, key=lambda j: (abs(P.age[i] - H.age[j]), tieH[H.ID[j]]))
    used.add(j)
    pairs.append(dict(pd_id=P.ID[i], hc_id=H.ID[j], sex=int(P.sex[i]), pd_age=float(P.age[i]), hc_age=float(H.age[j]),
                      age_diff=float(P.age[i] - H.age[j])))
pr = pd.DataFrame(pairs); pr.index.name = "pair"; pr.to_csv(f"{OUT}/nv_matched_pairs.csv")
keep = set(pr.pd_id) | set(pr.hc_id)
zs = z[z.speaker.astype(str).isin(keep)]
info = dict(eligible_pd=len(P), eligible_hc=len(H), excluded=excl, max_matching_pairs=max_pairs, greedy_pairs=len(pr),
            greedy_is_maximum=bool(len(pr) == max_pairs),
            greedy_order="PD speakers by number of eligible healthy partners ascending, ties by default_rng(0).permutation(PD ids); "
                         "each takes the unmatched eligible healthy speaker with the smallest |age difference|, ties by the next default_rng(0).permutation(HC ids)",
            n_clips=int(len(zs)), n_clips_pd=int((zs.label == 1).sum()), n_clips_hc=int((zs.label == 0).sum()),
            n_speakers=len(keep), sex_counts={str(k): int(v) for k, v in pr.sex.value_counts().items()},
            mean_abs_age_diff=float(pr.age_diff.abs().mean()), pd_age_mean=float(pr.pd_age.mean()), hc_age_mean=float(pr.hc_age.mean()),
            full_set=dict(pd_age_mean=float(P.age.mean()), hc_age_mean=float(H.age.mean()),
                          pd_male_frac=float(P.sex.mean()), hc_male_frac=float(H.sex.mean())),
            metadata=[f"{MD}/data_pd.csv", f"{MD}/data_hc.csv"], clip_list=ZS)
json.dump(info, open(f"{OUT}/nv_matched_pairs.json", "w"), indent=1)
zs[["clip", "speaker", "label"]].to_csv(f"{OUT}/nv_matched_clips.csv", index=False)
print(json.dumps({k: v for k, v in info.items() if k != "excluded"}, indent=1)); print("excluded", excl)
print(pr.to_string())
