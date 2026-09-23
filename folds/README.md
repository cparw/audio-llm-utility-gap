# Speaker fold assignments

Every probe in the paper is scored out of fold with scikit-learn `GroupKFold(n_splits=5)` grouped by speaker. The files here give the fold of every clip for each split that was actually run. So a probe number can be rebuilt without re-deriving the folds.

Files marked CONFIRMED were checked by refitting the saved probe with them. The result was then compared with the saved out-of-fold predictions, clip by clip. Files whose names end in `_UNVERIFIED` could not be checked that way, because no per-clip out-of-fold file was saved for those splits. They reproduce the saved per-repeat AUCs only (within 0.008), as each entry below says.

## Columns

- `clip_id`: the clip exactly as the run's states file names it.
- `speaker_id`: the group label exactly as the run passed it to GroupKFold.
- `fold`: 0 to 4, in the order GroupKFold yields its test folds.

## Why one dataset has several files

With shuffling off, GroupKFold sorts speakers by clip count with `numpy.argsort`. It hands them out, largest first, to the fold that has the fewest clips so far. When speakers have the same clip count, the order argsort leaves them in decides the split.

On ADReSSo, ADReSS-2020, MDVR-KCL and E-DAIC every speaker has exactly one clip, so that order decides the whole split. On Pitt, PC-GITA and NeuroVoz it decides most of it.

That order was different on the Mac used for some runs (numpy 2.2.6, arm64) and on the Linux GPU pods used for the others. So the same code, clips and speaker labels gave two different splits. Both are released:

- `_mac`: GroupKFold exactly as installed on the Mac (numpy default argsort).
- `_pod`: the same algorithm with `numpy.argsort(counts, kind="stable")`, reversed as GroupKFold reverses it. This reproduces every pod run that could be checked. It describes the result. The pods' numpy build itself was not inspected.
- `_seed0` to `_seed4`: `nested_repeats_all.py` (and part16 `perlayer_nested.py`) first renumber the speakers with `numpy.random.default_rng(seed).permutation(numpy.unique(ids))`. Then they run GroupKFold on the new numbers, once per seed. The Table 1 probe values are the mean over these five splits. The same tie rule applies after renumbering.
- `_shuffle_rs0` to `_shuffle_rs4`: the E-DAIC full-window probe used `GroupKFold(5, shuffle=True, random_state=r)`, which does not depend on argsort ties.

## How the files were checked

For each run whose hidden states are on disk, the probe was refitted with the file's folds. The refit uses StandardScaler, then LogisticRegression with max_iter 2000 and balanced class weights, as in the run's script. For nested probes, each outer fold was refitted at the layer the run saved, and the predictions were compared clip by clip. The inner GroupKFold(4) layer choice was then rerun with the same tie rule, to check that it picks the same layers.

Runs made on the Mac match exactly (largest difference 1.1e-16). Runs made on pods cannot match to 1e-6 on the Mac. The logistic regression stops at a gradient tolerance, and different numeric libraries stop at slightly different points. On identical states and folds the largest pod-versus-Mac difference is 0.083. Casting the same features to float64 on the Mac alone moves the same predictions by up to 0.038.

Every checked pod run also re-derives the saved layer choices (exceptions listed below) and agrees on AUC. The other tie rule is off by at least 0.55 on every run checked, so the two rules are never confused.

`CONFIRMED` means reproduced clip by clip as above. A name ending in `_UNVERIFIED` means no per-clip out-of-fold file was saved for that split, so it could not be checked clip by clip. The entry says what it does reproduce.

Pitt: one Pitt participant is listed in both diagnosis groups, so the files carry 228 speaker labels for 227 people. The labels are kept exactly as the runs used them. So in some splits this one person sits in two folds, and each Pitt entry says where.

The Qwen3-Omni part16 POD3 Pitt run wrote the Pitt ids without zero padding (the group name plus the bare number, not three digits). These ids sort differently and give different splits, so that run has its own files, the `_unpaddedids_` files.

## Files

### Single split, Mac tie order

**`pitt_groupkfold5_mac.csv`** (Pitt. 468 clips, 228 speaker labels, fold sizes 94/94/94/93/93). CONFIRMED. sklearn GroupKFold(5) as installed on the Mac.
- Reproduces `q2a_probe2_pitt` (Qwen2-Audio, enc/llm/ans). Max abs diff 1.11e-16. Saved layers re-derived.
- Reproduces `q2a_probe2_podruns_pitt` (Qwen2-Audio, enc). Max abs diff 1.11e-16. Saved layers re-derived. This pod_runs file is a byte copy of the Mac output probe2/pitt_enc_nested_oof.csv, not a pod result.
- Reproduces `egemaps_pitt` (eGeMAPS baseline). Max abs diff 1.11e-16.
- The participant listed twice: folds 1 and 3.

**`adresso_groupkfold5_mac.csv`** (ADReSSo. 237 clips, 237 speaker labels, fold sizes 48/48/47/47/47). CONFIRMED. sklearn GroupKFold(5) as installed on the Mac.
- Reproduces `q2a_probe2_adresso` (Qwen2-Audio, enc/llm/ans). Max abs diff 1.11e-16. Saved layers re-derived.
- Reproduces `egemaps_adresso` (eGeMAPS baseline). Max abs diff 1.11e-16.

**`adress2020_groupkfold5_mac.csv`** (ADReSS-2020. 156 clips, 156 speaker labels, fold sizes 32/31/31/31/31). CONFIRMED. sklearn GroupKFold(5) as installed on the Mac.
- Reproduces `q2a_probe2_adress2020` (Qwen2-Audio, enc/llm/ans). Max abs diff 1.11e-16. Saved layers re-derived.
- Reproduces `egemaps_adress2020` (eGeMAPS baseline). Max abs diff 1.11e-16.

**`pcgita_groupkfold5_mac.csv`** (PC-GITA. 1100 clips, 100 speaker labels, fold sizes 220/220/220/220/220). CONFIRMED. sklearn GroupKFold(5) as installed on the Mac.
- Reproduces `q2a_probe2_pcgita` (Qwen2-Audio, enc/llm/ans). Max abs diff 1.11e-16. Saved layers re-derived.
- Reproduces `egemaps_pcgita` (eGeMAPS baseline). Max abs diff 1.11e-16.

**`neurovoz_groupkfold5_mac.csv`** (NeuroVoz. 1270 clips, 107 speaker labels, fold sizes 256/251/251/251/261). CONFIRMED. sklearn GroupKFold(5) as installed on the Mac.
- Reproduces `q2a_probe2_neurovoz` (Qwen2-Audio, enc/llm/ans). Max abs diff 1.11e-16. Saved layers re-derived.
- Reproduces `egemaps_neurovoz` (eGeMAPS baseline). Max abs diff 1.11e-16.

**`kcl_groupkfold5_mac.csv`** (MDVR-KCL. 37 clips, 37 speaker labels, fold sizes 8/8/7/7/7). CONFIRMED. sklearn GroupKFold(5) as installed on the Mac.
- Reproduces `q2a_probe2_kcl` (Qwen2-Audio, enc/llm/ans). Max abs diff 1.11e-16. Saved layers re-derived.
- Reproduces `egemaps_kcl` (eGeMAPS baseline). Max abs diff 1.11e-16.

**`edaic_groupkfold5_mac.csv`** (E-DAIC first 30 s. 275 clips, 275 speaker labels, fold sizes 55/55/55/55/55). CONFIRMED. sklearn GroupKFold(5) as installed on the Mac.
- Reproduces `q2a_probe2_edaic` (Qwen2-Audio, enc/llm/ans). Max abs diff 1.11e-16. Saved layers re-derived.
- Reproduces `egemaps_edaic` (eGeMAPS baseline). Max abs diff 1.11e-16.

### Single split, pod tie order

**`pitt_groupkfold5_pod.csv`** (Pitt. 468 clips, 228 speaker labels, fold sizes 94/94/94/93/93). CONFIRMED. GroupKFold(5) with the stable tie order.
- Reproduces `omni_final_pitt` (Qwen2.5-Omni, enc/proj/llm/ans). Max abs diff 6.13e-02. Saved layers re-derived.
- The fold file `pitt468_folds.csv`, written on a pod by GroupKFold itself, agrees on 468 of 468 clips. It comes from part16 POD2 make_folds.py (pod), the same call as sft_projector.py.
- 9 per-layer AUC curves saved by pod runs (Kimi-Audio, Qwen2-Audio, Qwen2.5-Omni) agree with this split within 0.0013 AUC at every layer. The Mac order misses them by 0.0193 to 0.0637.
- The participant listed twice: folds 0 and 3.

**`adresso_groupkfold5_pod.csv`** (ADReSSo. 237 clips, 237 speaker labels, fold sizes 48/48/47/47/47). CONFIRMED. GroupKFold(5) with the stable tie order.
- Reproduces `q3o_new_adresso` (Qwen3-Omni, enc/proj/llm/ans). Max abs diff 3.06e-02. Saved layers re-derived.
- 3 per-layer AUC curves saved by pod runs (Qwen2-Audio) agree with this split within 0.0013 AUC at every layer. The Mac order misses them by 0.0165 to 0.0373.
- The Qwen2.5-Omni single-split probe on this dataset ran on the same pod with the same labels. Its states are not on disk, so for that run this split is inferred, not refitted.

**`adress2020_groupkfold5_pod.csv`** (ADReSS-2020. 156 clips, 156 speaker labels, fold sizes 32/31/31/31/31). CONFIRMED. GroupKFold(5) with the stable tie order.
- Reproduces `q3o_new_adress2020` (Qwen3-Omni, enc/proj/llm/ans). Max abs diff 1.40e-02. Saved layers re-derived.
- 3 per-layer AUC curves saved by pod runs (Qwen2-Audio) agree with this split within 0.0016 AUC at every layer. The Mac order misses them by 0.0296 to 0.1016.
- The Qwen2.5-Omni single-split probe on this dataset ran on the same pod with the same labels. Its states are not on disk, so for that run this split is inferred, not refitted.

**`pcgita_groupkfold5_pod.csv`** (PC-GITA. 1100 clips, 100 speaker labels, fold sizes 220/220/220/220/220). CONFIRMED. GroupKFold(5) with the stable tie order.
- Reproduces `q3o_new_pcgita` (Qwen3-Omni, enc/proj/llm/ans). Max abs diff 4.61e-02. Saved layers re-derived, except ans [27, 18, 4, 15, 1] vs [27, 18, 3, 15, 1].
- 5 per-layer AUC curves saved by pod runs (Kimi-Audio, Qwen2-Audio) agree with this split within 0.0009 AUC at every layer. The Mac order misses them by 0.0077 to 0.0269.
- The Qwen2.5-Omni single-split probe on this dataset ran on the same pod with the same labels. Its states are not on disk, so for that run this split is inferred, not refitted.

**`neurovoz_groupkfold5_pod.csv`** (NeuroVoz. 1270 clips, 107 speaker labels, fold sizes 256/251/251/251/261). CONFIRMED. GroupKFold(5) with the stable tie order.
- Reproduces `q3o_new_neurovoz` (Qwen3-Omni, enc/proj/llm/ans). Max abs diff 4.47e-02. Saved layers re-derived.
- 3 per-layer AUC curves saved by pod runs (Qwen2-Audio) agree with this split within 0.0006 AUC at every layer. The Mac order misses them by 0.0094 to 0.0306.
- Not confirmed: `q2a_probe2_podruns_neurovoz` (enc/llm/ans). It used states the pod extracted itself, which are not on disk. On the Mac states it sits closest to this split (max abs diff 2.49e-01, Mac order 9.88e-01).
- The Qwen2.5-Omni single-split probe on this dataset ran on the same pod with the same labels. Its states are not on disk, so for that run this split is inferred, not refitted.

**`kcl_groupkfold5_pod.csv`** (MDVR-KCL. 37 clips, 37 speaker labels, fold sizes 8/8/7/7/7). CONFIRMED. GroupKFold(5) with the stable tie order.
- Reproduces `q2a_probe2_kcl_local` (Qwen2-Audio, enc/llm). Max abs diff 2.66e-03. Saved layers re-derived.
- Reproduces `q3o_new_kcl` (Qwen3-Omni, enc/proj/llm/ans). Max abs diff 2.76e-03. Saved layers re-derived.
- 3 per-layer AUC curves saved by pod runs (Qwen2-Audio) agree with this split within 0.0060 AUC at every layer. The Mac order misses them by 0.0476 to 0.1012.
- The Qwen2.5-Omni single-split probe on this dataset ran on the same pod with the same labels. Its states are not on disk, so for that run this split is inferred, not refitted.

**`edaic_groupkfold5_pod.csv`** (E-DAIC first 30 s. 275 clips, 275 speaker labels, fold sizes 55/55/55/55/55). CONFIRMED. GroupKFold(5) with the stable tie order.
- Reproduces `omni_final_edaic` (Qwen2.5-Omni, enc/proj/llm/ans). Max abs diff 3.36e-02. Saved layers re-derived.
- The fold file `sft1c_folds.json`, written on a pod by GroupKFold itself, agrees on 275 of 275 clips. It comes from the part16 POD1 sft1c fold list (pod).
- 9 per-layer AUC curves saved by pod runs (Kimi-Audio, Qwen2-Audio, Qwen2.5-Omni) agree with this split within 0.0023 AUC at every layer. The Mac order misses them by 0.0528 to 0.1144.
- Not confirmed: `q2a_probe2_podruns_edaic` (enc/llm/ans). It used states the pod extracted itself, which are not on disk. On the Mac states it sits closest to this split (max abs diff 2.45e-01, Mac order 9.70e-01).

### Five renumbered splits, pod tie order (the Table 1 probe values)

Speakers renumbered with `default_rng(seed)`, then GroupKFold(5) with the stable tie order. These are the splits behind the five-repeat probe means in Table 1, for every model and dataset except the Qwen2.5-Omni Pitt encoder value (next section).

A per-clip out-of-fold file was saved only for the part16 POD3 Qwen3-Omni PC-GITA repeats. So only the PC-GITA files are confirmed clip by clip. The others reproduce the saved per-repeat AUCs only and carry `_UNVERIFIED`.

For every run and stream checked, the pod split is closer to the saved AUC than the Mac split. The Mac split was run on seed 0, and on all five seeds for the Qwen2-Audio Pitt encoder. The Qwen3-Omni Pitt and E-DAIC cells of Table 1 come from runs whose states are not on disk. So for those two cells these splits are inferred, not checked.

In each entry below, "saved per-repeat AUCs" lists the runs checked. No per-clip out-of-fold file exists for any `_UNVERIFIED` split.

- **`pitt_groupkfold5_pod_seed0_UNVERIFIED.csv`** (Pitt, seed 0). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen2.5-Omni enc) are reproduced within 0.0061. The Mac order misses by 0.0067 to 0.0406. The participant listed twice: both labels in fold 1.
- **`pitt_groupkfold5_pod_seed1_UNVERIFIED.csv`** (Pitt, seed 1). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen2.5-Omni enc) are reproduced within 0.0009. The Mac order misses by 0.0088. The participant listed twice: folds 4 and 1.
- **`pitt_groupkfold5_pod_seed2_UNVERIFIED.csv`** (Pitt, seed 2). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen2.5-Omni enc) are reproduced within 0.0011. The Mac order misses by 0.0248. The participant listed twice: folds 4 and 1.
- **`pitt_groupkfold5_pod_seed3_UNVERIFIED.csv`** (Pitt, seed 3). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen2.5-Omni enc) are reproduced within 0.0080. The Mac order misses by 0.0424. The participant listed twice: folds 4 and 1.
- **`pitt_groupkfold5_pod_seed4_UNVERIFIED.csv`** (Pitt, seed 4). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen2.5-Omni enc) are reproduced within 0.0022. The Mac order misses by 0.0185. The participant listed twice: folds 2 and 0.
- **`adresso_groupkfold5_pod_seed0_UNVERIFIED.csv`** (ADReSSo, seed 0). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0001. The Mac order misses by 0.0069 to 0.0227.
- **`adresso_groupkfold5_pod_seed1_UNVERIFIED.csv`** (ADReSSo, seed 1). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0001.
- **`adresso_groupkfold5_pod_seed2_UNVERIFIED.csv`** (ADReSSo, seed 2). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0006.
- **`adresso_groupkfold5_pod_seed3_UNVERIFIED.csv`** (ADReSSo, seed 3). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0002.
- **`adresso_groupkfold5_pod_seed4_UNVERIFIED.csv`** (ADReSSo, seed 4). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0002.
- **`adress2020_groupkfold5_pod_seed0_UNVERIFIED.csv`** (ADReSS-2020, seed 0). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0002. The Mac order misses by 0.0027 to 0.0548.
- **`adress2020_groupkfold5_pod_seed1_UNVERIFIED.csv`** (ADReSS-2020, seed 1). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0004.
- **`adress2020_groupkfold5_pod_seed2_UNVERIFIED.csv`** (ADReSS-2020, seed 2). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0002.
- **`adress2020_groupkfold5_pod_seed3_UNVERIFIED.csv`** (ADReSS-2020, seed 3). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0003.
- **`adress2020_groupkfold5_pod_seed4_UNVERIFIED.csv`** (ADReSS-2020, seed 4). UNVERIFIED. Saved per-repeat AUCs (5: Kimi-Audio llm, Qwen2-Audio ans, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0002.
- **`pcgita_groupkfold5_pod_seed0.csv`** (PC-GITA, seed 0). CONFIRMED. Per clip vs `q3o_POD3_pcgita` (ans/enc/llm/proj): max abs diff 7.07e-02. Saved layers re-derived, except llm [14, 15, 1, 0, 44] vs [14, 16, 1, 0, 44]. The other tie order misses by 0.98 or more. Saved per-repeat AUCs (4: Kimi-Audio llm, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0003. The Mac order misses by 0.0066 to 0.0336.
- **`pcgita_groupkfold5_pod_seed1.csv`** (PC-GITA, seed 1). CONFIRMED. Per clip vs `q3o_POD3_pcgita` (ans/enc/llm): max abs diff 4.00e-02. Saved layers re-derived. The other tie order misses by 0.96 or more. Saved per-repeat AUCs (4: Kimi-Audio llm, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0031.
- **`pcgita_groupkfold5_pod_seed2.csv`** (PC-GITA, seed 2). CONFIRMED. Per clip vs `q3o_POD3_pcgita` (ans/enc/llm): max abs diff 8.31e-02. Saved layers re-derived. The other tie order misses by 0.97 or more. Saved per-repeat AUCs (4: Kimi-Audio llm, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0024.
- **`pcgita_groupkfold5_pod_seed3.csv`** (PC-GITA, seed 3). CONFIRMED. Per clip vs `q3o_POD3_pcgita` (ans/enc/llm): max abs diff 5.16e-02. Saved layers re-derived. The other tie order misses by 0.99 or more. Saved per-repeat AUCs (4: Kimi-Audio llm, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0025.
- **`pcgita_groupkfold5_pod_seed4.csv`** (PC-GITA, seed 4). CONFIRMED. Per clip vs `q3o_POD3_pcgita` (ans/enc/llm): max abs diff 7.24e-02. Saved layers re-derived, except ans [3, 15, 44, 17, 27] vs [3, 15, 42, 17, 27], and llm [14, 0, 21, 14, 12] vs [14, 0, 21, 14, 13]. The other tie order misses by 0.91 or more. Saved per-repeat AUCs (4: Kimi-Audio llm, Qwen2-Audio enc, Qwen2-Audio llm, Qwen3-Omni enc) are reproduced within 0.0002.
- **`neurovoz_groupkfold5_pod_seed0_UNVERIFIED.csv`** (NeuroVoz, seed 0). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0001. The Mac order misses by 0.0077 to 0.0148.
- **`neurovoz_groupkfold5_pod_seed1_UNVERIFIED.csv`** (NeuroVoz, seed 1). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0015.
- **`neurovoz_groupkfold5_pod_seed2_UNVERIFIED.csv`** (NeuroVoz, seed 2). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0052.
- **`neurovoz_groupkfold5_pod_seed3_UNVERIFIED.csv`** (NeuroVoz, seed 3). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0005.
- **`neurovoz_groupkfold5_pod_seed4_UNVERIFIED.csv`** (NeuroVoz, seed 4). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0006.
- **`kcl_groupkfold5_pod_seed0_UNVERIFIED.csv`** (MDVR-KCL, seed 0). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0000. The Mac order misses by 0.0119 to 0.0744.
- **`kcl_groupkfold5_pod_seed1_UNVERIFIED.csv`** (MDVR-KCL, seed 1). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0000.
- **`kcl_groupkfold5_pod_seed2_UNVERIFIED.csv`** (MDVR-KCL, seed 2). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0000.
- **`kcl_groupkfold5_pod_seed3_UNVERIFIED.csv`** (MDVR-KCL, seed 3). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0000.
- **`kcl_groupkfold5_pod_seed4_UNVERIFIED.csv`** (MDVR-KCL, seed 4). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen3-Omni enc) are reproduced within 0.0000.
- **`edaic_groupkfold5_pod_seed0_UNVERIFIED.csv`** (E-DAIC first 30 s, seed 0). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen2.5-Omni enc) are reproduced within 0.0008. The Mac order misses by 0.0188 to 0.0568.
- **`edaic_groupkfold5_pod_seed1_UNVERIFIED.csv`** (E-DAIC first 30 s, seed 1). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen2.5-Omni enc) are reproduced within 0.0008.
- **`edaic_groupkfold5_pod_seed2_UNVERIFIED.csv`** (E-DAIC first 30 s, seed 2). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen2.5-Omni enc) are reproduced within 0.0004.
- **`edaic_groupkfold5_pod_seed3_UNVERIFIED.csv`** (E-DAIC first 30 s, seed 3). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen2.5-Omni enc) are reproduced within 0.0002.
- **`edaic_groupkfold5_pod_seed4_UNVERIFIED.csv`** (E-DAIC first 30 s, seed 4). UNVERIFIED. Saved per-repeat AUCs (3: Kimi-Audio llm, Qwen2-Audio enc, Qwen2.5-Omni enc) are reproduced within 0.0003.

### Five renumbered splits, Mac tie order (Qwen2.5-Omni Pitt encoder probe, 0.7706)

Speakers renumbered with `default_rng(seed)`, then sklearn GroupKFold(5) as installed on the Mac. These reproduce `overnight2/part10/pitt_enc_nested5_oof.npz` exactly. That file is behind the Table 1 Qwen2.5-Omni Pitt encoder value 0.7706. Its layer choices were not saved, so they were re-derived with the inner GroupKFold(4). The per-clip scores then match exactly.

The older value 0.7761 (`omni_final/omni_pitt_nested_repeats.json`) is the same five-repeat probe on the same states, run on a pod. So it uses the pod files of the previous section.

- **`pitt_groupkfold5_mac_seed0.csv`** (Pitt, seed 0). CONFIRMED. Per clip vs `omni_part10_pitt_nested5` (enc): max abs diff 0.00e+00. The other tie order misses by 0.99 or more. The participant listed twice: folds 1 and 4.
- **`pitt_groupkfold5_mac_seed1.csv`** (Pitt, seed 1). CONFIRMED. Per clip vs `omni_part10_pitt_nested5` (enc): max abs diff 0.00e+00. The other tie order misses by 0.96 or more. The participant listed twice: folds 0 and 2.
- **`pitt_groupkfold5_mac_seed2.csv`** (Pitt, seed 2). CONFIRMED. Per clip vs `omni_part10_pitt_nested5` (enc): max abs diff 0.00e+00. The other tie order misses by 1.00 or more. The participant listed twice: folds 1 and 0.
- **`pitt_groupkfold5_mac_seed3.csv`** (Pitt, seed 3). CONFIRMED. Per clip vs `omni_part10_pitt_nested5` (enc): max abs diff 0.00e+00. The other tie order misses by 0.99 or more. The participant listed twice: folds 3 and 1.
- **`pitt_groupkfold5_mac_seed4.csv`** (Pitt, seed 4). CONFIRMED. Per clip vs `omni_part10_pitt_nested5` (enc): max abs diff 0.00e+00. The other tie order misses by 0.99 or more. The participant listed twice: folds 2 and 3.

### Five renumbered splits, pod tie order, Pitt ids without zero padding (part16 POD3 Qwen3-Omni)

Same as the pod files above, but on the unpadded ids (the group name plus the bare number), as written in the part16 POD3 Qwen3-Omni Pitt states. Only seed 0 has a projector-stream repeat, so seeds 1 to 4 are checked on the other three streams. The file names say `unpaddedids` and hold no speaker id.

- **`pitt_groupkfold5_pod_unpaddedids_seed0.csv`** (Pitt, seed 0). CONFIRMED. Per clip vs `q3o_POD3_pitt` (enc/proj/llm/ans): max abs diff 3.08e-02. Saved layers re-derived. The other tie order misses by 0.94 or more. The participant listed twice: both labels in fold 3.
- **`pitt_groupkfold5_pod_unpaddedids_seed1.csv`** (Pitt, seed 1). CONFIRMED. Per clip vs `q3o_POD3_pitt` (enc/llm/ans): max abs diff 5.36e-02. Saved layers re-derived. The other tie order misses by 0.98 or more. The participant listed twice: folds 4 and 3.
- **`pitt_groupkfold5_pod_unpaddedids_seed2.csv`** (Pitt, seed 2). CONFIRMED. Per clip vs `q3o_POD3_pitt` (enc/llm/ans): max abs diff 3.26e-02. Saved layers re-derived. The other tie order misses by 0.91 or more. The participant listed twice: folds 0 and 3.
- **`pitt_groupkfold5_pod_unpaddedids_seed3.csv`** (Pitt, seed 3). CONFIRMED. Per clip vs `q3o_POD3_pitt` (enc/llm/ans): max abs diff 4.18e-02. Saved layers re-derived. The other tie order misses by 0.94 or more. The participant listed twice: folds 4 and 1.
- **`pitt_groupkfold5_pod_unpaddedids_seed4.csv`** (Pitt, seed 4). CONFIRMED. Per clip vs `q3o_POD3_pitt` (enc/llm/ans): max abs diff 3.92e-02. Saved layers re-derived. The other tie order misses by 0.98 or more. The participant listed twice: both labels in fold 4.

### E-DAIC full window, shuffled splits (part16 EDAICFULL)

`GroupKFold(5, shuffle=True, random_state=r)` as in `part16/EDAICFULL/probe_boot.py`. No tie order is involved. The saved per-clip file holds only the mean over the five splits, so the five files are checked together. Refitting all five at the saved layers and averaging reproduces the saved mean out-of-fold scores.

The five files share one check, so each entry below gives the same numbers. The larger llm and ans differences come from the inner layer choice, which differs in some folds at near ties.

- **`edaic_full_groupkfold5_shuffle_rs0.csv`** (E-DAIC full window, random_state 0). CONFIRMED. Max abs diff: enc 5.60e-03, proj 5.11e-03, llm 2.40e-02, ans 2.33e-02.
- **`edaic_full_groupkfold5_shuffle_rs1.csv`** (E-DAIC full window, random_state 1). CONFIRMED. Max abs diff: enc 5.60e-03, proj 5.11e-03, llm 2.40e-02, ans 2.33e-02.
- **`edaic_full_groupkfold5_shuffle_rs2.csv`** (E-DAIC full window, random_state 2). CONFIRMED. Max abs diff: enc 5.60e-03, proj 5.11e-03, llm 2.40e-02, ans 2.33e-02.
- **`edaic_full_groupkfold5_shuffle_rs3.csv`** (E-DAIC full window, random_state 3). CONFIRMED. Max abs diff: enc 5.60e-03, proj 5.11e-03, llm 2.40e-02, ans 2.33e-02.
- **`edaic_full_groupkfold5_shuffle_rs4.csv`** (E-DAIC full window, random_state 4). CONFIRMED. Max abs diff: enc 5.60e-03, proj 5.11e-03, llm 2.40e-02, ans 2.33e-02.

## Which runs used which split

| run | where | split |
|---|---|---|
| Qwen2-Audio single-split nested probes, `probe2/<ds>_{enc,llm,ans}_nested_oof.csv` (all seven datasets) | Mac | `_mac` |
| eGeMAPS baseline, `overnight/perclip/egemaps_<ds>_oof.csv` | Mac | `_mac` |
| Qwen2.5-Omni Pitt encoder probe 0.7706, `part10/pitt_enc_nested5_oof.npz` | Mac | `pitt_groupkfold5_mac_seed0..4` |
| Qwen2.5-Omni single-split nested probes, `omni_final/omni_<ds>_*_nested_oof.csv` | pod | `_pod` |
| Qwen3-Omni single-split nested probes, `overnight2/q3o_new/q3o_<ds>_*_nested_oof.csv` | pod | `_pod` |
| Qwen2-Audio MDVR-KCL `kcl_local` nested probe | pod | `kcl_groupkfold5_pod` |
| per-layer AUC curves written on pods (Qwen2-Audio, Qwen2.5-Omni, Kimi-Audio) | pod | `_pod` |
| five-repeat nested probes behind Table 1 (Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Kimi-Audio), all cells except the one above | pod | `_pod_seed0..4` |
| Qwen3-Omni part16 POD3 Pitt and PC-GITA repeats | pod | `pitt_groupkfold5_pod_unpaddedids_seed0..4`, `pcgita_groupkfold5_pod_seed0..4` |
| E-DAIC full window, part16 EDAICFULL | pod | `edaic_full_groupkfold5_shuffle_rs0..4` |

## Not covered

- Projector fine-tunes (`sft_projector.py`) use the same GroupKFold call but saved no fold record. A fine-tune cannot be refitted to check one. The pod-written `pitt468_folds.csv` from part16 POD2, made with the same call, equals `pitt_groupkfold5_pod.csv`.
- Qwen2.5-Omni states are on disk only for Pitt and E-DAIC. Qwen3-Omni states are on disk only for the five other datasets, plus the part16 POD3 Pitt and PC-GITA re-extractions. Splits of runs without states are named above by inference only.
- The older E-DAIC conflict manifest (390 rows, `edaic_conflict_new/`) has five-repeat probes but no per-clip out-of-fold file. Its saved AUCs sit near chance, where the layer choice is unstable. Its splits are not released here. The Part 14 conflict set has no nested probe.
- Runs made by others on other machines are not included.
