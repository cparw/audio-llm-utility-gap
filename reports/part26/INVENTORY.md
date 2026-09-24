# PART 26 inventory: encoder probe grid, Qwen2-Audio, Qwen3-Omni, Kimi-Audio

Written 24 Sep 2026, 00:31 to 00:45 UTC. Everything below was read from disk for this inventory. The cross-check script and its full output are in `scripts/part26/inventory/inv_check.py` and `scores/part26/inventory/inv_check.json`.

## Headline

- States ready for 13 of 21 cells: Qwen2-Audio on all 7 datasets, and Qwen3-Omni on 6 (every dataset except E-DAIC).
- States missing for 8 cells: Qwen3-Omni E-DAIC, and Kimi-Audio on all 7 datasets.
- Qwen3-Omni E-DAIC: a 30 s encoder state file was extracted on podD2 (the log shows `STATES {'enc': [275, 32, 1280], ...}`), but it was never pulled. I searched the whole release tree and the G-Drive paper folders and found no copy. Re-extracting it takes about 0.6 GPU hours, including the 30 minutes of setup. It fits before 05:00 UTC.
- Kimi-Audio: no encoder states exist anywhere, and no earlier script extracts them. Every earlier Kimi run hooked only the language model. The script docstring (`podD2_final/kimi_probe.py`) and all seven zero-shot sidecars give the reason: "Kimi sums discrete speech tokens and whisper features before the LM so there is no single encoder path to probe". Re-extracting means adding new forward hooks to the Whisper encoder inside kimia_infer, the part that builds `history.continuous_feature`. That is a new script and a method change, not a reuse. About 1.2 GPU hours covers all 7 datasets, including setup and a smoke test, so it fits before 05:00 UTC on time. **The authors need to decide** whether the paper should gain a Kimi encoder probe, because the current Method says the Kimi encoder side is skipped.
- Zero-shot per-clip files: all 21 exist. Each one gives exactly the master_lookup AUC when recomputed from its rows (4 dp).
- Fold files: the five pod-order repeat files exist for all 7 datasets, and each matches that dataset's clip list exactly.

## E-DAIC window and clip list

All three models use the **30 s window**: the first 30 s of `paper1_local_runs/drive_data/dcaps_proc/<pid>.wav`, cut in code as `x[:16000*30]`, on the same 275 participants. These are the master_lookup `edaic` zero-shot rows:
- Kimi-Audio row 24, "zero shot, 30 s window", `overnight2/part3/kimi_edaic_zeroshot.json`.
- Qwen2-Audio row 53, `probe2/edaic_zeroshot_scores.csv`, from `probe2/extract2.py` with the 30 s cut.
- Qwen3-Omni row 146, "zero shot, 30 s window", `overnight2/q3o_new/q3o_edaic_zeroshot.json`.

I did not use the 300 s window, the whole-recording window, the mid-recording window or the 390-row conflict manifest. The 390-row Qwen2-Audio and Qwen3-Omni conflict states in `edaic_conflict_new/` have a different clip list, so they do NOT count as E-DAIC states.

## Datasets (n from the fold files; identical in every zero-shot file)

| dataset | condition | n clips | n speakers | n positive |
|---|---|---|---|---|
| PC-GITA | PD | 1100 | 100 | 550 |
| NeuroVoz | PD | 1270 | 107 | 621 |
| MDVR-KCL | PD | 37 | 37 | 16 |
| E-DAIC (30 s) | MDD | 275 | 275 | 66 |
| Pitt | AD | 468 | 228 speaker labels | 290 |
| ADReSSo | AD | 237 | 237 | 122 |
| ADReSS-2020 | AD | 156 | 156 | 78 |

Pitt: participant 172 has two speaker labels, `Control172` and `Dementia172`, in every run (folds README). So 228 labels cover 227 people. I kept the labels exactly as the runs used them.

Class labels agree clip by clip across all three models' zero-shot files on every dataset.

## Fold files (release/folds)

Outer folds come from `<ds>_groupkfold5_pod_seed{0..4}[_UNVERIFIED].csv`. These are the Table 1 five-repeat splits: speakers renumbered with default_rng(seed), then GroupKFold(5) with the stable tie order.

- PC-GITA seed 0 to 4: CONFIRMED clip by clip against the Qwen3-Omni POD3 run.
- Pitt, Qwen3-Omni only: `pitt_groupkfold5_pod_Control15ids_seed{0..4}.csv`, CONFIRMED against the POD3 run. That run's states use ids without zero padding (`Control15`), and those ids sort differently. I checked that the padded ids in the Qwen3-Omni zero-shot file split the clips into exactly the same speaker groups (228 to 228, one to one).
- All other datasets, and Pitt for Qwen2-Audio and Kimi-Audio: `_UNVERIFIED`. They reproduce the saved per-repeat AUCs within 0.008, but no per-clip out-of-fold file ever existed to check them clip by clip.

Every fold file has exactly the zero-shot clip set, and exactly the states clip set wherever states exist.

## Searches run

- Every npz, npy, pt, h5 and pkl file over 1 MB under <local data dir>/release.
- Every `*states*`, `*.npz` over 500 KB, `.tgz` and `.tar` name under "<local data dir>" and "<local data dir>/DementiaBank" (finished, list kept in a local scratch folder).
- The part16 POD folders, the part20 POD folders, podD2_final and the part25 A and C pull folders.

The only Qwen3-Omni state files anywhere are the six listed below. The only Kimi-Audio state files are LM and answer-position states (`llm` 28 x 3584, `ans`), with no `enc` key. The E-DAIC 30 s state files in `leftovers_23sep/t5/inputs` are Qwen2.5-Omni (proj 3584, llm 29 x 3584), not one of these three models.

## Probe and extraction scripts to reuse

- Nested probe: `release_from_mac/scores/part25/A/podfiles/p25a_nested5.py`, the PART 25 A Qwen2.5-Omni encoder probe. It reads the saved outer folds, runs inner GroupKFold(4) with the pod tie rule, picks the layer by mean inner AUC, then fits StandardScaler plus LogisticRegression(max_iter=2000, class_weight='balanced'). It uses one BLAS thread and saves the out-of-fold scores for each repeat. It reads only the `enc`, `label`, `spk` and `name` keys, so it runs unchanged on the Qwen2-Audio and Qwen3-Omni state files. PART 25 A ran it on sklearn 1.9.1, numpy 2.1.2 and scipy 1.18.1 (sidecar `pod_versions`).
- Qwen3-Omni extraction: `release/podD2_final/extract_probe_layers.py`, the script behind q3o_new and the lost podD2 E-DAIC states. It takes WINDOW 30 s, hooks 32 encoder layers and runs in bf16. Command form, from the POD3 sidecar: `WINDOW_S=30 python3 extract_probe_layers.py MANIFEST OUT mdd Qwen/Qwen3-Omni-30B-A3B-Instruct`. The PART 25 A copy has a different md5 (it carries a cast patch). Use the podD2 copy for Qwen3-Omni.
- Kimi-Audio: `release/podD2_final/kimi_probe.py` hooks language model layers only. It would need new hooks on the Whisper encoder. No earlier script does this.
- Watchdog: `release_from_mac/scores/part24/watchdog24.sh` exists.

## The 21 cells

`states ok` means all of these hold: every clip in the dataset list, the same clip set as the fold files and the zero-shot file, speaker ids equal to the fold file's, labels equal, all encoder layers present, and every value finite.

| model | dataset | states ok | states file | enc layers | zero-shot per-clip file (AUC) | fold files | extract plan |
|---|---|---|---|---|---|---|---|
| Qwen2-Audio | PC-GITA | yes | paper1_local_runs/probe2/pcgita_states.npz | 33 x 1280 | probe2/pcgita_zeroshot_scores.csv (0.5369) | pcgita_pod_seed0..4 (CONFIRMED) | none |
| Qwen2-Audio | NeuroVoz | yes | probe2/neurovoz_states.npz | 33 | probe2/neurovoz_zeroshot_scores.csv (0.5519) | neurovoz_pod_seed0..4_UNVERIFIED | none |
| Qwen2-Audio | MDVR-KCL | yes | probe2/kcl_states.npz | 33 | probe2/kcl_zeroshot_scores.csv (0.5923) | kcl_pod_seed0..4_UNVERIFIED | none |
| Qwen2-Audio | E-DAIC | yes (30 s) | probe2/edaic_states.npz | 33 | probe2/edaic_zeroshot_scores.csv (0.6464) | edaic_pod_seed0..4_UNVERIFIED | none |
| Qwen2-Audio | Pitt | yes | probe2/pitt_states.npz | 33 | probe2/pitt_zeroshot_scores.csv (0.6173) | pitt_pod_seed0..4_UNVERIFIED | none |
| Qwen2-Audio | ADReSSo | yes | probe2/adresso_states.npz | 33 | probe2/adresso_zeroshot_scores.csv (0.6649) | adresso_pod_seed0..4_UNVERIFIED | none |
| Qwen2-Audio | ADReSS-2020 | yes | probe2/adress2020_states.npz | 33 | probe2/adress2020_zeroshot_scores.csv (0.6824) | adress2020_pod_seed0..4_UNVERIFIED | none |
| Qwen3-Omni | PC-GITA | yes | paper1_local_runs/part16_POD3_states/q3o_pcgita_states.npz (POD3, the master Table 1 source) | 32 x 1280 | overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv (0.5967) | pcgita_pod_seed0..4 (CONFIRMED vs POD3) | none |
| Qwen3-Omni | NeuroVoz | yes | release/overnight2/q3o/q3o_neurovoz_states.npz | 32 | q3o_new/q3o_neurovoz_zeroshot_scores.csv (0.6393) | neurovoz_pod_seed0..4_UNVERIFIED | none |
| Qwen3-Omni | MDVR-KCL | yes | overnight2/q3o/q3o_kcl_states.npz | 32 | q3o_new/q3o_kcl_zeroshot_scores.csv (0.7738) | kcl_pod_seed0..4_UNVERIFIED | none |
| Qwen3-Omni | E-DAIC | **NO** | missing (podD2 extracted it, never pulled) | | q3o_new/q3o_edaic_zeroshot_scores.csv (0.6290), 30 s | edaic_pod_seed0..4_UNVERIFIED | re-extract, about 0.6 GPU h, fits |
| Qwen3-Omni | Pitt | yes | release/edaic_rerun/part16/pod_sync/q3o_pitt_states.npz (POD3; the G-Drive copy has an identical enc, max diff 0.0) | 32 | q3o_new/q3o_pitt_zeroshot_scores.csv (0.7619) | pitt_pod_Control15ids_seed0..4 (CONFIRMED vs POD3) | none |
| Qwen3-Omni | ADReSSo | yes | overnight2/q3o/q3o_adresso_states.npz | 32 | q3o_new/q3o_adresso_zeroshot_scores.csv (0.7774) | adresso_pod_seed0..4_UNVERIFIED | none |
| Qwen3-Omni | ADReSS-2020 | yes | overnight2/q3o/q3o_adress2020_states.npz | 32 | q3o_new/q3o_adress2020_zeroshot_scores.csv (0.7097) | adress2020_pod_seed0..4_UNVERIFIED | none |
| Kimi-Audio | PC-GITA | **NO** | none (LM-only states in overnight2/kimi) | | overnight2/part3/kimi_pcgita_zeroshot_scores.csv (0.6147) | pcgita_pod_seed0..4 | new Whisper-encoder hook; decision needed |
| Kimi-Audio | NeuroVoz | **NO** | none (LM only) | | overnight2/kimi_new/kimi_neurovoz_zeroshot_scores.csv (0.6204) | neurovoz_pod_seed0..4_UNVERIFIED | same |
| Kimi-Audio | MDVR-KCL | **NO** | none (LM only) | | kimi_new/kimi_kcl_zeroshot_scores.csv (0.7113) | kcl_pod_seed0..4_UNVERIFIED | same |
| Kimi-Audio | E-DAIC | **NO** | none (LM only) | | overnight2/part3/kimi_edaic_zeroshot_scores.csv (0.6293), 30 s | edaic_pod_seed0..4_UNVERIFIED | same |
| Kimi-Audio | Pitt | **NO** | none (LM only) | | overnight2/part3/kimi_pitt_zeroshot_scores.csv (0.7180, the canonical file) | pitt_pod_seed0..4_UNVERIFIED | same |
| Kimi-Audio | ADReSSo | **NO** | none (LM only) | | kimi_new/kimi_adresso_zeroshot_scores.csv (0.7531) | adresso_pod_seed0..4_UNVERIFIED | same |
| Kimi-Audio | ADReSS-2020 | **NO** | none (LM only) | | kimi_new/kimi_adress2020_zeroshot_scores.csv (0.7053) | adress2020_pod_seed0..4_UNVERIFIED | same |

Path roots:
- Qwen2-Audio state files: `<local data dir>/paper1_local_runs/probe2/`.
- `overnight2`, `edaic_rerun` and `folds`: under `<local data dir>/release/`.

## Notes for each cell that matter for the run

- Qwen2-Audio PC-GITA: the zero-shot file's `speaker` column is the leaky clip stem, with 1100 ids. The bootstrap must group speakers by the fold file or the states `spk`, which have 100 speakers. The states `spk` and the fold `speaker_id` agree on all 1100 clips.
- Qwen2-Audio enc has 33 stages (extract2.py saves the encoder input stage plus 32 layers). Qwen3-Omni enc has 32, the extractor's `encoder_layers` value in every q3o sidecar.
- Qwen3-Omni PC-GITA has two full state sets. They have the same clips, but their encoder values differ by up to 28.0 in absolute terms.
  - The POD3 re-extraction is the one master_lookup row 168 uses, and the saved folds are confirmed against it. I chose POD3.
  - The other set, `overnight2/q3o/q3o_pcgita_states.npz`, is the older q3o_new extraction.
  - POD3 p_yes differs from the q3o_new zero-shot answer by up to 0.21 (separate runs, same prompt, same 30 s window). As with Pitt below, this does not touch the encoder states. The answer side stays on the q3o_new file, as master_lookup does.
- Qwen3-Omni Pitt: in the states file, p_yes differs from the q3o_new zero-shot answer by up to 0.38.
  - Both runs used the same dementia prompt and the same 30 s window, but they are separate runs (POD3 ran on transformers 5.17.0).
  - The encoder only sees the audio, so the probe side does not depend on this. The answer side uses the q3o_new file, as master_lookup does.
- Every other Qwen2-Audio and Qwen3-Omni state file's p_yes equals its zero-shot file to 1.1e-16. So the states and the answer come from the same forward pass on the same window.

## Re-extraction estimates (count from 00:45 UTC, plus 30 min of setup, so work starts 01:15 UTC)

Per-clip times come from the pod logs of the earlier extractions (`release/podD2_final/logs`). The time is counted from the first logged clip to the last, after the model has loaded.

| job | clips | s per clip (source) | model load | setup | GPU hours | finish (est.) | fits before 05:00 |
|---|---|---|---|---|---|---|---|
| Qwen3-Omni E-DAIC 30 s | 275 | 0.38 (q3o_edaic_extract.log: 104 s for clips 2 to 275) | 143 s | 30 min: env, 30B-A3B download, upload | about 0.6 | about 01:20 UTC | yes |
| Kimi-Audio encoder, 7 datasets | 3543 | 0.10 for 30 s clips (adresso 23 s/236, adress2020 15 s/155, kcl 4 s/36); 0.056 NeuroVoz (71 s/1269); PC-GITA, Pitt and E-DAIC logs not on disk, taken as 0.10 | about 40 s per dataset | 30 min (KIMI_SETUP 10:03 to 10:09, flash-attn fix to 10:17, plus download) plus about 30 min to write and smoke-test the new hook | about 1.2 | about 02:30 UTC | yes on time; needs the new script and an author decision |

Extraction time for each Kimi dataset, not counting setup:

| dataset | GPU hours |
|---|---|
| PC-GITA | 0.04 |
| NeuroVoz | 0.03 |
| MDVR-KCL | 0.01 |
| E-DAIC | 0.02 |
| Pitt | 0.02 |
| ADReSSo | 0.02 |
| ADReSS-2020 | 0.02 |

Before uploading the E-DAIC audio, cut it to the first 30 s on the Mac. `drive_data/dcaps_proc` is 8.1 GB in full; 275 cut files come to about 265 MB. The extractor cuts `x[:16000*30]` itself, so pre-cutting does not change the window. Build the manifest in the q3o_new zero-shot row order. After re-extraction, check that each clip's re-extracted p_yes matches the q3o_new zero-shot file. That proves the same audio and the same window.

Cost at earlier pod rates: H200 $4.59/h, H100 NVL $3.19/h (part25/A launch responses). Qwen3-Omni E-DAIC is about $3. The Kimi encoder job is about $6. Both are well under the $90 cap.

## Pods and balance at 00:41 UTC

- RunPod balance: $126.72. Current spend: $4.33/h.
- Two PART 25 pods are still running. Other runs own them, and I did not touch them:
  - `14r2usrba60esc` p25A-nvl-adress2020, $3.19/h, created 00:40:59 UTC.
  - `w08zs5dqipvkry` p25C-7, $1.12/h, created 00:10:55 UTC.
- PART 26 spend should be counted from this balance.
