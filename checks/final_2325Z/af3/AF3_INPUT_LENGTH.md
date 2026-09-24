# AF3 Pitt input length (check af3, item 4)

## Answer

**AF3 heard the first 30 s of each clip, not the full 30 to 56 s segment.** The part10 script cut every Pitt window to its first 480,000 samples (30.0 s) before the processor saw it. For each clip the processor built one 30 s window: 3,000 mel frames and 750 audio tokens.

Table 2's AF3 Pitt cells (conflict 0.61 from 0.6075, agreement 0.59 from 0.5855) are already the first 30 s condition. **Table 2 keeps 0.61 / 0.59.**

**The "30 s control" premise is wrong.** POD4c has no `af3_*30*` files. The values quoted as the control (all 0.5920, conflict 0.5417, agreement 0.6140) come from the `af3_noinv_*` files. That run is AF3 on the **interviewer-free cut**: participant speech only, with clips from 0.68 to 29.24 s. None of those clips was long enough for the 30 s window to cut. Printing 0.54 / 0.61 in Table 2 would replace the paper's condition with a different one.

## 1. Score file, script and command

| What | Value | Where it comes from |
|---|---|---|
| Score file | `<local data dir>/release/overnight2/part10/af3_pitt.csv`, sha256 `c421d988...77dd23b`, 468 rows | `shasum -a 256` |
| Sidecar | `.../overnight2/part10/af3_pitt.json`: `"script": "/workspace/af3_score.py"`, `"window_seconds": 30`, `"pre_cut_to_single_window": true`, `"median_source_duration_s": 33.28`, `"max_source_duration_s": 55.56`, `"date": "2026-09-20"` | `cat` |
| Same file under final/ | `overnight2/final/af3_pitt.csv` is byte identical | `cmp part10/af3_pitt.csv final/af3_pitt.csv` |
| Script | The pod copy is gone. The copy that was uploaded to the pod is kept on the authors' machine (4,031 bytes, last modified 19 Sep 08:33 PDT, sha256 `3a93070a...e6fd93b`) and is copied here as `af3/provenance/af3_score.py`. | The local upload log, which sends it with `scp ... $DIR/finish_all.sh $DIR/extract_probe_layers.py $DIR/af3_score.py ... root@$IP:/workspace/` |
| Driver | `finish_all.sh` from the same local folder (sha256 `d339dbbb...85e93f6c`), copied here as `af3/provenance/finish_all.sh` | |
| Command | `python3 af3_score.py mf_pitt.csv out/af3_pitt.csv ad > logs/af3_pitt.log 2>&1` (finish_all.sh line 52). There is **no `--nocut`**. | |
| transformers | `pip install ... transformers==5.5.4 ...` (finish_all.sh line 3) | |
| Model id and revision | `nvidia/audio-flamingo-3-hf`, fetched unpinned with `snapshot_download` (finish_all.sh line 11), 27 files (`logs/hf_af3.log`). The HF main branch has been `7d4bae64ee29878af6504ae6f6bb3e40492838ad` since 2026-04-13, so that is the revision the run got. | `curl https://huggingface.co/api/models/nvidia/audio-flamingo-3-hf/commits/main` |

Four things tie this script to the part10 file:
- The sidecar keys `window_seconds`, `pre_cut_to_single_window` and `median_source_duration_s` are written by lines 49 to 56 of this script.
- The log line `AF3 loaded in 34s | sound token 151669 | yes [...]` has the format of line 25.
- The log line `1/468 conflict_Con_015-0_67805.wav p_yes 0.706 mass 0.968 35s` has the format of line 43.
- The line `RESULT af3 mf_pitt.csv n=468 AUC=0.5894 ...` has the format of line 57.

The script file was last changed at 08:33 PDT. The driver was written at 18:09 PDT, and the run ended at 01:50 UTC on 20 Sep (18:50 PDT on 19 Sep, `logs/DRIVER.log`). The script did not change between upload and run.

These are the lines that load the audio and call the processor (af3_score.py; line 9 is `NOCUT = "--nocut" in sys.argv` and line 15 is `proc = AutoProcessor.from_pretrained(MID)`):

```
28    x, _ = librosa.load(r["path"], sr=16000)
29    durs.append(len(x) / 16000)
30    if not NOCUT: x = x[:16000 * 30]
31    cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
32    conv = [{"role": "user", "content": [{"type": "audio", "path": cut}, {"type": "text", "text": P}]}]
33    inp = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
34                                   return_dict=True, return_tensors="pt")
```

The processor call passes no `max_length`, truncation or audio kwargs. In transformers 5.5.4 the processor defaults are:
- `sampling_rate` 16000, `padding` "max_length" and `return_attention_mask` True;
- audio split into 30 s windows (`chunk_length` 30);
- a cap at `max_audio_len` 600 s.

The sidecar's median of 33.28 s and maximum of 55.56 s come from line 29, before the cut. They describe the source windows, not what the model heard.

## 2. Audio the run used

- `mf_pitt.csv` was built by finish_all.sh lines 25 to 31 from `data/clips_for_af3/pitt468_manifest.csv`. It points at `/workspace/data/clips_for_af3/pitt468/*.wav`.
- Local copies are in `<local data dir>/paper1_local_runs/clips_local/clips_for_af3/pitt468`. All 468 match `POD4c/work/orig_hashes.csv` on file sha256 and sample count, so there are 0 mismatches. All are 16 kHz mono.
- Durations run from 30.010 to 55.560 s, with a median of 33.285 s. **All 468 are longer than 30 s.** This matches the sidecar's 33.28 and 55.56.
  Command: `cd "<af3>" && /usr/local/bin/python3 durations.py "<af3>/pitt468_durations.csv"`. Output is in `durations.log` and `pitt468_durations.csv`.
- Arms come from the `set` column of `<local data dir>/DementiaBank/pitt_conflict_manifest.csv`, keyed by the basename of `segment_path`.
- The three clips checked: `agreement_Con_118-1_6475.wav` at 30.01 s (nearest to 30 s), `conflict_Con_227-0_460.wav` at 33.27 s (nearest to the median) and `conflict_Dem_639-0_9189.wav` at 55.56 s (the longest).

## 3. Processor check (CPU, processor only, no weights)

The check reproduces lines 28 to 34 exactly: the same `librosa.load`, the same slice, the same `sf.write` of a temp wav and the same `apply_chat_template` call. It used transformers 5.5.4, torch 2.8.0, librosa 1.0.0 and soundfile 0.14.0, with the processor at revision `7d4bae64...`. A patch on `AudioFlamingo3Processor.__call__` records the waveform length the processor actually receives. For contrast, each clip was also run with no cut.

| Clip | Segment (s) | Mode | Samples to processor | input_features | Mask sum (frames) | Windows | Audio tokens | Seconds reaching model |
|---|---|---|---|---|---|---|---|---|
| agreement_Con_118-1_6475 | 30.01 | **as run (x[:16000*30])** | **480,000** | **1x128x3000** | **3000** | **1** | **750** | **30.00** |
| | | no cut (contrast) | 480,160 | 2x128x3000 | 3000+1 | 2 | 750+0 | 30.01 |
| conflict_Con_227-0_460 | 33.27 | **as run** | **480,000** | **1x128x3000** | **3000** | **1** | **750** | **30.00** |
| | | no cut (contrast) | 532,320 | 2x128x3000 | 3000+327 | 2 | 750+82 = 832 | 33.27 |
| conflict_Dem_639-0_9189 | 55.56 | **as run** | **480,000** | **1x128x3000** | **3000** | **1** | **750** | **30.00** |
| | | no cut (contrast) | 888,960 | 2x128x3000 | 3000+2556 | 2 | 750+639 = 1389 | 55.56 |

"Seconds reaching model" is the mask sum times the 160-sample hop over 16 kHz. This version of the model only scatters the unpadded frames into the `<sound>` positions (`get_audio_features`: `valid_mask` from `input_features_mask`).

- Command: `hdiutil attach -nobrowse "<af3>/venv_image.sparseimage"`, then `cd "<af3>" && HF_HOME="<af3>/hf_home" "<af3>/venv/bin/python" af3_input_length.py "<af3>/af3_pitt_three_clips_input_length.csv" agreement_Con_118-1_6475.wav conflict_Con_227-0_460.wav conflict_Dem_639-0_9189.wav`
- Output is in `af3_pitt_three_clips_input_length.csv` and `af3_input_length.log`. A first run from a second venv with the same pins gave a byte-identical csv (`cmp`).

## 4. Cross-checks

- **Sidecar, written at run time:** `pre_cut_to_single_window` is `not NOCUT`, and it says true. The command line had no `--nocut`.
- **part10 logs:** No per-clip length is logged. In 5.5.4 the processor only warns about length for audio over 600 s, so a missing warning tells us nothing here. `logs/af3_pitt.log` has only HF hub notices.
- **POD4c val10:** 10 of the same original windows (30.27 to 42.20 s) were re-scored with `WINDOW_S=30` (af3_zs.py: `if WIN > 0: x = x[:int(16000 * WIN)]`; `scored_s` = 30.0 for all ten; transformers 5.17.0). Their p_yes matches part10 within **2.85e-06**. The whole-array run from 16 Sep differs from those same ten clips by up to 0.1856 (median 0.0739).
- **Whole-array run from 16 Sep** (`<local data dir>/paper1_local_runs/runpod_results/pitt468_af3.csv`). The script passed `{"type":"audio","audio":x}` with no slice (local run log). One clip gets no extra audio tokens from the whole array: agreement_Con_118-1_6475 at 30.01 s, whose second window has 1 frame and 0 tokens. That clip matches part10 to 7e-06. Clips with any extra audio differ, with a median |dp_yes| of 0.060 over all 468. So part10 behaves like the 30 s cut, not like the whole segment.
  Command: `cd "<af3>" && /usr/local/bin/python3 crosscheck_pyes.py`. Output is in `crosscheck_pyes.log`.

## 5. AUCs, recomputed from per-clip files

Method: 2000 speaker-bootstrap draws, a fresh `numpy.random.default_rng(0)` per cell, speakers resampled with replacement, and the 2.5 and 97.5 percentiles. AUC uses scipy rankdata with ties averaged. For paired cells, each draw uses one speaker sample and recomputes both AUCs on it. All 2000 draws were usable in every cell.

- Command: `cd "<af3>" && /usr/local/bin/python3 boot_af3_pitt.py "<af3>/af3_pitt_bootstrap.csv"`. Output is in `boot_af3_pitt.log`.
- The per-clip inputs are the part10 csv and `POD4c/work/af3_noinv468.csv`. The script asserts that the POD4c per-arm files hold the same p_yes values.

| Cell | All (n 468, 228 spk) | Conflict (n 146, 100 spk) | Agreement (n 322, 175 spk) |
|---|---|---|---|
| part10, original windows, first 30 s (**Table 2**) | 0.5894 [0.5302, 0.6476] | **0.6075** [0.5063, 0.7129] | **0.5855** [0.5144, 0.6554] |
| POD4c `af3_noinv`, interviewer-free cut (the "0.54 / 0.61" values) | 0.5920 [0.5293, 0.6577] | 0.5417 [0.4202, 0.6621] | 0.6140 [0.5427, 0.6798] |
| Paired, interviewer-free minus part10 | +0.0026 [-0.0548, 0.0600] | -0.0658 [-0.1653, 0.0359] | +0.0285 [-0.0404, 0.0959] |

These match POD4c's own files to 4 dp, with the sign of the difference flipped. POD4c reports part10 minus interviewer-free.

**There is no AF3 "30 s minus full segment" pair built the way the task assumed**, because part10 already is the 30 s run. The only full-segment AF3 Pitt run on disk is the 16 Sep whole-array file. It is given here as context only. It used the same clips and prompt, and matches part10 where the inputs coincide (section 4). Its YES/NO sets have four token variants each against five in part10, and its transformers version is not recorded next to the file.

- Command: `cd "<af3>" && /usr/local/bin/python3 boot_af3_fullseg.py "<af3>/af3_pitt_fullseg_bootstrap.csv"`. Output is in `boot_af3_fullseg.log`.

| Cell | All | Conflict | Agreement |
|---|---|---|---|
| Full segment, 16 Sep whole array | 0.5751 [0.5136, 0.6347] | 0.6202 [0.5135, 0.7209] | 0.5619 [0.4931, 0.6278] |
| Paired, full segment minus part10 first 30 s | -0.0142 [-0.0607, 0.0318] | +0.0128 [-0.0797, 0.0947] | -0.0236 [-0.0792, 0.0291] |

## What Table 2 should print

AF3 row, Alzheimer's columns: **Conflict 0.61, Agree. 0.59**, unchanged. They match the setup text ("Models hear the first 30 s") and the caption ("30 s segments"). The 0.54 / 0.61 pair is the interviewer-free cut and does not belong in this row. Hearing the full segment would move the cells by +0.01 (conflict) and -0.02 (agreement), and both intervals include zero.

## Environment note

The venv was `<af3>/venv`, a symlink to `<local data dir>/af3venv/venv`. That volume is an APFS sparse image on the authors' drive (sha256 `da0de8e6...ec85fc68f`, not in this repository), attached with `hdiutil attach -nobrowse`.

It was built this way because installing directly onto the exFAT external drive wrote about 1,400 of 19,000 files in 30 minutes, so that install was stopped. The build commands were `uv venv --python 3.12 venv` and `UV_LINK_MODE=copy uv pip install "transformers==5.5.4" "torch==2.8.0" librosa soundfile numpy`. The full freeze is in `venv_freeze.txt`.

`<af3>` = `checks/final_2325Z/af3`. Nothing in `final_overleaf_2325Z` or `omni_final` was touched, and no pod was used for this check.
