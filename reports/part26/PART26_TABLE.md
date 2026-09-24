# PART 26: probe grid on the other three models

Written 04:10 UTC on 24 Sep. All 21 cells landed and all 21 are verified. No PART 26 pod exists now.

Each cell: probe = mean of the five per-repeat nested encoder probe AUCs (layer chosen inside the training folds). Answer = zero-shot answer AUC of the same model on the same clips. Gap = probe minus answer, with the 95% speaker bootstrap interval (2000 draws, fresh default_rng(0), unique speakers resampled with all their clips, 2.5 and 97.5 percentiles). All 2000 draws were usable in every cell.

## The table

| model | PC-GITA | NeuroVoz | MDVR-KCL | E-DAIC | Pitt | ADReSSo | ADReSS-2020 |
|---|---|---|---|---|---|---|---|
| Qwen2-Audio | probe 0.9097<br>answer 0.5369<br>gap +0.3727 [+0.3102, +0.4271]<br>above 0<br>VERIFIED | probe 0.9324<br>answer 0.5519<br>gap +0.3805 [+0.3278, +0.4325]<br>above 0<br>VERIFIED | probe 0.7714<br>answer 0.5923<br>gap +0.1792 [-0.0503, +0.4305]<br>covers 0<br>VERIFIED | probe 0.5673<br>answer 0.6464<br>gap -0.0791 [-0.1664, +0.0120]<br>covers 0<br>VERIFIED | probe 0.7847<br>answer 0.6173<br>gap +0.1674 [+0.0945, +0.2433]<br>above 0<br>VERIFIED | probe 0.8537<br>answer 0.6649<br>gap +0.1888 [+0.1154, +0.2682]<br>above 0<br>VERIFIED | probe 0.8315<br>answer 0.6824<br>gap +0.1491 [+0.0640, +0.2368]<br>above 0<br>VERIFIED |
| Qwen3-Omni | probe 0.8969<br>answer 0.5967<br>gap +0.3002 [+0.2298, +0.3695]<br>above 0<br>VERIFIED | probe 0.9372<br>answer 0.6393<br>gap +0.2980 [+0.2360, +0.3578]<br>above 0<br>VERIFIED | probe 0.7661<br>answer 0.7738<br>gap -0.0077 [-0.1618, +0.1385]<br>covers 0<br>VERIFIED | probe 0.5828<br>answer 0.6290<br>gap -0.0462 [-0.1412, +0.0527]<br>covers 0<br>VERIFIED | probe 0.8122<br>answer 0.7619<br>gap +0.0503 [-0.0087, +0.1095]<br>covers 0<br>VERIFIED | probe 0.8750<br>answer 0.7774<br>gap +0.0976 [+0.0362, +0.1616]<br>above 0<br>VERIFIED | probe 0.8445<br>answer 0.7097<br>gap +0.1348 [+0.0417, +0.2335]<br>above 0<br>VERIFIED |
| Kimi-Audio | probe 0.8768<br>answer 0.6147<br>gap +0.2621 [+0.1810, +0.3364]<br>above 0<br>VERIFIED (new Kimi encoder states) | probe 0.9319<br>answer 0.6204<br>gap +0.3115 [+0.2515, +0.3674]<br>above 0<br>VERIFIED (new Kimi encoder states) | probe 0.7631<br>answer 0.7113<br>gap +0.0518 [-0.1637, +0.2901]<br>covers 0<br>VERIFIED (new Kimi encoder states) | probe 0.5823<br>answer 0.6293<br>gap -0.0469 [-0.1393, +0.0533]<br>covers 0<br>VERIFIED (new Kimi encoder states) | probe 0.8112<br>answer 0.7180<br>gap +0.0932 [+0.0331, +0.1547]<br>above 0<br>VERIFIED (new Kimi encoder states) | probe 0.8682<br>answer 0.7531<br>gap +0.1151 [+0.0481, +0.1831]<br>above 0<br>VERIFIED (new Kimi encoder states) | probe 0.8881<br>answer 0.7053<br>gap +0.1829 [+0.0879, +0.2738]<br>above 0<br>VERIFIED (new Kimi encoder states) |

Qwen3-Omni is Qwen3-Omni-30B-A3B. Per-repeat AUCs, n, positives and speakers for every cell are in scores/part26/PART26_TABLE.tsv.

Kimi row: no earlier run extracted Kimi encoder states, and the paper Method says the Kimi encoder side is skipped. PART 26 added new forward hooks on the 32 Whisper-large-v3 encoder layers inside kimia_infer (the continuous-feature path into the LM), mean over the frames Kimi keeps, first 30 s. The discrete GLM-4-Voice token path is not probed. The inventory flagged this as an author decision. The numbers are verified as computed. Using this row in the paper needs the Method text changed.

## Parkinson's and Alzheimer's count

18 cells (every dataset except E-DAIC). All 18 are verified.

- Interval above zero: 14 of 18 verified, and 14 of 18 overall.
  - Qwen2-Audio 5 of 6: PC-GITA, NeuroVoz, Pitt, ADReSSo, ADReSS-2020.
  - Qwen3-Omni 4 of 6: PC-GITA, NeuroVoz, ADReSSo, ADReSS-2020.
  - Kimi-Audio 5 of 6: PC-GITA, NeuroVoz, Pitt, ADReSSo, ADReSS-2020.
- Interval covers zero (4): MDVR-KCL for all three models (n = 37), and Qwen3-Omni Pitt (lower end -0.0087).
- Without the Kimi row: 9 of 12.

## E-DAIC, per model

All three intervals cover zero. The answer beats the probe at the point estimate in all three.

- Qwen2-Audio: probe 0.5673, answer 0.6464, gap -0.0791 [-0.1664, +0.0120]. 95.7% of draws at or below 0.
- Qwen3-Omni: probe 0.5828, answer 0.6290, gap -0.0462 [-0.1412, +0.0527]. 82.6% of draws at or below 0.
- Kimi-Audio: probe 0.5823, answer 0.6293, gap -0.0469 [-0.1393, +0.0533]. 82.4% of draws at or below 0.

## Missing states and re-extraction

13 of 21 cells had states on disk: all 7 Qwen2-Audio and 6 Qwen3-Omni. 8 were missing. All 8 were re-extracted and finished before 05:00 UTC (the last one at 03:36).

| model, dataset | GPU h estimated | re-extracted | GPU pod hours used | notes |
|---|---|---|---|---|
| Qwen3-Omni E-DAIC | 0.6 | yes, 00:47 to 01:08 | 0.34 (H100 80GB HBM3) | podD2_final/extract_probe_layers.py unchanged. Re-extracted p_yes equals the q3o_new zero-shot file exactly, and the single split equals the saved podD2 OOF exactly. Cause: podD2 extracted the states but never pulled them. |
| Kimi-Audio PC-GITA | 0.04 | yes | 0.60 | new hook script |
| Kimi-Audio NeuroVoz | 0.03 | yes | 0.34 | new hook script |
| Kimi-Audio MDVR-KCL | 0.01 | yes | 0.22 | new hook script |
| Kimi-Audio E-DAIC | 0.02 | yes | 0.35 | new hook script |
| Kimi-Audio Pitt | 0.02 | yes | 0.21 | new hook script |
| Kimi-Audio ADReSSo | 0.02 | yes | 0.44 | new hook script |
| Kimi-Audio ADReSS-2020 | 0.02 | yes, try 2 | 1.70 over 3 pods | Try 1 (0.66 h): the upload never arrived and the watchdog killed the idle pod. g2 (0.84 h): superseded by g3, still installing packages. g3 (0.21 h): did the extraction. |

The inventory estimate for Kimi was 1.2 GPU h in total (about 1.0 h shared setup plus the per-dataset times above). The actual total is 3.86 GPU pod hours, because each dataset ran its own setup and ADReSS-2020 needed three pods. Qwen3-Omni plus Kimi together used 4.20 GPU pod hours.

Kimi extraction scripts: the cells wrote four variants of the same hook (sha256 prefixes fa201c10 for PC-GITA, NeuroVoz and Pitt; cad19140 for ADReSSo; 006d5560 for E-DAIC; 7517f4b1 for MDVR-KCL and ADReSS-2020). All hook the same 32 layers and pool over the same kept frames, token_len*4 with token_len = (L-1)//1280+1. The only difference is that two variants take the mean in bf16 and then cast to fp32, and two cast first. That is a rounding difference only. For Pitt and E-DAIC, a separate fp32 Hugging Face Whisper recompute on the Mac matched the pod states (cosine 0.9999 or higher on every layer of the test clips).

## Pod cost for PART 26

Source: GET https://rest.runpod.io/v1/billing/pods (read only, hour buckets, 00:00 to 06:00 UTC), filtered to the 35 PART 26 pod ids from the create responses in each cell folder.

- Billed so far (GPU pods, as of 04:06 UTC): $12.49. Billing still lags. g2 (wikwa0qc0uw1b9) shows 1455 of its 3016 s, and g3 (mmqahrpwpu0m00) has no rows yet.
- The billing API returns no rows for CPU pods (queried by pod id too). CPU cost is rate times create-to-delete time.
- Time-based totals:
  - 10 GPU pods: $14.66. The closed pods match their billed amounts to about 1%.
  - 24 CPU pods with known delete times: $3.32.
  - p26-kimi-neurovoz-cpu (xm65txjjvqzq5q): up to $1.24. It was created at 01:22:44 and found gone at 02:40, and no delete time was recorded.
- PART 26 total: about $18.0, at most $19.2. That is under the $90 cap.
- Balance: $101.84 at 04:06 UTC, spending $0/h. It was $126.72 at 00:41. The $24.88 drop also covers other runs' PART 25 pods: p25A-nvl-adress2020 (14r2usrba60esc), p25C-7 (w08zs5dqipvkry) and the PART 25 A retry pods 348aftq6eytbv7, fty6q62utrqzmj, yk0kdi7s6k90j5 and zut4h14otaka38 (listed in part25/A/retry_tf554/launch/created_ids.txt).

No PART 26 pod still exists. At 03:55 UTC a GET on each of the 35 PART 26 pod ids returned HTTP 404. At 04:06 UTC the pod list endpoint returned 0 pods.

## How each cell was verified

1. Probe run. All 21 pod runs record Linux, sklearn 1.9.1, numpy 2.1.2 and scipy 1.18.1. The script sets OMP, OPENBLAS and MKL threads to 1. 20 cells ran part25/A p25a_nested5.py unchanged (md5 615ced03). Qwen3-Omni Pitt ran a copy (p26_nested5.py, md5 5311dd1a) with three edits that read the Control15-id fold files and skip the single split. The estimator is unchanged.
2. Pod-side check (p25a_podverify.py on the same stack). It refits at the saved layers and re-derives the inner layer choice for every outer fold.
   - PASS in 17 cells.
   - Qwen2-Audio MDVR-KCL and Kimi MDVR-KCL each flag one outer fold. There, two layers tie exactly on inner AUC. The saved layer is the probe script's own argmax, and the predictions differ by 0.
   - Qwen2-Audio ADReSSo and Kimi ADReSSo have no pod-side file. Their Mac refits picked the same layer in all 25 outer folds.
3. Each cell's own Mac check, plus a separate independent verifier with its own code, confirmed every number in the cell.
   - The only failed items are Mac refits on sklearn 1.7.2, compared against the pinned Linux pod. These refits are information only.
   - The refits chose the same layers. Per-clip scores differ by up to 0.04, and per-repeat AUCs by up to 7e-4. That is solver tolerance across builds.
4. This table's own recompute (scripts/part26/table26/verify_table26.py, run at 04:01 UTC, new code), for all 21 cells:
   - Recomputed from the per-clip csv, the zero-shot file named in part25/C/C_build_cells.csv (sha256 matched) and the release fold files, read directly.
   - Checks: per-repeat AUCs, answer AUC, gap and the full 2000-draw bootstrap.
   - All checks pass in every cell: 22 per cell, or 21 for Kimi E-DAIC, whose npz has no speaker index.
   - My draws equal the saved npz to 5.6e-16, and every number in the table appears in the cell sidecar.

Nothing under either omni_final tree changed after 00:37 UTC.

## Caveats

- Pitt, all three models: 228 speaker labels cover 227 people. Participant 172 is labeled both Control172 and Dementia172, kept as the earlier runs did.
- Qwen2-Audio Pitt: the master Table 1 value is 0.7846, and this run gives 0.7847. The PART 25 C gap was +0.1673 [+0.0944, +0.2432], and this table gives +0.1674 [+0.0945, +0.2433]. Pick one source so the table and the gap figure agree.
- Qwen2-Audio PC-GITA: seed0 is 0.9240 against the master's 0.9242, and the mean agrees at 0.9097. The zero-shot file's speaker column is the clip stem, so the bootstrap uses the 100 fold-file speakers.
- MDVR-KCL, all three models (n = 37): the inner AUC often ties.
  - Qwen3-Omni: in seed0 fold 3, two layers tie and float rounding breaks the tie. Exact arithmetic picks layer 13, which gives probe 0.7619 and gap -0.0119 [-0.1636, +0.1321].
  - Kimi: 14 of 30 outer folds tie. Swapping in any tied layer moves the gap between +0.0429 and +0.0560.
  - Qwen2-Audio: 5 of 25 tie. One swap would drop seed1 from 0.8095 to 0.7530.
  - All of these intervals still cover zero.
  - Qwen3-Omni: the cell's own check loosened one tolerance after it failed. The Mac refit probability tolerance of 1e-6 became "same layers and per-repeat AUC within 1e-9". Both the change and the first failure are written in its verify json.
- Qwen3-Omni Pitt: the lower end is -0.0087. With the zero-padded speaker ids it is [-0.0058, +0.1108], which still covers zero.
- ADReSSo, all three models: n = 237 includes adrso216. The lab list is 236 (empty transcript). This matches Table 1. Dropping it, Qwen2-Audio gives 0.8530, 0.6648 and +0.1882.
- Fold files: the repeat fold files are still named _UNVERIFIED in release/folds, except for PC-GITA and the Qwen3-Omni Pitt Control15-id files. Every cell's fold columns equal them clip by clip.
- Qwen3-Omni PC-GITA uses the part16 POD3 states, the master row source.
- Kimi answer AUCs use the canonical zero-shot files. The p_yes from the re-extraction pass drifts from them (E-DAIC median 0.02, max 0.18). With it, E-DAIC would be -0.0430 [-0.1368, +0.0569], PC-GITA +0.2638 [+0.1821, +0.3395] and ADReSSo +0.1139 [+0.0468, +0.1819]. None of these changes which side of zero the interval falls on.
- part25/C/C_other_models.tsv gained six Kimi encoder rows at 03:06 UTC, and they match this table. Its Kimi ADReSS-2020 encoder row is still blank.

## Files

In this repository:

- reports/part26/PART26_TABLE.md (this file), scores/part26/PART26_TABLE.tsv and scores/part26/PART26_TABLE.sidecar.json.
- scores/part26/<model>_<dataset>/: the per-clip csv, sidecar json and draws npz for each cell, with the pod probe outputs and the verify json. The exact paths are in the tsv. The scripts of each cell are in scripts/part26/<model>_<dataset>/.
- scripts/part26/table26/ and scores/part26/table26/: verify_table26.py and verify_table26_out.json (the table recompute), cost26.py and cost26_out.json (the cost), billing_p26_window.json (raw billing rows), pods_found.json (the 35 PART 26 pods) and pod_get_status.json (the 404 checks).
- The per-clip csv files have no clinical label column. The README section on clinical labels says how to join the labels from a licensed copy of each corpus.
