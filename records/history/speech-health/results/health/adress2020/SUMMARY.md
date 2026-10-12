# ADReSS-2020, the standard alzheimers benchmark split (a balanced subset of pitt)

Official split: 108 train recordings (54 control, 54 dementia), 48 test (24/24). The test
labels are not in the download; they were recovered by matching each test transcript to the
pitt transcripts (test_labels_recovered.csv). The same matcher on the 108 train recordings,
whose labels are known, scored 107 correct, 0 wrong, 1 undecided.

30 second clips, qwen2-audio encoder states, layer picked by cross validation inside train only.

- probe, official split: layer 22, test AUC 0.778, test accuracy 0.708
  (the challenge's own acoustic baseline accuracy was 0.625)
- probe, pooled 156 clips, 5 fold: best layer 27, AUC 0.824
- the model's own answer: 0.653 pooled (0.764 on the 48 test clips, 0.603 on the 108 train clips,
  the split-level numbers swing because the splits are small)

Same gap as pitt: probe 0.82, answer 0.65, on the benchmark everyone reports on.

## Patient-onset window (11 Sep): official split and pooled, one script (scripts/paper1/pooled_and_nested_probes.py)
- probe, official split, layer chosen inside the 108 train recordings: layer 27, test AUC 0.800, accuracy 0.729 (adress2020_probe_patient.csv, run_patient.log)
- probe, pooled 156, 5 fold, StratifiedKFold seed 42: peak layer 32 AUC 0.865; nested (layer chosen inside each training fold) 0.822 (adress2020_pooled_patient_probe.csv). The paper quotes nested.
- the model's own answer: pooled 0.681 [0.603, 0.762]; on the 48 test recordings 0.782 [0.635, 0.902]; on the 108 train recordings 0.624 [0.514, 0.730] (adress2020_answers_patient.csv, bootstrap_cis.csv). The split-level swing is why the pooled number is the one to compare.
- challenge baselines (Luz et al. 2020): accuracy 0.625 acoustic, 0.75 linguistic on manual transcripts.
