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
