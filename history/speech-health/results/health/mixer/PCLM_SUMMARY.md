# PCLM on clinical speech: learned layer weighting vs honest single-layer selection

Qwen2-Audio encoder, 33 layers, mean-pooled per clip. GroupKFold(5) by speaker,
zero speaker overlap asserted every fold. Pooled out-of-fold AUC. 95% CI from
2000 speaker-clustered bootstrap resamples. Shuffle = mean AUC under speaker-level
label permutation. perm p from that permutation null.

## Main table

| dataset | n clips | n spk | method | AUC | 95% CI | shuffle | perm p |
|---|---|---|---|---|---|---|---|
| pcgita:read | 1117 | 117 | single_nested | 0.916 | 0.865-0.961 | 0.481 | 0.020 |
| pcgita:read | 1117 | 117 | single_oracle | 0.930 | 0.878-0.971 | 0.548 | 0.020 |
| pcgita:read | 1117 | 117 | mean_layers | 0.916 | 0.861-0.962 | 0.481 | 0.020 |
| pcgita:read | 1117 | 117 | concat_layers | 0.915 | 0.857-0.962 | 0.482 | 0.020 |
| pcgita:read | 1117 | 117 | mixer_static | 0.909 | 0.853-0.957 | 0.482 | 0.048 |
| pcgita:read | 1117 | 117 | mixer_inputcond | 0.860 | 0.794-0.919 | 0.482 | 0.048 |
| neurovoz:read | 1270 | 107 | single_nested | 0.930 | 0.893-0.962 | 0.485 | 0.020 |
| neurovoz:read | 1270 | 107 | single_oracle | 0.949 | 0.920-0.973 | 0.543 | 0.020 |
| neurovoz:read | 1270 | 107 | mean_layers | 0.932 | 0.891-0.967 | 0.485 | 0.020 |
| neurovoz:read | 1270 | 107 | concat_layers | 0.940 | 0.901-0.971 | 0.486 | 0.020 |
| neurovoz:read | 1270 | 107 | mixer_static | 0.885 | 0.824-0.939 | 0.489 | 0.048 |
| neurovoz:read | 1270 | 107 | mixer_inputcond | 0.854 | 0.790-0.910 | 0.491 | 0.048 |
| dcaps:interview | 275 | 275 | single_nested | 0.642 | 0.565-0.718 | 0.509 | 0.020 |
| dcaps:interview | 275 | 275 | single_oracle | 0.697 | 0.622-0.768 | 0.577 | 0.020 |
| dcaps:interview | 275 | 275 | mean_layers | 0.658 | 0.584-0.734 | 0.508 | 0.020 |
| dcaps:interview | 275 | 275 | concat_layers | 0.668 | 0.598-0.743 | 0.509 | 0.020 |
| dcaps:interview | 275 | 275 | mixer_static | 0.608 | 0.529-0.684 | 0.505 | 0.048 |
| dcaps:interview | 275 | 275 | mixer_inputcond | 0.593 | 0.513-0.674 | 0.501 | 0.048 |

## The question: does the mixer beat nested single-layer selection?

| dataset | comparison | delta AUC | 95% CI | P(delta>0) |
|---|---|---|---|---|
| pcgita:read | mixer static minus single nested | -0.0067 | -0.0385..+0.0262 | 0.326 |
| pcgita:read | mixer inputcond minus single nested | -0.0553 | -0.1121..-0.0056 | 0.013 |
| pcgita:read | mixer static minus mean layers | -0.0071 | -0.0286..+0.0127 | 0.242 |
| pcgita:read | mixer inputcond minus mean layers | -0.0557 | -0.1031..-0.0146 | 0.004 |
| pcgita:read | mixer static minus concat layers | -0.0062 | -0.0275..+0.0133 | 0.269 |
| pcgita:read | mixer inputcond minus concat layers | -0.0548 | -0.1036..-0.0130 | 0.004 |
| pcgita:read | single oracle minus single nested | +0.0133 | -0.0053..+0.0348 | 0.913 |
| neurovoz:read | mixer static minus single nested | -0.0441 | -0.0792..-0.0138 | 0.002 |
| neurovoz:read | mixer inputcond minus single nested | -0.0754 | -0.1193..-0.0382 | 0.000 |
| neurovoz:read | mixer static minus mean layers | -0.0460 | -0.0820..-0.0166 | 0.001 |
| neurovoz:read | mixer inputcond minus mean layers | -0.0772 | -0.1224..-0.0371 | 0.000 |
| neurovoz:read | mixer static minus concat layers | -0.0532 | -0.0929..-0.0199 | 0.001 |
| neurovoz:read | mixer inputcond minus concat layers | -0.0844 | -0.1336..-0.0424 | 0.000 |
| neurovoz:read | single oracle minus single nested | +0.0188 | +0.0044..+0.0354 | 0.996 |
| dcaps:interview | mixer static minus single nested | -0.0332 | -0.1094..+0.0438 | 0.202 |
| dcaps:interview | mixer inputcond minus single nested | -0.0486 | -0.1294..+0.0329 | 0.113 |
| dcaps:interview | mixer static minus mean layers | -0.0496 | -0.1201..+0.0183 | 0.083 |
| dcaps:interview | mixer inputcond minus mean layers | -0.0650 | -0.1490..+0.0209 | 0.071 |
| dcaps:interview | mixer static minus concat layers | -0.0593 | -0.1323..+0.0125 | 0.051 |
| dcaps:interview | mixer inputcond minus concat layers | -0.0747 | -0.1564..+0.0069 | 0.038 |
| dcaps:interview | single oracle minus single nested | +0.0550 | -0.0155..+0.1264 | 0.934 |

## Where the attention mass sits

| dataset | mixer | peak layer | peak weight | peak/uniform | entropy/max | mass L0-3 | mass L4-15 | mass L16-32 |
|---|---|---|---|---|---|---|---|---|
| pcgita:read | static | 15 | 0.0369 | 1.22x | 0.997 | 0.101 | 0.407 | 0.492 |
| pcgita:read | inputcond | 32 | 0.1572 | 5.19x | 0.927 | 0.109 | 0.359 | 0.532 |
| neurovoz:read | static | 16 | 0.0354 | 1.17x | 0.998 | 0.104 | 0.382 | 0.514 |
| neurovoz:read | inputcond | 32 | 0.0642 | 2.12x | 0.994 | 0.099 | 0.384 | 0.516 |
| dcaps:interview | static | 14 | 0.0343 | 1.13x | 0.999 | 0.107 | 0.383 | 0.510 |
| dcaps:interview | inputcond | 25 | 0.0366 | 1.21x | 0.999 | 0.118 | 0.366 | 0.516 |

Uniform weight would be 0.0303 per layer; entropy/max = 1.000 means perfectly flat.

This is the central diagnostic. The learned static attention is close to uniform,
so the mixer is doing little more than averaging the layers, which is why
mixer_static and mean_layers land on nearly the same AUC.

## Do the peaks agree across datasets?

### static

| | pcgita:read | neurovoz:read | dcaps:interview |
|---|---|---|---|
| pcgita:read | 1.000 | 0.809 | 0.759 |
| neurovoz:read | 0.809 | 1.000 | 0.815 |
| dcaps:interview | 0.759 | 0.815 | 1.000 |

### inputcond

| | pcgita:read | neurovoz:read | dcaps:interview |
|---|---|---|---|
| pcgita:read | 1.000 | 0.815 | 0.305 |
| neurovoz:read | 0.815 | 1.000 | 0.491 |
| dcaps:interview | 0.305 | 0.491 | 1.000 |

Pearson r between the per-dataset 33-length attention profiles. Values near 0
mean the mixers are not agreeing on a shared readout depth.

## Single-layer probe AUC per layer (for reference, honest OOF, no selection)

| dataset | argmax layer | max AUC | AUC at L0 | AUC at L16 | AUC at L32 |
|---|---|---|---|---|---|
| pcgita:read | 12 | 0.930 | 0.841 | 0.928 | 0.860 |
| neurovoz:read | 24 | 0.949 | 0.844 | 0.931 | 0.917 |
| dcaps:interview | 31 | 0.697 | 0.559 | 0.632 | 0.667 |

Nested layer picks per outer fold:
  pcgita:read          [16, 17, 3, 16, 16]   (oracle layer 12)
  neurovoz:read        [19, 26, 20, 26, 28]   (oracle layer 24)
  dcaps:interview      [31, 18, 19, 20, 17]   (oracle layer 31)

## Prompt-conditioned mixer (attention weights conditioned on elicitation task)


### pcgita, all tasks pooled (6317 clips, 117 speakers, tasks ddk, monologue, read, vowel, words)

| method | AUC | 95% CI | shuffle | perm p |
|---|---|---|---|---|
| single_nested | 0.839 | 0.784-0.892 | 0.504 | 0.143 |
| mean_layers | 0.868 | 0.817-0.917 | 0.505 | 0.143 |
| mixer_static | 0.864 | 0.807-0.914 | 0.485 | 0.143 |
| mixer_taskcond | 0.864 | 0.807-0.915 | 0.480 | 0.143 |

| comparison | delta AUC | 95% CI | P(delta>0) |
|---|---|---|---|
| taskcond minus static | +0.0004 | -0.0018..+0.0027 | 0.630 |
| taskcond minus single nested | +0.0250 | -0.0018..+0.0557 | 0.965 |
| static minus single nested | +0.0247 | -0.0029..+0.0558 | 0.961 |

Per-task attention peaks learned by the prompt-conditioned mixer:

| task | peak layer | peak weight | mass L0-3 | mass L4-15 | mass L16-32 |
|---|---|---|---|---|---|
| ddk | 13 | 0.043 | 0.106 | 0.458 | 0.436 |
| monologue | 20 | 0.038 | 0.120 | 0.370 | 0.510 |
| read | 13 | 0.045 | 0.092 | 0.472 | 0.436 |
| vowel | 13 | 0.045 | 0.104 | 0.449 | 0.448 |
| words | 13 | 0.043 | 0.092 | 0.466 | 0.442 |

### neurovoz, all tasks pooled (2903 clips, 108 speakers, tasks ddk, read, vowel)

| method | AUC | 95% CI | shuffle | perm p |
|---|---|---|---|---|
| single_nested | 0.886 | 0.843-0.923 | 0.497 | 0.062 |
| mean_layers | 0.891 | 0.842-0.929 | 0.490 | 0.062 |
| mixer_static | 0.879 | 0.826-0.925 | 0.500 | 0.062 |
| mixer_taskcond | 0.879 | 0.827-0.924 | 0.505 | 0.062 |

| comparison | delta AUC | 95% CI | P(delta>0) |
|---|---|---|---|
| taskcond minus static | -0.0002 | -0.0028..+0.0023 | 0.448 |
| taskcond minus single nested | -0.0066 | -0.0339..+0.0195 | 0.317 |
| static minus single nested | -0.0064 | -0.0343..+0.0200 | 0.327 |

Per-task attention peaks learned by the prompt-conditioned mixer:

| task | peak layer | peak weight | mass L0-3 | mass L4-15 | mass L16-32 |
|---|---|---|---|---|---|
| ddk | 15 | 0.037 | 0.099 | 0.380 | 0.521 |
| read | 15 | 0.039 | 0.093 | 0.389 | 0.518 |
| vowel | 9 | 0.037 | 0.107 | 0.416 | 0.477 |
