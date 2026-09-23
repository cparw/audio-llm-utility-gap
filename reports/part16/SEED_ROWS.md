| 1a | Omni transcript-only, 966 conflict clips, pooled | 0.6077 | NA | 966 | 138 | part16/pod_sync/p14_o25_text.csv |
| 1a | Omni transcript-only, conflict arm | 0.1498 | [0.1131, 0.1887] | 483 | 138 | part15/B_text_conflict.csv |
| 1a | Omni transcript-only, agreement arm | 0.9596 | [0.9306, 0.9823] | 483 | 138 | part15/B_text_conflict.csv |
| 1a | Qwen2-Audio transcript-only, pooled | 0.5995 | NA | 966 | 138 | part16/pod_sync/p14_q2a_text.csv |
| 1a | Qwen2-Audio transcript-only, conflict arm | 0.2359 | [0.1851, 0.2886] | 483 | 138 | part15/B_text_conflict.csv |
| 1a | Qwen2-Audio transcript-only, agreement arm | 0.9081 | [0.8515, 0.9547] | 483 | 138 | part15/B_text_conflict.csv |
| 1a | Omni text vs audio Pearson r | 0.8442 | NA | 966 | 138 | part15/B_text_conflict.json |
| 1a | Omni text vs audio same Yes/No decision | 0.8634 | NA | 966 | 138 | part15/B_text_conflict.json |
| 3a | Qwen3-Omni Pitt encoder, nested, 5 repeats | 0.8122 | per-repeat [0.8199,0.8016,0.8142,0.8342,0.7914] | 468 | 228 | part16/pod_sync/q3o_pitt_nested_repeats.json |
| 3a | Qwen3-Omni Pitt projector, nested | 0.7820 | 1 repeat only | 468 | 228 | part16/pod_sync/q3o_pitt_nested_repeats.json |
| 3a | Qwen3-Omni Pitt LM best, nested, 5 repeats | 0.8445 | per-repeat [0.8397,0.8460,0.8393,0.8505,0.8468] | 468 | 228 | part16/pod_sync/q3o_pitt_nested_repeats.json |
| 3a | Qwen3-Omni Pitt answer state, nested, 5 repeats | 0.8237 | per-repeat [0.8144,0.8180,0.8221,0.8442,0.8200] | 468 | 228 | part16/pod_sync/q3o_pitt_nested_repeats.json |
| 3a | Qwen3-Omni Pitt zero-shot answer (THIS RUN) | 0.7673 | NA | 468 | 228 | part16/pod_sync/q3o_pitt_zeroshot_scores.csv |
| 3b | Qwen3-Omni Pitt cos(probe dir, Yes-No unembed) | 0.0051 | random floor 0.0177 [0.0008, 0.0498] | 468 | 228 | part16/pod_sync/readout_direction_q3o.csv |
| 3b | Qwen3-Omni Pitt probe AUC | 0.8307 | NA | 468 | 228 | part16/pod_sync/readout_direction_q3o.csv |
| 3b | Qwen3-Omni Pitt AUC along readout direction | 0.7656 | NA | 468 | 228 | part16/pod_sync/readout_direction_q3o.csv |
| 3b | Qwen3-Omni Pitt probe after projecting out readout dir | 0.8305 | NA | 468 | 228 | part16/pod_sync/readout_direction_q3o.csv |
