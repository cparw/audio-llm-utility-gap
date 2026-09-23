#!/usr/bin/env bash
cd /workspace
export HF_HOME=/workspace/hf_cache
P1="Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
P2="Listen to this recording. Does the speaker have depression? Answer Yes or No."
P3="From the audio alone, is this speaker likely to have depression? Reply with one word, Yes or No."

# 0: pipeline validation - rebuilt 0.70 set must reproduce the Part 14 audio scores
PROMPT_OVERRIDE="$P1" WINDOW_S=30 python3 -u score_audio.py p14_thr070_check_pod.csv out16/p14_thr070_check.csv > logs/s_check.log 2>&1
echo "CHECK done" >> logs/drive.log
# 1d
PROMPT_OVERRIDE="$P1" WINDOW_S=30 python3 -u score_audio.py p14_phq10_pod.csv out16/p14_phq10_omni.csv > logs/s_phq10.log 2>&1
echo "PHQ10 done" >> logs/drive.log
# 1e
PROMPT_OVERRIDE="$P1" WINDOW_S=30 python3 -u score_audio.py p14_thr060_pod.csv out16/p14_thr060_omni.csv > logs/s_060.log 2>&1
echo "THR060 done" >> logs/drive.log
PROMPT_OVERRIDE="$P1" WINDOW_S=30 python3 -u score_audio.py p14_thr080_pod.csv out16/p14_thr080_omni.csv > logs/s_080.log 2>&1
echo "THR080 done" >> logs/drive.log
# 1b
PROMPT_OVERRIDE="$P2" WINDOW_S=30 python3 -u score_audio.py mf_p14_pod.csv out16/p14_o25_audio_p2.csv > logs/s_p2.log 2>&1
echo "P2 done" >> logs/drive.log
PROMPT_OVERRIDE="$P3" WINDOW_S=30 python3 -u score_audio.py mf_p14_pod.csv out16/p14_o25_audio_p3.csv > logs/s_p3.log 2>&1
echo "P3 done" >> logs/drive.log
echo "ALL DONE" >> logs/drive.log
