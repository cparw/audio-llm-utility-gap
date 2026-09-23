#!/bin/bash
cd /workspace/pod4; export HF_HOME=/workspace/hf OMP_NUM_THREADS=8; unset HF_HUB_ENABLE_HF_TRANSFER
L=/workspace/pod4/logs/DRIVER_o25.log
until [ -f /workspace/pod4/CUT_OK ]; do sleep 5; done
echo "o25 noinv start $(date -u +%T)" >> $L
python3 score_zs.py Qwen/Qwen2.5-Omni-7B bf16 man_cut.csv,out/o25_noinv_recording.csv,recording > logs/o25_noinv.log 2>&1 || echo "o25 noinv FAILED" >> $L
python3 finalize.py o25_noinv_recording > logs/fin_o25_noinv.log 2>&1 || echo "fin o25 noinv FAILED" >> $L
until [ -f /workspace/pod4/ORIG_OK ]; do sleep 5; done
echo "o25 rerun start $(date -u +%T)" >> $L
python3 score_zs.py Qwen/Qwen2.5-Omni-7B bf16 man_orig.csv,out/o25_orig_recording_rerun.csv,recording > logs/o25_rerun.log 2>&1 || echo "o25 rerun FAILED" >> $L
echo "o25 voice start $(date -u +%T)" >> $L
python3 score_zs.py Qwen/Qwen2.5-Omni-7B bf16 man_orig.csv,out/o25_orig_voice.csv,voice > logs/o25_voice.log 2>&1 || echo "o25 voice FAILED" >> $L
python3 finalize.py o25_orig_voice > logs/fin_o25_voice.log 2>&1 || echo "fin o25 voice FAILED" >> $L
python3 finalize.py o25_orig_recording_rerun > logs/fin_o25_rerun.log 2>&1 || echo "fin o25 rerun FAILED" >> $L
echo "o25 ALL DONE $(date -u +%T)" >> $L; touch /workspace/pod4/O25_DONE
