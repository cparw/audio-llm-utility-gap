#!/bin/bash
cd /workspace/pod4; export HF_HOME=/workspace/hf OMP_NUM_THREADS=8; unset HF_HUB_ENABLE_HF_TRANSFER
L=/workspace/pod4/logs/DRIVER_q2a.log
until [ -f /workspace/pod4/CUT_OK ]; do sleep 5; done
echo "q2a noinv start $(date -u +%T)" >> $L
python3 score_zs.py Qwen/Qwen2-Audio-7B-Instruct fp16 man_cut.csv,out/q2a_noinv_recording.csv,recording > logs/q2a_noinv.log 2>&1 || echo "q2a noinv FAILED" >> $L
python3 finalize.py q2a_noinv_recording > logs/fin_q2a_noinv.log 2>&1 || echo "fin q2a noinv FAILED" >> $L
until [ -f /workspace/pod4/ORIG_OK ]; do sleep 5; done
echo "q2a rerun start $(date -u +%T)" >> $L
python3 score_zs.py Qwen/Qwen2-Audio-7B-Instruct fp16 man_orig.csv,out/q2a_orig_recording_rerun.csv,recording > logs/q2a_rerun.log 2>&1 || echo "q2a rerun FAILED" >> $L
python3 finalize.py q2a_orig_recording_rerun > logs/fin_q2a_rerun.log 2>&1 || echo "fin q2a rerun FAILED" >> $L
echo "q2a bf16 start $(date -u +%T)" >> $L
python3 score_zs.py Qwen/Qwen2-Audio-7B-Instruct bf16 man_orig.csv,out/q2a_orig_recording_rerun_bf16.csv,recording man_cut.csv,out/q2a_noinv_recording_bf16.csv,recording > logs/q2a_bf16.log 2>&1 || echo "q2a bf16 FAILED" >> $L
python3 finalize.py q2a_noinv_recording_bf16 > logs/fin_q2a_noinv_bf16.log 2>&1 || echo "fin q2a noinv bf16 FAILED" >> $L
python3 finalize.py q2a_orig_recording_rerun_bf16 > logs/fin_q2a_rerun_bf16.log 2>&1 || echo "fin q2a rerun bf16 FAILED" >> $L
echo "q2a ALL DONE $(date -u +%T)" >> $L; touch /workspace/pod4/Q2A_DONE
