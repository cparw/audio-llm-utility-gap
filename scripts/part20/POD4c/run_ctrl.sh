#!/bin/bash
cd /workspace; export HF_HOME=/workspace/hf; unset HF_HUB_ENABLE_HF_TRANSFER
L=/workspace/logs/driver_ctrl.log
echo "ctrl start $(date -u +%T)" >> $L
WINDOW_S=30 /workspace/af2venv/bin/python af2_zs_win.py orig30/score_manifest.csv work/af2_orig30_468.csv > logs/af2_orig30.log 2>&1
echo "af2 orig30 exit $? $(date -u +%T)" >> $L
until grep -q "analyze af2 exit" /workspace/logs/driver.log; do sleep 10; done
python3 analyze_ctrl.py af2 work/af2_noinv468.csv work/af2_orig30_468.csv > logs/analyze_ctrl.log 2>&1
echo "analyze ctrl exit $? $(date -u +%T)" >> $L
echo "CTRL_DONE $(date -u +%T)" >> $L
