#!/bin/bash
cd /workspace/verify22
export HF_HOME=/workspace/hf
python3 P22_verify.py score 305 679 466 705 --mode cut300 --out s2_cut300.csv > s2_cut300.log 2>&1
python3 P22_verify.py score 305 679 466 705 --mode default --out s2_default.csv > s2_default.log 2>&1
echo DONE > s2.DONE
