#!/bin/bash
cd /workspace/verify22
export HF_HOME=/workspace/hf
python3 P22_verify.py score --mode default 305 679 466 705 --out s2_default.csv > s2_default.log 2>&1
python3 P22_verify.py score --mode cut300 305 679 466 705 --out s2_cut300.csv > s2_cut300.log 2>&1
python3 P22_verify.py score --mode lift 305 679 466 705 --out s2_lift.csv > s2_lift.log 2>&1
python3 P22_verify.py score --mode lift --pidfile pids_all.txt --out all_lift.csv > all_lift.log 2>&1
python3 P22_verify.py score --mode default --pidfile pids_all.txt --out all_default.csv > all_default.log 2>&1
echo DONE > scores.DONE
