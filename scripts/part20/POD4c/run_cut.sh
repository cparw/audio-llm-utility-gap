#!/bin/bash
# POD4c driver: verify tarball, extract, verify clips, score AF3 + AF2 concurrently, analyze.
cd /workspace; export HF_HOME=/workspace/hf; unset HF_HUB_ENABLE_HF_TRANSFER
L=/workspace/logs/driver.log
echo "start $(date -u +%T)" >> $L
H=$(sha256sum pitt_noinv_clips.tgz | cut -d" " -f1)
[ "$H" = "b995bd9afe8a45b1c08472f7333c5855767d60de23307a7881b5f000c73d8a07" ] || { echo "TGZ_SHA_BAD $H" >> $L; exit 1; }
mkdir -p cut && tar --no-same-owner -xzf pitt_noinv_clips.tgz -C cut && echo "extracted $(ls cut/clips | wc -l)" >> $L
python3 prep_cut.py >> $L 2>&1 || { echo PREP_FAIL >> $L; exit 1; }
WINDOW_S=30 nohup python3 af3_zs.py cut/score_manifest.csv work/af3_noinv468.csv > logs/af3_noinv.log 2>&1 &
P3=$!
nohup /workspace/af2venv/bin/python af2_zs.py cut/score_manifest.csv work/af2_noinv468.csv > logs/af2_noinv.log 2>&1 &
P2=$!
echo "launched af3 $P3 af2 $P2 $(date -u +%T)" >> $L
wait $P3; echo "af3 exit $? $(date -u +%T)" >> $L
python3 analyze.py af3 work/af3_noinv468.csv > logs/analyze_af3.log 2>&1; echo "analyze af3 exit $? $(date -u +%T)" >> $L
wait $P2; echo "af2 exit $? $(date -u +%T)" >> $L
python3 analyze.py af2 work/af2_noinv468.csv > logs/analyze_af2.log 2>&1; echo "analyze af2 exit $? $(date -u +%T)" >> $L
echo "DRIVER_DONE $(date -u +%T)" >> $L
