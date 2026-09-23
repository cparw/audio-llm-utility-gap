#!/bin/bash
cd /workspace
python3 fetch_lmh.py > logs/fetch_lmh.log 2>&1
python3 pod3b_spread_wo.py > logs/spread_wo.log 2>&1
sha256sum out/* > out/SHA256SUMS.txt
touch ALL_DONE
