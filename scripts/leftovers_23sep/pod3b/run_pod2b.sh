#!/bin/bash
cd /workspace
SKIP_SPREAD=1 VOUT=/workspace/out/pod3b_verify.json python3 pod3b_verify.py > logs/verify.log 2>&1
sha256sum out/* > out/SHA256SUMS.txt
touch ALL_DONE
