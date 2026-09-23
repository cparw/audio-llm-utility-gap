#!/bin/bash
cd /workspace
python3 pod3b_verify.py > logs/verify.log 2>&1
rm -rf /workspace/hf/.cache
sha256sum out/* > out/SHA256SUMS.txt
touch ALL_DONE
