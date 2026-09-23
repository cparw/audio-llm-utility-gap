#!/bin/bash
cd /workspace
python3 -c "
import numpy as np
W=np.load('/workspace/lmh/q3o_lm_head.npy'); np.save('/workspace/lmh/q3o_lm_head_fp16.npy', W.astype(np.float16))" > logs/fp16dump.log 2>&1
python3 pod3b_gap.py /workspace/lmh/q3o_lm_head_fp16.npy > logs/gap.log 2>&1
sha256sum out/* > out/SHA256SUMS.txt
touch ALL_DONE
