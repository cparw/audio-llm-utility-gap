#!/bin/bash
# lo-pod3b-1 driver: unmodified direction_by_arm.py twice (fp16 dump as POD3 made it; bf16-exact), then the extended primary.
cd /workspace; set -x
python3 -c "
import numpy as np
W=np.load('/workspace/lmh/q3o_lm_head.npy')
np.save('/workspace/lmh/q3o_lm_head_fp16.npy', W.astype(np.float16))   # rescore_probe.py line 22: W.astype(np.float16)
print('fp16 dump', W.shape)
" > logs/fp16dump.log 2>&1
ln -sf /workspace/q3o_pitt_states.npz /workspace/out/q3o_pitt_states.npz
for V in fp16 bf16exact; do
  if [ $V = fp16 ]; then ln -sf /workspace/lmh/q3o_lm_head_fp16.npy /workspace/out/q3o_lm_head.npy; else ln -sf /workspace/lmh/q3o_lm_head.npy /workspace/out/q3o_lm_head.npy; fi
  python3 direction_by_arm.py > logs/orig_direction_by_arm_$V.log 2>&1
  mv out/readout_direction_q3o_by_arm.json out/orig_direction_by_arm_$V.json
  mv out/readout_direction_q3o_by_arm.csv out/orig_direction_by_arm_$V.csv
done
rm -f /workspace/out/q3o_lm_head.npy /workspace/out/q3o_pitt_states.npz
python3 pod3b_primary.py /workspace/lmh/q3o_lm_head_fp16.npy fp16 > logs/primary_fp16.log 2>&1 &
python3 pod3b_primary.py /workspace/lmh/q3o_lm_head.npy bf16exact > logs/primary_bf16exact.log 2>&1 &
wait
sha256sum out/* > out/SHA256SUMS.txt
touch ALL_DONE
