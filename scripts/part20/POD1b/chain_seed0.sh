#!/bin/bash
# lowest-priority extra check: seed 0 = the original sft_projector.py clip order RandomState(fold*10+ep), run only after seeds 1,2,3 finish
D=/workspace/scores/part20/POD1b
until [ -f $D/std_ft_seed1_pitt_oof.csv ] && [ -f $D/std_ft_seed2_pitt_oof.csv ] && [ -f $D/std_ft_seed3_pitt_oof.csv ]; do sleep 10; done
echo "seed0 start $(date -u)"
cd $D && HF_HOME=/workspace/hf INPUT_CACHE=/root/pitt468_inputs.pt OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 /workspace/code/std_ft_seed.py /workspace/code/mf_pitt468_pod.csv /workspace/code/pitt_groupkfold5_pod.csv 0 $D/std_ft_seed0_pitt > $D/std_ft_seed0.log 2>&1
echo "seed0 exit $? $(date -u)"
