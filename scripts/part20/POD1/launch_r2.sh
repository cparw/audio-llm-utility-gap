#!/bin/bash
# Repeat 2 of the whole 5-fold run, balanced and uniform, launched after balanced fold 3 collapsed in repeat 1
# and did not collapse when rerun alone (GPU nondeterminism). Same script, same folds, same order, same weights.
cd /workspace/scores/part20/POD1
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
S=/workspace/scores/part20/POD1
for FL in 0,1,2 3,4; do
  setsid nohup python3 $S/balanced_ft_pitt.py $S/mf_pitt468_p20.csv $S/pitt_groupkfold5_pod.csv $S/balanced_r2_folds balanced balanced $FL > /workspace/logs/balanced_r2_${FL//,/_}.log 2>&1 < /dev/null &
  setsid nohup python3 $S/balanced_ft_pitt.py $S/mf_pitt468_p20.csv $S/pitt_groupkfold5_pod.csv $S/control_r2_folds uniform uniform $FL > /workspace/logs/control_r2_${FL//,/_}.log 2>&1 < /dev/null &
done
echo launched
