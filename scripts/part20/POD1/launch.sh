#!/bin/bash
cd /workspace/scores/part20/POD1
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
S=/workspace/scores/part20/POD1
for FL in 0,3 1,4 2; do
  nohup python3 $S/balanced_ft_pitt.py $S/mf_pitt468_p20.csv $S/pitt_groupkfold5_pod.csv $S/balanced_folds balanced balanced $FL > /workspace/logs/balanced_${FL/,/_}.log 2>&1 &
  echo "balanced $FL pid $!"
done
nohup python3 $S/balanced_ft_pitt.py $S/mf_pitt468_p20.csv $S/pitt_groupkfold5_pod.csv $S/control_folds uniform uniform 0,1,2 > /workspace/logs/control_0_1_2.log 2>&1 &
echo "control 0,1,2 pid $!"
nohup python3 $S/balanced_ft_pitt.py $S/mf_pitt468_p20.csv $S/pitt_groupkfold5_pod.csv $S/control_folds uniform uniform 4,3 > /workspace/logs/control_4_3.log 2>&1 &   # launched after balanced fold 2 finished
nohup python3 $S/balanced_ft_pitt.py $S/mf_pitt468_p20.csv $S/pitt_groupkfold5_pod.csv $S/balanced_rerun_fold3 balanced balanced 3 > /workspace/logs/balanced_rerun_fold3.log 2>&1 < /dev/null &   # repeatability check of balanced fold 3 (it collapsed), launched about 12:42Z
