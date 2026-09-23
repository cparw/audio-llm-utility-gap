#!/bin/bash
# PART24 driver, one outer fold per pod.  usage: nohup bash run_fold24.sh FOLD > logs/run_fold.out 2>&1 &
# Copied from PART23 run_fold.sh (build + fold check unchanged); the ctrl300 branch is dropped; seed 0 run, then the
# on-pod mass rule: median answer_mass over this fold's test clips < 0.5 -> rerun with seed 1 (both kept). No manual step needed.
K=$1; cd /workspace; L=/workspace/logs/DRIVER.log
export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 GC=on CK=55
unset EPOCHS_OVERRIDE
t(){ date -u +%H:%M:%S; }
fail(){ echo "FAILED $1 $(t)" >> $L; echo "$1" > /workspace/FAILED; touch /workspace/ALL_DONE; echo "ALL_DONE (after failure) $(t)" >> $L; exit 1; }
echo "DRIVER_START fold $K $(t) pod ${RUNPOD_POD_ID:-?}" >> $L
while [ ! -f /workspace/FETCH_DONE ] || [ ! -f /workspace/PIP_DONE ] || [ ! -f /workspace/HF_DONE ]; do sleep 10; done
# smoke (smoke24.sh, same pod, runs in the fetch window) gates this pod's fold run. Max wait 10 min.
n=0; while [ -f /workspace/SMOKE_START ] && [ ! -f /workspace/SMOKE_OK ] && [ ! -f /workspace/SMOKE_FAIL ] && [ $n -lt 600 ]; do sleep 5; n=$((n+5)); done
if [ -f /workspace/SMOKE_OK ]; then echo "SMOKE_OK seen $(t)" >> $L
elif [ -f /workspace/SMOKE_FAIL ]; then
  echo "SMOKE_FAIL seen, waiting up to 300 s for /workspace/SMOKE_OVERRIDE $(t)" >> $L
  n=0; while [ ! -f /workspace/SMOKE_OVERRIDE ] && [ $n -lt 300 ]; do sleep 5; n=$((n+5)); done
  [ -f /workspace/SMOKE_OVERRIDE ] && echo "SMOKE_OVERRIDE present, continuing $(t)" >> $L || fail SMOKE
else echo "SMOKE not run or no verdict after wait, continuing $(t)" >> $L; fi
if [ ! -f /workspace/BUILD_OK ]; then
  python3 /workspace/build_windows.py > /workspace/logs/build.log 2>&1
  head -4 /workspace/logs/build.log >> $L
  if grep -q "built 275 / 275   failures 0" /workspace/logs/build.log && [ "$(grep -c 'n within 0.05s 275/275' /workspace/logs/build.log)" = "2" ]; then
     (cd /workspace/edaicfull/cut && sha256sum *.wav) > /workspace/edaicfull/cut_sha256.txt
     if cmp -s /workspace/edaicfull/cut_sha256.txt /workspace/p23/cut_sha256_p23.txt; then echo "CUT_SHA identical to PART23 (275 windows) $(t)" >> $L
     else echo "CUT_SHA DIFFERS from PART23: $(diff /workspace/edaicfull/cut_sha256.txt /workspace/p23/cut_sha256_p23.txt | grep -c '^<') lines $(t)" >> $L; fi
     touch /workspace/BUILD_OK; echo "BUILD_OK $(t)" >> $L
  else fail BUILD; fi
fi
python3 /workspace/p23/check_folds.py >> $L 2>&1
run_seed(){ s=$1
  for a in 1 2 3; do
    echo "SEED${s}_START fold $K attempt $a $(t)" >> $L
    python3 /workspace/sft_whole24.py /workspace/p23/mf_whole.csv /workspace/p23/edaic_groupkfold5_pod.csv $K $s /workspace/out >> /workspace/logs/whole_s${s}_fold$K.log 2>&1
    echo "SEED${s}_EXIT $? fold $K attempt $a $(t)" >> $L
    [ -f /workspace/out/FT_whole_s${s}_fold$K.RESULT ] && return 0
    sleep 5
  done; return 1; }
run_seed 0 || fail SEED0
touch /workspace/SEED0_DONE; echo "SEED0_DONE $(cat /workspace/out/FT_whole_s0_fold$K.RESULT) $(t)" >> $L
MED=$(python3 -c "
import csv, numpy as np
v = [float(r['answer_mass']) for r in csv.DictReader(open('/workspace/out/FT_whole_s0_fold${K}_oof.csv'))]
print(repr(float(np.median(v))), len(v))")
[ -n "${MED% *}" ] || { echo "MASS_CHECK_ERROR fold $K could not read answer_mass $(t)" >> $L; MED="nan 0"; }
echo "MASS_CHECK fold $K seed 0 median_answer_mass ${MED% *} n ${MED#* } rule: rerun with seed 1 if < 0.5 $(t)" >> $L
echo "fold $K seed 0 median_answer_mass ${MED% *} n ${MED#* }" > /workspace/out/MASS_CHECK_fold$K.txt
if python3 -c "import sys, math; m = float('${MED% *}'); sys.exit(0 if (m < 0.5 or math.isnan(m)) else 1)"; then
  echo "MASS_RULE fold $K: median < 0.5 -> RERUN seed 1 $(t)" >> $L; echo "decision rerun_seed1" >> /workspace/out/MASS_CHECK_fold$K.txt
  if run_seed 1; then touch /workspace/SEED1_DONE; echo "SEED1_DONE $(cat /workspace/out/FT_whole_s1_fold$K.RESULT) $(t)" >> $L
  else echo "SEED1_FAILED fold $K $(t)" >> $L; echo SEED1 > /workspace/FAILED; fi
else
  echo "MASS_RULE fold $K: median >= 0.5 -> no rerun $(t)" >> $L; echo "decision no_rerun" >> /workspace/out/MASS_CHECK_fold$K.txt
fi
touch /workspace/ALL_DONE; echo "ALL_DONE $(t)" >> $L
