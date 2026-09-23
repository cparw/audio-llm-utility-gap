#!/bin/bash
# T5 pod driver. Reads /workspace/JOBS (space separated job ids), /workspace/MODE (main|verify), /workspace/VARIANTS (full|basic).
cd /workspace; mkdir -p out logs
export PYTHONNOUSERSITE=1
L=logs/DRIVER.log; POD=$(cat POD_NAME); MODE=$(cat MODE); VAR=$(cat VARIANTS); JOBS=$(cat JOBS)
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
log "START pod $POD mode $MODE variants $VAR jobs $JOBS host $(hostname)"
python3 -m pip install --break-system-packages -q scikit-learn==1.9.1 numpy==2.1.2 scipy==1.18.1 pandas > logs/pip.log 2>&1; log "pip rc $?"
touch PIP_DONE
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 t5_env.py out/env_${POD}.json >> $L 2>&1
sha256sum curves_from_states.py t5_*.py inputs/*.npz folds/*.csv > out/input_sha256_${POD}.txt
fds(){ case $1 in pitt) echo pitt;; edaic30|edaicfull) echo edaic;; *) echo $1;; esac; }
for J in $JOBS; do
  DS=${J%%_*}; FD=$(fds $DS); NPZ=inputs/${J}_states.npz
  if [ "$MODE" = "main" ]; then
    ( export OMP_NUM_THREADS=1; python3 curves_from_states.py $NPZ out/${J} > logs/${J}_curves.log 2>&1 ); log "$J exact script rc $? | $(grep -h 'best auc_oof' logs/${J}_curves.log | tr '\n' ';')"
    python3 t5_foldcheck.py $J $NPZ folds $FD out >> logs/${J}_foldcheck.log 2>&1; log "$J foldcheck rc $?"
    for T in pod mac; do
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 t5_variant.py $J $NPZ folds/${FD}_groupkfold5_${T}.csv ${T}file out f32 >> logs/${J}_variants.log 2>&1; log "$J variant ${T}file rc $?"
    done
    if [ "$VAR" = "full" ]; then
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 t5_variant.py $J $NPZ folds/${FD}_groupkfold5_pod.csv podfile_f64 out f64 >> logs/${J}_variants.log 2>&1; log "$J variant podfile_f64 rc $?"
      OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 t5_variant.py $J $NPZ folds/${FD}_groupkfold5_pod.csv podfile_thr8 out f32 4 >> logs/${J}_variants.log 2>&1; log "$J variant podfile_thr8 rc $?"
      for CT in Haswell SkylakeX Zen SapphireRapids Sandybridge Prescott; do
        OPENBLAS_CORETYPE=$CT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 t5_variant.py $J $NPZ folds/${FD}_groupkfold5_pod.csv podfile_ct${CT} out f32 >> logs/${J}_variants.log 2>&1; log "$J variant podfile_ct${CT} rc $?"
      done
    fi
    python3 t5_vrefit.py $J $NPZ folds/${FD}_groupkfold5_pod.csv out > logs/${J}_vrefit.log 2>&1; log "$J same-pod verifier refit rc $?"
  else
    python3 t5_vrefit.py $J $NPZ folds/${FD}_groupkfold5_pod.csv out > logs/${J}_vrefit.log 2>&1; log "$J vrefit rc $?"
  fi
done
sha256sum out/* > out/output_sha256_${POD}.txt 2>/dev/null
log "ALL DONE"
touch ALL_DONE
