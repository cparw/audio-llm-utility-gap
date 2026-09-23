#!/bin/bash
# t5 verifier pod driver. /workspace/JOBS lines: "<npz basename> <streams> <mode>" ; mode exact|variants|native
cd /workspace; mkdir -p out logs
L=logs/DRIVER.log; POD=$(cat POD_NAME); NP=$(( $(nproc) - 2 ))
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
log "START $POD host $(hostname) nproc $(nproc)"
while [ ! -f PIP_DONE ]; do sleep 3; done; log "pip: $(tail -1 logs/pip.log)"
while [ ! -f UPLOAD_DONE ]; do sleep 3; done; log "upload done"
sha256sum in/*.npz folds/*.csv scripts/*.py > out/input_sha256_${POD}.txt
ARGS=""; while read -r f s m; do [ -z "$f" ] && continue; ds=${f%%_*}; case $ds in edaic30|edaicfull) fd=edaic;; *) fd=$ds;; esac; ARGS="$ARGS in/$f.npz:$fd"; done < JOBS
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/v5_envfolds.py out/envfolds_${POD}.json $ARGS > logs/envfolds.log 2>&1; log "envfolds rc $?"
run(){ # tag npz streams folds dtype nproc threads coretype
  local tag=$1 npz=$2 st=$3 fo=$4 dt=$5 np=$6 th=$7 ct=$8
  if [ -n "$ct" ]; then
    OPENBLAS_CORETYPE=$ct OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th MKL_NUM_THREADS=$th python3 scripts/v5_refit.py $npz $st $fo $dt $np $tag out >> logs/$tag.log 2>&1
  else
    OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th MKL_NUM_THREADS=$th python3 scripts/v5_refit.py $npz $st $fo $dt $np $tag out >> logs/$tag.log 2>&1
  fi
  log "$tag rc $? $(tail -1 logs/$tag.log)"; }
while read -r f s m; do
  [ -z "$f" ] && continue; ds=${f%%_*}; case $ds in edaic30|edaicfull) fd=edaic;; *) fd=$ds;; esac
  run ${f}__exact in/$f.npz $s mine f32 $NP 1 ""
  if [ "$m" = "variants" ]; then
    run ${f}__podfile in/$f.npz enc folds/${fd}_groupkfold5_pod.csv f32 $NP 1 ""
    run ${f}__macfile in/$f.npz enc folds/${fd}_groupkfold5_mac.csv f32 $NP 1 ""
    run ${f}__f64 in/$f.npz enc mine f64 $NP 1 ""
    run ${f}__thr8 in/$f.npz enc mine f32 4 8 ""
    for CT in Haswell Zen Sandybridge Prescott SkylakeX SapphireRapids; do run ${f}__ct$CT in/$f.npz enc mine f32 $NP 1 $CT; done
  fi
done < JOBS
sha256sum out/* > out/output_sha256_${POD}.txt 2>/dev/null
log "ALL DONE"; touch ALL_DONE
