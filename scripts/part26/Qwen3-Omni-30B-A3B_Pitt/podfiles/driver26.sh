#!/bin/bash
# PART26 CPU pod driver, Qwen3-Omni-30B-A3B Pitt encoder probe (saved POD3 states, no extraction).
# 1 pinned pip install (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1), versions asserted, CPU and avx512f logged;
# 2 wait for the Mac upload (UPLOAD_OK), check every uploaded file against the Mac sha256 list;
# 3 nested five-repeat encoder probe on the saved fold files (p26_nested5.py, one BLAS thread per fit);
# 4 pod-side independent check (p26_podverify.py); 5 sha256 of out/; ALL_DONE.
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-30}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export P26_FOLD_STEM=pod_Control15ids P26_TAGS="seed0 seed1 seed2 seed3 seed4"
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START q3o pitt"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib > logs/pip.log 2>&1 || fail "pip install failed"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
python3 -c "import sklearn,numpy,scipy; assert (sklearn.__version__,numpy.__version__,scipy.__version__)==('1.9.1','2.1.2','1.18.1')" || fail "version pins not met"
grep -m1 "model name" /proc/cpuinfo >> $L; grep -q avx512f /proc/cpuinfo && log "avx512f present" || log "avx512f ABSENT"
log "nproc $(nproc)"
python3 -c "import numpy; numpy.show_config()" > logs/numpy_config.log 2>&1
touch PIP_DONE
for i in $(seq 1 120); do [ -f UPLOAD_OK ] && break; sleep 10; done
[ -f UPLOAD_OK ] || fail "upload not finished after 20 min"
sha256sum -c files.sha256 > logs/upload_check.log 2>&1 || fail "upload sha256 check failed"
log "UPLOAD CHECK OK $(grep -c ': OK' logs/upload_check.log) files"
python3 p26_nested5.py q3o_pitt_encstates.npz pitt folds out/p26_q3o_pitt > logs/probe.log 2>&1 || fail "probe failed"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe.log)"
python3 p26_podverify.py q3o_pitt_encstates.npz pitt folds out/p26_q3o_pitt > logs/podverify.log 2>&1 || log "podverify crashed"
log "PODVERIFY: $(grep PODVERIFY logs/podverify.log)"
( cd out && sha256sum * > ../logs/out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
