#!/bin/bash
# PART26 Qwen3-Omni-30B-A3B MDVR-KCL CPU pod driver.
# 1 pinned libraries (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1), stops if the versions differ; 2 waits for in/UPLOAD_OK
# and checks every uploaded file against the Mac sha256 list; 3 nested five-repeat encoder probe with the saved fold
# files (p25a_nested5.py, the PART25 A script, unchanged); 4 pod-side independent check (p25a_podverify.py, unchanged);
# 5 ALL_DONE (the Mac watchdog pulls and deletes the pod).
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-30} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib > logs/pip.log 2>&1 || fail "pip install failed"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
python3 -c "import sklearn,numpy,scipy,sys; sys.exit(0 if (sklearn.__version__,numpy.__version__,scipy.__version__)==('1.9.1','2.1.2','1.18.1') else 1)" || fail "library versions differ from 1.9.1/2.1.2/1.18.1"
python3 -c "import numpy; numpy.show_config()" > logs/numpy_config.log 2>&1
grep -m1 "model name" /proc/cpuinfo >> $L; log "nproc $(nproc)"
touch PIP_DONE
for i in $(seq 1 240); do [ -f in/UPLOAD_OK ] && break; sleep 5; done
[ -f in/UPLOAD_OK ] || fail "upload not ready after 20 min"
( cd in && sha256sum -c files.sha256 ) > logs/upload_check.log 2>&1 || fail "upload sha256 check failed"
log "UPLOAD CHECK OK $(grep -c ': OK' logs/upload_check.log) files"
python3 p25a_nested5.py in/q3o_kcl_states.npz kcl in/folds out/p26_q3o_kcl > logs/probe_kcl.log 2>&1 || fail "probe failed: $(tail -1 logs/probe_kcl.log | cut -c1-200)"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe_kcl.log)"
python3 p25a_podverify.py in/q3o_kcl_states.npz kcl in/folds out/p26_q3o_kcl > logs/podverify_kcl.log 2>&1 || log "podverify crashed"
log "PODVERIFY: $(grep PODVERIFY logs/podverify_kcl.log)"
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
