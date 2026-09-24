#!/bin/bash
# PART 26 CPU pod driver, one cell: Qwen3-Omni-30B-A3B on ADReSSo, encoder stream.
# 1 pinned fit libraries (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1), one BLAS thread per fit;
# 2 wait for the Mac upload (in/UPLOAD_OK) and check every uploaded file against the Mac sha256 list;
# 3 nested five-repeat encoder probe on the saved fold files (p25a_nested5.py, the PART 25 A script, unchanged);
# 4 pod-side independent check (p25a_podverify.py, unchanged); 5 sha256 of out/; ALL_DONE (the Mac watchdog pulls
# and deletes the pod).
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 P25_NJ=${P25_NJ:-30}
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START q3o adresso enc"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib > logs/pip.log 2>&1 || fail "pip install failed"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
python3 -c "import sklearn,numpy,scipy,sys;sys.exit(0 if (sklearn.__version__,numpy.__version__,scipy.__version__)==('1.9.1','2.1.2','1.18.1') else 1)" || fail "pinned versions not installed"
log "CPU $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) ncpu $(nproc) avx512f $(grep -q avx512f /proc/cpuinfo && echo yes || echo no)"
touch PIP_DONE
for i in $(seq 1 360); do [ -f in/UPLOAD_OK ] && break; sleep 5; done
[ -f in/UPLOAD_OK ] || fail "upload not done after 30 min"
( cd in && sha256sum -c files.sha256 ) > logs/upload_check.log 2>&1 || fail "uploaded file sha256 check failed"
log "UPLOAD CHECK OK $(grep -c ': OK' logs/upload_check.log) files"
python3 p25a_nested5.py in/q3o_adresso_states.npz adresso in/folds out/p26_q3o_adresso > logs/probe.log 2>&1 || fail "probe failed: $(tail -2 logs/probe.log | tr '\n' ' ')"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe.log)"
python3 p25a_podverify.py in/q3o_adresso_states.npz adresso in/folds out/p26_q3o_adresso > logs/podverify.log 2>&1 || log "podverify crashed: $(tail -2 logs/podverify.log | tr '\n' ' ')"
log "PODVERIFY: $(grep PODVERIFY logs/podverify.log)"
( cd out && sha256sum * > ../out_sha256.txt ) && cp out_sha256.txt logs/out_sha256.txt
log "ALL DONE"
touch ALL_DONE
