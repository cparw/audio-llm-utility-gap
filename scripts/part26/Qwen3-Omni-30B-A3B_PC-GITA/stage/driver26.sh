#!/bin/bash
# PART 26 CPU pod driver, Qwen3-Omni-30B-A3B PC-GITA encoder nested probe.  usage: bash driver26.sh
# 1 pinned pip install (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1); 2 wait for in/UPLOAD_OK; 3 check the states sha256
# against in/expected.sha256; 4 p25a_nested5.py unchanged (PART 25 A script, md5 615ced03...) on the saved fold files;
# 5 p25a_podverify.py unchanged (pod-side independent refit and layer re-derivation); 6 sha256 of out/; ALL_DONE.
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-32} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; ( cd out && sha256sum * > ../out_sha256.txt 2>/dev/null ); touch ALL_DONE; exit 1; }
log "DRIVER START"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib > logs/pip.log 2>&1 || fail "pip install failed"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
python3 -c "import sklearn,numpy,scipy,sys;sys.exit(0 if (sklearn.__version__,numpy.__version__,scipy.__version__)==('1.9.1','2.1.2','1.18.1') else 1)" || fail "pinned versions not installed"
python3 -c "import numpy; numpy.show_config()" > logs/numpy_config.log 2>&1
log "CPU $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) nproc $(nproc) $(grep -q avx512f /proc/cpuinfo && echo avx512f present || echo avx512f ABSENT)"
touch PIP_DONE
for i in $(seq 1 240); do [ -f in/UPLOAD_OK ] && break; sleep 5; done
[ -f in/UPLOAD_OK ] || fail "upload not complete after 20 min"
( cd in && sha256sum -c expected.sha256 ) > logs/upload_check.log 2>&1 || fail "states or fold sha256 check failed"
log "UPLOAD CHECK OK: $(grep -c ': OK' logs/upload_check.log) files"
python3 p25a_nested5.py in/q3o_pcgita_encstates.npz pcgita in/folds out/p26_q3o_pcgita > logs/probe_pcgita.log 2>&1 || fail "probe failed: $(tail -2 logs/probe_pcgita.log)"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe_pcgita.log)"
python3 p25a_podverify.py in/q3o_pcgita_encstates.npz pcgita in/folds out/p26_q3o_pcgita > logs/podverify_pcgita.log 2>&1 || log "podverify crashed: $(tail -2 logs/podverify_pcgita.log)"
log "PODVERIFY: $(grep PODVERIFY logs/podverify_pcgita.log)"
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
