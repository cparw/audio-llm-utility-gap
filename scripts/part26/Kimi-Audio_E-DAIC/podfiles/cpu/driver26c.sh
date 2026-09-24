#!/bin/bash
# PART 26 Kimi-Audio E-DAIC CPU pod driver (same design as the part26 Qwen3-Omni E-DAIC CPU driver): pinned fit libraries
# (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1), then the PART 25 A nested five-repeat encoder probe (p25a_nested5.py,
# unchanged) on the re-extracted Kimi Whisper-encoder states with the saved release fold files, one BLAS thread per fit,
# then the independent pod-side check (p25a_podverify.py, unchanged). ALL_DONE.
cd /workspace; mkdir -p logs out in
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=32 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; ( cd out && sha256sum * > ../out_sha256.txt 2>/dev/null ); touch ALL_DONE; exit 1; }
log "DRIVER START kimi edaic encoder probe"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" pandas joblib > logs/pip.log 2>&1 || fail "pip failed"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
python3 -c "import numpy; numpy.show_config()" > logs/numpy_config.log 2>&1
grep -m1 "model name" /proc/cpuinfo >> $L; grep -q avx512f /proc/cpuinfo && log "avx512f present" || log "avx512f ABSENT"; log "nproc $(nproc) cpus_in_cpuinfo $(grep -c ^processor /proc/cpuinfo)"
touch PIP_DONE
for i in $(seq 1 240); do [ -f in/UPLOAD_OK ] && break; sleep 15; done
[ -f in/UPLOAD_OK ] || fail "states not uploaded after 60 min"
( cd in && sha256sum -c files.sha256 ) > logs/in_check.log 2>&1 || fail "input sha256 check failed"
log "INPUT CHECK OK: $(tr '\n' ' ' < logs/in_check.log)"
python3 p25a_nested5.py in/p26_kimi_edaic_encstates.npz edaic folds out/p26_kimi_edaic > logs/probe_edaic.log 2>&1 || fail "probe failed: $(tail -2 logs/probe_edaic.log | tr '\n' ' ')"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe_edaic.log)"
python3 p25a_podverify.py in/p26_kimi_edaic_encstates.npz edaic folds out/p26_kimi_edaic > logs/podverify_edaic.log 2>&1 || log "podverify crashed: $(tail -2 logs/podverify_edaic.log | tr '\n' ' ')"
log "PODVERIFY: $(grep PODVERIFY logs/podverify_edaic.log)"
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
