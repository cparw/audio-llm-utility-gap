#!/bin/bash
# PART26 Kimi-Audio Pitt GPU pod driver (adapted from part26/Kimi-Audio_PC-GITA/podfiles/run_gpu26.sh).
# 1 Kimi env and model (setup_kimi.sh, the PC-GITA cell copy of the part20 POD4d setup); 2 wait for the Mac upload,
# check sha256 of every input and every wav; 3 smoke run on 3 clips; 4 encoder states for all 468 Pitt clips
# (kimi_enc_extract.py, new PART26 script, copy of the PC-GITA cell file); 5 rerun of the first 10 clips in a fresh
# process (determinism); ALL_DONE.
cd /workspace; mkdir -p logs out in data
export HF_HOME=/workspace/hf PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
unset HF_HUB_ENABLE_HF_TRANSFER
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START kimi pitt"
nohup bash guard26.sh > /dev/null 2>&1 &
bash setup_kimi.sh
[ -f KIMI_PIP_DONE ] || fail "kimi pip setup failed (see logs/KIMI_SETUP.log)"
log "PIP OK: $(grep '^VERS' logs/KIMI_SETUP.log | tail -1)"
for i in $(seq 1 480); do [ -f KIMI_DL_DONE ] && break; sleep 5; done
[ -f KIMI_DL_DONE ] || fail "model download not done after 40 min"
log "MODEL OK: $(tail -1 logs/KIMI_DL.log)"
for i in $(seq 1 360); do [ -f in/UPLOAD_OK ] && break; sleep 5; done
[ -f in/UPLOAD_OK ] || fail "upload not done after 30 min"
sha256sum -c in/in_sha256.txt > logs/upload_check.log 2>&1 || fail "input sha256 check failed"
sha256sum -c in/audio.sha256 > logs/wav_check.log 2>&1 || fail "wav sha256 check failed"
log "DATA OK $(grep -c ': OK' logs/wav_check.log) wavs, $(grep -c ': OK' logs/upload_check.log) inputs"
/opt/kimienv/bin/python kimi_enc_extract.py in/mf_pitt.csv out/smoke_kimi_pitt ad 3 > logs/smoke.log 2>&1 || fail "smoke failed: $(tail -3 logs/smoke.log | tr '\n' ' ' | cut -c1-300)"
log "SMOKE OK: $(grep RESULT logs/smoke.log)"
touch SMOKE_OK
/opt/kimienv/bin/python kimi_enc_extract.py in/mf_pitt.csv out/kimi_pitt ad > logs/extract.log 2>&1 || fail "extract failed: $(tail -3 logs/extract.log | tr '\n' ' ' | cut -c1-300)"
log "EXTRACT DONE: $(grep RESULT logs/extract.log)"
/opt/kimienv/bin/python kimi_enc_extract.py in/mf_pitt.csv out/rerun10_kimi_pitt ad 10 > logs/rerun10.log 2>&1 || log "rerun10 failed"
log "RERUN10: $(grep RESULT logs/rerun10.log)"
/opt/kimienv/bin/pip freeze > logs/kimienv_freeze.txt 2>/dev/null
( cd out && sha256sum * > ../logs/out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
