#!/bin/bash
# PART26 Kimi-Audio NeuroVoz GPU pod driver (copy of the Kimi-Audio_PC-GITA run_gpu26.sh with the dataset swapped).
# 1 Kimi env and model (setup_kimi.sh, same file as the PC-GITA cell); 2 wait for the Mac upload, check the sha256 of every
# wav (1270) and of the clip list; 3 smoke run on 3 clips; 4 encoder states for all 1270 clips (kimi_enc_extract.py, the
# PART26 script, same file as the PC-GITA cell); 5 rerun of the first 10 clips in a fresh process (determinism); ALL_DONE.
cd /workspace; mkdir -p logs out in data
export HF_HOME=/workspace/hf PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
unset HF_HUB_ENABLE_HF_TRANSFER
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START kimi neurovoz"
pgrep -f guard26.sh > /dev/null || nohup bash guard26.sh > /dev/null 2>&1 &
# restart-safe: setup is skipped when its markers exist (first attempt 01:10 UTC stopped at untar: chown not permitted on the volume)
{ [ -f KIMI_PIP_DONE ] && [ -f KIMI_DL_DONE ]; } && log "SETUP SKIPPED (markers present)" || bash setup_kimi.sh
[ -f KIMI_PIP_DONE ] || fail "kimi pip setup failed (see logs/KIMI_SETUP.log)"
log "PIP OK: $(grep '^VERS' logs/KIMI_SETUP.log | tail -1)"
for i in $(seq 1 480); do [ -f KIMI_DL_DONE ] && break; sleep 5; done
[ -f KIMI_DL_DONE ] || fail "model download not done after 40 min"
log "MODEL OK: $(tail -1 logs/KIMI_DL.log)"
for i in $(seq 1 360); do [ -f in/UPLOAD_OK ] && break; sleep 5; done
[ -f in/UPLOAD_OK ] || fail "upload not done after 30 min"
sha256sum -c in/in_sha256.txt > logs/upload_check.log 2>&1 || fail "clip list sha256 check failed"
tar --no-same-owner -xf in/neurovoz1270.tar -C data || fail "untar failed"
( cd data && sha256sum -c ../in/wav_sha256.txt > ../logs/wav_check.log 2>&1 ) || fail "wav sha256 check failed"
n=$(grep -c ': OK$' logs/wav_check.log); [ "$n" = "1270" ] || fail "wav check count $n, expected 1270"
log "DATA OK $n wavs"
/opt/kimienv/bin/python kimi_enc_extract.py in/mf_neurovoz_pod.csv out/smoke_kimi_neurovoz pd 3 > logs/smoke.log 2>&1 || fail "smoke failed"
log "SMOKE OK: $(grep RESULT logs/smoke.log)"
touch SMOKE_OK
/opt/kimienv/bin/python kimi_enc_extract.py in/mf_neurovoz_pod.csv out/kimi_neurovoz pd > logs/extract.log 2>&1 || fail "extract failed"
log "EXTRACT DONE: $(grep RESULT logs/extract.log)"
/opt/kimienv/bin/python kimi_enc_extract.py in/mf_neurovoz_pod.csv out/rerun10_kimi_neurovoz pd 10 > logs/rerun10.log 2>&1 || log "rerun10 failed"
log "RERUN10: $(grep RESULT logs/rerun10.log)"
/opt/kimienv/bin/pip freeze > logs/kimienv_freeze.txt 2>/dev/null
( cd out && sha256sum * > ../logs/out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
