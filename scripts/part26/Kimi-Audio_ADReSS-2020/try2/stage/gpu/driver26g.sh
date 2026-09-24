#!/bin/bash
# PART26 GPU pod driver, Kimi-Audio ADReSS-2020 encoder states (try2; copy of the MDVR-KCL driver26g.sh that landed).
# Try 1 failed because the Mac upload never arrived and the watchdog idle-killed the pod while the driver slept.
# Here the setup runs first and the Mac uploads during the setup; then the driver waits for UPLOAD_OK with a python
# sleep (so the pod never looks idle), checks every uploaded file against the Mac sha256 list, runs kimi_enc_probe.py
# (podD2_final/kimi_probe.py plus Whisper encoder hooks, the script behind the KCL cell) on the 156 ADReSS-2020 clips,
# 30 s window, ad prompt, then records the environment and the sha256 of out/; ALL_DONE.
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONNOUSERSITE=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; ( cd out && sha256sum * > ../out_sha256.txt 2>/dev/null ); touch ALL_DONE; exit 1; }
log "DRIVER START kimi adress2020 enc"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> $L 2>&1
bash setup_kimi.sh
[ -f KIMI_SETUP_DONE ] || fail "setup did not finish"
grep -q KIMI_IMPORT_OK logs/KIMI_SETUP.log || fail "kimia_infer import failed"
log "SETUP DONE $(grep -m1 KIMI_SRC_COMMIT logs/KIMI_SETUP.log | tr -d '+')"
python3 -c "
import os,time
for i in range(240):
    if os.path.exists('/workspace/UPLOAD_OK'): break
    time.sleep(10)
" 
[ -f UPLOAD_OK ] || fail "upload not done 40 min after setup"
sha256sum -c upload.sha256 > logs/upload_check.log 2>&1 || fail "upload sha256 check failed"
log "UPLOAD CHECK OK $(grep -c ': OK' logs/upload_check.log) files"
source /workspace/kimienv/bin/activate
python kimi_enc_probe.py mf_adress2020.csv out/p26_kimi_adress2020 ad > logs/kimi_adress2020_enc.log 2>&1 || fail "extraction failed: $(tail -3 logs/kimi_adress2020_enc.log | tr '\n' ' ' | cut -c1-300)"
log "EXTRACT DONE: $(grep -h '^RESULT\|^STATES' logs/kimi_adress2020_enc.log | tr '\n' ' ')"
pip freeze > logs/pip_freeze.txt 2>&1
python -c "import torch,transformers,flash_attn,sys;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,'flash_attn',flash_attn.__version__,'py',sys.version.split()[0],'gpu',torch.cuda.get_device_name(0))" > logs/env.txt 2>&1
ls /workspace/hf_cache/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/ >> logs/env.txt
git -C /workspace/kimi_src rev-parse HEAD >> logs/env.txt
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
