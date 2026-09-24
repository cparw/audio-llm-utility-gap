#!/bin/bash
# PART26 GPU pod driver, Kimi-Audio MDVR-KCL encoder states.
# 1 sha256 check of every uploaded file against the Mac list; 2 setup_kimi.sh (PART26 copy of the edaic_conflict_new Kimi setup,
# snapshot pinned); 3 kimi_enc_probe.py (kimi_probe.py plus Whisper encoder hooks) on the 37 KCL clips, 30 s window, pd prompt;
# 4 environment record; 5 sha256 of out/; ALL_DONE.
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONNOUSERSITE=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; ( cd out && sha256sum * > ../out_sha256.txt 2>/dev/null ); touch ALL_DONE; exit 1; }
log "DRIVER START kimi kcl enc"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> $L 2>&1
sha256sum -c upload.sha256 > logs/upload_check.log 2>&1 || fail "upload sha256 check failed"
log "UPLOAD CHECK OK $(grep -c ': OK' logs/upload_check.log) files"
bash setup_kimi.sh
[ -f KIMI_SETUP_DONE ] || fail "setup did not finish"
grep -q KIMI_IMPORT_OK logs/KIMI_SETUP.log || fail "kimia_infer import failed"
log "SETUP DONE $(grep -m1 KIMI_SRC_COMMIT logs/KIMI_SETUP.log | tr -d '+')"
source /workspace/kimienv/bin/activate
python kimi_enc_probe.py mf_kcl.csv out/p26_kimi_kcl pd > logs/kimi_kcl_enc.log 2>&1 || fail "extraction failed: $(tail -3 logs/kimi_kcl_enc.log | tr '\n' ' ' | cut -c1-300)"
log "EXTRACT DONE: $(grep -h '^RESULT\|^STATES' logs/kimi_kcl_enc.log | tr '\n' ' ')"
pip freeze > logs/pip_freeze.txt 2>&1
python -c "import torch,transformers,flash_attn,sys;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,'flash_attn',flash_attn.__version__,'py',sys.version.split()[0],'gpu',torch.cuda.get_device_name(0))" > logs/env.txt 2>&1
ls /workspace/hf_cache/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/ >> logs/env.txt
git -C /workspace/kimi_src rev-parse HEAD >> logs/env.txt
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
