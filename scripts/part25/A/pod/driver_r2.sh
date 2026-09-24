#!/bin/bash
# PART25 A retry driver.  usage: bash driver_r2.sh DATASET COND(pd|ad)
# 1 wait for setup + data; 2 sha256 of every wav vs the Mac list; 3 extract with the release extract_probe_layers.py
# (unchanged, sha 745f451e...) under transformers 5.5.4; 4 zero-shot p_yes vs the saved omni_final file, clip by clip;
# 5 the original Table 1 script nested_repeats_all.py on the encoder states (pod sklearn GroupKFold, as run on 18 Sep) and,
# alongside, p25a_nested5.py with the release fold files, saving the per-repeat out-of-fold vectors; 6 pod-side check.
DS=$1; COND=$2
cd /workspace; mkdir -p logs out
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-$(( $(nproc) - 2 ))}
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START $DS $COND NJ=$P25_NJ"
for i in $(seq 1 240); do [ -f PIP_DONE ] && [ -f HF_DONE ] && [ -f DATA_OK ] && break; sleep 15; done
[ -f PIP_DONE ] && [ -f HF_DONE ] && [ -f DATA_OK ] || fail "setup or data not ready after 60 min"
( cd data/$DS && sha256sum -c ../files.sha256 ) > logs/data_check.log 2>&1 || fail "audio sha256 check failed"
log "DATA CHECK OK $(grep -c ': OK' logs/data_check.log) files"
python3 extract_probe_layers.py data/mf_${DS}.csv out/r2_omni_${DS} $COND "Qwen/Qwen2.5-Omni-7B" > logs/extract_${DS}.log 2>&1 || fail "extract failed: $(grep -m1 -E 'Error' logs/extract_${DS}.log | cut -c1-200)"
log "EXTRACT DONE: $(grep RESULT logs/extract_${DS}.log | tail -1)"
python3 zs_compare.py out/r2_omni_${DS}_zeroshot_scores.csv saved/omni_${DS}_zeroshot_scores.csv out/r2_${DS}_zs_compare.json > logs/zs_compare.log 2>&1
log "$(cat logs/zs_compare.log | cut -c1-300)"
python3 - <<PY || fail "enc-only npz failed"
import numpy as np
z = np.load("out/r2_omni_${DS}_states.npz", allow_pickle=True)
np.savez_compressed("out/r2_omni_${DS}_encstates.npz", **{k: z[k] for k in ("enc", "label", "spk", "p_yes", "name")})
z2 = np.load("out/r2_omni_${DS}_encstates.npz", allow_pickle=True)
assert all(np.array_equal(z[k], z2[k]) for k in ("enc", "label", "spk", "p_yes", "name"))
print("ENCSTATES OK", z2["enc"].shape)
PY
mkdir -p states_full; mv out/r2_omni_${DS}_states.npz states_full/
( python3 nested_repeats_all.py out/r2_omni_${DS}_encstates.npz out/r2orig_${DS} 5 > logs/orig_repeats_${DS}.log 2>&1; echo "ORIG_RC $?" >> logs/orig_repeats_${DS}.log ) &
OP=$!
python3 p25a_nested5.py out/r2_omni_${DS}_encstates.npz $DS folds out/r2_${DS} > logs/probe_${DS}.log 2>&1 || fail "probe failed"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe_${DS}.log)"
wait $OP
log "ORIGINAL SCRIPT: $(grep -E '^enc:' logs/orig_repeats_${DS}.log) $(grep ORIG_RC logs/orig_repeats_${DS}.log)"
python3 p25a_podverify.py out/r2_omni_${DS}_encstates.npz $DS folds out/r2_${DS} > logs/podverify_${DS}.log 2>&1 || log "podverify crashed"
log "PODVERIFY: $(grep PODVERIFY logs/podverify_${DS}.log)"
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
