#!/bin/bash
# PART 26 Kimi-Audio E-DAIC GPU pod driver.
# 1 wait for the Kimi setup, the tarball and the small Mac upload (manifest, full-file sha256 list, reference zero-shot);
# 2 tarball sha256 must equal drive_data/checksums.txt (83ea8873...); extract; every one of the 275 wavs must equal the
#   Mac dcaps_proc file by sha256; 3 kimi_enc_probe.py (release kimi_probe.py plus the new Whisper encoder hooks; it cuts
#   x[:16000*30] itself, cond mdd, bf16); 4 enc-only npz (byte-identical arrays); 5 p_yes check against the canonical
#   Kimi E-DAIC zero-shot file (overnight2/part3); ALL_DONE.
cd /workspace; mkdir -p logs out
export HF_HOME=/workspace/hf_cache PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; ( cd out && sha256sum * > ../out_sha256.txt 2>/dev/null ); touch ALL_DONE; exit 1; }
log "DRIVER START kimi edaic encoder states"
for i in $(seq 1 240); do [ -f KIMI_SETUP_DONE ] && grep -qE "GDOWN_DONE|GDOWN_FAILED" logs/gdown_edaic.log 2>/dev/null && [ -f data/UPLOAD_OK ] && break; sleep 15; done
[ -f KIMI_SETUP_DONE ] && [ -f KIMI_IMPORT_OK ] || fail "Kimi setup not done or import failed: $(tail -3 logs/KIMI_SETUP.log | tr '\n' ' ')"
[ -f data/UPLOAD_OK ] || fail "Mac upload not done after 60 min"
source /workspace/kimienv/bin/activate
log "KIMI SETUP OK: $(grep -E '^torch|^flash_attn|model at' logs/KIMI_SETUP.log | tr '\n' ' ') src $(git -C kimi_src rev-parse HEAD)"
grep -q GDOWN_DONE logs/gdown_edaic.log || fail "gdown failed: $(tail -3 logs/gdown_edaic.log | tr '\n' ' ')"
TS=$(cut -d' ' -f1 data/tarball.sha256)
[ "$TS" = "83ea8873a27a3ffc5babcc6315028d232e42109264eb90e8010d845a5a5160e8" ] || fail "tarball sha256 $TS differs from drive_data/checksums.txt"
log "TARBALL SHA OK $TS"
mkdir -p data/edaic_x && tar -xzf data/E-DAIC_audio.tar.gz -C data/edaic_x --no-same-owner > logs/untar.log 2>&1 || fail "untar failed"
DP=$(find /workspace/data/edaic_x -type d -name dcaps_proc | head -1); [ -n "$DP" ] || fail "no dcaps_proc dir in the tarball"
log "dcaps_proc at $DP: $(ls $DP | wc -l) files"
( cd "$DP" && sha256sum -c /workspace/data/dcaps_proc_full.sha256 ) > logs/data_check.log 2>&1 || fail "wav sha256 check against the Mac files failed"
log "DATA CHECK OK $(grep -c ': OK' logs/data_check.log) files equal the Mac dcaps_proc files"
ln -sfn "$DP" /workspace/data/dcaps
python - <<'PY' || fail "manifest check failed"
import csv, os
rows = list(csv.DictReader(open("/workspace/data/mf_edaic.csv")))
assert len(rows) == 275 and all(os.path.exists(r["path"]) for r in rows)
print(len(rows))
PY
cp data/mf_edaic.csv out/p26_kimi_mf_edaic.csv
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> $L
if [ ! -f out/p26_kimi_edaic_states.npz ]; then
  python kimi_enc_probe.py data/mf_edaic.csv out/p26_kimi_edaic mdd > logs/extract_edaic.log 2>&1 \
    || fail "extract failed: $(grep -m1 -E 'Error|error' logs/extract_edaic.log | cut -c1-200)"
fi
log "EXTRACT DONE: $(grep RESULT logs/extract_edaic.log | tail -1) $(grep STATES logs/extract_edaic.log | tail -1)"
python - <<'PY' >> $L 2>&1 || fail "enc-only npz or p_yes check failed"
import numpy as np, csv, json
z = np.load("out/p26_kimi_edaic_states.npz", allow_pickle=True)
K = ("enc", "label", "spk", "p_yes", "name")
np.savez_compressed("out/p26_kimi_edaic_encstates.npz", **{k: z[k] for k in K})
z2 = np.load("out/p26_kimi_edaic_encstates.npz", allow_pickle=True)
assert all(np.array_equal(z[k], z2[k]) for k in K)
print("ENCSTATES OK", z2["enc"].shape, z2["enc"].dtype, "finite", bool(np.isfinite(z2["enc"]).all()), "enc_frames", sorted(set(z["enc_frames"].tolist())))
ref = list(csv.DictReader(open("data/ref_kimi_edaic_zeroshot_scores.csv")))
names = [str(v) for v in z["name"]]
assert names == [r["clip"] for r in ref], "clip order differs from the Kimi zero-shot file"
assert [int(v) for v in z["label"]] == [int(r["label"]) for r in ref], "labels differ"
assert [str(v) for v in z["spk"]] == [r["speaker"] for r in ref], "speakers differ"
d = np.abs(z["p_yes"].astype(float) - np.array([float(r["p_yes"]) for r in ref]))
from sklearn.metrics import roc_auc_score
res = {"n": len(d), "max_abs_p_yes_diff": float(d.max()), "median_abs_p_yes_diff": float(np.median(d)),
       "n_over_1e-3": int((d > 1e-3).sum()), "n_over_1e-2": int((d > 1e-2).sum()),
       "auc_rerun": float(roc_auc_score(z["label"].astype(int), z["p_yes"].astype(float))),
       "auc_ref": float(roc_auc_score([int(r["label"]) for r in ref], [float(r["p_yes"]) for r in ref]))}
json.dump(res, open("out/p26_kimi_pyes_check.json", "w"), indent=1)
print("PYES_CHECK", json.dumps(res))
PY
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
