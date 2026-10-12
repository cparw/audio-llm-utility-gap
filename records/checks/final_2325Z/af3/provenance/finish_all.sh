set -x; cd /workspace; mkdir -p data out logs hf_cache
export PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/workspace/hf_cache
pip install --break-system-packages -q transformers==5.5.4 accelerate librosa soundfile scikit-learn pandas gdown joblib hf_transfer peft > logs/pip.log 2>&1
echo PIP_DONE >> logs/DRIVER.log
( python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct', allow_patterns=['*.json','*.txt','*.safetensors','*.model','*.py'])
print('Q3O OK')" > logs/hf_q3o.log 2>&1; echo Q3O_DL_DONE >> logs/DRIVER.log ) &
( python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/audio-flamingo-3-hf')
print('AF3 OK')" > logs/hf_af3.log 2>&1; echo AF3_DL_DONE >> logs/DRIVER.log ) &
( cd data && gdown <E-DAIC archive id> -O E-DAIC_audio.tar.gz > /workspace/logs/dl_ed.log 2>&1 && tar -xzf E-DAIC_audio.tar.gz && echo EDAIC_DL_DONE >> /workspace/logs/DRIVER.log ) &
wait
until [ -f data/clips_for_af3/pitt468_manifest.csv ]; do [ -f data/clips_for_af3.zip ] && unzip -q -o data/clips_for_af3.zip -d data/ 2>/dev/null; sleep 15; done
python3 - <<PY
import csv, os, collections
D="/workspace/data"
def index(root):
    ix=collections.defaultdict(list)
    for d,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".wav"): ix[f].append(os.path.join(d,f))
    return ix
ix=index(f"{D}/clips_for_af3/pitt468"); rows=[]
for r in csv.DictReader(open(f"{D}/clips_for_af3/pitt468_manifest.csv")):
    b=os.path.basename(r["clip_path"]); c=ix.get(b)
    if c: rows.append((c[0], r["label"], r["speaker_id"], "conflict" if b.startswith("conflict") else "agreement"))
with open("/workspace/mf_pitt.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["path","label","speaker","set"]); w.writerows(rows)
print("mf_pitt.csv", len(rows), "speakers", len({r[2] for r in rows}))
ix=index(f"{D}/dcaps_proc"); rows=[]
for r in csv.DictReader(open(f"{D}/dcaps.csv")):
    c=ix.get(os.path.basename(r["filepath"]))
    if c: rows.append((c[0], r["label"], r["speaker_id"]))
with open("/workspace/mf_edaic.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["path","label","speaker"]); w.writerows(rows)
print("mf_edaic.csv", len(rows))
PY
echo "MANIFESTS DONE $(date +%H:%M)" >> logs/DRIVER.log
Q3="Qwen/Qwen3-Omni-30B-A3B-Instruct"
# ---- GPU steps, one at a time, highest value first ----
for job in "pitt ad" "edaic mdd"; do
  set -- $job
  if python3 extract_probe_layers.py mf_$1.csv out/q3o_$1 $2 "$Q3" > logs/q3o_$1_extract.log 2>&1; then
    grep -hE "^RESULT" logs/q3o_$1_extract.log >> logs/DRIVER.log; echo "q3o $1 extracted $(date +%H:%M)" >> logs/DRIVER.log
    ( python3 curves_from_states.py out/q3o_$1_states.npz out/q3o_$1 > logs/q3o_$1_curves.log 2>&1
      python3 nested_repeats_all.py out/q3o_$1_states.npz out/q3o_$1 5 > logs/q3o_$1_repeats.log 2>&1
      echo "q3o $1 probes DONE $(date +%H:%M)" >> logs/DRIVER.log ) &
  else echo "q3o $1 EXTRACT FAILED $(date +%H:%M)" >> logs/DRIVER.log; fi
done
python3 af3_score.py mf_pitt.csv out/af3_pitt.csv ad > logs/af3_pitt.log 2>&1
grep -h RESULT logs/af3_pitt.log >> logs/DRIVER.log; echo "AF3 pitt done $(date +%H:%M)" >> logs/DRIVER.log
python3 af3_score.py mf_edaic.csv out/af3_edaic.csv mdd > logs/af3_edaic.log 2>&1
grep -h RESULT logs/af3_edaic.log >> logs/DRIVER.log; echo "AF3 edaic done $(date +%H:%M)" >> logs/DRIVER.log
python3 sft_projector.py mf_pitt.csv out/q3osft_pitt ad "$Q3" > logs/q3osft_pitt.log 2>&1
grep -h RESULT logs/q3osft_pitt.log >> logs/DRIVER.log; echo "q3o pitt SFT done $(date +%H:%M)" >> logs/DRIVER.log
wait
echo "ALL DONE $(date +%H:%M)" >> logs/DRIVER.log
