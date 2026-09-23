#!/bin/bash
# PART24 seed 1 pre-emptive driver, one outer fold per pod.  usage: nohup bash run_fold24_s1.sh FOLD > logs/run_fold.out 2>&1 &
# Copied from run_fold24.sh (same wait, smoke gate, build, fold check, run_seed with 3 attempts and checkpoint resume,
# same env). Changed only in: seed 0 and the on-pod mass rule are dropped; this pod runs ONLY seed 1 on fold FOLD.
# Seed 1 runs pre-emptively on a separate pod, in parallel with seed 0 on the p24-ft pod of the same fold.
# After the run: adds "note" to the sidecar json and logs p_yes, answer_mass and audio_tok per test clip.
K=$1; cd /workspace; L=/workspace/logs/DRIVER.log
export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 GC=on CK=55
unset EPOCHS_OVERRIDE
t(){ date -u +%H:%M:%S; }
fail(){ echo "FAILED $1 $(t)" >> $L; echo "$1" > /workspace/FAILED; touch /workspace/ALL_DONE; echo "ALL_DONE (after failure) $(t)" >> $L; exit 1; }
echo "DRIVER_START seed1-only fold $K $(t) pod ${RUNPOD_POD_ID:-?}" >> $L
while [ ! -f /workspace/FETCH_DONE ] || [ ! -f /workspace/PIP_DONE ] || [ ! -f /workspace/HF_DONE ]; do sleep 10; done
# smoke (smoke24.sh, same pod, runs in the fetch window) gates this pod's fold run. Max wait 10 min.
n=0; while [ -f /workspace/SMOKE_START ] && [ ! -f /workspace/SMOKE_OK ] && [ ! -f /workspace/SMOKE_FAIL ] && [ $n -lt 600 ]; do sleep 5; n=$((n+5)); done
if [ -f /workspace/SMOKE_OK ]; then echo "SMOKE_OK seen $(t)" >> $L
elif [ -f /workspace/SMOKE_FAIL ]; then
  echo "SMOKE_FAIL seen, waiting up to 300 s for /workspace/SMOKE_OVERRIDE $(t)" >> $L
  n=0; while [ ! -f /workspace/SMOKE_OVERRIDE ] && [ $n -lt 300 ]; do sleep 5; n=$((n+5)); done
  [ -f /workspace/SMOKE_OVERRIDE ] && echo "SMOKE_OVERRIDE present, continuing $(t)" >> $L || fail SMOKE
else echo "SMOKE not run or no verdict after wait, continuing $(t)" >> $L; fi
if [ ! -f /workspace/BUILD_OK ]; then
  python3 /workspace/build_windows.py > /workspace/logs/build.log 2>&1
  head -4 /workspace/logs/build.log >> $L
  if grep -q "built 275 / 275   failures 0" /workspace/logs/build.log && [ "$(grep -c 'n within 0.05s 275/275' /workspace/logs/build.log)" = "2" ]; then
     (cd /workspace/edaicfull/cut && sha256sum *.wav) > /workspace/edaicfull/cut_sha256.txt
     if cmp -s /workspace/edaicfull/cut_sha256.txt /workspace/p23/cut_sha256_p23.txt; then echo "CUT_SHA identical to PART23 (275 windows) $(t)" >> $L
     else echo "CUT_SHA DIFFERS from PART23: $(diff /workspace/edaicfull/cut_sha256.txt /workspace/p23/cut_sha256_p23.txt | grep -c '^<') lines $(t)" >> $L; fi
     touch /workspace/BUILD_OK; echo "BUILD_OK $(t)" >> $L
  else fail BUILD; fi
fi
python3 /workspace/p23/check_folds.py >> $L 2>&1
run_seed(){ s=$1
  for a in 1 2 3; do
    echo "SEED${s}_START fold $K attempt $a $(t)" >> $L
    python3 /workspace/sft_whole24.py /workspace/p23/mf_whole.csv /workspace/p23/edaic_groupkfold5_pod.csv $K $s /workspace/out >> /workspace/logs/whole_s${s}_fold$K.log 2>&1
    echo "SEED${s}_EXIT $? fold $K attempt $a $(t)" >> $L
    [ -f /workspace/out/FT_whole_s${s}_fold$K.RESULT ] && return 0
    sleep 5
  done; return 1; }
run_seed 1 || fail SEED1
python3 - "$K" >> $L 2>&1 <<'PY'
import csv, json, sys, os
k = sys.argv[1]
sc = f"/workspace/out/FT_whole_s1_fold{k}_oof.sidecar.json"; oof = f"/workspace/out/FT_whole_s1_fold{k}_oof.csv"
m = json.load(open(sc))
m["note"] = "seed 1 run pre-emptively on a separate pod, in parallel with seed 0, to save time; same scripts and seed rule"
json.dump(m, open(sc + ".tmp", "w"), indent=1); os.replace(sc + ".tmp", sc)
rows = list(csv.DictReader(open(oof)))
with open(f"/workspace/logs/perclip_s1_fold{k}.log", "w") as fh:
    for r in rows:
        line = f"CLIP fold {k} seed 1 pid {r['pid']} label {r['label']} p_yes {float(r['p_yes']):.6f} answer_mass {float(r['answer_mass']):.6f} audio_tok {r['audio_tok']}"
        print(line); fh.write(line + "\n")
print(f"SIDECAR_NOTE_ADDED fold {k} n_clips {len(rows)} auc {m['fold_auc_p_yes_two_logit']} median_mass {m['median_answer_mass']}")
PY
touch /workspace/SEED1_DONE; echo "SEED1_DONE $(cat /workspace/out/FT_whole_s1_fold$K.RESULT) $(t)" >> $L
touch /workspace/ALL_DONE; echo "ALL_DONE $(t)" >> $L
