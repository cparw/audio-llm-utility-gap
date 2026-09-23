#!/bin/bash
# usage: run_fold.sh FOLD   (nohup)
K=$1; cd /workspace; L=/workspace/logs/DRIVER.log
export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
while [ ! -f /workspace/FETCH_DONE ] || [ ! -f /workspace/PIP_DONE ] || [ ! -f /workspace/HF_DONE ]; do sleep 10; done
if [ ! -f /workspace/BUILD_OK ]; then
  python3 /workspace/build_windows.py > /workspace/logs/build.log 2>&1
  cat /workspace/logs/build.log | head -4 >> $L
  if grep -q "built 275 / 275   failures 0" /workspace/logs/build.log && grep -q "n within 0.05s 275/275" /workspace/logs/build.log; then
     (cd /workspace/edaicfull/cut && sha256sum *.wav) > /workspace/edaicfull/cut_sha256.txt; touch /workspace/BUILD_OK; echo "BUILD_OK $(date -u +%H:%M:%S)" >> $L
  else echo "BUILD_FAIL $(date -u +%H:%M:%S)" >> $L; exit 1; fi
fi
python3 /workspace/p23/check_folds.py >> $L 2>&1
echo "WHOLE_START fold $K $(date -u +%H:%M:%S)" >> $L
python3 /workspace/sft_whole.py /workspace/p23/mf_whole.csv /workspace/p23/edaic_groupkfold5_pod.csv $K whole /workspace/out >> /workspace/logs/whole_fold$K.log 2>&1
RC=$?; echo "WHOLE_EXIT $RC fold $K $(date -u +%H:%M:%S)" >> $L
[ -f /workspace/out/FT_whole_fold$K.RESULT ] || exit 1
touch /workspace/WHOLE_DONE
# CONTROL (300 s default processing) only if the estimate finishes before 17:30Z (hard stop 17:45Z)
GO=$(python3 - <<PY
import json, datetime, csv
m = json.load(open("/workspace/out/FT_whole_fold$K.json" if False else "/workspace/out/FT_whole_fold${K}_oof.json"))
d = [float(r["dur_s"]) for r in csv.DictReader(open("/workspace/out/FT_whole_fold${K}_oof.csv"))]
rows = [r for r in csv.DictReader(open("/workspace/edaicfull/build_report.csv"))]
w = [float(r["win_s"]) for r in rows]; ratio = sum(min(x, 300) for x in w) / sum(w)
est = (m["train_minutes"] + m["eval_minutes"]) * ratio * 1.15 + 3
now = datetime.datetime.utcnow(); end = now + datetime.timedelta(minutes=est)
print("1" if end.hour * 60 + end.minute < 17 * 60 + 30 and end.date() == now.date() else "0", round(est, 1))
PY
)
echo "CONTROL_DECISION $GO $(date -u +%H:%M:%S)" >> $L
if [ "${GO%% *}" = "1" ]; then
  echo "CTRL_START fold $K $(date -u +%H:%M:%S)" >> $L
  python3 /workspace/sft_whole.py /workspace/p23/mf_whole.csv /workspace/p23/edaic_groupkfold5_pod.csv $K ctrl300 /workspace/out >> /workspace/logs/ctrl_fold$K.log 2>&1
  echo "CTRL_EXIT $? fold $K $(date -u +%H:%M:%S)" >> $L
fi
touch /workspace/ALL_DONE
