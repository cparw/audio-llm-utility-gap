#!/bin/bash
# PART24 smoke: runs the real sft_whole24.py end to end on synthetic clips (1 epoch, CK=2) in the fetch window,
# then re-runs it to exercise the resume path, then checks the outputs. Writes SMOKE_OK or SMOKE_FAIL. ~3 min.
cd /workspace; L=/workspace/logs/DRIVER.log; SL=/workspace/logs/smoke24.log
t(){ date -u +%H:%M:%S; }
while [ ! -f /workspace/PIP_DONE ] || [ ! -f /workspace/HF_DONE ]; do sleep 5; done
touch /workspace/SMOKE_START; echo "SMOKE_START $(t)" >> $L
export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 GC=on CK=2 EPOCHS_OVERRIDE=1
ok=1
python3 /workspace/smoke_data.py >> $SL 2>&1 || ok=0
[ $ok = 1 ] && { timeout 600 python3 /workspace/sft_whole24.py /workspace/smoke/mf.csv /workspace/smoke/folds.csv 0 0 /workspace/smoke_out >> $SL 2>&1 || ok=0; }
[ $ok = 1 ] && cp /workspace/smoke_out/FT_whole_s0_fold0_oof.csv /workspace/smoke_out/run1_oof.csv
[ $ok = 1 ] && { timeout 600 python3 /workspace/sft_whole24.py /workspace/smoke/mf.csv /workspace/smoke/folds.csv 0 0 /workspace/smoke_out >> $SL 2>&1 || ok=0; }
if [ $ok = 1 ]; then
  python3 - >> $SL 2>&1 <<'PY' || ok=0
import csv, json, numpy as np
r = list(csv.DictReader(open("/workspace/smoke_out/FT_whole_s0_fold0_oof.csv"))); r1 = list(csv.DictReader(open("/workspace/smoke_out/run1_oof.csv")))
m = json.load(open("/workspace/smoke_out/FT_whole_s0_fold0_oof.sidecar.json"))
assert [x["pid"] for x in r] == ["s02", "s03", "s04", "s05"], [x["pid"] for x in r]
assert r == r1, "resume rewrite changed rows"
tok = {x["pid"]: int(x["audio_tok"]) for x in r}; assert tok["s02"] == 22500 and tok["s03"] == 500, tok
for x in r:
    ps, pm = float(x["p_yes_set"]), float(x["answer_mass"]); assert 0 <= ps <= 1 and 0 <= pm <= 1 and 0 <= float(x["p_yes"]) <= 1
    assert x["seed"] == "0" and x["fold"] == "0"
for k in ["seed", "order_sha256_per_epoch", "hf_snapshot", "command", "versions", "n", "n_speakers", "bootstrap_rule_for_pooled_report", "median_answer_mass", "yes_ids", "no_ids", "id_text"]:
    assert k in m, k
assert m["yes_ids"] == [7414, 9454, 9693, 9834, 14004] and m["no_ids"] == [902, 2152, 2308, 2753, 8996]
assert m["restarts_from_checkpoint"] == 1 and m["epochs"] == 1 and m["gradient_checkpointing_frozen_lm"] is True
st = json.load(open("/workspace/smoke_out/whole_s0_fold0_state.json")); assert len(st["steps"]) == 6, len(st["steps"])
g = np.random.RandomState(0); o = g.permutation(np.array([0, 5, 6, 7, 8, 9]))
assert st["orders_pid"][0] == [f"s{i+1:02d}" for i in o], (st["orders_pid"], o)
print("SMOKE_CHECKS_OK id_text", m["id_text"], "median_mass", m["median_answer_mass"], "peak", m["peak_alloc_gb_train"], "GB", "gpu", m["versions"]["gpu"])
PY
fi
if [ $ok = 1 ]; then touch /workspace/SMOKE_OK; echo "SMOKE_OK $(grep SMOKE_CHECKS_OK $SL | cut -c1-300) $(t)" >> $L
else touch /workspace/SMOKE_FAIL; echo "SMOKE_FAIL (see logs/smoke24.log) $(t)" >> $L; fi
