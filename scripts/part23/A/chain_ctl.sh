#!/bin/bash
cd /workspace/p23a
while ! grep -q "EXTRACTION DONE" ctl300/ctl300_extract.log 2>/dev/null; do
  if ! pgrep -f "ctl300/ctl300 mdd" >/dev/null && ! grep -q "EXTRACTION DONE" ctl300/ctl300_extract.log; then echo "CTL EXTRACT NOT RUNNING $(date -u +%T)"; exit 3; fi
  sleep 15
done
echo "ctl extract done $(date -u +%T)"
for k in enc proj llm ans; do CTL300=1 nohup python3 p23a_probe.py probe $k > probe_ctl300_$k.log 2>&1 & done
wait
echo "ctl probes done $(date -u +%T)"
python3 p23a_report.py ctl > report_ctl.log 2>&1; echo "ctl report exit $? $(date -u +%T)"
