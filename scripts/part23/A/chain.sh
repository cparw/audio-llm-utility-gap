#!/bin/bash
# PART 23 A chain: wait for extraction, then items 1-2 report, nested probes (4 streams in parallel), per-layer, best-stage, item 3 report.
cd /workspace/p23a
while [ ! -f A_extract.RESULT ]; do
  if ! pgrep -f "p23a_extract.py" >/dev/null; then echo "EXTRACT NOT RUNNING and no RESULT at $(date -u +%T)"; exit 3; fi
  sleep 20
done
echo "extract done $(date -u +%T)"
python3 p23a_report.py items12 > report12.log 2>&1; echo "items12 exit $? $(date -u +%T)"
for k in enc proj llm ans; do nohup python3 p23a_probe.py probe $k > probe_$k.log 2>&1 & done
wait
echo "nested probes done $(date -u +%T)"
python3 p23a_probe.py perlayer > perlayer.log 2>&1 &
for k in enc llm ans; do python3 p23a_probe.py fixed $k > fixed_$k.log 2>&1 & done
wait
echo "perlayer+fixed done $(date -u +%T)"
python3 p23a_report.py item3 > report3.log 2>&1; echo "item3 exit $? $(date -u +%T)"
[ -f A3s_edaic_whole_beststage.RESULT ] && touch ALL_DONE && cp ALL_DONE out/ALL_DONE && echo "ALL_DONE $(date -u +%T)"
