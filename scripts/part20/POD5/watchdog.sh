#!/bin/bash
date -u +%FT%TZ
ps aux | grep -E "extract_p20|sft_p20|nested_p20|direction_pod5" | grep -v grep | awk '{print $2, $3"%cpu", $11, $12, $13}' | head
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
for f in /workspace/p20/work/*.log; do echo "$f $(wc -c < $f) bytes | $(tail -1 $f | cut -c1-200)"; done
df -h /workspace /root | tail -2
