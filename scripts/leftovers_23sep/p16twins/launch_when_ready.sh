#!/bin/bash
# start twin_perlayer.py on each pod once its upload log reports a sha match
OUT="<local data dir>/leftovers_23sep/p16twins"
SC=<local data dir>/scratch/p16
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
todo="2:pitt:enc 3:pitt:llm 4:pitt:ans 6:pcgita:llm 7:pcgita:ans"
while [ -n "$todo" ]; do
  left=""
  for t in $todo; do
    k=${t%%:*}; r=${t#*:}; ds=${r%%:*}; st=${r#*:}
    if grep -q "match=yes" $SC/pod/up$k.log 2>/dev/null; then
      set -- $(awk -v n="lo-p16twins-$k" '$1==n{print $3, $4}' "$OUT/pods.txt")
      ssh -n $SSHO -p $2 root@$1 "cd /workspace; nohup bash -c 'python3 -u twin_perlayer.py $ds $st > logs/perlayer.log 2>&1; touch PL_DONE' >/dev/null 2>&1 &"
      echo "$(date -u +%H:%M:%S) started pod $k $ds $st"
    elif grep -q "match=NO" $SC/pod/up$k.log 2>/dev/null; then
      echo "$(date -u +%H:%M:%S) pod $k upload sha MISMATCH"; 
    else left="$left $t"; fi
  done
  todo=$(echo $left); [ -n "$todo" ] && sleep 10
done
echo "all launched"
