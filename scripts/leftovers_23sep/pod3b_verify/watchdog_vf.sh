#!/bin/bash
# PART24 watchdog. Runs on the Mac, independent of any other job. Every 3 min, for each pod in pods.txt
# ("name id ip port costPerHr"):
#   ALL_DONE on the pod           -> tar out/, logs/, driver files over ssh into pull/<name>.tar, confirm the file count
#                                    matches the pod, extract, then DELETE the pod (REST) and log it.
#   GPU 0% and no python >= 15 min -> pull the same way, delete, log "IDLE KILL".
#   GPU 0% and no log write >= 20 min (python present, hung) -> pull, delete, log "STALL KILL".
#   ssh unreachable >= 30 min while RUNNING -> REST stop (volume kept, GPU billing stops), log "UNREACHABLE STOP".
#   pull fails 4 times in a row when a kill is due -> REST stop (volume kept), log "PULL FAILED, STOPPED".
#   pod older than 4 h (first seen) -> pull, delete, log "MAX_AGE KILL".
#   balance < $4 -> pull every live pod (best effort), then delete them all.
# Exits when pods.txt has no live pods.   usage: bash watchdog24.sh [once]
R="<local data dir>/leftovers_23sep/verify/pod3b_vf"
PODS="$R/pods.txt"; PULL="$R/pull"; WD="$R/wd"; mkdir -p "$PULL" "$WD"
KEY=$(tr -d '\n\r ' < ~/.runpod_key)
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
ONCE=${1:-}
IDLE_S=900; STALL_S=1200; UNREACH_S=1800; MAXAGE_S=14400; BAL_MIN=4
log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
if [ -z "$ONCE" ]; then caffeinate -i -s -w $$ >/dev/null 2>&1 & log "watchdog start pid $$ (caffeinate keeps the Mac awake while this runs)"; fi
balance(){ curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" https://api.runpod.io/graphql \
  -d '{"query":"query { myself { clientBalance currentSpendPerHr } }"}' | /usr/local/bin/python3 -c "import json,sys;m=json.load(sys.stdin)['data']['myself'];print('%.4f %.3f'%(m['clientBalance'],m['currentSpendPerHr']))" 2>/dev/null; }
status_of(){ # id -> desiredStatus or GONE or ERR
  local c; c=$(curl -s -m 30 -o "$WD/st.json" -w "%{http_code}" -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$1")
  if [ "$c" = "404" ]; then echo GONE; elif [ "$c" = "200" ]; then /usr/local/bin/python3 -c "import json;d=json.load(open('$WD/st.json'));print(d.get('desiredStatus') or 'NONE')" 2>/dev/null || echo ERR; else echo "ERR$c"; fi; }
delete_pod(){ # name id reason
  local c s; c=$(curl -s -m 30 -X DELETE -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$2")
  sleep 3; s=$(status_of "$2"); log "$1 DELETE ($3) HTTP $c -> status now $s"
  if [ "$s" = "GONE" ] || [ "$c" = "200" ] || [ "$c" = "204" ]; then echo "$3 $(date -u +%H:%M:%SZ)" > "$WD/$1.gone"; return 0; fi; return 1; }
stop_pod(){ # name id reason
  local c; c=$(curl -s -m 30 -X POST -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$2/stop")
  log "$1 STOP ($3) HTTP $c (volume kept, GPU billing stops)"; echo "STOPPED $3 $(date -u +%H:%M:%SZ)" > "$WD/$1.gone"; }
pull_pod(){ # name ip port -> 0 when the tar holds exactly the pod's file list and the result csv hashes match
  local n ip port cnt got tarf; n=$1; ip=$2; port=$3; tarf="$PULL/$n.tar"
  cnt=$(ssh -n $SSHO -p $port root@$ip 'cd /workspace && { find out logs p23 smoke_out -type f 2>/dev/null; ls -1 *.sh *.py 2>/dev/null;
      for m in HF_DONE PIP_DONE FETCH_DONE BUILD_OK SMOKE_START SMOKE_OK SMOKE_FAIL SEED0_DONE SEED1_DONE ALL_DONE FAILED; do [ -f $m ] && echo $m; done;
      for f in edaicfull/windows_full.csv edaicfull/build_report.csv edaicfull/cut_sha256.txt edaicfull/tarball_sha256.tsv; do [ -f $f ] && echo $f; done; } > /root/pull.list;
      wc -l < /root/pull.list' 2>/dev/null | tr -d ' \r')
  [ -n "$cnt" ] && [ "$cnt" -gt 0 ] 2>/dev/null || { log "$n PULL list failed"; return 1; }
  ssh -n $SSHO -p $port root@$ip 'cd /workspace && tar -cf - -T /root/pull.list' > "$tarf.part" 2> "$WD/$n.tar.err" || { log "$n PULL tar stream failed: $(head -c 200 "$WD/$n.tar.err")"; return 1; }
  got=$(tar -tf "$tarf.part" 2>/dev/null | grep -vc '/$')
  if [ "$got" != "$cnt" ]; then log "$n PULL COUNT MISMATCH pod $cnt tar $got"; return 1; fi
  mv -f "$tarf.part" "$tarf"; rm -rf "$PULL/$n"; mkdir -p "$PULL/$n"; tar -xf "$tarf" -C "$PULL/$n" || { log "$n PULL extract failed"; return 1; }
  local ex; ex=$(find "$PULL/$n" -type f | wc -l | tr -d ' ')
  ssh -n $SSHO -p $port root@$ip 'cd /workspace && sha256sum out/*.csv out/*.json 2>/dev/null' > "$WD/$n.remote.sha" 2>/dev/null
  local bad=0; while read -r h p <&4; do [ -z "$p" ] && continue; l=$(shasum -a 256 "$PULL/$n/$p" 2>/dev/null | cut -d' ' -f1); [ "$l" = "$h" ] || bad=$((bad+1)); done 4< "$WD/$n.remote.sha"
  if [ $bad -ne 0 ]; then log "$n PULL SHA MISMATCH on $bad result files"; return 1; fi
  log "$n PULLED $tarf files pod $cnt tar $got extracted $ex, result csv/sidecar sha256 match ($(wc -l < "$WD/$n.remote.sha" | tr -d ' ') files), $(du -h "$tarf" | cut -f1)"
  return 0; }
kill_pod(){ # name id ip port reason : pull then delete; 4 failed pulls -> stop
  if pull_pod "$1" "$3" "$4"; then rm -f "$WD/$1.pullfail"; delete_pod "$1" "$2" "$5"; return; fi
  local f; f=$(( $(cat "$WD/$1.pullfail" 2>/dev/null || echo 0) + 1 )); echo $f > "$WD/$1.pullfail"
  log "$1 pull failed ($f/4) before $5"
  [ $f -ge 4 ] && stop_pod "$1" "$2" "PULL FAILED, STOPPED, $5"; }
while true; do
  now=$(date +%s); live=0
  B=$(balance); bal=${B%% *}; log "balance/spend ${B:-unavailable}"
  low=0; if [ -n "$bal" ] && /usr/local/bin/python3 -c "import sys;sys.exit(0 if float('$bal')<$BAL_MIN else 1)"; then low=1; log "BALANCE BELOW \$$BAL_MIN ($bal): pulling and deleting every pod"; fi
  while read -r name id ip port cost <&3; do
    [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
    [ -f "$WD/$name.gone" ] && continue
    [ -f "$WD/$name.first_seen" ] || echo $now > "$WD/$name.first_seen"
    s=$(status_of "$id")
    if [ "$s" = "GONE" ]; then log "$name $id no longer exists (deleted elsewhere)"; echo "GONE_EXTERNAL $(date -u +%H:%M:%SZ)" > "$WD/$name.gone"; continue; fi
    if [ "$s" = "EXITED" ] || [ "$s" = "TERMINATED" ]; then log "$name $id status $s (stopped outside the watchdog; volume kept, not live)"; echo "EXITED_EXTERNAL $(date -u +%H:%M:%SZ)" > "$WD/$name.gone"; continue; fi
    live=$((live+1))
    out=$(ssh -n $SSHO -p $port root@$ip 'cd /workspace 2>/dev/null; a=0; [ -f ALL_DONE ] && a=1;
      g=0;
      py=$(ps -eo args | grep -E "^(/[^ ]*/)?python[0-9.]*( |$)" | grep -v -e jupyter | wc -l);
      lm=$(find /workspace/logs -type f -printf "%T@\n" 2>/dev/null | sort -n | tail -1 | cut -d. -f1); nw=$(date +%s);
      f=-; [ -f FAILED ] && f=$(cat FAILED);
      mk=$(ls FETCH_DONE BUILD_OK SMOKE_OK SMOKE_FAIL SEED0_DONE SEED1_DONE ALL_DONE 2>/dev/null | tr "\n" ",");
      echo "ALL=$a GPU=${g:-NA} PY=$py LOGAGE=$((nw-${lm:-$nw})) FAILED=$f MK=$mk | $(tail -1 logs/DRIVER.log 2>/dev/null | cut -c1-150)"' 2>/dev/null | tr '\n' ' ')
    if [ -z "$out" ]; then
      [ -f "$WD/$name.unreach_since" ] || echo $now > "$WD/$name.unreach_since"; u=$(( now - $(cat "$WD/$name.unreach_since") ))
      log "$name ssh unreachable for ${u}s (REST status $s)"
      if [ $low = 1 ]; then delete_pod "$name" "$id" "BALANCE LOW, unreachable, no pull"; live=$((live-1)); continue; fi
      if [ $u -ge $UNREACH_S ] && [ "$s" = "RUNNING" ]; then stop_pod "$name" "$id" "UNREACHABLE STOP after ${u}s"; live=$((live-1)); fi
      continue
    fi
    rm -f "$WD/$name.unreach_since"; log "$name $out"
    ALL=$(echo "$out" | sed -n 's/.*ALL=\([01]\).*/\1/p'); GPU=$(echo "$out" | sed -n 's/.*GPU=\([0-9NA]*\).*/\1/p')
    PY=$(echo "$out" | sed -n 's/.* PY=\([0-9]*\).*/\1/p'); LA=$(echo "$out" | sed -n 's/.*LOGAGE=\([0-9-]*\).*/\1/p')
    age=$(( now - $(cat "$WD/$name.first_seen") ))
    if [ $low = 1 ]; then
      pull_pod "$name" "$ip" "$port" || pull_pod "$name" "$ip" "$port" || log "$name pull failed under low balance, deleting anyway"
      delete_pod "$name" "$id" "BALANCE LOW ($bal)"; live=$((live-1)); continue
    fi
    if [ "$ALL" = "1" ]; then kill_pod "$name" "$id" "$ip" "$port" "ALL_DONE"; [ -f "$WD/$name.gone" ] && live=$((live-1)); continue; fi
    if [ "$GPU" = "0" ] && [ "$PY" = "0" ]; then
      [ -f "$WD/$name.idle_since" ] || echo $now > "$WD/$name.idle_since"; i=$(( now - $(cat "$WD/$name.idle_since") ))
      log "$name idle (GPU 0%, no python) for ${i}s"
      if [ $i -ge $IDLE_S ]; then kill_pod "$name" "$id" "$ip" "$port" "IDLE KILL after ${i}s"; [ -f "$WD/$name.gone" ] && live=$((live-1)); fi
      continue
    fi
    rm -f "$WD/$name.idle_since"
    if [ "$GPU" = "0" ] && [ -n "$LA" ] && [ "$LA" -ge $STALL_S ] 2>/dev/null; then
      kill_pod "$name" "$id" "$ip" "$port" "STALL KILL (GPU 0%, python present, no log write for ${LA}s)"; [ -f "$WD/$name.gone" ] && live=$((live-1)); continue; fi
    if [ $age -ge $MAXAGE_S ]; then kill_pod "$name" "$id" "$ip" "$port" "MAX_AGE KILL after ${age}s"; [ -f "$WD/$name.gone" ] && live=$((live-1)); continue; fi
  done 3< "$PODS"
  log "live pods: $live"
  if [ $live -le 0 ]; then log "no live pods in pods.txt, watchdog exit"; exit 0; fi
  [ -n "$ONCE" ] && { log "once mode, exit after one iteration"; exit 0; }
  sl=$(( 180 - ($(date +%s) - now) )); [ $sl -lt 20 ] && sl=20; sleep $sl
done
