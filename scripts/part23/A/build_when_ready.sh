cd /workspace/p23a
while [ $(ls edaicfull/tmp/*.ok 2>/dev/null | wc -l) -lt 275 ]; do
  if ! pgrep -x xargs >/dev/null; then break; fi
  sleep 5
done
echo "downloads ok: $(ls edaicfull/tmp/*.ok | wc -l)"
python3 build_windows_p23a.py
