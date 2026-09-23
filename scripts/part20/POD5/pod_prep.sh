#!/bin/bash
set -e
cd /root/p20
cat parts/part_* > pitt_noinv_clips.tgz
H=$(sha256sum pitt_noinv_clips.tgz | cut -d' ' -f1); echo "tgz sha256 $H"
[ "$H" = "b995bd9afe8a45b1c08472f7333c5855767d60de23307a7881b5f000c73d8a07" ] || { echo TGZ_SHA_MISMATCH; exit 1; }
tar -xzf pitt_noinv_clips.tgz
python3 - <<'PY'
import csv, hashlib
m = list(csv.DictReader(open("/root/p20/manifest.csv"))); pm = list(csv.DictReader(open("/root/p20/scripts/pod_manifest_noinv.csv")))
bad = [r["clip"] for r in m if hashlib.sha256(open("/root/p20/clips/" + r["clip"], "rb").read()).hexdigest() != r["sha256"]]
assert len(m) == 468, len(m)
shaman = {r["clip"]: r["sha256"] for r in m}
bad2 = [r["path"] for r in pm if shaman[r["path"].split("/")[-1]] != r["sha256"]]
print("clips", len(m), "sha mismatches", len(bad), bad[:5], "pod-manifest mismatches", len(bad2))
assert not bad and not bad2
open("/root/p20/SHA_OK", "w").write("468 clips verified against manifest.csv sha256\n")
PY
