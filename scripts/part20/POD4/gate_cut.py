import hashlib, csv, os, sys, subprocess
W = "/workspace/pod4"
TGZ_SHA = "b995bd9afe8a45b1c08472f7333c5855767d60de23307a7881b5f000c73d8a07"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
subprocess.run(f"cat {W}/cut/parts/part_* > {W}/cut/pitt_noinv_clips.tgz", shell=True, check=True)
s = sha(f"{W}/cut/pitt_noinv_clips.tgz"); print("tgz sha256", s, "match", s == TGZ_SHA, flush=True)
assert s == TGZ_SHA
subprocess.run(f"tar xzf {W}/cut/pitt_noinv_clips.tgz -C {W}/cut", shell=True, check=True)
m = list(csv.DictReader(open(f"{W}/cut/manifest.csv")))
bad = [r["clip"] for r in m if sha(f"{W}/cut/clips/{r['clip']}") != r["sha256"]]
print("clips", len(m), "sha mismatches", len(bad), bad[:5], flush=True)
ref = list(csv.DictReader(open(f"{W}/ref/noinv_manifest.csv")))
print("pod manifest identical to Mac manifest:", [dict(r) for r in m] == [dict(r) for r in ref], flush=True)
assert not bad and len(m) == 468
open(f"{W}/CUT_OK", "w").write(f"468/468 clip sha256 match manifest; tgz sha256 {s}\n")
print("CUT_OK", flush=True)
