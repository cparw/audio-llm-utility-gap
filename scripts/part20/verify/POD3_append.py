"""Append FINAL rows (verified value agrees with compute value to 4 dp, both bounds too) to rows/POD3.tsv.
input: JSON list of {id, what, comp:[v,lo,hi], ver:[v,lo,hi], n, n_spk, file, ref(0.5 or 0)}"""
import json, sys, os
T = "reports/part20/rows/POD3.tsv"
H = ["id", "what", "value_computed", "value_verified", "lo", "hi", "n", "n_spk", "file", "two_dp", "vs_half", "agree_4dp"]
items = json.load(open(sys.argv[1]))
new = not os.path.exists(T)
have = set()
if not new:
    have = {l.split("\t")[0] for l in open(T).read().splitlines()[1:]}
with open(T, "a") as fh:
    if new: fh.write("\t".join(H) + "\n")
    for it in items:
        c, v = it["comp"], it["ver"]
        agree = all(round(a, 4) == round(b, 4) for a, b in zip(c, v))
        ref = it.get("ref", 0.5)
        vs = "above" if v[1] > ref else ("below" if v[2] < ref else "straddles")
        if ref != 0.5: vs += f"_{ref:g}"
        line = [it["id"], it["what"], f"{c[0]:.4f}", f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}", str(it["n"]), str(it["n_spk"]),
                it["file"], f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]", vs, str(agree)]
        if not agree:
            print("DISAGREE NOT FINAL", it["id"], "computed", c, "verified", v); continue
        if it["id"] in have:
            print("already present", it["id"]); continue
        fh.write("\t".join(line) + "\n")
        print("FINAL", "\t".join(line))
