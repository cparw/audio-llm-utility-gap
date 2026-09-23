import sys; sys.path.insert(0, "<local data dir>/release/scores/part20/verify")
from POD5_verify import *
W = "reports/part20/verify/POD5_work"
pc = f"{W}/POD5_zs_o25_noinv.csv"
rows = read_csv(pc)
# provenance: noinv column == extraction output; paper column == release omni_final file
src = {q["orig_clip_id"]: q for q in read_csv(f"{W}/o25noinv_pitt_zeroshot_scores.csv")}
pap = {q["clip"]: q for q in read_csv("<local data dir>/release/omni_final/omni_pitt_zeroshot_scores.csv")}
d1 = max(abs(float(q["p_yes_noinv"]) - float(src[q["orig_clip_id"]]["p_yes"])) for q in rows)
d2 = max(abs(float(q["p_yes_original_paper"]) - float(pap[q["orig_clip_id"]]["p_yes"])) for q in rows)
lab = all(int(q["label"]) == int(pap[q["orig_clip_id"]]["label"]) and q["speaker"] == pap[q["orig_clip_id"]]["speaker"] for q in rows)
print("provenance: noinv vs extraction max diff", d1, "| paper col vs omni_final max diff", d2, "| labels/speakers match paper file", lab,
      "| paper prompt", set(q["prompt"] for q in pap.values()) == {PROMPT})
spec = {}
for a, sfx in ((None, "overall"), ("conflict", "conflict"), ("agreement", "agreement")):
    spec[f"POD5_zs_o25_{sfx}"] = ("single", ["p_yes_noinv"], None, a)
    spec[f"POD5_zs_o25_paper_{sfx}"] = ("single", ["p_yes_original_paper"], None, a)
    spec[f"POD5_zs_o25_diff_{sfx}"] = ("paired", ["p_yes_original_paper"], ["p_yes_noinv"], a)
out, bc, pok = verify_cells("POD5_zs_o25_noinv", pc, f"{W}/POD5_zs_o25_noinv.sidecar.json", spec)
print("basic", bc, "prompt", pok)
allok = d1 < 1e-12 and d2 < 1e-12 and lab
DUMP = []
for d, ag, det in out:
    print(("AGREE " if ag and allok else "DISAGREE ") + det)
    if d is not None and ag and allok and "--write" in sys.argv:
        DUMP.append(d); print("PASTE", paste(d), flush=True)

json.dump({"rows": DUMP, "all_agree": all(ag for _, ag, _ in out), "n_cells": len(out)}, open(sys.argv[0].replace(".py", "") + "_" + (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "zs") + ".rows.json", "w"), indent=1)
