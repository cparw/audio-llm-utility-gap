"""PART20 POD5 verifier: nested RESULT cells. Own code; numbers from the per-clip csv only, plus provenance checks
of every column against the run's raw oof files (which were refit-checked on the pod) and the paper files."""
import sys; sys.path.insert(0, "<local data dir>/release/scores/part20/verify")
from POD5_verify import *
W = "reports/part20/verify/POD5_work"
REL = "<local data dir>/release"
res_id = sys.argv[1]            # e.g. POD5_nested_llm_pod_seed
tag = res_id.replace("POD5_nested_", "")
pc = f"{W}/{res_id}.csv"; sc = f"{W}/{res_id}.sidecar.json"
rows = read_csv(pc)
hdr = list(rows[0].keys())
noinv = [c for c in hdr if c.endswith("__noinv")]
other = [c for c in hdr if c.startswith("p_") and not c.endswith("__noinv")]
side = other[0].split("__")[1]
# provenance of the noinv columns
if tag.startswith("proj_pod_seed0"):
    raw_n, raw_o = f"{W}/noinvP_proj_pod_seed0_oof.csv", f"{W}/origP_proj_pod_seed0_oof.csv"
else:
    raw_n, raw_o = f"{W}/noinv_{tag}_oof.csv", f"{W}/orig_{tag}_oof.csv"
rn = {q["clip_id"]: q for q in read_csv(raw_n)}
prov = {}
prov["noinv_vs_raw"] = max(abs(float(q[c]) - float(rn[q["clip_id"]][c.split("__")[0]])) for q in rows for c in noinv)
if side == "paper_mac":
    z = np.load(f"{REL}/overnight2/part10/pitt_enc_nested5_oof.npz", allow_pickle=True)
    pm = {str(n): i for i, n in enumerate(z["name"])}
    prov["paper_vs_npz"] = max(abs(float(q[f"p_rep{k}__paper_mac"]) - float(z["oof"][k, pm[q["clip_id"]]])) for q in rows for k in range(5))
    prov["paper_labels_spk"] = all(int(z["label"][pm[q["clip_id"]]]) == int(q["label"]) and str(z["spk"][pm[q["clip_id"]]]) == q["speaker"] for q in rows)
else:
    ro = {q["clip_id"]: q for q in read_csv(raw_o)}
    prov["orig_vs_raw"] = max(abs(float(q[c]) - float(ro[q["clip_id"]][c.split("__")[0]])) for q in rows for c in other)
print(res_id, "noinv cols", noinv, "other", other, "side", side, "provenance", prov)
spec = {}
sidename = {"paper_mac": "paper", "orig_rerun": "original_rerun"}[side]
for a, sfx in ((None, "overall"), ("conflict", "conflict"), ("agreement", "agreement")):
    spec[f"POD5_{tag}_{sfx}"] = ("single", noinv, None, a)
    spec[f"POD5_{tag}_{sidename}_{sfx}"] = ("single", other, None, a)
    spec[f"POD5_{tag}_diff_{sfx}"] = ("paired", other, noinv, a)
out, bc, pok = verify_cells(res_id, pc, sc, spec, key="clip_id")
provok = all((v < 1e-12) if isinstance(v, float) else bool(v) for v in prov.values())
print("basic", bc, "prompt", pok, "provenance_ok", provok)
DUMP = []
for d, ag, det in out:
    print(("AGREE " if ag and provok else "DISAGREE ") + det, flush=True)
    if d is not None and ag and provok and "--write" in sys.argv:
        DUMP.append(d); print("PASTE", paste(d), flush=True)

json.dump({"rows": DUMP, "all_agree": all(ag for _, ag, _ in out), "n_cells": len(out)}, open(sys.argv[0].replace(".py", "") + "_" + (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "zs") + ".rows.json", "w"), indent=1)
