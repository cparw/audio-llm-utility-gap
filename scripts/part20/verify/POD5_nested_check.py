"""PART20 POD5 verifier: nested-probe values from the per-clip oof csvs (own code, see POD5_verify.py)."""
import sys, json
sys.path.insert(0, "<local data dir>/release/scores/part20/verify")
from POD5_verify import *
W = "reports/part20/verify/POD5_work"
S = json.load(open(f"{W}/noinv_nested_summary.json"))
out = {}
for tag, r in S["results"].items():
    rows = read_csv(f"{W}/{os.path.basename(r['oof_csv'])}")
    clips = [q["clip_id"] for q in rows]; spk = np.array([q["speaker"] for q in rows])
    y = np.array([int(q["label"]) for q in rows]); arm = np.array([q["set"] for q in rows])
    cols = [np.array([float(q[f"p_{s}"]) for q in rows]) for s in r["fold_sets"]]
    bc = basic_checks(clips, spk, y, arm)
    o = {"basic": bc, "overall": point_and_ci(spk, y, cols)}
    for a in ("conflict", "agreement"):
        o[a] = point_and_ci(spk, y, cols, mask=(arm == a))
    out[tag] = o
    print(f"{tag}: basic ok={bc['ok']} n={bc['n']} spk={bc['n_spk']} arms={bc['arms']}")
    ov = o["overall"]
    print(f"   per-repeat MW {[r4(v) for v in ov['per_mw']]} trap {[r4(v) for v in ov['per_trap']]} | run {r['per_repeat_auc']}")
    print(f"   overall {r4(ov['value'])} [{r4(ov['lo'])}, {r4(ov['hi'])}] usable {ov['usable']} | run mean {r['mean_auc']}")
    for a in ("conflict", "agreement"):
        q = o[a]
        print(f"   {a} {r4(q['value'])} [{r4(q['lo'])}, {r4(q['hi'])}] n={q['n']} spk={q['n_spk']} | alt-convention [{r4(q['lo_alt'])}, {r4(q['hi_alt'])}]")
json.dump(out, open(f"{W}/../POD5_nested_check_out.json", "w"), indent=1, default=float)
