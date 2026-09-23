"""PART20 POD5 verifier: projector fine-tune RESULT. Own code; numbers from the per-clip csv, the fold column checked
against the release fold file, the recipe p_yes column checked against the raw sft oof, the paper column against
omni_final/omnisft_pitt_oof.csv (read only)."""
import sys; sys.path.insert(0, "<local data dir>/release/scores/part20/verify")
from POD5_verify import *
W = "reports/part20/verify/POD5_work"
REL = "<local data dir>/release"
res_id = sys.argv[1]
pc = f"{W}/{res_id}.csv"; sc = json.load(open(f"{W}/{res_id}.sidecar.json"))
rows = read_csv(pc); hdr = list(rows[0].keys())
print("header", hdr)
key = "orig_clip_id" if "orig_clip_id" in hdr else "clip_id"
clips = [q[key] for q in rows]; spk = np.array([q["speaker"] for q in rows])
y = np.array([int(q["label"]) for q in rows]); arm = np.array([q["set"] for q in rows])
bc = basic_checks(clips, spk, y, arm)
fcol = "fold" if "fold" in hdr else None
mis, miss = check_folds(clips, spk, [q[fcol] for q in rows], f"{REL}/folds/pitt_groupkfold5_pod.csv") if fcol else (-1, -1)
raw = {q["orig_clip_id"]: q for q in read_csv(f"{W}/omnisft_noinv_pitt_oof.csv")}
pap = {os.path.basename(q["path"]): q for q in read_csv(f"{REL}/omni_final/omnisft_pitt_oof.csv")}
ncol = [c for c in hdr if "noinv" in c and c.startswith("p_yes")] or ["p_yes"]
pcol = [c for c in hdr if ("paper" in c or "original" in c) and c.startswith("p_yes")]
print("noinv col", ncol, "paper col", pcol)
d_raw = max(max(abs(float(q[ncol[0]]) - float(raw[q[key]]["p_yes"])), abs(float(q["p_yes_rule"]) - float(raw[q[key]]["p_yes_rule"]))) for q in rows)
d_rawfold = sum(int(q[fcol]) != int(raw[q[key]]["fold"]) for q in rows) if fcol else -1
d_pap = max(abs(float(q[pcol[0]]) - float(pap[q[key]]["p_yes"])) for q in rows) if pcol else -1
lab = all(int(pap[q[key]]["label"]) == int(q["label"]) and pap[q[key]]["speaker"] == q["speaker"] and pap[q[key]]["set"] == q["set"] for q in rows)
prompts = set(q.get("prompt", PROMPT) for q in rows) | {raw[c]["prompt"] for c in raw} | {sc.get("prompt_verbatim")}
fold_named = sc.get("fold_file", "") or json.dumps(sc)[:0]
print(f"basic {bc} | fold mismatches vs release pod file {mis} missing {miss} | fold vs raw {d_rawfold} | noinv col vs raw {d_raw:.2e} "
      f"| paper col vs omni_final {d_pap:.2e} | labels/spk/set match paper {lab} | prompts {prompts}")
base_ok = bc["ok"] and mis == 0 and miss == 0 and d_rawfold == 0 and d_raw < 1e-12 and 0 <= d_pap < 1e-12 and lab and prompts == {PROMPT}
cells = {c["id"]: c for c in sc["cells"]}
print("cells", list(cells))
DUMP = []; allag = True
col = lambda c: np.array([float(q[c]) for q in rows])
for cid, c in cells.items():
    a = "conflict" if cid.endswith("conflict") else ("agreement" if cid.endswith("agreement") else None)
    m = None if a is None else arm == a
    if "diff" in cid or "minus" in cid:
        r = paired_ci(spk, y, [col(pcol[0])], [col(ncol[0])], mask=m); tr_ok = True
    elif "rule" in cid:
        r = point_and_ci(spk, y, [col("p_yes_rule")], mask=m); tr_ok = abs(r["value"] - r["value_trap"]) < 1e-12
    elif "paper" in cid or "original" in cid:
        r = point_and_ci(spk, y, [col(pcol[0])], mask=m); tr_ok = abs(r["value"] - r["value_trap"]) < 1e-12
    else:
        r = point_and_ci(spk, y, [col(ncol[0])], mask=m); tr_ok = abs(r["value"] - r["value_trap"]) < 1e-12
    d, ag = row(cid, c["what"], c["value"], r["value"], c["lo"], c["hi"], r["lo"], r["hi"], r["n"], r["n_spk"], sc["per_clip_csv"])
    ag = ag and tr_ok and base_ok and c.get("n") == r["n"] and c.get("n_speakers") == r["n_spk"]; d["agree_4dp"] = "yes" if ag else "NO"
    allag = allag and ag
    print(("AGREE " if ag else "DISAGREE ") + f"{cid}: computed {c['value']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] n={c.get('n')}/{c.get('n_speakers')} | verified {r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] n={r['n']}/{r['n_spk']}", flush=True)
    if ag: DUMP.append(d); print("PASTE", paste(d), flush=True)
json.dump({"rows": DUMP, "all_agree": allag}, open(f"POD5_check_sft_{res_id}.rows.json", "w"), indent=1)
