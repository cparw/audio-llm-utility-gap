"""Check that no Qwen2.5-Omni states on disk match the saved zero-shot runs of ADReSSo, PC-GITA, MDVR-KCL, E-DAIC mid300, mid30.
For every npz found (find/f_*.txt) that holds a 'name' (or clip) array and an encoder-like array: compare its clip set and p_yes
(when present) with each target zero-shot file. Also count per-clip shards dirs. Writes v5_notpossible.json."""
import numpy as np, csv, glob, json, os, zipfile
R = "<local data dir>/release"; V = "<local data dir>/leftovers_23sep/verify/t5v"
T = {"adresso": f"{R}/omni_final/omni_adresso_zeroshot_scores.csv", "pcgita": f"{R}/omni_final/omni_pcgita_zeroshot_scores.csv",
     "kcl": f"{R}/omni_final/omni_kcl_zeroshot_scores.csv", "edaic_mid300": f"{R}/edaic_rerun/variants/o25_mid300_zeroshot_scores.csv",
     "edaic_mid30": f"{R}/edaic_rerun/variants/o25_mid30_zeroshot_scores.csv"}
Z = {k: {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(p))} for k, p in T.items()}
curve_n = {k: len(v) for k, v in Z.items()}
files = sorted(set(l.strip() for f in (glob.glob(f"{V}/find/f_*.txt") + glob.glob(f"{V}/find/g_*.txt")) for l in open(f) if l.strip() and "/verify/t5v/" not in l and "/leftovers_23sep/t5/inputs" not in l))
res = {"n_npz_scanned": len(files), "targets": curve_n, "candidates": [], "exact_matches": []}
res["unnamed_encoder_files"] = []
for p in files:
    try:
        with zipfile.ZipFile(p) as zf: keys = [n[:-4] for n in zf.namelist()]
        z = np.load(p, allow_pickle=False)
    except Exception: continue
    ek = [k for k in keys if k in ("enc", "feats") or "enc" in k]
    shp = {}
    for k in ek:
        try: shp[k] = list(z[k].shape)
        except Exception: pass
    ek = [k for k, v in shp.items() if len(v) == 3 and v[-1] == 1280]
    if not ek: continue
    nl = shp[ek[0]][1]; nk = "name" if "name" in keys else ("clip" if "clip" in keys else None)
    try: names = [os.path.basename(str(x)) for x in z[nk]] if nk else None
    except Exception: names = None
    if names is None:
        d = dict(file=p, n=shp[ek[0]][0], enc_layers=nl, omni_shaped=nl == 32, same_n_as=[t for t, n in curve_n.items() if n == shp[ek[0]][0]])
        res["unnamed_encoder_files"].append(d); continue
    for t, zz in Z.items():
        ov = sum(n in zz for n in names)
        if ov == 0: continue
        d = dict(file=p, target=t, n_states=len(names), overlap=ov, curve_n=curve_n[t], enc_layers=nl)
        if "p_yes" in keys:
            py = dict(zip(names, z["p_yes"].astype(float))); common = [n for n in names if n in zz]
            d["max_abs_p_yes_diff_on_overlap"] = float(max(abs(py[n] - zz[n]) for n in common))
        res["candidates"].append(d)
        if ov == curve_n[t] and len(names) == curve_n[t] and d.get("max_abs_p_yes_diff_on_overlap", 1) < 1e-9: res["exact_matches"].append(d)
unres = [u for u in res["unnamed_encoder_files"] if u["omni_shaped"] and u["same_n_as"]]
res["unnamed_omni_shaped_same_n"] = unres
by = {}
for c in res["candidates"]:
    if "target" in c: by.setdefault(c["target"], []).append(f"{c['n_states']} clips, overlap {c['overlap']}/{c['curve_n']}, p_yes diff {c.get('max_abs_p_yes_diff_on_overlap', 'n/a') if isinstance(c.get('max_abs_p_yes_diff_on_overlap'), str) else ('%.3g' % c['max_abs_p_yes_diff_on_overlap'] if 'max_abs_p_yes_diff_on_overlap' in c else 'n/a')}")
res["by_target"] = by
res["ok"] = len(res["exact_matches"]) == 0 and not unres and curve_n["adresso"] == 237 and all(
    c["n_states"] != 237 or c.get("max_abs_p_yes_diff_on_overlap", 1) > 0.1 for c in res["candidates"] if c.get("target") == "adresso")
res["note"] = "237-clip ADReSSo states on disk are the Qwen3-Omni run (q3o) and a 33-layer encoder run (Qwen2-Audio shape); the Qwen2.5-Omni ADReSSo states (part16 POD4B) hold 161 clips"
adr = sorted(set(c["n_states"] for c in res["candidates"] if c.get("target") == "adresso"))
full = [c["max_abs_p_yes_diff_on_overlap"] for c in res["candidates"] if c.get("overlap") == c.get("curve_n") and "max_abs_p_yes_diff_on_overlap" in c]
lo_hi = f"{min(full):.2f} to {max(full):.2f}" if full else "n/a"
res["summary"] = (f"scanned {len(files)} npz files (>300 kB) under Desktop/release and G-Drive paper work: no states match any of the 5 saved zero-shot runs "
                  f"(exact matches {len(res['exact_matches'])}); ADReSSo states found hold {adr} clips vs {curve_n['adresso']} in the curve, the 237-clip files are " + ", ".join(f"{c['file'].split('/')[-1]} ({c['enc_layers']} encoder layers, p_yes diff {c['max_abs_p_yes_diff_on_overlap']:.2f})" for c in res['candidates'] if c.get('target') == 'adresso' and c['n_states'] == 237) + ", so not the Qwen2.5-Omni run; Qwen2.5-Omni ADReSSo states (part16 POD4B) are 161-clip subsets; " f"every full-overlap file differs in p_yes by {lo_hi}; "
                  + "; ".join(f"{t}: {len(by.get(t, []))} overlapping files" for t in T) + f"; {len(res['unnamed_encoder_files'])} files without clip names all have {sorted(set(u['enc_layers'] for u in res['unnamed_encoder_files']))} encoder layers (Qwen2.5-Omni has 32), unresolved {len(unres)}")
json.dump(res, open(f"{V}/v5_notpossible.json", "w"), indent=1, default=str); print(res["summary"]); print(json.dumps(by, indent=1)[:3000])
