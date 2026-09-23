"""T5 prep (Mac, read only on sources). Writes per-job states npz subsets under the t5 inputs folder.
Each subset keeps the stream arrays byte for byte (same dtype, same clip order) plus label, spk, p_yes, name,
so scripts/curves_from_states.py, unchanged, computes only that job's stream(s)."""
import numpy as np, pandas as pd, hashlib, json, os, sys, shutil
R = "<local data dir>/release"; G = "<local data dir>/release_from_mac"
T5 = f"{G}/scores/leftovers_23sep/t5"; INP = f"{T5}/inputs"
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
SRC = {
 "pitt": dict(npz=f"{R}/overnight2/part10/o25_pitt_states.npz", zs=f"{R}/omni_final/omni_pitt_zeroshot_scores.csv", states_class="same_run"),
 "edaic30": dict(npz=f"{R}/overnight2/part10/o25_edaic_states.npz", zs=f"{R}/omni_final/omni_edaic_zeroshot_scores.csv", states_class="same_run"),
 "adress2020": dict(npz=f"{G}/edaic_rerun/part16/POD4B/states/p16_adress2020_orig_states.npz", zs=f"{R}/omni_final/omni_adress2020_zeroshot_scores.csv", states_class="other_extraction"),
 "neurovoz": dict(npz=f"{G}/scores/part20/POD6/B_neurovoz_full_check/omni_nvfull_states.npz", zs=f"{R}/omni_final/omni_neurovoz_zeroshot_scores.csv", states_class="other_extraction"),
 "edaicfull": dict(shards=f"{G}/edaic_rerun/part17/T7_scratch/shards", order=f"{R}/edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv", zs=f"{R}/edaic_rerun/variants/o25_full_zeroshot_scores.csv", states_class="other_extraction"),
}
JOBS = [("pitt", "encproj", ["enc", "proj"]), ("pitt", "llm", ["llm"]), ("pitt", "ans", ["ans"]),
        ("edaic30", "encproj", ["enc", "proj"]), ("edaic30", "llm", ["llm"]), ("edaic30", "ans", ["ans"]),
        ("adress2020", "encproj", ["enc", "proj"]), ("neurovoz", "encproj", ["enc", "proj"]), ("edaicfull", "encproj", ["enc", "proj"])]
man = {"sources": {}, "jobs": {}}
cache = {}
for ds, cfg in SRC.items():
    zs = pd.read_csv(cfg["zs"])
    if "npz" in cfg:
        z = np.load(cfg["npz"], allow_pickle=True); d = {k: z[k] for k in z.files}
        man["sources"][ds] = dict(file=cfg["npz"], sha256=sha_file(cfg["npz"]), keys={k: [list(v.shape), str(v.dtype)] for k, v in d.items()})
    else:
        order = pd.read_csv(cfg["order"]); clips = list(order["clip"].astype(str))
        assert clips == list(zs["clip"].astype(str)), "edaicfull order differs from variants zero-shot file"
        st = {k: [] for k in ("enc", "proj", "llm", "ans")}; shs = {}
        for c in clips:
            p = f"{cfg['shards']}/{c}.npz"; zz = np.load(p)
            for k in st: st[k].append(zz[k])
        d = {k: np.stack(v) for k, v in st.items()}
        d["label"] = order["label"].values.astype(np.int64); d["spk"] = order["speaker"].astype(str).values.astype("<U3")
        d["p_yes"] = order["p_yes"].values.astype(np.float64); d["name"] = np.array(clips)
        man["sources"][ds] = dict(file=cfg["shards"], n_shards=len(clips), order_file=cfg["order"], order_sha256=sha_file(cfg["order"]),
                                  keys={k: [list(v.shape), str(v.dtype)] for k, v in d.items()})
    names = [str(v) for v in d["name"]]
    same_order = names == list(zs["clip"].astype(str))
    pdiff = float(np.max(np.abs(d["p_yes"].astype(float) - zs["p_yes"].values))) if same_order else None
    spk_ok = same_order and all(a == str(b) for a, b in zip(d["spk"].astype(str), zs["speaker"].astype(str)))
    lab_ok = same_order and bool(np.all(d["label"].astype(int) == zs["label"].values.astype(int)))
    man["sources"][ds].update(zeroshot_file=cfg["zs"], zeroshot_sha256=sha_file(cfg["zs"]), clip_order_equals_zeroshot=same_order,
                              max_abs_p_yes_diff_vs_zeroshot=pdiff, speaker_equal=spk_ok, label_equal=lab_ok, states_class=cfg["states_class"], n=len(names))
    print(ds, "order", same_order, "p_yes diff", pdiff, "spk", spk_ok, "label", lab_ok, flush=True)
    cache[ds] = d
for ds, tag, keys in JOBS:
    d = cache[ds]; out = f"{INP}/{ds}_{tag}_states.npz"
    sub = {k: d[k] for k in keys + ["label", "spk", "p_yes", "name"]}
    np.savez(out, **sub)
    chk = np.load(out, allow_pickle=True)
    same = all(sha_arr(chk[k]) == sha_arr(d[k]) and chk[k].dtype == d[k].dtype for k in sub)
    man["jobs"][f"{ds}_{tag}"] = dict(file=out, sha256=sha_file(out), keys=keys, array_sha256={k: sha_arr(d[k]) for k in sub}, byte_identical_to_source_arrays=same)
    print(ds, tag, keys, "written", os.path.getsize(out) >> 20, "MB, arrays identical to source:", same, flush=True)
    assert same
json.dump(man, open(f"{INP}/inputs_manifest.json", "w"), indent=1)
print("PREP DONE")
