"""Verifier prep for t5 (own code). Reads the original states sources directly (not the t5 inputs) and writes
compressed per-job subsets with label, spk, name. Records sha256 of each source and each array."""
import numpy as np, csv, hashlib, json, os, sys
R = "<local data dir>/release"; G = "<local data dir>/release_from_mac"
OUT = f"{G}/scores/leftovers_23sep/verify/t5v/inputs"; os.makedirs(OUT, exist_ok=True)
def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def asha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
which = sys.argv[1:]
man = {}
def dump(tag, d, keys, src):
    p = f"{OUT}/{tag}.npz"
    sub = {k: d[k] for k in keys + ["label", "spk", "name"]}
    np.savez_compressed(p, **sub)
    man[tag] = dict(file=p, source=src, sha256=fsha(p), arrays={k: [list(v.shape), str(v.dtype), asha(v)] for k, v in sub.items()})
    print(tag, {k: v.shape for k, v in sub.items()}, os.path.getsize(p) >> 20, "MB", flush=True)
def load_npz(p):
    z = np.load(p, allow_pickle=False); return {k: z[k] for k in z.files}
if "pitt" in which:
    src = f"{R}/overnight2/part10/o25_pitt_states.npz"; d = load_npz(src); man["src_pitt"] = dict(file=src, sha256=fsha(src))
    dump("pitt_encproj", d, ["enc", "proj"], src); dump("pitt_llm", d, ["llm"], src); dump("pitt_ans", d, ["ans"], src)
if "edaic30" in which:
    src = f"{R}/overnight2/part10/o25_edaic_states.npz"; d = load_npz(src); man["src_edaic30"] = dict(file=src, sha256=fsha(src))
    dump("edaic30_encproj", d, ["enc", "proj"], src); dump("edaic30_llm", d, ["llm"], src); dump("edaic30_ans", d, ["ans"], src)
if "adress2020" in which:
    src = f"{G}/edaic_rerun/part16/POD4B/states/p16_adress2020_orig_states.npz"; d = load_npz(src); man["src_adress2020"] = dict(file=src, sha256=fsha(src))
    dump("adress2020_encproj", d, ["enc", "proj"], src)
if "neurovoz" in which:
    src = f"{G}/scores/part20/POD6/B_neurovoz_full_check/omni_nvfull_states.npz"; d = load_npz(src); man["src_neurovoz"] = dict(file=src, sha256=fsha(src))
    dump("neurovoz_encproj", d, ["enc", "proj"], src)
if "edaicfull" in which:
    order = f"{R}/edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv"; sh = f"{G}/edaic_rerun/part17/T7_scratch/shards"
    rows = list(csv.DictReader(open(order)))
    enc, proj = [], []
    for r in rows:
        z = np.load(f"{sh}/{r['clip']}.npz"); enc.append(z["enc"]); proj.append(z["proj"])
    d = dict(enc=np.stack(enc), proj=np.stack(proj), label=np.array([int(r["label"]) for r in rows], dtype=np.int64),
             spk=np.array([r["speaker"] for r in rows]), name=np.array([r["clip"] for r in rows]))
    man["src_edaicfull"] = dict(file=sh, order=order, order_sha256=fsha(order), n=len(rows))
    dump("edaicfull_encproj", d, ["enc", "proj"], sh)
json.dump(man, open(f"{OUT}/manifest_{'_'.join(which)}.json", "w"), indent=1)
print("PREP DONE", which)
