"""Finalize one seed: stats on the synced per-clip csv, sidecar json, paste lines, provisional rows.
usage: finalize20b.py SEED   (run with /usr/local/bin/python3.10, which has scipy)"""
import sys, os, json, hashlib, glob, csv, datetime, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats20b import *
SEED = int(sys.argv[1])
MAC = os.environ.get("MAC_OUT", "<local data dir>/release/scores/part20/POD1b")
HERE = os.path.dirname(os.path.abspath(__file__))
name = f"std_ft_seed{SEED}_pitt"
oof = f"{MAC}/{name}_oof.csv"; run = json.load(open(f"{MAC}/{name}_run.json"))
def sha1mb(p):
    with open(p, "rb") as fh: return hashlib.sha256(fh.read(1 << 20)).hexdigest()
SEG = "<local data dir>/DementiaBank/segments"
src = {"per_clip_csv": oof,
       "manifest": "<local data dir>/release/edaic_rerun/part16/POD2/mf_pitt468_pod.csv",
       "folds": "folds/pitt_groupkfold5_pod.csv",
       "folds_readme": "folds/README.md",
       "arm_reference": "<local data dir>/DementiaBank/pitt_conflict_manifest.csv",
       "recipe_source_script": "<local data dir>/scratch/p15/sft_projector.py",
       "training_script": f"{HERE}/std_ft_seed.py", "input_precompute_script": f"{HERE}/precompute_inputs.py",
       "stats_script": f"{HERE}/stats20b.py", "finalize_script": f"{HERE}/finalize20b.py",
       "run_json": f"{MAC}/{name}_run.json", "train_log": f"{MAC}/std_ft_seed{SEED}.log",
       "original_run_oof_READONLY": "<local data dir>/release/omni_final/omnisft_pitt_oof.csv",
       "original_run_sidecar_READONLY": "<local data dir>/release/omni_final/omnisft_pitt_sft.json"}
R = list(csv.DictReader(open(oof)))
_af = f"{HERE}/audio_sha1mb.json"
AUD = json.load(open(_af)) if os.path.exists(_af) else {os.path.basename(r["path"]): sha1mb(os.path.join(SEG, os.path.basename(r["path"]))) for r in R}
assert sorted(AUD) == sorted(os.path.basename(r["path"]) for r in R)
y = np.array([int(r["label"]) for r in R]); spk = np.array([r["speaker"] for r in R]); arm = np.array([r["set"] for r in R])
assert len(R) == 468 and len(set(spk)) == 228
out = {}
for col in ("p_yes", "p_yes_multivariant"):
    s = np.array([float(r[col]) for r in R]); res = {}
    for nm, m in (("overall", np.ones(len(R), bool)), ("conflict", arm == "conflict"), ("agreement", arm == "agreement")):
        lo, hi, nd = boot(y[m], s[m], spk[m])
        res[nm] = {"auc": auc(y[m], s[m]), "lo": lo, "hi": hi, "n": int(m.sum()), "n_spk": int(len(set(spk[m]))), "draws": nd}
    lo, hi, nd = boot_pair(y, s, spk, arm == "conflict", arm == "agreement")
    res["conflict_minus_agreement"] = {"diff": res["conflict"]["auc"] - res["agreement"]["auc"], "lo": lo, "hi": hi,
                                       "n": len(R), "n_spk": 228, "draws": nd}
    out[col] = res
am = np.array([float(r["answer_mass"]) for r in R])
fd = np.array([int(r["fold"]) for r in R]); ps = np.array([float(r["p_yes"]) for r in R])
per_fold = {str(k): {"n": int((fd == k).sum()), "auc_p_yes_within_fold": auc(y[fd == k], ps[fd == k]),
                     "answer_mass_min": float(am[fd == k].min()), "answer_mass_median": float(np.median(am[fd == k])),
                     "n_answer_mass_below_0.5": int((am[fd == k] < 0.5).sum()),
                     "train_loss_by_epoch": [run["loss_per_fold_epoch"].get(f"{k}_{e}") for e in range(3)]} for k in range(5)}
side = {"result": f"PART20 POD1b standard Pitt projector fine-tune, seed {SEED} (Qwen2.5-Omni-7B, out of fold)",
        "model_id": run["checkpoint"], "seed": SEED, "order_rng": run["order_rng"], "torch_manual_seed": SEED,
        "original_run_seed": "sft_projector.py has no seed argument and the omni_final sidecar has no seed field; its only RNG is np.random.RandomState(fold*10+epoch) for clip order (offset 0); torch unseeded; all dropout 0.0",
        "prompt_verbatim": run["prompt"], "n": len(R), "n_speakers": 228, "n_conflict": int((arm == "conflict").sum()),
        "n_agreement": int((arm == "agreement").sum()),
        "recipe": "sft_projector.py unchanged: projector only (thinker.audio_tower.proj, fp32 forward), AdamW lr 1e-4, 3 epochs, batch 1, CE over [' Yes',' No'] logits at answer position, bf16, 30 s window, no weighting",
        "folds": "loaded from release/folds/pitt_groupkfold5_pod.csv; pod GroupKFold(5) agrees on %d/468" % run["pod_groupkfold_agreement"],
        "scoring": {"p_yes": "recipe: softmax over [' Yes',' No'] at the first answer position (same as the original run's p_yes)",
                    "p_yes_multivariant": "P(Yes-variants)/(P(Yes-variants)+P(No-variants)), full-vocab softmax, single-token bare and leading-space Yes/yes/YES, No/no/NO; ids %s / %s" % (run["yes_set"], run["no_set"]),
                    "answer_mass": {"median": float(np.median(am)), "min": float(am.min()), "n_below_0.5": int((am < 0.5).sum())}},
        "per_fold_diagnostics": per_fold,
        "statistics": "AUC by scipy.stats.rankdata rank formula (ties averaged); 2000 speaker-bootstrap draws, fresh numpy.random.default_rng(0) per cell, percentile 2.5/97.5; arm cells resample that arm's speakers; paired diff = one speaker draw per replicate",
        "results": out,
        "sources_absolute": src,
        "sha256_first_1MB": {k: sha1mb(v) for k, v in src.items() if os.path.exists(v)},
        "audio": {"mac_source_dir": SEG, "pod_dir": "/workspace/data/pitt468",
                  "pod_copy_check": "468 files, 516863704 bytes, md5-of-md5s 6f9b44dc0967d0cd9a345e87dcadaead identical on Mac and pod",
                  "sha256_first_1MB_per_clip": AUD},
        "run": {k: run[k] for k in ("loss_per_fold_epoch", "minutes_per_fold", "peak_gpu_gb", "transformers", "torch", "sklearn", "numpy", "date")},
        "command": run["command"] + "   (env HF_HOME=/workspace/hf INPUT_CACHE=/root/pitt468_inputs.pt OMP_NUM_THREADS=8; cache from: python3 precompute_inputs.py mf_pitt468_pod.csv /root/pitt468_inputs.pt)",
        "pod": {"name": "POD1b", "runpod_id": "gqgam006faroq3", "gpu": "H100 NVL 94GB", "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                "concurrency": "three seeds (1,2,3) trained concurrently on the one GPU"},
        "date": datetime.date.today().isoformat()}
json.dump(side, open(f"{MAC}/{name}_oof.sidecar.json", "w"), indent=1)
def vs(lo, hi):
    return "interval above 0.5" if lo > 0.5 else ("interval below 0.5" if hi < 0.5 else "interval includes 0.5")
rows_f = os.environ.get("ROWS_F", "reports/part20/rows/POD1b_provisional.tsv")
os.makedirs(os.path.dirname(rows_f), exist_ok=True)
new = not os.path.exists(rows_f)
with open(rows_f, "a") as fh:
    if new: fh.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\ttwo_dp\tvs_half\n")
    for col in ("p_yes", "p_yes_multivariant"):
        tag = "" if col == "p_yes" else "_mv"
        for cell in ("overall", "conflict", "agreement"):
            c = out[col][cell]; v = "above 0.5" if c["auc"] > 0.5 else "below 0.5"
            line = (f"std_ft_seed{SEED}_{cell}{tag}", f"Pitt projector FT seed {SEED} OOF AUC {cell} ({col})",
                    f"{c['auc']:.4f}", f"{c['lo']:.4f}", f"{c['hi']:.4f}", c["n"], c["n_spk"], oof, f"{c['auc']:.2f}", f"{v}; {vs(c['lo'], c['hi'])}")
            fh.write("\t".join(map(str, line)) + "\n")
            print(f"{line[0]} | AUC {c['auc']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] | n={c['n']} n_spk={c['n_spk']} | {oof} | {c['auc']:.2f}, {v} ({vs(c['lo'], c['hi'])})")
        d = out[col]["conflict_minus_agreement"]; ex = "interval excludes zero" if (d["lo"] > 0 or d["hi"] < 0) else "interval includes zero"
        line = (f"std_ft_seed{SEED}_conf_minus_agr{tag}", f"Pitt projector FT seed {SEED} conflict minus agreement, paired ({col})",
                f"{d['diff']:.4f}", f"{d['lo']:.4f}", f"{d['hi']:.4f}", d["n"], d["n_spk"], oof, f"{d['diff']:.2f}", ex)
        fh.write("\t".join(map(str, line)) + "\n")
        print(f"{line[0]} | diff {d['diff']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}] | n={d['n']} n_spk={d['n_spk']} | {oof} | {d['diff']:.2f}, {ex}")
print(f"answer_mass median {np.median(am):.4f} min {am.min():.4f} n<0.5 {(am < 0.5).sum()}")
for k, v in per_fold.items(): print(f"  fold {k}: within-fold AUC {v['auc_p_yes_within_fold']:.4f} mass median {v['answer_mass_median']:.4f} n<0.5 {v['n_answer_mass_below_0.5']} loss {[round(x, 3) for x in v['train_loss_by_epoch']]}")
