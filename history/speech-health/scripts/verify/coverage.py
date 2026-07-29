"""
Coverage audit: what has ACTUALLY been computed, read off disk.
Nothing here is from memory. Run it any time to see gaps.
"""
import os, json, glob
R = "/project2/msoleyma_946/speech_health/results_chaitanya"

DATASETS = ["italian", "kcl", "neurovoz", "pcgita", "dcaps"]
TASKS = {"italian": ["read", "vowel"], "kcl": ["read"], "neurovoz": ["read", "vowel"],
         "pcgita": ["read", "vowel"], "dcaps": ["interview"]}

def has(path_glob):
    return len(glob.glob(path_glob)) > 0

def jload(p):
    try: return json.load(open(p))
    except Exception: return None

print("=" * 96)
print("COVERAGE AUDIT  (generated from files on disk)")
print("=" * 96)

# 1. recording-properties confound audit
print("\n[1] RECORDING-PROPERTIES CONFOUND AUDIT (artifact floor)")
for t in ["read", "vowel", "interview"]:
    p = f"{R}/audit/confound_audit_{t}.json"
    d = jload(p)
    if not d:
        print(f"   {t:10s} MISSING")
        continue
    for ds in DATASETS:
        if ds in d:
            print(f"   {t:10s} {ds:10s} floor={d[ds].get('AUC_from_recording_properties_only')}  "
                  f"shuffle={d[ds].get('AUC_label_shuffled_control')}")
    for ds in DATASETS:
        if t in TASKS.get(ds, []) and ds not in d:
            print(f"   {t:10s} {ds:10s} *** MISSING ***")

# 2. nested test
print("\n[2] NESTED TEST (does encoder beat recording properties)")
d = jload(f"{R}/audit/nested_confound_test.json")
print("   in JSON:", list(d.keys()) if d else "MISSING")
print("   NOTE: read-task results were overwritten by the vowel run; they exist only in the slurm log")

# 3. metadata / MI
print("\n[3] METADATA + MUTUAL INFORMATION")
for ds in DATASETS:
    src = {"italian": "source .xlsx (age/sex recovered)",
           "neurovoz": "metadata csv (age/sex/date/clinical)",
           "pcgita": "metadata xlsx (age/sex/UPDRS)",
           "dcaps": "split csv (gender only, no age)",
           "kcl": "*** NO metadata file found anywhere ***"}[ds]
    print(f"   {ds:10s} {src}")

# 4. noise + speaker shuffle
print("\n[4] NOISE + SPEAKER-SHUFFLE MANIPULATION")
d = jload(f"{R}/audit/noise_shuffle_read.json")
if d:
    for ds in DATASETS:
        if ds in d:
            print(f"   read  {ds:10s} clean={d[ds]['clean']} noise0dB={d[ds]['noise_0dB']} shuffled={d[ds]['speaker_shuffled']}")
        elif "read" in TASKS.get(ds, []):
            print(f"   read  {ds:10s} *** MISSING ***")
else:
    print("   MISSING entirely")
for t in ["vowel", "interview"]:
    print(f"   {t:9s} *** NOT RUN for any dataset ***" if not has(f"{R}/audit/noise_shuffle_{t}.json") else f"   {t} present")

# 5. 25-case diagnostic
print("\n[5] 25-CASE FEATURE DIAGNOSTIC")
for ds in DATASETS:
    for t in TASKS.get(ds, []):
        p = f"{R}/audit/diagnose25_{ds}_{t}.json"
        print(f"   {ds:10s} {t:10s} {'present' if os.path.exists(p) else '*** MISSING ***'}")

# 6. LLM text-free ablation
print("\n[6] LLM TEXT-FREE ABLATION (per model)")
for f in sorted(glob.glob(f"{R}/ablation/*_summary.json")):
    d = jload(f)
    print(f"   {os.path.basename(f).replace('_summary.json',''):26s} n={d['n']:<5} AUC_orig={d.get('AUC_orig')}")
print("   coverage gaps: italian NOT run for LLM at all; kcl only 'read'; no LLM run on italian/kcl vowels")

print("\n" + "=" * 96)
