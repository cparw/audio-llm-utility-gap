"""Shared helpers for the Part 12 tex-vs-files lookup.

Only the Python standard library is used, so the tools run with /usr/local/bin/python3 as is.
Nothing here writes anywhere; callers choose their own output paths.
"""
import csv
import os
import re

RELEASE = "<local data dir>/release"
EDAIC = RELEASE + "/edaic_rerun"
PREP = EDAIC + "/part17/part12_prep"
OLD_PREFIX = "<local data dir>/paper1_local_runs/"
NEW_PREFIX = "<local data dir>/paper work/paper1_local_runs/"
TA_BATCH = "<local data dir>/paper work/ashutosh_batch2/"   # never canonical
READ_ONLY_ROOTS = [RELEASE + "/omni_final"]                     # never written by any Part 12 tool

MODELS = ["Qwen2-Audio", "Qwen2.5-Omni", "Qwen3-Omni", "Audio Flamingo 2", "Audio Flamingo 3", "Kimi-Audio"]
SIX_DATASETS = ["pcgita", "neurovoz", "kcl", "pitt", "adresso", "adress2020"]
PD = ["pcgita", "neurovoz", "kcl"]
AD = ["pitt", "adresso", "adress2020"]
DATASET_LABEL = {"pcgita": "PC-GITA", "neurovoz": "NeuroVoz", "kcl": "MDVR-KCL", "edaic": "E-DAIC (first 30 s)",
                 "edaic_full": "E-DAIC (full, 900 s cap)", "edaic_mid30": "E-DAIC (mid 30 s)",
                 "edaic_mid300": "E-DAIC (300 s)", "pitt": "Pitt 468", "pitt_all": "Pitt all 708",
                 "adresso": "ADReSSo", "adress2020": "ADReSS-2020",
                 "edaic_conflict_v2": "E-DAIC conflict set (Part 14, 483 pairs)",
                 "edaic_conflict_390": "E-DAIC conflict set (older 390 rows)"}


# ----------------------------------------------------------------------------- paths
def rewrite_path(p):
    """Rewrite the stale G-Drive prefix. Returns (new_path, rewritten_flag)."""
    if p and p.startswith(OLD_PREFIX):
        return NEW_PREFIX + p[len(OLD_PREFIX):], True
    return p, False


def resolve_path(p, bases=()):
    """Return an existing absolute path for p if one can be found, else the best guess."""
    if not p:
        return p
    p = p.strip()
    p = p.split(" [")[0].strip()            # master_lookup appends "[SUPERSEDES ...]" notes to one path
    p = os.path.expanduser(p)
    p, _ = rewrite_path(p)
    if p.startswith("R/"):
        p = RELEASE + "/" + p[2:]
    if p.startswith("G-Drive Pro/paper1_local_runs/"):
        p = NEW_PREFIX + p[len("G-Drive Pro/paper1_local_runs/"):]
    if os.path.isabs(p):
        return p
    for b in list(bases) + [EDAIC + "/part16", EDAIC, RELEASE, EDAIC + "/part15", EDAIC + "/part17",
                            EDAIC + "/variants", EDAIC + "/part14", EDAIC + "/pitt_all", RELEASE + "/overnight2",
                            RELEASE + "/omni_final", "<local data dir>", NEW_PREFIX.rstrip("/")]:
        cand = os.path.join(b, p)
        if os.path.exists(cand):
            return cand
    return os.path.join(RELEASE, p)


def first_path(s):
    """Pick the first path-like token out of a free-text 'file' cell (some cells hold 'a + b')."""
    if not s:
        return ""
    s = s.strip()
    part = re.split(r" \+ | \| ", s)[0].strip()
    return part


# ----------------------------------------------------------------------------- numbers
def to_float(x):
    if x is None:
        return None
    s = str(x).strip().replace("−", "-")
    if s in ("", "NA", "NaN", "nan", "None", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def auc_rank(y, s):
    """Tie-averaged rank AUC (Mann-Whitney), identical to sklearn roc_auc_score for binary y."""
    pairs = [(float(v), int(l)) for v, l in zip(s, y)]
    n1 = sum(1 for _, l in pairs if l == 1)
    n0 = len(pairs) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    r1 = sum(ranks[i] for i in range(len(pairs)) if pairs[i][1] == 1)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def rnd(x, dp):
    """Round half away from zero at dp decimals on the 4 dp file value (avoids binary float surprises)."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-dp)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def read_csv(path, delim=","):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))


# ----------------------------------------------------------------------------- canonical fields
def canon_model(text):
    t = (text or "").lower()
    if re.search(r"qwen3|q3o\b|q3o_|\bq3o", t):
        return "Qwen3-Omni"
    if re.search(r"qwen2\.5|\bo25|o25_|o25t|qwen2\.5-omni|\bomni\b|omni[-_ ]|omnisft|\bomni$", t):
        return "Qwen2.5-Omni"
    if re.search(r"qwen2-audio|qwen2audio|qwen2 audio|\bq2a|q2a_|q2at", t):
        return "Qwen2-Audio"
    if re.search(r"flamingo ?2|audioflamingo2|\baf2", t):
        return "Audio Flamingo 2"
    if re.search(r"flamingo ?3|audioflamingo3|\baf3", t):
        return "Audio Flamingo 3"
    if "kimi" in t:
        return "Kimi-Audio"
    if "egemaps" in t:
        return "eGeMAPS"
    return ""


def canon_dataset(text):
    t = (text or "").lower().replace("‐", "-")
    if re.search(r"adress-?_?2020|adress 2020", t):
        return "adress2020"
    if "adresso" in t:
        return "adresso"
    if re.search(r"pc-?gita|pcgita", t):
        return "pcgita"
    if "neurovoz" in t:
        return "neurovoz"
    if re.search(r"mdvr|\bkcl", t):
        return "kcl"
    if re.search(r"pitt[_ ]all|all[- ]708|\b708\b|pitt_all", t):
        return "pitt_all"
    if "pitt" in t or "dementiabank" in t:
        return "pitt"
    if re.search(r"e-?daic|edaic|depression", t):
        if re.search(r"part ?14|conflict_v2|p14_|a_edaic_part14|table 2|\b483\b|\b966\b|conflict-vs-agreement|conflict minus agreement paired", t):
            return "edaic_conflict_v2"
        if re.search(r"edaic_conflict|conflict set|390", t):
            return "edaic_conflict_390"
        if re.search(r"mid300|300 ?s\b|middle 300", t):
            return "edaic_mid300"
        if re.search(r"mid30\b|mid30_|middle 30", t):
            return "edaic_mid30"
        if re.search(r"edaic_?full|edaicfull|full window|full,|900 ?s|full stitched|\bfull\b", t):
            return "edaic_full"
        return "edaic"
    return ""


STREAM_RULES = [
    ("lora_projector_ft", r"lora plus projector|lora\+projector|abl2|both"),
    ("lora_ft", r"\blora\b|abl_pitt_lora|abl_pitt_oof"),
    ("egemaps", r"egemaps"),
    ("age_alone", r"age alone|age_alone"),
    ("transcript", r"transcript|text-prompt|text only|text_only|\btext\b|_text\b|o25t|q2at"),
    ("projector_ft", r"projector[- ]?(only )?fine|projector-ft|projector ft|\bsft\b|omnisft|fine[- ]tune|finetune|retrain"),
    ("projector_probe", r"projector probe|proj probe|\bproj\b|_proj"),
    ("lm_probe", r"\blm probe|llm probe|\bllm\b|language[- ]model|_llm"),
    ("answer_state_probe", r"answer-state|answer state|\bans probe|ans_nested|ans_final|\bans\b|_ans\b|_ans_"),
    ("encoder_probe", r"encoder probe|enc probe|\benc\b|_enc|encoder"),
    ("zero_shot", r"zero[- ]?shot|zeroshot|audio answer|answer auc|audio conflict|audio agreement|\baudio\b"),
]


MODEL_NAME_RE = re.compile(r"qwen2-audio(-7b-instruct)?|kimi-audio|qwen2\.5-omni(-7b)?|qwen3-omni(-30b-a3b)?|audio ?flamingo ?\d|audioflamingo\d", re.I)


def canon_stream(text):
    t = MODEL_NAME_RE.sub(" ", (text or "")).lower()
    for name, pat in STREAM_RULES:
        if re.search(pat, t):
            return name
    return ""


def canon_arm(text):
    t = (text or "").lower().strip()
    if re.search(r"conflict[_ -]minus[_ -]agreement|conflict-agreement|conflict minus agreement", t):
        return "conflict-agreement"
    tail = re.sub(r"\s*(arm|auc|\(.*\))\s*$", "", t).strip()
    if tail.endswith("conflict"):
        return "conflict"
    if tail.endswith("agreement"):
        return "agreement"
    if tail.endswith("pooled"):
        return "all"
    if "conflict" in t and "agreement" not in t:
        return "conflict"
    if "agreement" in t and "conflict" not in t:
        return "agreement"
    return "all"


def is_delta(text):
    t = (text or "").lower()
    return bool(re.search(r"minus|paired|delta|\bgap\b|_gap|change vs|diff", t))


METRIC_RULES = [
    ("answer_mass", r"answer mass|answer_mass|\bmass\b"), ("yes_rate", r"yes[_ ]rate|share .*yes|yes on"),
    ("kappa", r"kappa"), ("raw_agreement", r"raw agreement"), ("cosine", r"\bcos\b|cos\(|cosine|random-direction floor|randfloor"),
    ("pearson", r"pearson|spearman"), ("share", r"\bshare\b|same_decision|same decision"), ("wer", r"\bwer\b|disagree"),
    ("count", r"layers \(of|number of|pairs\b|clips\b\s*\||speakers\b\s*\||trainable params|minutes|wall-clock|memory|resolvable|containing"),
    ("threshold", r"threshold"), ("mean_feature", r"mean_pd|mean_control|logprob|compression|n_words|_mean_|median_"),
]


def canon_metric(text):
    t = (text or "").lower()
    for name, pat in METRIC_RULES:
        if re.search(pat, t):
            return name
    return "auc"


def canon_level(text):
    t = (text or "").lower()
    return "speaker" if re.search(r"speaker[- ]level|speaker level|mean p_yes\)", t) else "clip"
