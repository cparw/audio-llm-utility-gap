#!/usr/bin/env python3
"""Find the independent-verifier twin of every PART 16 and PART 17 fragment row in summary_all.csv.

Reads only. For each fragment row it searches the verifier outputs of that task (the verify/ folders,
pod-side VERIFY files, sibling verify logs) for a record where the VERIFIER's own number equals the
fragment value at 4 dp, and, when the row has an interval, where the verifier's own lo and hi sit in
the same record. Numbers that are only a copy of the claimed value ("claimed", "theirs", "original",
"orig", "task", "POD4_value", "their_*") are never used as the twin; they are used only to spot a
verifier that recomputed a different number (a disagreement).

Output: twin_map.csv (one line per fragment row) in OUTDIR.
"""
import csv, json, re, os, sys, glob, math
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

R = '<local data dir>/release'
P16 = R + '/edaic_rerun/part16'
P17 = R + '/edaic_rerun/part17'
SUMMARY = '<local data dir>/release_from_mac/summary/summary_all.csv'
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else '.'

NUM = r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'
THEIR_KEY = re.compile(r'(^master$|^master_n$|^json_auc$|^figure_answer$|^saved|^published|^primary$|claimed|theirs|their_|^their|original|^orig$|^orig_|_orig$|^task$|POD4_|published|fragment|_pod$|per_repeat_pod|value_computed|^expected)', re.I)
MINE_KEY = re.compile(r'(mine|^my_|_my_|recomputed|VERIFIER|verifier|value_verified|^MW$|trap)', re.I)

# --------------------------------------------------------------- candidate twin files per task
def G(*pats):
    out = []
    for p in pats:
        out += sorted(glob.glob(p))
    return [f for f in out if os.path.isfile(f)]

V16, V17 = P16 + '/verify', P17 + '/verify'
TWIN_FILES = {
    'EDAICFULL': G(V17 + '/T4_verify.log', V17 + '/T7_verify.json', V17 + '/T7b_verify_out/T7b_verify.json', V16 + '/MEDAIC_verify_results.json'),
    'M1': G(V16 + '/M1_verify.json'),
    'M10': G(V16 + '/M10_verify.json'),
    'M11': G(V16 + '/M11_verify_perclip_mine.json', V16 + '/M11_verify_asr_mine.json', V16 + '/M11_verify_asr.log'),
    'M2': G(V16 + '/M2_verify_out.json'),
    'M3': G(V17 + '/T5_verify.log', V17 + '/T1_verify.log'),
    'M4': G(V16 + '/M4_verify_out.json', P16 + '/M4_verify.json'),
    'M5': G(V16 + '/M5_verify_out.json', V17 + '/T3_verify.log'),
    'M7': G(V16 + '/M7_verify_out.json'),
    'M7fix': G(V17 + '/T4_verify.log'),
    'M8': G(V16 + '/M8_verify_compare.csv', V16 + '/M8_verify_cells.csv'),
    'M9': G(V16 + '/M9_verify_cells.json'),
    'MEDAIC': G(V16 + '/MEDAIC_verify_results.json'),
    'POD1': G(V16 + '/POD1_verify_out.json', P16 + '/POD1/POD1_VERIFY.csv'),
    'POD2': G(V16 + '/POD2_verify_out.json'),
    'POD3': G(V16 + '/POD3_verify_out.json', V16 + '/POD3_partF_out.json', V17 + '/T3_verify_others.log', V17 + '/T4_verify.log'),
    'POD4': G(V16 + '/POD4_VERIFIER_SUMMARY.tsv', V16 + '/POD4_verify_results.json', P16 + '/POD4/VERIFY_POD4*.json', P16 + '/POD4/VERIFY_POD4_SUMMARY.tsv'),
    'POD4B': G(V16 + '/POD4B_verify_out_integers.json', V16 + '/POD4B_verify_adresso_align.json', V16 + '/POD4B_verify_cha.json', V16 + '/POD4B_verify_pittoverlap.json'),
    'Q3Ofix': G(V17 + '/T3_verify_others.log', V17 + '/T3_verify_others.json'),
    'SEED': G(V16 + '/POD1_verify_out.json', P16 + '/POD1/POD1_VERIFY.csv', V16 + '/POD3_verify_out.json', V16 + '/POD3_partF_out.json', V17 + '/T1_verify_comparison.csv', V17 + '/T3_verify_others.log', V17 + '/T4_verify.log'),
    'T1': G(V17 + '/T1_verify_comparison.csv', V17 + '/T1_verify.json', V17 + '/T1_verify.log'),
    'T2': G(V17 + '/T2_verify_compiled.json'),
    'T3': G(V17 + '/T3_verify.log', V17 + '/T3_verify.json', V17 + '/T3_verify_extra.log'),
    'T4': G(V17 + '/T4_verify.log', V17 + '/T4_verify.json', V17 + '/T7b_verify_out/T7b_verify.json'),
    'T5': G(V17 + '/T5_verify.log', V17 + '/T5_verify.json'),
    'T6': G(V17 + '/T6_verify_cells.csv', V17 + '/T6_verify.json', V17 + '/T6_verify.log'),
    'T7': G(V17 + '/T7_verify.json', V17 + '/T7_verify_sk190.json', V17 + '/T7b_verify_out/T7b_verify.json'),
    'T7b': G(V17 + '/T7b_verify_out/T7b_verify.json'),
}

# --------------------------------------------------------------- record extraction
def fnum(s):
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

class Rec:
    __slots__ = ('file', 'loc', 'ctx', 'mine', 'their', 'pairs', 'text', 'keys')
    def __init__(self, file, loc):
        self.file, self.loc, self.ctx, self.mine, self.their, self.pairs, self.text, self.keys = file, loc, [], [], [], [], '', {}

PAIR_SUBS = [('claimed_ci', 'mine_ci'), ('claimed', 'mine'), ('theirs', 'mine'), ('their_', 'my_'), ('their_', 'mine_'),
             ('original_value', 'recomputed_value'), ('orig', 'mine'), ('POD4_', 'VERIFIER_'), ('task', 'verifier'),
             ('per_repeat_pod', 'per_repeat_mine'), ('primary', 'verify'), ('_pod', '_mine'), ('value_computed', 'value_verified')]

def flat(v, key):
    """numbers under a key; lists become key[i]; numeric strings with spaces split."""
    out = []
    if isinstance(v, bool) or v is None:
        return out
    if isinstance(v, (int, float)):
        if math.isfinite(v): out.append((key, float(v)))
    elif isinstance(v, str):
        parts = v.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
        if parts and all(fnum(p) is not None for p in parts) and len(parts) <= 6:
            if len(parts) == 1: out.append((key, float(parts[0])))
            else: out += [('%s[%d]' % (key, i), float(p)) for i, p in enumerate(parts)]
    elif isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v) and len(v) <= 8:
        out += [('%s[%d]' % (key, i), float(x)) for i, x in enumerate(v)]
    return out

def side(key):
    k = key.split('[')[0]
    if THEIR_KEY.search(k) and not MINE_KEY.search(k): return 'their'
    return 'mine'

def add_pairs(rec, d):
    keys = list(d.keys())
    for k in keys:
        for a, b in PAIR_SUBS:
            if a in k:
                k2 = k.replace(a, b)
                if k2 in d and k2 != k:
                    A, B = dict(flat(d[k], k)), dict(flat(d[k2], k2))
                    for (ka, va), (kb, vb) in zip(sorted(A.items()), sorted(B.items())):
                        rec.pairs.append((va, vb, '%s->%s' % (ka, kb)))

def recs_json(path):
    out = []
    def walk(o, loc, parent_ctx):
        if isinstance(o, dict):
            r = Rec(path, loc)
            r.ctx = parent_ctx + [loc.split('/')[-1]] + [str(k) for k in o.keys()] + [str(v) for v in o.values() if isinstance(v, str)]
            for k, v in o.items():
                for kk, x in flat(v, str(k)):
                    (r.mine if side(kk) == 'mine' else r.their).append((kk, x))
            add_pairs(r, o)
            r.text = json.dumps({k: v for k, v in o.items() if not isinstance(v, (dict,)) and not (isinstance(v, list) and len(v) > 8)})[:400]
            r.keys = {str(k): v for k, v in o.items()}
            if r.mine or r.their: out.append(r)
            for k, v in o.items():
                if isinstance(v, (dict, list)): walk(v, loc + '/' + str(k), parent_ctx + [str(k)])
        elif isinstance(o, list):
            if all(isinstance(x, (str, int, float)) for x in o) and len(o) in (3, 4) and isinstance(o[0], str) and all(fnum(x) is not None for x in o[1:]):
                # e.g. M5 disagree entries [what, mine, theirs]
                r = Rec(path, loc); r.ctx = parent_ctx + [o[0]]
                r.mine.append(('list[1]', float(o[1]))); r.their.append(('list[2]', float(o[2]))); r.pairs.append((float(o[2]), float(o[1]), 'list'))
                out.append(r)
            for i, v in enumerate(o):
                if isinstance(v, (dict, list)): walk(v, loc + '[%d]' % i, parent_ctx)
    walk(json.load(open(path)), '', [os.path.basename(path)])
    return out

def recs_table(path):
    out = []
    with open(path, newline='') as f:
        txt = f.read()
    delim = '\t' if path.endswith('.tsv') else ','
    rd = list(csv.reader(txt.splitlines(), delimiter=delim))
    hdr = rd[0]
    for i, row in enumerate(rd[1:], start=2):
        d = dict(zip(hdr, row))
        r = Rec(path, 'line %d' % i)
        r.ctx = [os.path.basename(path)] + [c for c in row if fnum(c.split(' ')[0]) is None]
        for k, v in d.items():
            for kk, x in flat(v, k):
                (r.mine if side(kk) == 'mine' else r.their).append((kk, x))
        add_pairs(r, d)
        r.text = (delim.join(row))[:400]
        if r.mine or r.their: out.append(r)
    return out

CLAIM_PAT = [re.compile(r'claimed\s+(%s)\s*\|\s*MW\s+(%s)' % (NUM, NUM)),
             re.compile(r'claimed\s+(%s)\s+mine\s+(%s)' % (NUM, NUM)),
             re.compile(r'claimed\s+(%s)\s*\|\s*computed\s+(%s)' % (NUM, NUM))]

def recs_log(path):
    out = []
    lines = open(path, errors='replace').read().split('\n')
    COPY = re.compile(r'->|^\s*[+~]\s|SUPERSEDES|cols changed|^added \d|^changed \d|^\[src\]|^dropped|^rows \(excl|duplicate keys|figure script answers|run json|master_lookup \(|readout csv|json llm per_repeat|json enc per_repeat|^old json|pod_sync json')
    for i, l in enumerate(lines):
        if not re.search(r'\d', l) or COPY.search(l): continue
        r = Rec(path, 'line %d' % (i + 1)); r.text = l.strip()[:400]
        for p in CLAIM_PAT:
            for m in p.finditer(l):
                r.pairs.append((float(m.group(1)), float(m.group(2)), 'claimed->mine'))
        l2 = re.sub(r'\(claimed[^)]*\)', ' ', l)
        l2 = re.sub(r'claimed\s+(\[[^\]]*\]|%s|None)' % NUM, ' ', l2)
        for m in re.finditer(r'\(claimed[^)]*\)|claimed\s+%s' % NUM, l):
            for x in re.findall(NUM, m.group(0)): r.their.append(('claimed', float(x)))
        l2 = re.sub(r'np\.float64\(', '(', l2)
        for x in re.findall(NUM, l2):
            r.mine.append(('log', float(x)))
        r.ctx = [os.path.basename(path)] + re.findall(r'[A-Za-z][A-Za-z0-9_\-]+', l)
        if r.mine or r.their: out.append(r)
    # three-line windows so a value and its interval on neighbouring lines can co-match
    win = []
    for i in range(len(out) - 2):
        a, b, c = out[i], out[i + 1], out[i + 2]
        la, lc = int(a.loc.split()[1]), int(c.loc.split()[1])
        if lc - la > 3: continue
        r = Rec(path, 'lines %d-%d' % (la, lc)); r.ctx = a.ctx + b.ctx + c.ctx
        r.mine = a.mine + b.mine + c.mine; r.their = a.their + b.their + c.their; r.pairs = a.pairs + b.pairs + c.pairs
        r.text = ' || '.join([a.text, b.text, c.text])[:600]
        r.loc += ' (window)'
        win.append(r)
    return out + win

def records(path):
    if path.endswith('.json'): return recs_json(path)
    if path.endswith('.csv') or path.endswith('.tsv'): return recs_table(path)
    return recs_log(path)

# --------------------------------------------------------------- matching
def dec(s):
    try: return Decimal(str(s).strip().lstrip('+'))
    except (InvalidOperation, ValueError): return None

def q4(x):
    return Decimal(repr(float(x))).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

def eq(x, v):
    """verifier number x equals fragment value v (Decimal) at the fragment's precision, 4 dp at most."""
    if v is None or not v.is_finite() or abs(x) > 1e12 or abs(v) > Decimal(10) ** 12: return False
    e = -v.as_tuple().exponent if v.as_tuple().exponent < 0 else 0
    if e == 0:
        return abs(x - float(v)) < 1e-9
    e = min(e, 4)
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-e), rounding=ROUND_HALF_UP) == v.quantize(Decimal(1).scaleb(-e), rounding=ROUND_HALF_UP) or abs(x - float(v)) <= 0.5 * 10 ** (-e) + 1e-12

TOK = re.compile(r'[a-z0-9]+')
STOP = set('the a an of and or in on to for with by from at vs is auc all arm'.split())
def toks(s):
    return set(t for t in TOK.findall(s.lower()) if t not in STOP and len(t) > 1)

GENERIC_IDS = {'M10', 'M11', 'M4', 'M8', 'M7fix', '3b_fix'}

def keyed(row, r):
    rid = row['id'].split('#')[0]
    if rid in GENERIC_IDS or len(rid) < 5: return False
    t = ' '.join(r.ctx) + ' ' + r.text
    return re.search(r'(?<![A-Za-z0-9_])' + re.escape(rid) + r'(?![A-Za-z0-9_])', t) is not None

def ci_elsewhere(row, r, recs, lo, hi):
    """lo and hi recomputed by the verifier in other records of the same file that belong to this row:
    (a) JSON: the same key name holds [lo, hi] in a sibling dict (M10 point / ci_integers);
    (b) a record keyed to the row id; (c) list-of-checks files: records whose text is this record's
    text with the quantity word swapped for 'CI lo' / 'CI hi' (POD1 verifier)."""
    same = [x for x in recs if x.file == r.file and x is not r]
    # (a)
    for k, v in r.keys.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and eq(float(v), dec(row['value_computed'])):
            for x in same:
                w = x.keys.get(k)
                if isinstance(w, list) and len(w) == 2 and all(isinstance(z, (int, float)) for z in w) and eq(w[0], lo) and eq(w[1], hi):
                    return 'same key %s in %s' % (k, x.loc)
    # (b)
    kl = [x for x in same if keyed(row, x)]
    fl = any(any(eq(z, lo) for _, z in x.mine) for x in kl); fh = any(any(eq(z, hi) for _, z in x.mine) for x in kl)
    if kl and fl and fh: return 'record keyed to the row id'
    # (c)
    base = None
    what = r.keys.get('what') if r.keys else None
    if isinstance(what, str):
        base = re.sub(r'\s*(AUC|auc)\s*$', '', what).strip()
    elif r.text:
        mm = re.match(r'^(.*?)(?:,|\s)(\S*AUC)\b', r.text)
        base = mm.group(1).strip() if mm else None
    rid = row['id'].split('#')[0]
    if rid.endswith('_auc'):
        stem = rid[:-4]
        fl = any(x.text.startswith(stem + '_lo') and any(eq(z, lo) for _, z in x.mine) for x in same)
        fh = any(x.text.startswith(stem + '_hi') and any(eq(z, hi) for _, z in x.mine) for x in same)
        if fl and fh: return 'id_lo / id_hi rows of the same verifier table'
    if base and len(base) > 6:
        gl = gh = False
        for x in same:
            t = (x.keys.get('what') if x.keys and isinstance(x.keys.get('what'), str) else x.text) or ''
            if t.startswith(base) and re.search(r'CI lo|\blo\b', t) and any(eq(z, lo) for _, z in x.mine): gl = True
            if t.startswith(base) and re.search(r'CI hi|\bhi\b', t) and any(eq(z, hi) for _, z in x.mine): gh = True
        if gl and gh: return 'CI lo / CI hi checks of the same quantity'
    return ''

def match_row(row, recs):
    v, lo, hi = dec(row['value_computed']), dec(row['lo']), dec(row['hi'])
    rt = toks(row['id'] + ' ' + row['what'])
    best = None
    for r in recs:
        mvk = [(k, x) for k, x in r.mine if eq(x, v)]
        if not mvk: continue
        mv = [x for k, x in mvk]
        ml = lo is not None and any(eq(x, lo) for k, x in r.mine)
        mh = hi is not None and any(eq(x, hi) for k, x in r.mine)
        ov = len(rt & toks(' '.join(r.ctx) + ' ' + r.text))
        ky = keyed(row, r)
        score = 10 + 5 * ml + 5 * mh + 12 * ky + min(ov, 8) - (2 if 'window' in r.loc else 0)
        cand = [score, ml, mh, ov, r, mv[0], ky, '', mvk[0][0]]
        if best is None or cand[0] > best[0]: best = cand
    if best is not None and lo is not None and not (best[1] and best[2]):
        how = ci_elsewhere(row, best[4], recs, lo, hi)
        if how: best[1] = best[2] = True; best[7] = how
    dis = None
    if best is None or (lo is not None and not (best[1] and best[2])):
        for r in recs:
            for tv, mvv, lab in r.pairs:
                if eq(tv, v) and not eq(mvv, v):
                    ov = len(rt & toks(' '.join(r.ctx) + ' ' + r.text)) + 12 * keyed(row, r)
                    c = (ov, r, mvv, lab)
                    if dis is None or c[0] > dis[0]: dis = c
    return best, dis

def main():
    rows = [r for r in csv.DictReader(open(SUMMARY, newline='')) if r['part'] in ('16', '17')]
    cache = {}
    out = []
    for i, row in enumerate(rows):
        t = row['task/pod']
        files = TWIN_FILES.get(t, [])
        recs = []
        for f in files:
            if f not in cache:
                try: cache[f] = records(f)
                except Exception as e: print('READ FAIL', f, e); cache[f] = []
            recs += cache[f]
        best, dis = match_row(row, recs)
        o = dict(part=row['part'], task=t, id=row['id'], value=row['value_computed'], lo=row['lo'], hi=row['hi'], role=row['paper_role'])
        if best:
            score, ml, mh, ov, r, x, ky, how, mkey = best
            o.update(twin_value=repr(x), twin_file=r.file, twin_loc=r.loc, ci_twinned=('yes' if (ml and mh) else ('no interval in row' if not row['lo'] else 'no')),
                     ci_how=how, keyed=ky, twin_key=mkey, ctx_overlap=ov, score=score, twin_text=r.text, twin_lo='', twin_hi='')
            if ml and mh:
                o['twin_lo'] = row['lo']; o['twin_hi'] = row['hi']
        else:
            o.update(twin_value='', twin_file='', twin_loc='', ci_twinned='', ci_how='', keyed='', twin_key='', ctx_overlap='', score='', twin_text='', twin_lo='', twin_hi='')
        if dis:
            ov, r, mvv, lab = dis
            o.update(dis_value=repr(mvv), dis_file=r.file, dis_loc=r.loc, dis_pair=lab, dis_overlap=ov, dis_text=r.text)
        else:
            o.update(dis_value='', dis_file='', dis_loc='', dis_pair='', dis_overlap='', dis_text='')
        out.append(o)
    cols = list(out[0].keys())
    with open(os.path.join(OUTDIR, 'twin_map_auto.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)
    from collections import Counter
    print('rows', len(out))
    print('matched', sum(1 for o in out if o['twin_value']), 'of which CI yes', sum(1 for o in out if o['ci_twinned'] == 'yes'))
    print('no match', sum(1 for o in out if not o['twin_value']), 'disagree-candidates', sum(1 for o in out if o['dis_value'] and not o['twin_value']))
    print(Counter((o['task'], bool(o['twin_value'])) for o in out))

if __name__ == '__main__':
    main()
