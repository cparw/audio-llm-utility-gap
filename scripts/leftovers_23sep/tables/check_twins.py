#!/usr/bin/env python3
"""Second pass over twin_decisions.csv: for every row with a twin, reopen the twin file and confirm that a number
in it equals value_verified (and lo_verified / hi_verified when given) at the stated precision. Independent of
the matcher: plain regex over the file text, no record structure."""
import csv, re, sys, os
from decimal import Decimal, ROUND_HALF_UP
R = '<local data dir>/release'
NUM = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')
cache = {}
def nums(p):
    if p not in cache:
        t = open(p, errors='replace').read()
        cache[p] = [x for x in NUM.findall(t)]
    return cache[p]
def found(p, target):
    if target == '':
        return True
    if re.match(r'^(PASSED|FAILED)_', target):
        target = target.split('_', 1)[1]
    try:
        tv = Decimal(target)
    except Exception:
        return target in open(p, errors='replace').read()
    e = -tv.as_tuple().exponent if tv.as_tuple().exponent < 0 else 0
    if 'e' in target.lower():
        return any(abs(float(x) - float(tv)) <= abs(float(tv)) * 5e-4 + 1e-300 for x in nums(p) if x not in ('+', '-'))
    q = Decimal(1).scaleb(-e)
    for x in nums(p):
        try:
            if Decimal(x).quantize(q, rounding=ROUND_HALF_UP) == tv:
                return True
        except Exception:
            pass
    return False
bad = []; n = 0
for r in csv.DictReader(open(sys.argv[1], newline='')):
    if not r['twin_file']:
        continue
    p = r['twin_file'].replace('release/', R + '/', 1) if r['twin_file'].startswith('release/') else r['twin_file']
    n += 1
    ok = found(p, r['value_verified']) and found(p, r['lo_verified']) and found(p, r['hi_verified'])
    if not ok:
        bad.append((r['id'], r['value_verified'], r['lo_verified'], r['hi_verified'], p))
print('rows with a twin checked:', n, 'not found in twin file:', len(bad))
for b in bad: print(b)
