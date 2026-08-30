import json, collections
from pathlib import Path
rows=[]
for p in Path('results').glob('*.jsonl'):
    for ln in p.read_text().splitlines():
        if ln.strip(): rows.append(json.loads(ln))
# schema consistency across files
keys=collections.Counter(tuple(sorted(r)) for r in rows)
for k,v in keys.items(): print(v, k)
print("--- labels ---", collections.Counter(r['label'] for r in rows))
# fields used by dashboard
for f in ['tokens','ttft_s','gen_tps','prefill_tps','total_s','accuracy','correct','total','category','ts','model']:
    miss=[r for r in rows if f not in r]
    if miss: print('MISSING',f,len(miss))
# ts format sanity
import re
bad=[r['ts'] for r in rows if not re.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\+00:00|Z)$', r['ts'])]
print('odd ts:', bad[:5])
