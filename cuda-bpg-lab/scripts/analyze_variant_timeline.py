#!/usr/bin/env python3
"""Analyze ordered single-kernel cases produced by common collect(), not arbitrary traces."""
import argparse
import csv
import json
import math
import sqlite3
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('sqlite')
p.add_argument('csv')
p.add_argument('--symbol', required=True)
p.add_argument('--variants', nargs='+', required=True)
a = p.parse_args()
samples = list(csv.DictReader(open(a.csv)))
assert {r['variant'] for r in samples} == set(a.variants)
c = sqlite3.connect(f'file:{a.sqlite}?mode=ro', uri=True)
kernels = c.execute('SELECT k.start,k.end,k.gridX,k.blockX,k.streamId,s.value,k.gridY,k.gridZ,k.blockY,k.blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                    'JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').fetchall()
assert kernels and all(a.symbol in r[5] for r in kernels)

def stats(values):
    v = sorted(values)
    return {'min': v[0], 'median': statistics.median(v), 'p95': v[math.ceil(.95*len(v))-1]}

result = {'scope': 'instrumented trace; source collect() ordering checked; all times us', 'cases': []}
offset = 0
for variant in a.variants:
    subset = [r for r in samples if r['variant'] == variant]
    raw = [r for r in subset if r['statistic'] == 'raw']
    first = raw[0]
    warmup, repeats, iterations = (int(first[k]) for k in ('warmup','repeats','iterations'))
    count = 1 + repeats * (warmup+iterations+1)
    rows = kernels[offset:offset+count]
    assert len(rows) == count and len({r[4] for r in rows}) == 1
    expected_grid = (int(first.get('grid_x', first.get('grid', '0'))), int(first.get('grid_y','1')), 1)
    expected_block = (int(first.get('block_x',first['block'])), int(first.get('block_y','1')), 1)
    assert all((r[2],r[6],r[7]) == expected_grid and (r[3],r[8],r[9]) == expected_block for r in rows)
    batches = []
    for repeat in range(repeats):
        start = 1 + repeat*(warmup+iterations+1) + warmup
        batch = rows[start:start+iterations]
        batches.append({'kernel_enclosing_span_per_op': (batch[-1][1]-batch[0][0])/(1000*iterations),
                        'kernel_active_mean': sum(r[1]-r[0] for r in batch)/(1000*iterations),
                        'event_span_per_op': float(raw[repeat]['event_span_us_per_op'])})
    result['cases'].append({'variant': variant, 'kernel_count': count,
                            'all_kernel_active': stats([(r[1]-r[0])/1000 for r in rows]), 'batches': batches})
    offset += count
assert offset == len(kernels), 'unexpected extra kernels'
print(json.dumps(result, indent=2))
