#!/usr/bin/env python3
"""Read-only check of the specific two-pass, single-stream Stage 4 NSYS trace."""
import argparse
import json
import math
import sqlite3
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('sqlite')
p.add_argument('--warmup', type=int, default=5)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--iterations', type=int, default=100)
p.add_argument('--first-grid', type=int, default=5)
a = p.parse_args()
c = sqlite3.connect(f'file:{a.sqlite}?mode=ro', uri=True)
rows = c.execute('SELECT k.start,k.end,k.gridX,k.streamId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                 'JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').fetchall()
ops = 1 + a.repeats * (a.warmup + a.iterations + 1)
assert len(rows) == 2 * ops and len({r[3] for r in rows}) == 1
assert all('reduce_pass' in r[4] for r in rows)
assert [r[2] for r in rows] == [a.first_grid, 1] * ops
pairs = list(zip(rows[::2], rows[1::2]))
assert all(first[1] <= second[0] for first, second in pairs)


def summary(values):
    values = sorted(values)
    return dict(min=values[0], median=statistics.median(values), p95=values[math.ceil(.95 * len(values)) - 1])


result = {'kernel_count': len(rows), 'logical_op_count': ops, 'units': 'us', 'scope': 'instrumented trace only',
          'pass1_active': summary([(x[1]-x[0])/1000 for x,y in pairs]),
          'pass2_active': summary([(y[1]-y[0])/1000 for x,y in pairs]),
          'paired_active_sum': summary([(x[1]-x[0]+y[1]-y[0])/1000 for x,y in pairs]),
          'paired_enclosing_span': summary([(y[1]-x[0])/1000 for x,y in pairs]),
          'formal_batch_enclosing_span_per_op': []}
for repeat in range(a.repeats):
    first_op = 1 + repeat * (a.warmup + a.iterations + 1) + a.warmup
    batch = pairs[first_op:first_op+a.iterations]
    result['formal_batch_enclosing_span_per_op'].append((batch[-1][1][1]-batch[0][0][0]) / (1000*a.iterations))
print(json.dumps(result, indent=2))
