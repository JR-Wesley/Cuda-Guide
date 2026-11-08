#!/usr/bin/env python3
"""Independently recompute common experiment CSV summaries from raw samples."""
import csv
import math
from pathlib import Path
import statistics
import sys


def check(path):
    rows = list(csv.DictReader(Path(path).open()))
    assert rows, f"empty CSV: {path}"
    for variant in {r['variant'] for r in rows}:
        subset = [r for r in rows if r['variant'] == variant]
        raw = [r for r in subset if r['statistic'] == 'raw']
        summary = {r['statistic']: r for r in subset if r['statistic'] != 'raw'}
        assert set(summary) == {'min', 'median', 'p95'}
        assert len(raw) == int(raw[0]['repeats']) and len(subset) == len(raw) + 3
        assert {int(r['sample']) for r in raw} == set(range(len(raw)))
        for key in ('event_span_us_per_op', 'host_batch_us_per_op', 'host_e2e_us'):
            values = sorted(float(r[key]) for r in raw)
            assert all(math.isfinite(x) and x > 0 for x in values)
            expected = {'min': values[0], 'median': statistics.median(values),
                        'p95': values[math.ceil(.95 * len(values)) - 1]}
            for stat, value in expected.items():
                assert math.isclose(float(summary[stat][key]), value, rel_tol=1e-9), (path, variant, key, stat)
        for row in subset:
            assert row['validation'] == 'PASS' and row['gpu'] and row['compiler'] and row['build_flags']
            latency = float(row['event_span_us_per_op'])
            for numerator, rate in (('requested_bytes', 'effective_GBps_at_event_stat'), ('flops', 'GFLOPs_at_event_stat')):
                assert math.isclose(float(row[rate]), float(row[numerator]) / (1000 * latency), rel_tol=1e-9)
    print(f'PASS: {path}: raw counts, metadata, units, Min/Median/P95')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: check_samples.py samples.csv [...]')
    for arg in sys.argv[1:]:
        check(arg)
