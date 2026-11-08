"""GPU sample schema and boundary checks."""
import csv
import math

import statistics
from config import SMOKE

def verify_csv(path, copies, repeats):
    rows = list(csv.DictReader(path.open()))
    variants = {"scalar", "grid_stride", "runtime"} if copies else {"scalar", "grid_stride"}
    assert {r["variant"] for r in rows} == variants
    assert len(rows) == len(variants) * (repeats + 3)
    for r in rows:
        assert r["validation"] == "PASS" and r["gpu"] and r["build_flags"] and r["compiler"]
        size = int(r["n"]) * 4 * (2 if copies else 3)
        assert int(r["requested_bytes"]) == size
        latency = float(r["event_span_us_per_op"])
        assert latency > 0 and float(r["host_e2e_us"]) > 0
        assert abs(float(r["effective_GBps_at_event_stat"]) - size / (latency * 1000)) < 1e-6


def verify_global_csv(path, repeats):
    rows = list(csv.DictReader(path.open()))
    variants = {r["variant"] for r in rows}
    assert len(rows) == len(variants) * (repeats + 3)
    for r in rows:
        assert r["validation"] == "PASS" and r["gpu"] and r["compiler"]
        n, stride, offset, width = (int(r[k]) for k in ("n", "stride", "offset_elements", "vector_width"))
        assert int(r["requested_bytes"]) == 8 * n
        counts = []
        for read in (True, False):
            active = r["direction"] != ("scatter" if read else "gather")
            s, off = (stride, offset) if active else (1, 0)
            counts.append((n - 1) * s + off + 1)
            sectors = {(4 * (off + i * width * s) + b) // 32
                       for i in range(min(32, n // width)) for b in range(width * 4)}
            key = "model_read_sectors_first_instruction" if read else "model_write_sectors_first_instruction"
            assert int(r[key]) == len(sectors)
        assert int(r["allocation_bytes"]) == sum(counts) * 4
        t = float(r["event_span_us_per_op"])
        assert t > 0 and math.isclose(float(r["effective_GBps_at_event_stat"]), 8 * n / (1000 * t), rel_tol=1e-9)
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        summary = {r["statistic"]: r for r in subset if r["statistic"] != "raw"}
        for key in ("event_span_us_per_op", "host_batch_us_per_op", "host_e2e_us"):
            raw = sorted(float(r[key]) for r in subset if r["statistic"] == "raw")
            assert len(raw) == repeats and all(math.isfinite(x) and x > 0 for x in raw)
            for statistic, expected in (("min", raw[0]), ("median", statistics.median(raw)), ("p95", raw[math.ceil(.95 * len(raw)) - 1])):
                assert math.isclose(float(summary[statistic][key]), expected, rel_tol=1e-9)


def verify_global(session, binary):
    cases = [dict(n=7, block=1), dict(n=4099, block=31), dict(n=4099, block=1024),
             dict(n=4099, offset=4), dict(n=33, offset=2, variant="float2")]
    for direction in ("gather", "scatter", "both"):
        cases.append(dict(n=4099, stride=3, offset=7, direction=direction, variant="scalar"))
    for i, case in enumerate(cases):
        path = session.path / f"global_copy_case{i}.csv"
        opts = [x for k, v in case.items() for x in ("--" + k, str(v))]
        session.command([binary, *opts, "--warmup", "1", "--repeats", "2", "--iterations", "3", "--csv", path])
        verify_global_csv(path, 2)
    dest = session.path / "global_invalid.csv"
    for invalid in (("--n", "0"), ("--stride", "0"), ("--offset", "-1"), ("--n", "18446744073709551615"),
                    ("--variant", "float4", "--offset", "1"), ("--variant", "float2", "--stride", "2"),
                    ("--n", "1000000000"), ("--warmup", "0"), ("--variant", "unknown"), ("--direction", "invalid")):
        session.command([binary, *invalid, "--csv", dest], expected=1)
        assert not dest.exists()
    existing = session.path / "global_copy_case0.csv"
    before = existing.read_bytes()
    session.command([binary, "--csv", existing], expected=1)
    assert existing.read_bytes() == before


def verify_hierarchy(session, binary, name):
    dest = session.path / f"{name}_smoke.csv"
    session.command([binary, *SMOKE[name], "--warmup", "1", "--repeats", "2", "--iterations", "3", "--csv", dest])
    rows = list(csv.DictReader(dest.open()))
    for row in rows:
        assert row['validation'] == 'PASS' and float(row['event_span_us_per_op']) > 0
        n = int(row['n'])
        expected_bytes = 4 * n * (int(row['steps']) + 1) if name == 'cache_working_set' else 8 * n
        assert int(row['requested_bytes']) == expected_bytes
        if name == 'transpose':
            assert n == int(row['rows']) * int(row['cols'])
            if row['variant'] != 'naive':
                pitch = int(row['shared_pitch_words'])
                assert int(row['static_shared_bytes']) == 32 * pitch * 4
                assert int(row['model_distinct_banks_transposed_full_warp']) == len({i*pitch % 32 for i in range(32)})
        if name == 'local_array':
            local = int(row['local_bytes_per_thread'])
            assert local == 0 if row['variant'] == 'static_index' else local > 0
            if row['variant'] == 'register_limited':
                assert int(row['registers']) <= int(row['register_limit'])
    invalids = {
        'transpose': [('--rows','0'), ('--rows','65536','--cols','65536'), ('--block','128'), ('--n','17')],
        'cache_working_set': [('--working-set','33'), ('--n','1','--steps','1','--working-set','32'), ('--steps','0')],
        'local_array': [('--n','65537'), ('--rounds','0'), ('--shift','32')],
    }
    for invalid in invalids[name]:
        bad = session.path / f'{name}_invalid.csv'
        session.command([binary, *invalid, '--csv', bad], expected=1)
        assert not bad.exists()



def verify_execution(session, binary, name):
    from check_samples import check
    dest = session.path / f'{name}_smoke.csv'
    session.command([binary, *SMOKE[name], '--warmup', '1', '--repeats', '2', '--iterations', '3', '--csv', dest])
    check(dest)
    rows = list(csv.DictReader(dest.open()))
    for row in rows:
        n, block = int(row['n']), int(row['block'])
        occupancy = float(row['theoretical_occupancy'])
        assert 0 < occupancy <= 1
        if name == 'reduction':
            current, expected_bytes, passes = n, 0, 0
            while True:
                following = (current + block * int(row['items_per_thread']) - 1) // (block * int(row['items_per_thread']))
                expected_bytes += 4 * (current + following)
                passes += 1
                if following == 1:
                    break
                current = following
            assert int(row['requested_bytes']) == expected_bytes and int(row['passes']) == passes
            assert float(row['max_absolute_error']) <= float(row['absolute_tolerance'])
        else:
            assert int(row['requested_bytes']) == 8 * n
            assert float(row['flops']) == 2 * n * 64 * int(row['rounds'])
            if row['register_limit']:
                assert int(row['registers']) <= int(row['register_limit'])
            assert int(row['theoretical_active_blocks_per_sm']) <= int(row['rough_resident_blocks_bound'])
    invalids = {'reduction': [('--block','31'), ('--block','96'), ('--items','0'), ('--n','16777217'), ('--pattern','bad')],
                'register_pressure': [('--n','65537'), ('--rounds','0')]}
    for invalid in [*invalids[name], ('--warmup','0'), ('--n','0'), ('--variant','bad')]:
        bad = session.path / f'{name}_invalid.csv'
        session.command([binary, *invalid, '--csv', bad], expected=1)
        assert not bad.exists()
    before = dest.read_bytes()
    session.command([binary, '--csv', dest], expected=1)
    assert dest.read_bytes() == before


def verify_instructions(session, binary, name):
    from check_samples import check
    dest = session.path / f'{name}_smoke.csv'
    session.command([binary, *SMOKE[name], '--warmup', '1', '--repeats', '2', '--iterations', '3', '--csv', dest])
    check(dest)
    rows = list(csv.DictReader(dest.open()))
    expected_variants = {'integer_division': {'runtime','constant','manual_shift'},
                         'loop_unroll': {'u1','u4','u8'}, 'branch_flow': {'uniform','interleaved'}}
    assert {r['variant'] for r in rows} == expected_variants[name]
    for row in rows:
        n, rounds = int(row['n']), int(row['rounds'])
        assert row['validation_mode'] == 'all uint32 output words exact'
        assert float(row['flops']) == 0
        count = n * rounds * int(row.get('chains', '1'))
        assert int(row['logical_divisions' if name == 'integer_division' else 'logical_updates']) == count
        assert int(row['requested_bytes']) == n * 4 * (1 + int(row.get('chains', '1')))
        if name == 'branch_flow':
            assert int(row['path_b_tasks']) == 32
            assert int(row['model_mixed_warps']) == (0 if row['variant'] == 'uniform' else 2)
    invalids = {
        'integer_division': [('--divisor','0'), ('--signed','2'), ('--divisor','7','--variant','manual_shift'), ('--divisor','3','--variant','constant')],
        'loop_unroll': [('--chains','2'), ('--rounds','0')],
        'branch_flow': [('--block','31'), ('--body','short','--rounds','2'), ('--body','bad')],
    }
    for invalid in [*invalids[name], ('--n','1048577'), ('--warmup','0'), ('--variant','bad')]:
        bad = session.path / f'{name}_invalid.csv'
        session.command([binary, *invalid, '--csv', bad], expected=1)
        assert not bad.exists()
    before = dest.read_bytes()
    session.command([binary, '--csv', dest], expected=1)
    assert dest.read_bytes() == before


def verify_async(session, binary, name):
    from check_samples import check
    dest = session.path / f'{name}_smoke.csv'
    session.command([binary, *SMOKE[name], '--warmup', '1', '--repeats', '2', '--iterations', '3', '--csv', dest])
    check(dest)
    rows = list(csv.DictReader(dest.open()))
    wanted = {'host_transfer': {'pageable','pinned'},
              'stream_pipeline': {'pageable_serial','pinned_serial','pinned_pipeline','pageable_pipeline'},
              'concurrent_kernels': {'serial','concurrent'}}[name]
    if name == 'concurrent_kernels' and rows[0]['concurrent_kernels_capability'] == '0':
        wanted = {'serial'}
    assert {r['variant'] for r in rows} == wanted
    for row in rows:
        n = int(row['n'])
        assert row['stream_flags'] == 'cudaStreamNonBlocking'
        assert int(row['requested_bytes']) == n * (4 if name == 'host_transfer' else 8)
        if name == 'stream_pipeline':
            chunk, buffers = int(row['chunk_elements']), int(row['buffers'])
            assert int(row['chunks_per_op']) == (n + chunk - 1) // chunk
            assert int(row['allocation_bytes']) == 8 * min(n, chunk) * buffers
            assert int(row['h2d_bytes_per_op']) == int(row['d2h_bytes_per_op']) == 4*n
            expected_workers = 1 if 'serial' in row['variant'] else int(row['streams_requested'])
            assert int(row['worker_streams']) == expected_workers
        if name != 'host_transfer':
            assert float(row['flops']) == 2 * n * int(row['rounds'])
        if name == 'concurrent_kernels':
            assert int(row['min_elements_per_job']) == n // int(row['jobs'])
    invalids = {
        'host_transfer': [('--direction','bad'), ('--n','8388609')],
        'stream_pipeline': [('--chunk','0'), ('--streams','0'), ('--buffers','0'), ('--order','bad'), ('--n','4097','--chunk','1'),
                            ('--max-mib','1','--rounds','1','--n','1048576','--chunk','262144','--buffers','3')],
        'concurrent_kernels': [('--n','1','--jobs','2'), ('--jobs','9'), ('--rounds','0')],
    }
    for invalid in [*invalids[name], ('--warmup','0'), ('--variant','bad')]:
        bad = session.path / f'{name}_invalid.csv'
        session.command([binary, *invalid, '--csv', bad], expected=1)
        assert not bad.exists()
    before = dest.read_bytes()
    session.command([binary, '--csv', dest], expected=1)
    assert dest.read_bytes() == before


def verify_gemm(session, binary):
    from check_samples import check
    dest = session.path / 'gemm_smoke.csv'
    session.command([binary, *SMOKE['gemm'], '--warmup', '1', '--repeats', '2', '--iterations', '2', '--csv', dest])
    check(dest)
    rows = list(csv.DictReader(dest.open()))
    assert {r['variant'] for r in rows} == {'naive', 'coalesced', 'shared_tiled', 'register_tiled', 'unrolled'}
    for r in rows:
        m,n,k = (int(r[x]) for x in ('matrix_m','matrix_n','matrix_k'))
        assert int(r['n']) == m*n and float(r['flops']) == 2*m*n*k
        assert int(r['allocation_bytes']) == 4*(m*k+k*n+m*n)
        tile,bm,bn = (int(r[x]) for x in ('tile','output_tile_rows','output_tile_cols'))
        direct = r['variant'] in ('naive','coalesced')
        loads = 2*m*n*k if direct else m*k*((n+bn-1)//bn)+k*n*((m+bm-1)//bm)
        assert int(r['model_global_load_elements']) == loads
        assert int(r['requested_bytes']) == 4*(loads+m*n)
        assert int(r['static_shared_bytes']) == (0 if direct else (bm+bn)*tile*4)
        assert float(r['validation_max_scaled_error']) <= 1
    for invalid in [('--rows','0'),('--cols','4097'),('--depth','0'),('--tile','17'),('--tile','32'),
                    ('--pattern','bad'),('--reg-rows','3'),('--n','17'),('--rows','4096','--cols','4096','--depth','4096'),
                    ('--max-mib','1','--rows','512','--cols','512','--depth','512'),('--warmup','0'),('--variant','bad')]:
        bad = session.path / 'gemm_invalid.csv'
        session.command([binary, *invalid, '--csv', bad], expected=1)
        assert not bad.exists()
    before = dest.read_bytes()
    session.command([binary, '--csv', dest], expected=1)
    assert dest.read_bytes() == before
