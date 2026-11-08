#!/usr/bin/env python3
"""Validate Stage 6 source ordering using runtime correlation, then measure interval unions.

Only stream_pipeline/concurrent_kernels traces from the checked collection protocol.
All output times are us; initialization/warmup/batch/e2e remain explicitly labeled.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
import sqlite3
import statistics


def intervals(kernels, copies):
    events = defaultdict(lambda: [0, 0])
    for kind, rows in enumerate((kernels, copies)):
        for r in rows:
            assert r['end'] > r['start']
            events[r['start']][kind] += 1
            events[r['end']][kind] -= 1
    active = [0, 0]
    acc = dict(kernel_union=0, copy_union=0, copy_compute_overlap=0,
               kernel_concurrent_ge2=0, activity_union=0)
    last = min(events)
    peak = 0
    for t, changes in sorted(events.items()):
        dt = t-last
        acc['kernel_union'] += dt*(active[0] > 0)
        acc['copy_union'] += dt*(active[1] > 0)
        acc['copy_compute_overlap'] += dt*(active[0] > 0 and active[1] > 0)
        acc['kernel_concurrent_ge2'] += dt*(active[0] >= 2)
        acc['activity_union'] += dt*(sum(active) > 0)
        active = [active[i]+changes[i] for i in (0, 1)]
        assert min(active) >= 0
        peak = max(peak, active[0])
        last = t
    assert active == [0, 0]
    acc['kernel_active_sum'] = sum(r['end']-r['start'] for r in kernels)
    acc['copy_active_sum'] = sum(r['end']-r['start'] for r in copies)
    acc['activity_enclosing_span'] = max(events)-min(events)
    out = {k+'_us': v/1000 for k, v in acc.items()}
    out['peak_active_kernels'] = peak
    return out


def stats(values):
    v = sorted(values)
    return dict(min=v[0], median=statistics.median(v), p95=v[math.ceil(.95*len(v))-1])


def analyze(database, sample_path):
    samples = list(csv.DictReader(open(sample_path)))
    raw = [r for r in samples if r['statistic'] == 'raw']
    variants = list(dict.fromkeys(r['variant'] for r in raw))
    experiment = raw[0]['experiment']
    assert experiment in ('stream_pipeline', 'concurrent_kernels')
    c = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    # These experiments are single process, one host submission thread.
    runtime = list(c.execute('SELECT r.*,s.value AS api_name FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
                             'JOIN StringIds s ON s.id=r.nameId'))
    assert len({r['globalTid'] for r in runtime}) == 1
    apis = {r['correlationId']: dict(r) for r in runtime}
    assert len(apis) == len(runtime), 'ambiguous correlation IDs'
    names = dict(c.execute('SELECT id,value FROM StringIds'))
    copy_kinds = dict(c.execute('SELECT id,name FROM ENUM_CUDA_MEMCPY_OPER'))
    activities = []
    for table, kind in [('KERNEL', 'kernel'), ('MEMCPY', 'copy')]:
        for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_'+table):
            r = dict(r)
            api = apis[r['correlationId']]
            assert api['returnValue'] == 0
            assert ('cudaLaunchKernel' if kind == 'kernel' else 'cudaMemcpyAsync') in api['api_name']
            r.update(kind=kind, api_start=api['start'], api_duration_us=(api['end']-api['start'])/1000)
            if kind == 'kernel':
                assert 'asynchronous::transform' in names[r['demangledName']]
            else:
                r['direction'] = copy_kinds[r['copyKind']]
            activities.append(r)
    assert len({(r['deviceId'], r['contextId'], r['globalPid']) for r in activities}) == 1
    activities.sort(key=lambda r: r['api_start'])
    kernels = [r for r in activities if r['kind'] == 'kernel']
    copies = [r for r in activities if r['kind'] == 'copy']
    cursor = 0
    cases = []
    for variant in variants:
        vr = [r for r in raw if r['variant'] == variant]
        first = vr[0]
        n, block, warmup, repeats, iterations, workers = (int(first[k]) for k in
            ('n','block','warmup','repeats','iterations','worker_streams'))
        assert first['measurement_context'] == 'nsys'
        assert len(vr) == repeats
        labels = ['validation']
        for repeat in range(repeats):
            labels += [f'warmup_{repeat}']*warmup + [f'batch_{repeat}']*iterations + [f'e2e_{repeat}']
        parts = int(first['chunks_per_op'] if experiment == 'stream_pipeline' else first['jobs'])
        operations, all_k, all_c = [], [], []
        streams = {}
        previous_op_end = None
        for op, phase in enumerate(labels):
            if experiment == 'stream_pipeline':
                rows = activities[cursor:cursor+3*parts]
                cursor += 3*parts
                assert len(rows) == 3*parts
                ko, co = [], []
                slot_done = {}
                if first.get('submission_order', 'interleaved') == 'interleaved':
                    expected_order = [(part,phase) for part in range(parts) for phase in range(3)]
                else:
                    buffers = int(first['buffers'])
                    expected_order = [(part,phase) for base in range(0,parts,buffers)
                                      for phase in range(3) for part in range(base,min(parts,base+buffers))]
                by_chunk = {part: {} for part in range(parts)}
                for (part,stage), row in zip(expected_order,rows):
                    assert row['kind'] == ('kernel' if stage == 1 else 'copy')
                    by_chunk[part][stage] = row
                for part in range(parts):
                    h, k, d = (by_chunk[part][phase] for phase in range(3))
                    assert [r['kind'] for r in (h,k,d)] == ['copy','kernel','copy']
                    assert h['direction'] == 'CUDA_MEMCPY_KIND_HTOD' and d['direction'] == 'CUDA_MEMCPY_KIND_DTOH'
                    count = min(int(first['chunk_elements']), n-part*int(first['chunk_elements']))
                    assert h['bytes'] == d['bytes'] == count*4
                    assert h['end'] <= k['start'] and k['end'] <= d['start'], 'chunk dependency violated'
                    assert h['streamId'] == k['streamId'] == d['streamId']
                    slot = part % int(first['buffers'])
                    if slot in slot_done:
                        assert slot_done[slot] <= h['start'], 'slot reuse before completion'
                    slot_done[slot] = d['end']
                    ko.append(k); co.extend((h,d))
            else:
                ko = kernels[cursor:cursor+parts]
                cursor += parts
                assert len(ko) == parts
                co = []  # Transfers are outside the device-resident operation being analyzed.
            for part, k in enumerate(ko):
                count = (min(int(first['chunk_elements']), n-part*int(first['chunk_elements']))
                         if experiment == 'stream_pipeline' else n*(part+1)//parts-n*part//parts)
                assert k['gridX'] == (count+block-1)//block and k['blockX'] == block
                assert k['gridY'] == k['gridZ'] == k['blockY'] == k['blockZ'] == 1
                worker = part % workers
                streams.setdefault(worker, k['streamId'])
                assert streams[worker] == k['streamId']
            begin, end = min(r['start'] for r in ko+co), max(r['end'] for r in ko+co)
            if previous_op_end is not None:
                assert previous_op_end <= begin, 'unexpected overlap across joined operations'
            previous_op_end = end
            operations.append(dict(op=op, phase=phase, **intervals(ko,co)))
            all_k.extend(ko); all_c.extend(co)
        assert len(set(streams.values())) == min(workers,parts)
        case = dict(variant=variant, submission_order=first.get('submission_order','interleaved (legacy CSV)'),
                    operations=operations, kernel_count=len(all_k), copy_count=len(all_c),
                    observed_worker_streams=streams, all_operation_activity=intervals(all_k,all_c))
        case['batch_operation_metrics'] = {
            key: stats([o[key] for o in operations if o['phase'].startswith('batch_')])
            for key in operations[0] if key.endswith('_us')}
        case['instrumented_event_span_per_op'] = [float(r['event_span_us_per_op']) for r in vr]
        case['kernel_active_duration_us'] = stats([ (r['end']-r['start'])/1000 for r in all_k])
        if all_c:
            case['copy_directions'] = {}
            for direction in ('CUDA_MEMCPY_KIND_HTOD', 'CUDA_MEMCPY_KIND_DTOH'):
                cr = [r for r in all_c if r['direction'] == direction]
                case['copy_directions'][direction] = dict(count=len(cr), bytes=sum(r['bytes'] for r in cr),
                    device_duration_us=stats([(r['end']-r['start'])/1000 for r in cr]),
                    host_api_duration_us=stats([r['api_duration_us'] for r in cr]))
        cases.append(case)
    assert cursor == (len(activities) if experiment == 'stream_pipeline' else len(kernels)), 'extra activity'
    if experiment == 'concurrent_kernels':
        assert len(copies) == sum(2+3*int(next(r for r in raw if r['variant']==v)['repeats']) for v in variants)
        # Same full-input/output transfer size per selected configuration.
        assert all(r['bytes'] == int(raw[0]['n'])*4 for r in copies)
    return dict(scope='instrumented trace; source order via runtime correlation; all times us; transfer/initialization scope explicit',
                experiment=experiment, cases=cases)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite'); p.add_argument('csv')
    a = p.parse_args()
    print(json.dumps(analyze(a.sqlite,a.csv), indent=2))
