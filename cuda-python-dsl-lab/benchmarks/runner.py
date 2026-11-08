"""Small correctness-first GPU runner; steady-state timing is opt-in only."""
import argparse
from dataclasses import dataclass, field

import importlib.metadata
import json
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
from benchmarks.core import make_case, validate


@dataclass
class Workload:
    launch: Callable[[], Any]
    output: Any
    framework: str = 'torch'
    artifacts: Callable[[], dict] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Experiment:
    id: str
    op: str
    shape: tuple[int, ...]
    config: dict
    prepare: Callable[[tuple[np.ndarray, ...], dict], Workload]


def adapter(work):
    if work.framework == 'torch':
        import torch
        stream = torch.cuda.current_stream(work.output.device)
        return dict(sync=stream.synchronize,
                    poison=lambda: work.output.fill_(float('nan')),
                    cpu=lambda: work.output.detach().cpu().numpy(),
                    contiguous=work.output.is_contiguous,
                    event=lambda: torch.cuda.Event(enable_timing=True),
                    stream=stream, stream_id=stream.cuda_stream,
                    device=torch.cuda.get_device_name(work.output.device),
                    capability=torch.cuda.get_device_capability(work.output.device),
                    runtime=torch.version.cuda,
                    strides=[s*work.output.element_size() for s in work.output.stride()])
    if work.framework == 'cupy':
        import cupy as cp
        stream = cp.cuda.get_current_stream()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        return dict(sync=stream.synchronize,
                    poison=lambda: work.output.fill(cp.nan),
                    cpu=lambda: cp.asnumpy(work.output),
                    contiguous=lambda: work.output.flags.c_contiguous,
                    event=cp.cuda.Event, stream=stream, stream_id=stream.ptr,
                    device=props['name'].decode(), capability=[props['major'], props['minor']],
                    runtime=cp.cuda.runtime.runtimeGetVersion(), strides=work.output.strides)
    raise ValueError(f'Unknown framework: {work.framework}')


def versions():
    result = {'python': platform.python_version(), 'executable': sys.executable}
    for name in ('numpy', 'torch', 'triton', 'nvidia-cutlass-dsl', 'cuda-tile', 'cupy-cuda13x'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def plan_ids(exp):
    explicit = exp.config.get('plan_ids')
    if explicit:
        return explicit
    match = re.match(r'([ETGCUH]\d+\.\d+)', exp.id)
    return [match.group(1)] if match else []


def matches(identifier, selected):
    return any(identifier == item or identifier.startswith((item + '.', item + '-')) for item in selected)


def run_cli(experiments):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', help='comma-separated exact IDs or plan prefixes, e.g. T1.1,C4.4')
    parser.add_argument('--bench', action='store_true', help='explicitly enable event timing')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--batches', type=int, default=5)
    args = parser.parse_args()
    if min(args.warmup, args.repeat, args.batches) < 1:
        parser.error('warmup/repeat/batches must be positive')
    if args.ids:
        selected = args.ids.split(',')
        if any(not any(matches(e.id, [item]) for e in experiments) for item in selected):
            parser.error('unknown experiment ID/prefix')
        experiments = [e for e in experiments if matches(e.id, selected)]
    # Refuse overwriting evidence, including an interrupted prior run.
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / 'results.jsonl'
    log = result_file.open('x')
    report = dict(schema_version=1, command=sys.argv, versions=versions(),
                  mode='benchmark' if args.bench else 'functional-only',
                  cache='existing project cache; not cleared; cold-cache state not asserted',
                  correctness_repeats=2, cases=[])
    try:
        for index, exp in enumerate(experiments):
            label = re.sub(r'[^A-Za-z0-9_.-]', '_', exp.id)
            case_dir = args.output / f'{index:03d}-{label}'
            case_dir.mkdir()
            record = dict(id=exp.id, plan_ids=plan_ids(exp), op=exp.op, shape=exp.shape,
                          config=exp.config, status='FAIL', evidence=case_dir.name,
                          timing={'status': 'not measured: functional-only run'},
                          hardware_counters='not measured: no profiler in functional run')
            try:
                case = make_case(exp.op, exp.shape)
                record.update(input_id=case.input_id, inputs=[dict(shape=a.shape, strides=a.strides,
                              dtype=str(a.dtype)) for a in case.inputs],
                              reference='CPU NumPy; float64 reduction/softmax/GEMM, cast to output dtype',
                              precision='FP16 input/output, FP32 accumulation' if exp.op == 'gemm' else 'FP32')
                np.savez(case_dir / 'inputs.npz', **{f'input{i}': a for i, a in enumerate(case.inputs)})
                np.save(case_dir / 'reference.npy', case.expected)
                t0 = time.perf_counter()
                work = exp.prepare(case.inputs, exp.config)
                device = adapter(work)
                device['sync']()
                prepare_ms = (time.perf_counter() - t0)*1000
                record.update(metadata=work.metadata, device=device['device'],
                              capability=device['capability'], runtime=device['runtime'],
                              stream=device['stream_id'], output_strides=device['strides'])
                if not device['contiguous']():
                    raise AssertionError('GPU output is a strided view, not contiguous storage')
                errors = []
                for repeat in range(2):
                    device['poison']()
                    device['sync']()
                    first = time.perf_counter()
                    work.launch()
                    device['sync']()
                    elapsed = (time.perf_counter()-first)*1000
                    if repeat == 0:
                        record['startup'] = dict(prepare_ms=prepare_ms,
                            first_launch_sync_ms=elapsed, prepare_plus_first_launch_ms=prepare_ms+elapsed,
                            scope='prepare includes upload/allocation and eager compile; first launch includes lazy JIT; not kernel timing')
                    actual = device['cpu']()
                    np.save(case_dir / f'actual-{repeat}.npy', actual)
                    errors.append(validate(actual, case))
                record['correctness'] = {'status': 'PASS', 'repeats': errors}
                if work.artifacts:
                    artifact_dir = case_dir / 'artifacts'
                    artifact_dir.mkdir()

                    for name, data in work.artifacts().items():
                        relative = Path(name)
                        if relative.is_absolute() or '..' in relative.parts:
                            raise ValueError(f'Unsafe artifact path: {name}')
                        destination = artifact_dir / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        payload = data.encode() if isinstance(data, str) else data
                        destination.write_bytes(payload)

                if args.bench:
                    for _ in range(args.warmup):
                        work.launch()
                    device['sync']()
                    times = []
                    record['timing'] = dict(status='measuring', scope='batch launch average; entire closure; not profiler single kernel',
                        warmup=args.warmup, repeat=args.repeat, batches=args.batches,
                        batch_average_ms=times, input_reuse=True,
                        order='catalog order; not alternating performance candidates')
                    record['benchmark_correctness'] = []
                    for batch in range(args.batches):
                        # In the same stream, before the timed interval. A stale correct
                        # output must not let a no-op launch pass validation.
                        device['poison']()
                        start, end = device['event'](), device['event']()
                        start.record(device['stream'])
                        for _ in range(args.repeat):
                            work.launch()
                        end.record(device['stream'])
                        end.synchronize()
                        if work.framework == 'torch':
                            batch_ms = start.elapsed_time(end)
                        else:
                            import cupy as cp
                            batch_ms = cp.cuda.get_elapsed_time(start, end)
                        times.append(batch_ms/args.repeat)
                        actual = device['cpu']()
                        np.save(case_dir / f'actual-benchmark-{batch}.npy', actual)
                        check = {'batch': batch, 'status': 'FAIL'}
                        record['benchmark_correctness'].append(check)
                        check.update(validate(actual, case), status='PASS')
                    median = statistics.median(times)
                    record['timing'].update(status='measured', median_ms=median, min_ms=min(times), max_ms=max(times))
                    if median > 0 and case.algorithm_bytes is not None:
                        record['timing']['algorithm_effective_GB_s'] = case.algorithm_bytes / (median*1e6)
                    if median > 0 and case.flops is not None:
                        record['timing']['TFLOP_s'] = case.flops/(median*1e9)
                record['status'] = 'PASS'
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
                (case_dir / 'error.log').write_text(traceback.format_exc())
                print(traceback.format_exc(), flush=True)
            report['cases'].append(record)
            log.write(json.dumps(record, default=str) + '\n')
            log.flush()
            print(f"{record['status']} {exp.id} {exp.op} {exp.shape}", flush=True)
    finally:
        log.close()
        report['passed'] = sum(r['status'] == 'PASS' for r in report['cases'])
        report['failed'] = sum(r['status'] != 'PASS' for r in report['cases'])
        (args.output / 'results.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(f"Functional cases: {report['passed']} passed, {report['failed']} failed; {args.output}")
    if report['failed']:
        raise SystemExit(1)
