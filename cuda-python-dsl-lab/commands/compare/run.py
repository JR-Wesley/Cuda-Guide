import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

"""CPU-only cross-DSL input-identity audit, not a performance comparison."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    from command_support import Record
    args.output = Record("compare").path
    records, groups = [], {}
    for run in args.runs:
        for backend in ('triton', 'gluon', 'cute', 'cutile'):
            path = run / backend / 'results.json'
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            records.append(dict(path=str(path)))
            for case in data['cases']:
                key = (case['op'], tuple(case['shape']))
                groups.setdefault(key, []).append(dict(backend=backend, id=case['id'],
                    input_id=case.get('input_id'), status=case['status'],
                    evidence=str(path.parent / case['evidence'])))
    pairs, failed = [], False
    for (op, shape), cases in sorted(groups.items()):
        backends = sorted({c['backend'] for c in cases})
        if len(backends) < 2:
            continue
        same = len({c['input_id'] for c in cases}) == 1 and cases[0]['input_id'] is not None
        ok = same and all(c['status'] == 'PASS' for c in cases)
        failed |= not ok
        pairs.append(dict(op=op, shape=shape, backends=backends, same_input_hash=same,
                          status='PASS' if ok else 'FAIL', cases=cases))
    if not pairs:
        failed = True
    report = dict(command=sys.argv, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  inputs=records,
                  scope='recorded CPU input hashes and prior functional PASS only; no new GPU execution',
                  comparison=pairs, status='FAIL' if failed else 'PASS',
                  performance='not measured', cpp_baseline='not implemented', library_baseline='not implemented')
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# 跨 DSL 小规模功能覆盖（非性能报告）', '',
             '| 算子 | shape | 后端 | 同输入及原功能验证 |', '| --- | --- | --- | --- |']
    for pair in pairs:
        lines.append(f'| {pair["op"]} | {pair["shape"]} | {", ".join(pair["backends"])} | {pair["status"]} |')
    lines += ['', '仅核对原记录的 CPU 输入 SHA256 和功能状态；未重新运行 kernel。',
              '未测量性能；未提供 CUDA C++/cuBLAS 性能基线；不构成第 10 周完整验收。', '']
    (args.output / 'SUMMARY.md').write_text('\n'.join(lines))
    message = f'{report["status"]}: {len(pairs)} shared operator/shape groups; {args.output}'
    print(message)
    (args.output / 'run.log').write_text('$ ' + shlex.join([sys.executable, *sys.argv]) + '\n' + message + f'\nexit_code={int(failed)}\n')
    (args.output / 'exit.json').write_text(json.dumps({'exit_code': int(failed)}) + '\n')
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
