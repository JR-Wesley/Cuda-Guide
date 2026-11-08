"""Summarize this action's learning results; never merge historic performance."""
import json
from pathlib import Path
import sys


def main():
    run = Path(sys.argv[1])
    lines = ['# 小规模功能验证记录', '',
             '仅按本目录实际结果汇总；PASS 是该输入/配置的功能通过，不等于学习计划整项验收或性能结论。', '',
             '| DSL | 通过 | 失败 | 原始结果 |', '| --- | ---: | ---: | --- |']
    cases = []
    for name in ('triton', 'gluon', 'cute', 'cutile'):
        path = run / name / 'results.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        lines.append(f'| {name} | {result["passed"]} | {result["failed"]} | [{name}/results.json]({name}/results.json) |')
        for case in result['cases']:
            cases.append((name, case))
    lines += ['', '## 编号与用例', '', '| 编号 | 算子/shape | 状态 | 证据目录 |', '| --- | --- | --- | --- |']
    for name, case in cases:
        path = f'{name}/{case["evidence"]}'
        lines.append(f'| {case["id"]} | {case["op"]} {case["shape"]} | {case["status"]} | [{path}]({path}) |')
    lines += ['', '## 边界', '',
              '- 不做性能排名；功能测试默认未启用事件计时、参数扫描或大规模数据。',
              '- 资源信息只以各 case 的 compiler artifact 为依据，不由耗时推断 occupancy/cache/DRAM。',
              '- H7.1–H7.8：本机 SM89，不满足 H20/SM90 验收；未运行。',
              '- C4.10 多 Stage、CUDA C++/库性能基线、完整 Profile、跨 DSL 性能报告仍未完成。', '']
    (run / 'SUMMARY.md').write_text('\n'.join(lines))
    print(f'Summary: {run / "SUMMARY.md"}')


if __name__ == '__main__':
    main()
