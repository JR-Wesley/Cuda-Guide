"""Validate scope within named PTX kernels, plus asynchronous copy instructions."""
import re
import sys
from pathlib import Path

def entry(text, name):
    m = re.search(r"\.visible\s+\.entry\s+" + name + r"\s*\(", text)
    if not m:
        raise RuntimeError(f"missing kernel {name}")
    next_entry = text.find(".visible .entry", m.end())
    return text[m.start():next_entry if next_entry != -1 else len(text)]

scopes, copies = [Path(p).read_text() for p in sys.argv[1:3]]
sass = Path(sys.argv[3]).read_text()
checks = {}
for name, scope in [('count_cta', 'cta'), ('count_gpu', 'gpu'), ('count_sys', 'sys'), ('count_builtin', 'gpu')]:
    body = entry(scopes, name)
    opcodes = re.findall(r'\batom(?:\.[A-Za-z0-9_]+)+', body)
    checks[f'{name}: atomic {scope}'] = any({'add', 'relaxed', scope}.issubset(op.split('.')) for op in opcodes)
checks['copy PTX cp.async'] = bool(re.search(r'cp\.async\.(ca|cg)\.shared\.global', copies))
checks['copy PTX commit group'] = 'cp.async.commit_group' in copies
checks['copy PTX wait group'] = 'cp.async.wait_group' in copies
checks['pipeline kernel'] = bool(entry(copies, 'copy_pipeline'))
checks['SASS asynchronous load/store'] = 'LDGSTS' in sass
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'} {name}")
sys.exit(0 if all(checks.values()) else 1)
