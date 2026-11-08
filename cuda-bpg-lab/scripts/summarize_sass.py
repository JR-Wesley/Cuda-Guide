#!/usr/bin/env python3
"""Summarize listed SM70+ SASS instructions; counts are static, never dynamic metrics."""
import collections
import json
from pathlib import Path
import re
import sys

def summarize(path):
    text = Path(path).read_text()
    functions = []
    for chunk in text.split('Function :')[1:]:
        name = chunk.splitlines()[0].strip()
        instructions = [(int(m.group(1), 16), m.group(2).strip())
                        for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+([^\n;]+);', chunk)]
        assert instructions and instructions[0][0] == 0, (path, name, 'missing instruction PCs')
        assert all(pc == 16*i for i, (pc, _) in enumerate(instructions)), (path, name, 'unexpected PC spacing')
        counts = collections.Counter()
        backedges = []
        conditional_branches = []
        for pc, op in instructions:
            mnemonic = re.sub(r'^@!?[A-Z]+\d*\s+', '', op).split()[0]
            counts[mnemonic] += 1
            branch = re.search(r'\bBRA(?:\.[A-Z]+)?\s+(0x[0-9a-f]+)', op)
            if branch:
                target = int(branch.group(1), 16)
                if target < pc:
                    backedges.append({'pc': hex(pc), 'target': hex(target),
                                      'listed_body_instructions': (pc-target)//16+1})
                if op.startswith('@'):
                    conditional_branches.append({'pc': hex(pc), 'target': hex(target)})
        functions.append({'symbol': name, 'listed_bytes_including_padding': instructions[-1][0]+16,
                          'static_instruction_count_including_padding': len(instructions),
                          'nonself_backedges': backedges, 'conditional_branches': conditional_branches,
                          'mnemonics': dict(sorted(counts.items()))})
    assert functions, f'no functions in {path}'
    return {'path': path, 'scope': 'static listed instructions; includes padding; not executed counts or cache traffic',
            'functions': functions}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: summarize_sass.py *.sass.txt')
    print(json.dumps([summarize(p) for p in sys.argv[1:]], indent=2))
