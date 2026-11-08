"""Fail closed if this experiment's generated instruction/resource contract changes."""
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
ptx = (out / "setmaxnreg.ptx").read_text()
sass = (out / "setmaxnreg.sass").read_text()
resources = pathlib.Path(sys.argv[2]).read_text()
assert "setmaxnreg.dec.sync.aligned.u32 64;" in ptx
assert "setmaxnreg.inc.sync.aligned.u32 192;" in ptx
assert "setmaxnreg.dec.sync.aligned.u32 128;" in ptx
assert "setmaxnreg.inc.sync.aligned.u32 128;" in ptx
assert "barrier.sync 1, 128;" in ptx, "WG-only barrier must use the non-.aligned form"
sections = re.split(r"Function\s*:\s*", sass)[1:]
assert len(sections) == 3, f"Expected three kernels, found {len(sections)}"
for mode in range(3):
    section, = [s for s in sections if f"experimentILi{mode}E" in s.splitlines()[0]]
    instructions = re.findall(r"USETMAXREG[^;]*;", section)
    if mode == 0:
        assert not instructions, "Baseline unexpectedly contains USETMAXREG"
    else:
        assert len(instructions) == (2 if mode == 1 else 4), instructions
        required = [("DEALLOC", "40"), ("TRY_ALLOC", "80")]
        if mode == 2:
            required += [("TRY_ALLOC", "c0"), ("DEALLOC", "80")]
        for action, count in required:
            assert any(action in s and re.search(rf"0x{count}\s*;", s) for s in instructions), (mode, action, count)
    # These are static opcode counts, not measured dynamic memory traffic.
    local = {op: len(re.findall(rf"\b{op}(?:\.|\s)", section)) for op in ("LDL", "STL")}
    registers = [int(n) for n in re.findall(r"\bR(\d+)\b", section)]
    if mode == 2:
        assert 128 <= max(registers) < 192, "Expected actual use of enlarged register range"
    print(f"mode={mode} USETMAXREG={instructions} max_R={max(registers)} local_opcodes={local}")
    resource = re.search(rf"Function\s+\S*experimentILi{mode}E\S*\s*:\s*([^\n]+)", resources)
    assert resource, f"Mode {mode}: resource metadata missing"
    if mode:
        assert re.search(r"REG:128\b", resource[1]), resource[1]
    print(f"mode={mode} resources={resource[1].strip()}")
print("PASS static contract (instruction presence and initial allocation, not runtime correctness)")
