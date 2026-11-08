"""Compile only: never launch deliberately invalid instruction examples."""
import pathlib
import re
import shlex

import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from lab_state import compile_command

out = pathlib.Path(sys.argv[1])
ptxas = sys.argv[2]
work = pathlib.Path(sys.argv[3])
probes = [
    ("valid_min", "sm_90a", "24", True, ""),
    ("valid_max", "sm_90a", "256", True, ""),
    ("below_min", "sm_90a", "16", False, r"range|between|24"),
    ("not_multiple", "sm_90a", "65", False, r"multiple|aligned|8"),
    ("above_max", "sm_90a", "264", False, r"range|between|256"),
    ("unsupported_sm89", "sm_89", "64", False, r"sm_90a|not supported|requires"),
    ("unsupported_sm90", "sm_90", "64", False, r"sm_90a|not supported|requires"),
    ("wrong_direction_dec256", "sm_90a", "256", False, r"C7406|larger than"),
]
for name, arch, count, success, diagnostic in probes:
    source = out / f"probe_{name}.ptx"
    # inc 256 is accepted syntax; it is NOT a runnable budget configuration.
    # dec 256 from .maxnreg 128 is rejected by current ptxas (see DEBUGGING.md).
    action = "inc" if name == "valid_max" else "dec"
    source.write_text(f""".version 8.0
.target {arch}
.address_size 64
.visible .entry probe()
.maxnreg 128
.reqntid 256
{{
    setmaxnreg.{action}.sync.aligned.u32 {count};
    bar.sync 0;
    ret;
}}
""")
    command = [ptxas, "-arch=" + arch, "-v", str(source), "-o", str(out / "bin" / f"probe_{name}.cubin")]
    print(shlex.join(command), flush=True)
    # Positive and negative controls always query the current compiler.
    result = compile_command(command)
    with (work / "status.tsv").open("a") as status:
        status.write(f"probe_{name}\t{result.returncode}\n")
    print(f"PROBE {name} expected={'success' if success else 'failure'} exit={result.returncode}")
    print(result.stdout, end="")
    if (result.returncode == 0) != success or (not success and not re.search(diagnostic, result.stdout, re.I)):
        raise SystemExit(f"FAIL unexpected compiler outcome for {name}")
print("PASS compile probes; accepted syntax is not proof of valid runtime direction/budget")
