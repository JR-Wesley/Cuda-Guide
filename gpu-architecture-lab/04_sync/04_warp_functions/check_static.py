"""Check instructions in the relevant kernel, not incidental library code."""
import pathlib
import re
import sys


def check(text, patterns, label):
    for pattern in patterns:
        count = len(re.findall(pattern, text))
        if not count:
            raise RuntimeError(f"{label}: missing {pattern}")
        print(f"PASS {label}: {pattern} count={count}")


def main():
    ptx, sass = (pathlib.Path(p).read_text() for p in sys.argv[1:])
    entries = re.split(r"(?=\.visible\s+\.entry\s+)", ptx)[1:]
    functions = re.split(r"(?=Function\s*:)" , sass)[1:]
    if not entries or not functions:
        raise RuntimeError("Could not parse PTX entries / SASS functions")

    def kernel(parts, name):
        matches = [p for p in parts if name in p.splitlines()[0]]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {name}, found {len(matches)}")
        return matches[0]

    check(kernel(entries, "collectives_kernel"),
          [r"activemask\.b32", r"vote\.sync\.ballot", r"vote\.sync\.all", r"vote\.sync\.any",
           r"match\.any\.sync", r"match\.all\.sync"] +
          [rf"redux\.sync\.{op}\." for op in ("add", "min", "max", "and", "or", "xor")], "collectives PTX")
    check(kernel(entries, "shuffle_kernel"), [rf"shfl\.sync\.bfly", r"shfl\.sync\.idx",
          r"shfl\.sync\.up", r"shfl\.sync\.down"], "shuffle PTX")
    check(kernel(entries, "shared_exchange_kernel"), [r"bar\.warp\.sync", r"st\.shared", r"ld\.shared"], "shared PTX")
    check(kernel(functions, "collectives_kernel"), [r"\bVOTE", r"\bMATCH", r"\bREDUX", r"\bSHFL"], "collectives SASS")
    check(kernel(functions, "shuffle_kernel"), [r"\bSHFL"], "shuffle SASS")
    check(kernel(functions, "shared_exchange_kernel"), [r"\bWARPSYNC", r"\bSTS", r"\bLDS"], "shared SASS")


if __name__ == "__main__":
    main()
