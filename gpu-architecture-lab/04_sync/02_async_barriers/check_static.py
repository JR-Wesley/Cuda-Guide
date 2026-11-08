"""Check actual PTX/SASS, not just successful compilation."""
import re
import sys
from pathlib import Path

ptx = Path(sys.argv[1]).read_text()
sass = Path(sys.argv[2]).read_text()
checks = {
    "PTX barrier init": "mbarrier.init" in ptx,
    "PTX arrival": "mbarrier.arrive" in ptx,
    "PTX explicit parity": bool(re.search(r"mbarrier\.(test|try)_wait\.parity", ptx)),
    "PTX async copy": bool(re.search(r"cp\.async\.(ca|cg)\.shared\.global", ptx)),
    "PTX copy/barrier binding": "cp.async.mbarrier.arrive" in ptx,
    "SASS async copy": "LDGSTS" in sass,
}
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'} {name}")
sys.exit(0 if all(checks.values()) else 1)
