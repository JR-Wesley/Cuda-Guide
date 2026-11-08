"""C4.2/C4.3 CPU-only CuTe index algebra, plus copy/MMA partition checks.

Uses real 4.8.1 CuTe operations in an MLIR context. No JIT compilation, torch,
CUDA device query, device allocation, or kernel launch is performed.
"""

import argparse
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys


def _equal(actual, expected, label):
    # Do not use assert: these checks must survive python -O.
    if actual != expected:
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def run_checks():
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir import ir
    from cutlass.cute.nvgpu import CopyUniversalOp
    from cutlass.cute.nvgpu.warp import MmaF16BF16Op

    version = importlib.metadata.version("nvidia-cutlass-dsl")
    _equal(version, "4.8.1", "installed DSL version")
    report = {"status": "passed", "execution": "CPU static index algebra only",
              "cutlass_dsl": version, "gpu_execution": False, "checks": []}
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            def check_map(name, layout, coords, expected, domain, formula):
                rows = []
                for coord in coords:
                    actual, reference = int(layout(coord)), int(expected(coord))
                    _equal(actual, reference, f"{name} {coord}")
                    rows.append({"coordinate": coord, "index": actual, "expected": reference})
                report["checks"].append({
                    "id": name, "shape": layout.shape, "stride": layout.stride,
                    "domain": domain, "codomain": "integer element offsets",
                    "formula": formula, "points_checked": len(rows), "points": rows,
                })

            check_map("C4.2.1d", cute.make_layout(8, stride=2), range(8),
                      lambda i: 2 * i, "i in [0,8)", "2*i (strided, not compact)")
            check_map("C4.2.row-major", cute.make_layout((3, 4), stride=(4, 1)),
                      itertools.product(range(3), range(4)), lambda c: c[0] * 4 + c[1],
                      "(r,c) in [0,3)x[0,4)", "4*r+c")
            check_map("C4.2.column-major", cute.make_layout((3, 4), stride=(1, 3)),
                      itertools.product(range(3), range(4)), lambda c: c[0] + 3 * c[1],
                      "(r,c) in [0,3)x[0,4)", "r+3*c")
            nested_coords = [((i, j), k) for i, j, k in
                             itertools.product(range(2), range(3), range(4))]
            check_map("C4.2.nested", cute.make_layout(((2, 3), 4), stride=((1, 8), 2)),
                      nested_coords, lambda c: c[0][0] + 8 * c[0][1] + 2 * c[1],
                      "((i,j),k) in ([0,2)x[0,3))x[0,4)", "i+8*j+2*k")

            outer = cute.make_layout((4, 6), stride=(6, 1))
            inner = cute.make_layout((4, 3), stride=(1, 8))
            composed = cute.composition(outer, inner)
            coords = list(itertools.product(range(4), range(3)))
            for coord in coords:
                _equal(composed(coord), outer(inner(coord)), "outer(inner(coord))")
            check_map("C4.3.composition", composed, coords, lambda c: 6 * c[0] + 2 * c[1],
                      "inner: [0,4)x[0,3) -> scalar coordinates [0,24) of outer",
                      "outer(inner(i,j)); inner=i+8*j; outer(q)=6*(q%4)+q//4")
            report["checks"][-1]["operands"] = {
                "outer": {"shape": outer.shape, "stride": outer.stride},
                "inner": {"shape": inner.shape, "stride": inner.stride},
                "order": "right operand first; not inner(outer(coord))",
            }
            matrix = cute.make_layout((6, 8), stride=(8, 1))
            divided = cute.logical_divide(matrix, (2, 4))
            div_coords = [((r, tr), (c, tc)) for r, tr, c, tc in
                          itertools.product(range(2), range(3), range(4), range(2))]
            check_map("C4.3.logical-divide", divided, div_coords,
                      lambda p: (p[0][0] + 2 * p[0][1]) * 8 + p[1][0] + 4 * p[1][1],
                      "((r,tile_r),(c,tile_c)): ((2,3),(4,2))",
                      "(r+2*tile_r)*8 + c+4*tile_c; no rounding/tails in this example")
            _equal(sorted(int(divided(c)) for c in div_coords), list(range(48)),
                   "logical-divide exact coverage")

            # Partition actual identity tensors, not hand-written thread formulas.
            for values in (1, 4):
                atom = cute.make_copy_atom(CopyUniversalOp(), cutlass.Float32,
                                           num_bits_per_copy=32 * values)
                tiled = cute.make_tiled_copy_tv(
                    atom, cute.make_layout((1, 128), stride=(128, 1)),
                    cute.make_layout((1, values)),
                )
                identity = cute.make_identity_tensor((1, 512))
                seen, partitions = [], []
                for tid in range(128):
                    src = tiled.get_slice(tid).partition_S(identity)
                    dst = tiled.get_slice(tid).partition_D(identity)
                    points = [tuple(src[j]) for j in range(cute.size(src))]
                    _equal(points, [tuple(dst[j]) for j in range(cute.size(dst))],
                           f"copy v{values} source/destination thread {tid}")
                    expected = ([(0, tid + 128 * j) for j in range(4)] if values == 1
                                else [(0, 4 * tid + j) for j in range(4)])
                    _equal(points, expected, f"copy v{values} thread {tid}")
                    seen.extend(p[1] for p in points)
                    partitions.append({"thread": tid, "shape": src.shape,
                                       "stride": str(src.stride), "coordinates": points})
                _equal(sorted(seen), list(range(512)), f"copy v{values} no gaps/overlap")
                report["checks"].append({
                    "id": f"C4.3.partition.copy-v{values}",
                    "domain": "128 threads x 4 owned elements",
                    "codomain": "(row=0,column) of a 1x512 identity tensor",
                    "atom_bits": 32 * values, "points_checked": len(seen),
                    "partitions": partitions, "coverage": "exactly once",
                })

            mma = cute.make_tiled_mma(
                MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)), (1, 1, 1))
            for bm, bn in ((16, 8), (16, 16), (32, 32)):
                for operand, shape in (("A", (bm, 16)), ("B", (bn, 16)), ("C", (bm, bn))):
                    identity = cute.make_identity_tensor(shape)
                    seen, partitions = [], []
                    for tid in range(32):
                        thread = mma.get_slice(tid)
                        partition = getattr(thread, f"partition_{operand}")(identity)
                        points = [tuple(partition[j]) for j in range(cute.size(partition))]
                        seen.extend(points)
                        partitions.append({"lane": tid, "shape": partition.shape,
                                           "coordinates": points})
                    expected = list(itertools.product(range(shape[0]), range(shape[1])))
                    _equal(sorted(seen), expected, f"MMA {bm}x{bn} {operand} exact coverage")
                    report["checks"].append({
                        "id": f"C4.7.partition.mma-{bm}x{bn}-{operand}",
                        "domain": "32 lanes x instruction-defined fragment elements",
                        "codomain": f"{operand} coordinates, shape {shape}",
                        "points_checked": len(seen), "partitions": partitions,
                        "coverage": "exactly once; mapping only, not numerical MMA validation",
                    })
        module.operation.verify()
        algebra_ir = str(module)
    report["points_checked"] = sum(c["points_checked"] for c in report["checks"])
    return report, algebra_ir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / "layout-checks.json", args.output / "layout-algebra.mlir"]
    if any(path.exists() for path in paths):
        parser.error("Use a new output directory; existing evidence is not overwritten")
    report, algebra_ir = run_checks()
    report["command"] = [sys.executable, "-m", "experiments.cute_layouts", *sys.argv[1:]]
    with paths[0].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with paths[1].open("x", encoding="utf-8") as handle:
        handle.write(algebra_ir)
    print(f"PASS: {len(report['checks'])} checks, {report['points_checked']} static indices; no GPU")


if __name__ == "__main__":
    main()
