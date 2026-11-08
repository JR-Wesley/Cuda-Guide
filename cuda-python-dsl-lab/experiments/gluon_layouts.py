"""CPU-only BlockedLayout derivations, with an explicit opt-in GPU owner probe.

The CPU path uses only the Python standard library and never imports Triton or
Torch. Register slots below are logical per-thread element slots, not physical
SASS register numbers. Full CTA tiles only: replicated/small layouts are rejected
rather than silently treating replication as unique ownership.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools

import json
import math
from pathlib import Path
import sys


@dataclass(frozen=True)
class LayoutCase:
    name: str
    shape: tuple[int, ...]
    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]

    @property
    def tile(self):
        return tuple(s * t * w for s, t, w in zip(
            self.size_per_thread, self.threads_per_warp, self.warps_per_cta))

    def validate(self):
        rank = len(self.shape)
        if rank not in (1, 2) or sorted(self.order) != list(range(rank)):
            raise ValueError("Only rank-1/2 layouts with a permutation order are supported")
        for values in (self.shape, self.size_per_thread,
                       self.threads_per_warp, self.warps_per_cta):
            if len(values) != rank or any(v <= 0 or v & (v - 1) for v in values):
                raise ValueError("All extents must be positive powers of two of matching rank")
        if math.prod(self.threads_per_warp) != 32 or math.prod(self.warps_per_cta) != 4:
            raise ValueError("These exercises target 32-lane warps and four-warps CTAs")
        if any(n < t or n % t for n, t in zip(self.shape, self.tile)):
            raise ValueError("Sub-tile/replicated layouts are outside this derivation")


def decode(index, extents, order):
    """Mixed-radix digits, with order[0] the fastest-changing dimension."""
    result = [0] * len(extents)
    for dim in order:
        result[dim] = index % extents[dim]
        index //= extents[dim]
    if index:
        raise ValueError("index exceeds mixed-radix domain")
    return tuple(result)


def encode(coords, extents, order):
    index, stride = 0, 1
    for dim in order:
        index += coords[dim] * stride
        stride *= extents[dim]
    return index


def derive(case):
    case.validate()
    repeats = tuple(n // t for n, t in zip(case.shape, case.tile))
    local_count = math.prod(case.size_per_thread)
    slots = math.prod(case.shape) // 128
    records = []
    for warp, lane, slot in itertools.product(range(4), range(32), range(slots)):
        local = decode(slot % local_count, case.size_per_thread, case.order)
        repeat = decode(slot // local_count, repeats, case.order)
        lane_coord = decode(lane, case.threads_per_warp, case.order)
        warp_coord = decode(warp, case.warps_per_cta, case.order)
        coord = tuple(local[d] + case.size_per_thread[d] * (
            lane_coord[d] + case.threads_per_warp[d] * warp_coord[d]) +
            case.tile[d] * repeat[d] for d in range(len(case.shape)))
        records.append({"coordinate": coord, "warp": warp, "lane": lane,
                        "slot": slot, "thread": warp * 32 + lane})
    return records


def inverse(case, coord):
    local = tuple(c % s for c, s in zip(coord, case.size_per_thread))
    lane = tuple((c // s) % t for c, s, t in zip(
        coord, case.size_per_thread, case.threads_per_warp))
    warp = tuple((c // (s * t)) % w for c, s, t, w in zip(
        coord, case.size_per_thread, case.threads_per_warp, case.warps_per_cta))
    repeat = tuple(c // t for c, t in zip(coord, case.tile))
    repeats = tuple(n // t for n, t in zip(case.shape, case.tile))
    return (encode(warp, case.warps_per_cta, case.order),
            encode(lane, case.threads_per_warp, case.order),
            encode(local, case.size_per_thread, case.order) +
            math.prod(case.size_per_thread) * encode(repeat, repeats, case.order))


def check(case, records):
    expected = set(itertools.product(*(range(n) for n in case.shape)))
    actual = [r["coordinate"] for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise AssertionError(f"{case.name}: missing, repeated, or out-of-bounds elements")
    for r in records:
        if inverse(case, r["coordinate"]) != (r["warp"], r["lane"], r["slot"]):
            raise AssertionError(f"{case.name}: inverse mapping disagrees")
        if len(case.shape) == 1:
            s = case.size_per_thread[0]
            formula = s * (32 * r["warp"] + r["lane"]) + r["slot"] % s + 128 * s * (r["slot"] // s)
            if r["coordinate"] != (formula,):
                raise AssertionError("1-D closed form disagrees")
        elif case.name in ("2d-row-order", "2d-column-order"):
            w, l, q = r["warp"], r["lane"], r["slot"]
            if case.order == (1, 0):
                formula = (l // 8 + 4 * (w // 2) + 8 * (q // 4),
                           2 * (l % 8 + 8 * (w % 2)) + q % 2 + 32 * ((q // 2) % 2))
            else:
                formula = (l % 4 + 4 * (w % 2) + 8 * ((q // 2) % 2),
                           2 * (l // 4 + 8 * (w // 2)) + q % 2 + 32 * (q // 4))
            if r["coordinate"] != formula:
                raise AssertionError("2-D closed form disagrees")
    return {"elements": len(actual), "threads": 128,
            "slots_per_thread": len(actual) // 128,
            "missing": 0, "duplicates": 0, "inverse_check": "passed"}


def layout_cases():
    return [
        LayoutCase("1d-128-spt1", (128,), (1,), (32,), (4,), (0,)),
        LayoutCase("1d-256-spt1", (256,), (1,), (32,), (4,), (0,)),
        LayoutCase("1d-256-spt2", (256,), (2,), (32,), (4,), (0,)),
        LayoutCase("1d-512-spt2", (512,), (2,), (32,), (4,), (0,)),
        LayoutCase("2d-row-order", (16, 64), (1, 2), (4, 8), (2, 2), (1, 0)),
        LayoutCase("2d-column-order", (16, 64), (1, 2), (4, 8), (2, 2), (0, 1)),
        LayoutCase("transpose-load", (32, 32), (1, 1), (4, 8), (4, 1), (1, 0)),
        LayoutCase("transpose-store", (32, 32), (1, 1), (8, 4), (1, 4), (0, 1)),
        LayoutCase("sum-warp-per-row", (4, 256), (1, 1), (1, 32), (4, 1), (1, 0)),
        LayoutCase("sum-four-warps-per-row", (4, 256), (1, 1), (1, 32), (1, 4), (1, 0)),
    ]


def diagram(case, records):
    owners = {r["coordinate"]: r for r in records}
    lines = [f"## {case.name}", "", "Cell: warp:lane:logical-slot. Full mapping is in the CSV.", "", "```text"]
    if len(case.shape) == 1:
        for start in range(0, case.shape[0], 16):
            cells = [owners[(i,)] for i in range(start, min(start + 16, case.shape[0]))]
            lines.append(f"{start:3d}: " + " ".join(f"{r['warp']}:{r['lane']:02d}:{r['slot']}" for r in cells))
    else:
        lines.append("Top-left crop, at most 8 rows x 16 columns:")
        for row in range(min(8, case.shape[0])):
            cells = [owners[row, col] for col in range(min(16, case.shape[1]))]
            lines.append(f"{row:2d}: " + " ".join(f"{r['warp']}:{r['lane']:02d}:{r['slot']}" for r in cells))
    return "\n".join(lines + ["```", ""])


def verify_gpu(case, records, output_dir):
    """Opt-in: observe physical thread IDs, not guessed hardware register names."""
    import torch
    from experiments.gluon_suite import compiled_artifacts
    from experiments.transpose.gluon_ops import mapping_2d_kernel
    from experiments.vector.gluon_ops import mapping_1d_kernel

    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("Mapping probes target NVIDIA SM80+")
    expected = torch.empty(case.shape, dtype=torch.int32)
    for record in records:
        expected[record["coordinate"]] = record["thread"]
    owners = torch.empty(case.shape, device="cuda", dtype=torch.int32)
    for poison in (-1, -2):
        owners.fill_(poison)
        if len(case.shape) == 1:
            kernel = mapping_1d_kernel[(1,)](owners, case.shape[0], case.size_per_thread[0], num_warps=4)
        else:
            kernel = mapping_2d_kernel[(1,)](
                owners, *case.shape, case.size_per_thread, case.threads_per_warp,
                case.warps_per_cta, case.order, num_warps=4)
        actual = owners.cpu()
        if not torch.equal(actual, expected):
            raise AssertionError(f"{case.name}: physical thread ownership disagrees with CPU derivation")
    artifacts_dir = output_dir / case.name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, value in compiled_artifacts(kernel).items():
        path = artifacts_dir / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value, encoding="utf-8")
    return {"thread_ownership": "passed", "poisoned_runs": 2,
            "physical_register_assignment": "not measured"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-gpu", action="store_true",
                        help="explicitly launch tiny owner probes; default is CPU only")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "command": sys.orig_argv, "executable": sys.executable, "python": sys.version,

               "gpu_verification": "not requested", "exit_code": 0, "cases": []}
    diagrams = ["# BlockedLayout ownership maps", ""]
    all_cases = []
    for case in layout_cases():
        records = derive(case)
        result = {"layout": asdict(case), "tile": case.tile, "checks": check(case, records)}
        summary["cases"].append(result)
        all_cases.append((case, records, result))
        with (args.output / f"{case.name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([*(f"dim{d}" for d in range(len(case.shape))), "warp", "lane", "logical_slot", "thread"])
            for r in sorted(records, key=lambda item: item["coordinate"]):
                writer.writerow([*r["coordinate"], r["warp"], r["lane"], r["slot"], r["thread"]])
        diagrams.append(diagram(case, records))
    (args.output / "maps.md").write_text("\n".join(diagrams), encoding="utf-8")
    try:
        if args.verify_gpu:
            summary["gpu_verification"] = "running"
            for case, records, result in all_cases:
                result["gpu"] = verify_gpu(case, records, args.output)
            summary["gpu_verification"] = "passed"
    except Exception as error:
        summary["gpu_verification"] = "failed"
        summary["exit_code"] = 1
        summary["error"] = repr(error)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"CPU mapping checks passed: {len(all_cases)} cases; evidence: {args.output}")
    print(f"GPU verification: {summary['gpu_verification']}")


if __name__ == "__main__":
    main()
