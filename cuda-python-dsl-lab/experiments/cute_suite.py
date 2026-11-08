"""C4 GPU cases. Run only after the parent has serialized GPU access.

Default: small correctness cases. --tile-comparison adds two fixed tile cases;
benchmarking remains opt-in through the common runner's --bench flag.
"""

import sys


def experiment_specs(tile_comparison=False):
    """CPU-only case discovery; no torch/CuTe imports and no GPU probing."""
    specs = []
    for n in (1, 255, 256, 4099):
        specs.append((f"C4.1.add.n{n}", "add", (n,), {"kind": "add", "threads": 256}))
    for n in (1, 255, 256, 4099):
        specs.append((f"C4.4.copy.n{n}", "copy", (n,), {"kind": "copy", "threads": 256}))
    for m, n in ((32, 32), (37, 53)):
        for shared in (False, True):
            name = "shared" if shared else "naive"
            specs.append((f"C4.4.transpose.{name}.{m}x{n}", "transpose", (m, n),
                          {"kind": "transpose", "shared": shared, "threads": 256,
                           "tile": [32, 32] if shared else None,
                           "shared_stride": [33, 1] if shared else None}))
    for stage, values in (("C4.5", 1), ("C4.6", 4)):
        for n in (1, 255, 256, 4099):
            specs.append((f"{stage}.shared-copy.v{values}.n{n}", "copy", (n,),
                          {"kind": "shared_copy", "values": values,
                           "threads": 128, "tile": [1, 512],
                           "thread_layout": "(1,128):(128,1)",
                           "value_layout": f"(1,{values})",
                           "atom_bits": 32 * values}))
    for case, shape, tile in (
        ("C4.7.mma.16x8x16", (16, 8, 16), (16, 8, 16)),
        ("C4.8.gemm.32x32x32", (32, 32, 32), (32, 32, 16)),
        ("C4.9.gemm.37x53x29", (37, 53, 29), (32, 32, 16)),
    ):
        specs.append((case, "gemm", shape,
                      {"kind": "gemm", "tile": list(tile), "stages": 1,
                       "threads": 32, "mma": [16, 8, 16]}))
    if tile_comparison:
        for tile in ((16, 16, 16), (32, 32, 16)):
            label = "x".join(map(str, tile))
            specs.append((f"C4.10.gemm.tile{label}.37x53x29", "gemm", (37, 53, 29),
                          {"kind": "gemm", "tile": list(tile), "stages": 1,
                           "threads": 32, "mma": [16, 8, 16]}))
    return specs


def get_experiments(tile_comparison=False):
    from benchmarks.runner import Experiment
    from experiments.cute.workloads import prepare

    return [Experiment(id=case, op=op, shape=shape, config=config, prepare=prepare)
            for case, op, shape, config in experiment_specs(tile_comparison)]


def main():
    tile_comparison = "--tile-comparison" in sys.argv
    if tile_comparison:
        sys.argv.remove("--tile-comparison")
    from benchmarks.runner import run_cli

    run_cli(get_experiments(tile_comparison))


if __name__ == "__main__":
    main()
