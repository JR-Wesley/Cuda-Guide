"""Inputs common to the two profiler actions."""
DEFAULT_VARIANTS = {"transpose": "naive", "local_array": "static_index", "reduction": "interleaved",
                    "register_pressure": "unlimited", "integer_division": "runtime", "loop_unroll": "u1",
                    "branch_flow": "uniform", "host_transfer": "pageable", "stream_pipeline": "pinned_serial",
                    "concurrent_kernels": "serial", "gemm": "naive"}


def benchmark(session, directory, name, extra):
    return [directory / "bin" / name, "--variant", DEFAULT_VARIANTS.get(name, "scalar"),
            "--warmup", "1", "--repeats", "1", "--iterations", "3", *extra,
            "--csv", session.path / "instrumented_samples.csv"]
