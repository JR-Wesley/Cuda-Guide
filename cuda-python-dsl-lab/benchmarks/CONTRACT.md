# Experiment adapter contract

Existing environments only; default is SMALL correctness checks, no performance sweep.

`benchmarks.runner` exports:

```python
@dataclass
class Workload:
    launch: Callable[[], Any]  # reuse preallocated inputs/outputs; no allocation here
    output: Any               # torch.Tensor or cupy.ndarray, contiguous expected
    framework: str = 'torch'   # 'torch' or 'cupy'
    artifacts: Callable[[], dict] | None = None  # after first launch; keys filenames, values str/bytes
    metadata: dict = field(default_factory=dict)

@dataclass
class Experiment:
    id: str
    op: str
    shape: tuple[int, ...]
    config: dict
    prepare: Callable[[tuple[np.ndarray, ...], dict], Workload]

run_cli(experiments: list[Experiment])
```

CLI: `--output DIR` required, `--ids ID,ID` optional, `--bench` opt-in, `--warmup 10 --repeat 50 --batches 5`. No benchmarks by default. All cases run twice with output poisoned before each launch. CPU generator/reference shared. FP32 inputs except GEMM (FP16 inputs, FP32 accumulator, FP16 output); contiguous arrays only.

Supported `op`: `copy`, `add`, `axpy` (a*0.5+b, unfused FP32), `transpose`, `sum` (row sum FP32), `softmax` (finite input only), `gemm`, `fusion` (a*0.5+b, unfused FP32), `scale` (a*0.5). Shape is input vector/matrix shape or GEMM (M,N,K). Same op+shape always yields same CPU arrays and SHA256 across frameworks. Tolerances in shared runner; do not weaken to hide failures.

Each suite is runnable as `python -m experiments.triton_suite` / `gluon_suite` / `cute_suite` / `cutile_suite`. Imports are resolved from project root. Kernels may live in per-operator directories with unique backend filenames. Do not modify common runner/core, other backend files, lab.py, README.md or LEARNING.md. CPU mapping exercises may be separate modules with their own CLI and output evidence. Keep defaults small (vectors 1,255,256,4099; transpose 32x32,37x53; sum/softmax 1x127,3x129,2x4097; GEMM 32x32x32 and 37x53x29). Config scans optional, not part of default run.

For compiler artifacts, return actual exported PTX/cubin/IR, not assertions about what compiler should emit. Closures may capture compiled kernel from launch. Include metadata for mapping/stages/precision. Unsupported advanced APIs must be explicitly documented as unimplemented/unverified, never silently substitute scalar GEMM for MMA or synchronous code for async pipelines. No GPU tests in parallel: parent runs suites in learning order. Do not install packages or alter environments.
