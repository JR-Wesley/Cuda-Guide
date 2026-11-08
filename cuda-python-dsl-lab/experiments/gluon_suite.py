"""G3.1-G3.8 adapters for the shared correctness-first experiment runner.

Run from the project root with the existing .venvs/triton environment.
No allocation occurs in Workload.launch; no CUDA work occurs on import.
"""

import json

import triton

from experiments.reduction.gluon_ops import row_sum_kernel
from experiments.transpose.gluon_ops import transpose_kernel
from experiments.vector.gluon_ops import add_kernel, copy_kernel


def compiled_artifacts(kernel):
    """Export actual compiler outputs, without inventing unavailable IR stages."""
    artifacts = {
        f"kernel.{name}": value
        for name, value in kernel.asm.items()
        if isinstance(value, (str, bytes))
    }
    metadata = kernel.metadata
    if hasattr(metadata, "_asdict"):
        metadata = metadata._asdict()
    elif hasattr(metadata, "__dict__"):
        metadata = vars(metadata)
    artifacts["compiler.json"] = json.dumps({
        "triton_version": triton.__version__,
        "available_asm": list(kernel.asm),
        "metadata": metadata,
        "n_regs": getattr(kernel, "n_regs", None),
        "n_spills": getattr(kernel, "n_spills", None),
    }, indent=2, default=str) + "\n"
    return artifacts


def prepare(arrays, config):
    import torch
    from benchmarks.runner import Workload

    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("This suite targets NVIDIA SM80+ (SM89 learning target)")
    inputs = tuple(torch.as_tensor(a, device="cuda") for a in arrays)
    kind = config["kind"]
    shape = tuple(arrays[0].shape)
    if kind == "sum":
        output_shape = (shape[0],)
    elif kind == "transpose":
        output_shape = (shape[1], shape[0])
    else:
        output_shape = shape
    output = torch.empty(output_shape, device="cuda", dtype=torch.float32)
    compiled = None

    def launch():
        nonlocal compiled
        if kind == "add":
            compiled = add_kernel[(triton.cdiv(shape[0], config["block"]),)](
                inputs[0], inputs[1], output, shape[0], config["block"], config["spt"],
                num_warps=4, enable_fp_fusion=False)
        elif kind == "sum":
            compiled = row_sum_kernel[(triton.cdiv(shape[0], 4),)](
                inputs[0], output, shape[0], shape[1], triton.next_power_of_2(shape[1]),
                config["warps_per_row"], num_warps=4, enable_fp_fusion=False)
        elif kind == "transpose":
            compiled = transpose_kernel[(triton.cdiv(shape[0], 32), triton.cdiv(shape[1], 32))](
                inputs[0], output, shape[0], shape[1], 32, config["convert"], num_warps=4)
        elif kind == "copy":
            compiled = copy_kernel[(triton.cdiv(shape[0], config["block"] * config["tiles"]),)](
                inputs[0], output, shape[0], config["block"], config["tiles"], config["mode"],
                num_warps=4, enable_fp_fusion=False)
        else:
            raise ValueError(f"unknown kind: {kind}")
        return compiled

    def artifacts():
        if compiled is None:
            raise RuntimeError("Compiler artifacts are available only after the first launch")
        return compiled_artifacts(compiled)

    return Workload(launch=launch, output=output, framework="torch", artifacts=artifacts,
                    metadata={"backend": "gluon", "num_warps": 4,
                              "precision": "fp32", "config": dict(config),
                              "plan_ids": list(config["plan_ids"]),
                              "target": "NVIDIA SM80+; designed for SM89"})


def build_experiments():
    from benchmarks.runner import Experiment

    cases = []
    for n in (1, 255, 256, 4099):
        for spt in (1, 2):
            plan_id = "G3.1" if spt == 1 else "G3.3"
            cases.append(Experiment(
                id=f"{plan_id}-add-spt{spt}-{n}", op="add", shape=(n,),
                config={"kind": "add", "block": 256, "spt": spt,
                        "plan_ids": ["G3.1", "G3.3", "G3.8"],
                        "layout": [[spt], [32], [4], [0]]}, prepare=prepare))
        for mode in ("registers", "shared_sync", "async_single", "async_double"):
            plan_id = "G3.7" if mode == "async_double" else "G3.6"
            plan_ids = ["G3.6", "G3.7", "G3.8"] if mode.startswith("async") else ["G3.6", "G3.8"]
            cases.append(Experiment(
                id=f"{plan_id}-copy-{mode}-{n}", op="copy", shape=(n,),
                config={"kind": "copy", "block": 128, "tiles": 4, "mode": mode,
                        "plan_ids": plan_ids,
                        "buffers": 2 if mode == "async_double" else (0 if mode == "registers" else 1),
                        "async": mode.startswith("async"),
                        "layout": [[1], [32], [4], [0]]}, prepare=prepare))
    for m, n in ((1, 127), (3, 129), (2, 4097)):
        for warps in (1, 4):
            plan_id = "G3.1" if warps == 1 else "G3.4"
            cases.append(Experiment(
                id=f"{plan_id}-sum-w{warps}-{m}x{n}", op="sum", shape=(m, n),
                config={"kind": "sum", "warps_per_row": warps, "rows_per_cta": 4,
                        "plan_ids": ["G3.1", "G3.4", "G3.8"],
                        "layout": [[1, 1], [1, 32], [4 // warps, warps], [1, 0]]},
                prepare=prepare))
    for m, n in ((32, 32), (37, 53)):
        for convert in (False, True):
            cases.append(Experiment(
                id=f"G3.5-transpose-{'convert' if convert else 'direct'}-{m}x{n}",
                op="transpose", shape=(m, n),
                config={"kind": "transpose", "tile": 32, "convert": convert,
                        "plan_ids": ["G3.5", "G3.8"],
                        "load_layout": [[1, 1], [4, 8], [4, 1], [1, 0]],
                        "converted_layout": [[1, 1], [8, 4], [1, 4], [0, 1]]}, prepare=prepare))
    # Stable numeric plan order retains the existing tiny shape order per exercise.
    cases.sort(key=lambda case: tuple(int(part) for part in case.id.split("-", 1)[0][1:].split(".")))
    return cases


def main():
    from benchmarks.runner import run_cli

    run_cli(build_experiments())


if __name__ == "__main__":
    main()
