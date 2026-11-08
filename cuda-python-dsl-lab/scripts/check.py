"""Import checks are separate from GPU execution checks."""
import argparse
import importlib
import importlib.metadata as md
import platform

parser = argparse.ArgumentParser()
parser.add_argument("target", choices=("triton", "cute", "cutile"))
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()
print("Python:", platform.python_version())
modules = {"triton": ("triton", "triton.experimental.gluon", "torch"),
           "cute": ("cutlass", "cutlass.cute", "torch"),
           "cutile": ("cuda.tile", "cupy")}
for module in modules[args.target]:
    imported = importlib.import_module(module)
    print("IMPORT OK:", module, getattr(imported, "__version__", ""), imported.__file__)
if args.target == "cute":
    print("nvidia-cutlass-dsl:", md.version("nvidia-cutlass-dsl"))
    from cutlass.base_dsl.version_info import CUDA_VERSION
    print("CuTe compiler CUDA:", CUDA_VERSION)
if args.gpu:
    if args.target == "cutile":
        import cupy as cp
        print("CUDA driver API:", cp.cuda.runtime.driverGetVersion())
        print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
        print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"])
        print("Compute capability:", cp.cuda.Device(0).compute_capability)
        cp.cuda.runtime.deviceSynchronize()
    else:
        import torch
        assert torch.cuda.is_available(), "CUDA device unavailable"
        print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
        torch.cuda.synchronize()
print("PASS:", "imports + GPU context" if args.gpu else "imports only (GPU not tested)")
