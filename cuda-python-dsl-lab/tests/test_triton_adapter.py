"""CPU-only contract/wiring tests. Fake launches do NOT validate GPU numerics."""

from contextlib import ExitStack
from dataclasses import dataclass, field
import inspect
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments import triton_suite as suite
from experiments.gemm import triton_ops as gemm
from experiments.reduction import triton_ops as reduction
from experiments.transpose import triton_ops as transpose
from experiments.vector import triton_ops as vector


@dataclass
class ContractExperiment:
    id: str
    op: str
    shape: tuple
    config: dict
    prepare: object


@dataclass
class ContractWorkload:
    launch: object
    output: object
    framework: str = "torch"
    artifacts: object = None
    metadata: dict = field(default_factory=dict)


class FakeKernel:
    def __init__(self):
        self.calls = []
        self.handle = SimpleNamespace(
            asm={"ptx": "synthetic test PTX", "cubin": b"synthetic test cubin", "ttir": "test IR"},
            metadata=SimpleNamespace(num_warps=4, num_stages=2, shared=0, target="test-only"),
            name="fake_kernel", n_regs=8, n_spills=0,
        )

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return self.handle
        return launch


class TritonAdapterTests(unittest.TestCase):
    def setUp(self):
        # The runner may be under construction; use precisely its published constructor API.
        runner = ModuleType("benchmarks.runner")
        runner.Experiment = ContractExperiment
        runner.Workload = ContractWorkload
        runner.run_cli = Mock()
        self.runner = runner
        self.modules = patch.dict(sys.modules, {"benchmarks.runner": runner})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_catalog_order_shapes_and_scan(self):
        defaults = suite.build_experiments()
        scans = suite.build_experiments(scan=True)
        self.assertEqual(len(defaults), 40)
        self.assertEqual(len(scans), 48)
        self.assertEqual(list(dict.fromkeys(case.id for case in defaults)),
                         ["E0.1", "E0.2", "E0.3", "E0.4", "T1.1", "T1.2", "T1.3", "T1.4",
                          "T1.5", "T2.1", "T2.2", "T2.3", "T2.4"])
        keys = [(c.id, c.op, c.shape, json.dumps(c.config, sort_keys=True)) for c in scans]
        self.assertEqual(len(keys), len(set(keys)))
        for op in ("copy", "add", "axpy"):
            self.assertEqual([c.shape for c in defaults if c.id == "T1.1" and c.op == op],
                             [(1,), (255,), (256,), (4099,)])
        for op in ("sum", "softmax"):
            self.assertEqual([c.shape for c in defaults if c.id == "T1.3" and c.op == op],
                             [(1, 127), (3, 129), (2, 4097)])
        self.assertEqual([c.shape for c in defaults if c.id == "T1.4"],
                         [(32, 32, 32), (37, 53, 29)])
        for case in defaults:
            inspect.signature(case.prepare).bind((), case.config)
        vector_configs = [c.config for c in scans if c.id == "T1.2"]
        self.assertEqual({c["block"] for c in vector_configs}, {128, 256, 512, 1024})
        self.assertEqual({c["num_warps"] for c in vector_configs}, {4, 8})
        for case in (c for c in scans if c.id == "T2.2"):
            base = next(c.config for c in defaults if c.id == "T2.2")
            self.assertLessEqual(sum(case.config[key] != value for key, value in base.items()), 1)

    def test_cli_forwards_only_shared_arguments(self):
        argv = ["triton_suite", "--scan", "--output", "not-created", "--ids", "T1.2"]
        with patch.object(sys, "argv", argv[:]):
            self.runner.run_cli.side_effect = lambda cases: self.assertEqual(
                sys.argv, ["triton_suite", "--output", "not-created", "--ids", "T1.2"])
            suite.main()
            self.assertEqual(sys.argv, argv)
        self.assertEqual(len(self.runner.run_cli.call_args.args[0]), 48)

    def test_input_boundaries(self):
        suite._validate_inputs("gemm", (np.zeros((37, 29), np.float16), np.zeros((29, 53), np.float16)))
        suite._validate_inputs("softmax", (np.zeros((2, 4097), np.float32),))
        invalid = [
            ("copy", (np.zeros(0, np.float32),)),
            ("copy", (np.zeros(3, np.float16),)),
            ("add", (np.zeros(3, np.float32),)),
            ("add", (np.zeros(3, np.float32), np.zeros(4, np.float32))),
            ("transpose", (np.zeros((3, 4), np.float32).T,)),
            ("softmax", (np.array([[np.inf]], np.float32),)),
            ("softmax", (np.array([[np.nan]], np.float32),)),
            ("gemm", (np.zeros((2, 3), np.float16), np.zeros((4, 2), np.float16))),
        ]
        for op, arrays in invalid:
            with self.subTest(op=op), self.assertRaises(ValueError):
                suite._validate_inputs(op, arrays)

    def test_prepare_workload_contract_on_cpu(self):
        def cpu_transfer(tensor, *, device):
            self.assertEqual(device, "cuda")
            return tensor

        def fake_make_launch(inputs, output, config, compiled):
            def launch():
                compiled["test"] = FakeKernel().handle
            return launch, {"launches_per_call": 1}

        backend = SimpleNamespace(make_launch=fake_make_launch)
        examples = [
            ("copy", (np.zeros(255, np.float32),), (255,)),
            ("sum", (np.zeros((3, 129), np.float32),), (3,)),
            ("softmax", (np.zeros((3, 129), np.float32),), (3, 129)),
            ("transpose", (np.zeros((37, 53), np.float32),), (53, 37)),
            ("gemm", (np.zeros((37, 29), np.float16), np.zeros((29, 53), np.float16)), (37, 53)),
        ]
        with patch.object(torch.Tensor, "to", cpu_transfer), patch.object(suite, "import_module", return_value=backend):
            for op, arrays, shape in examples:
                with self.subTest(op=op):
                    workload = suite._prepare(op, arrays, {})
                    self.assertIsInstance(workload, ContractWorkload)
                    self.assertEqual(tuple(workload.output.shape), shape)
                    self.assertTrue(workload.output.is_contiguous())
                    self.assertEqual(workload.output.dtype, torch.float16 if op == "gemm" else torch.float32)
                    self.assertEqual(workload.framework, "torch")
                    with self.assertRaises(RuntimeError):
                        workload.artifacts()
                    workload.launch()
                    self.assertIn("test.cubin", workload.artifacts())
        self.assertFalse(torch.cuda.is_initialized())

    def test_artifact_collection(self):
        with self.assertRaises(RuntimeError):
            suite._artifacts({})
        fake = FakeKernel().handle
        artifacts = suite._artifacts({"scale": fake, "add": fake})
        self.assertEqual(artifacts["scale.ptx"], fake.asm["ptx"])
        self.assertEqual(artifacts["add.cubin"], fake.asm["cubin"])
        self.assertEqual(json.loads(artifacts["add.metadata.json"])["registers_per_thread"], 8)
        del fake.asm["cubin"]
        with self.assertRaises(RuntimeError):
            suite._artifacts({"missing": fake})

    def test_launch_wiring_without_device_or_tensor_allocations(self):
        with ExitStack() as stack:
            fakes = {}
            for module, names in ((vector, ["vector_kernel"]),
                                  (transpose, ["naive_transpose", "tiled_transpose"]),
                                  (reduction, ["row_sum", "row_softmax"]), (gemm, ["gemm_kernel"])):
                for name in names:
                    fakes[name] = FakeKernel()
                    stack.enter_context(patch.object(module, name, fakes[name]))
            for case in suite.build_experiments(scan=True):
                with self.subTest(id=case.id, op=case.op, shape=case.shape, config=case.config):
                    if case.op == "gemm":
                        m, n, k = case.shape
                        inputs = (torch.empty((m, k), dtype=torch.float16), torch.empty((k, n), dtype=torch.float16))
                        output = torch.empty((m, n), dtype=torch.float16)
                    else:
                        inputs = (torch.empty(case.shape, dtype=torch.float32),)
                        if case.op in {"add", "axpy", "fusion"}:
                            inputs += (torch.empty_like(inputs[0]),)
                        shape = case.shape[::-1] if case.op == "transpose" else case.shape
                        if case.op == "sum":
                            shape = (case.shape[0],)
                        output = torch.empty(shape, dtype=torch.float32)
                    module = {"vector": vector, "transpose": transpose, "reduction": reduction,
                              "gemm": gemm}[suite._BACKENDS[case.op]]
                    compiled = {}
                    launch, metadata = module.make_launch(inputs, output, {**case.config, "op": case.op}, compiled)
                    before = sum(len(fake.calls) for fake in fakes.values())
                    with patch.object(torch, "empty", side_effect=AssertionError("allocation in launch")), \
                         patch.object(torch, "empty_like", side_effect=AssertionError("allocation in launch")):
                        launch()
                        launch()
                    after = sum(len(fake.calls) for fake in fakes.values())
                    self.assertEqual(after - before, 2 * metadata["launches_per_call"])
                    self.assertEqual(len(compiled), metadata["launches_per_call"])
                    self.assertTrue(output.is_contiguous())
                    if case.op == "fusion" and case.config["variant"] == "unfused":
                        scale, add = fakes["vector_kernel"].calls[-2:]
                        self.assertEqual((scale[2]["OP"], add[2]["OP"]), ("scale", "add"))
                        self.assertIs(scale[1][2], add[1][0])
                        self.assertIs(add[1][2], output)
            for fake in fakes.values():
                for _, _, kwargs in fake.calls:
                    self.assertIs(kwargs["enable_fp_fusion"], False)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
