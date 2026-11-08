"""Framework-independent, deterministic small inputs and CPU references (E0.3)."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass
class Case:
    inputs: tuple
    expected: np.ndarray
    input_id: str
    rtol: float
    atol: float
    algorithm_bytes: int | None
    flops: int | None


def make_case(op, shape):
    shape = tuple(shape)
    rank = 3 if op == 'gemm' else (2 if op in ('transpose', 'sum', 'softmax') else 1)
    supported = ('copy', 'add', 'axpy', 'scale', 'fusion', 'transpose', 'sum', 'softmax', 'gemm')
    if op not in supported or len(shape) != rank or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError(f'Unsupported op/positive shape: {op}, {shape}')
    # Seed depends on the input family, not the DSL, layout, or candidate config.
    family = 'vector' if rank == 1 else ('gemm' if op == 'gemm' else 'matrix')
    key = json.dumps([family, shape, 'numpy-PCG64-v1'], separators=(',', ':')).encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
    rng = np.random.Generator(np.random.PCG64(seed))
    if op == 'gemm':
        m, n, k = shape
        arrays = (rng.standard_normal((m, k)).astype(np.float16),
                  rng.standard_normal((k, n)).astype(np.float16))
        expected = (arrays[0].astype(np.float64) @ arrays[1].astype(np.float64)).astype(np.float16)
        rtol, atol, traffic, flops = 2e-3, 2e-3, None, 2*m*n*k
    else:
        a = rng.standard_normal(shape).astype(np.float32)
        # Signed, large finite values exercise stable softmax. All matrix operators
        # see identical arrays at identical shape (not independently generated RNGs).
        if rank == 2:
            a *= np.float32(20)
        b = rng.standard_normal(shape).astype(np.float32) if op in ('add', 'axpy', 'fusion') else None
        arrays = (a, b) if b is not None else (a,)
        traffic, flops = None, None
        rtol, atol = 1e-6, 1e-6
        if op == 'copy':
            expected, traffic = a.copy(), 2*a.nbytes
            rtol = atol = 0
        elif op == 'add':
            expected, traffic = a + b, 3*a.nbytes
        elif op in ('axpy', 'fusion'):
            expected = (a * np.float32(0.5)) + b
            traffic = 3*a.nbytes if op == 'axpy' else None
        elif op == 'scale':
            expected, traffic = a * np.float32(0.5), 2*a.nbytes
        elif op == 'transpose':
            expected, traffic = a.T.copy(), 2*a.nbytes
            rtol = atol = 0
        elif op == 'sum':
            expected = a.astype(np.float64).sum(axis=1).astype(np.float32)
            rtol, atol = 2e-5, 2e-3
        elif op == 'softmax':
            shifted = a.astype(np.float64) - a.max(axis=1, keepdims=True).astype(np.float64)
            exps = np.exp(shifted)
            expected = (exps / exps.sum(axis=1, keepdims=True)).astype(np.float32)
            rtol, atol = 2e-5, 2e-6
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order='C'))
    return Case(arrays, expected, digest.hexdigest(), rtol, atol, traffic, flops)


def validate(actual, case):
    if actual.shape != case.expected.shape or actual.dtype != case.expected.dtype:
        raise AssertionError(f'Output shape/dtype {actual.shape}/{actual.dtype}, expected '
                             f'{case.expected.shape}/{case.expected.dtype}')
    if not actual.flags.c_contiguous:
        raise AssertionError('Output must be materialized C-contiguous')
    if not np.isfinite(actual).all():
        raise AssertionError('Output contains NaN/Inf (unwritten or invalid elements)')
    delta = np.abs(actual.astype(np.float64) - case.expected.astype(np.float64))
    relative = delta / np.maximum(np.abs(case.expected.astype(np.float64)), 1e-12)
    metrics = {'max_abs_error': float(delta.max()), 'max_rel_error': float(relative.max()),
               'relative_error_denominator_floor': 1e-12, 'rtol': case.rtol, 'atol': case.atol}
    np.testing.assert_allclose(actual, case.expected, rtol=case.rtol, atol=case.atol,
                               equal_nan=False)
    return metrics
