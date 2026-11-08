import unittest
import numpy as np

from benchmarks.core import make_case, validate
from benchmarks.runner import matches


class SharedInputsTests(unittest.TestCase):
    def test_reproducible_and_shared(self):
        for op, shape in [('add', (255,)), ('gemm', (37, 53, 29)), ('softmax', (2, 4097))]:
            a, b = make_case(op, shape), make_case(op, shape)
            self.assertEqual(a.input_id, b.input_id)
            for x, y in zip(a.inputs, b.inputs):
                np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(make_case('copy', (255,)).inputs[0], make_case('add', (255,)).inputs[0])
        self.assertEqual(make_case('axpy', (255,)).input_id, make_case('fusion', (255,)).input_id)

    def test_supported_outputs(self):
        shapes = {'copy': (1,), 'add': (255,), 'scale': (256,), 'axpy': (4099,),
                  'fusion': (255,), 'transpose': (37, 53), 'sum': (3, 129),
                  'softmax': (2, 4097), 'gemm': (37, 53, 29)}
        for op, shape in shapes.items():
            with self.subTest(op=op):
                case = make_case(op, shape)
                self.assertEqual(validate(case.expected, case)['max_abs_error'], 0)
        softmax = make_case('softmax', (2, 4097)).expected
        np.testing.assert_allclose(softmax.sum(axis=1), 1, rtol=1e-6)

    def test_rejects_invalid_and_nan(self):
        for op, shape in [('copy', (0,)), ('sum', (1,)), ('gemm', (1, -2, 3)), ('unknown', (1,))]:
            with self.assertRaises(ValueError):
                make_case(op, shape)
        case = make_case('copy', (4,))
        for actual in [np.full(4, np.nan, dtype=np.float32), np.zeros(4, dtype=np.float64), np.zeros(5, dtype=np.float32)]:
            with self.assertRaises(AssertionError):
                validate(actual, case)

    def test_transpose_is_materialized(self):
        case = make_case('transpose', (37, 53))
        self.assertTrue(case.expected.flags.c_contiguous)
        with self.assertRaises(AssertionError):
            validate(case.inputs[0].T, case)

    def test_exact_copy_and_plan_prefix(self):
        case = make_case('copy', (1,))
        with self.assertRaises(AssertionError):
            validate(case.expected + np.float32(1e-4), case)
        self.assertTrue(matches('C4.4.copy.n1', ['C4.4']))
        self.assertTrue(matches('G3.7-copy-async_double-1', ['G3.7']))
        self.assertFalse(matches('C4.10.gemm', ['C4.1']))


if __name__ == '__main__':
    unittest.main()
