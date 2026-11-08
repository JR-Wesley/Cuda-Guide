"""CPU doubles exercise logging/timing failure paths without a GPU or benchmark."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from benchmarks.core import make_case
from benchmarks.runner import Experiment, Workload, run_cli


class Event:
    def record(self, stream):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 1.0


def cpu_adapter(work):
    return dict(sync=lambda: None, poison=lambda: work.output.fill(np.nan),
                cpu=lambda: work.output.copy(), contiguous=lambda: True,
                event=Event, stream=None, stream_id=0, device='CPU test double',
                capability=None, runtime='test double', strides=work.output.strides)


class RunnerTests(unittest.TestCase):
    def execute(self, directory, mode, bench=False):
        calls = 0

        def prepare(inputs, config):
            output = np.empty_like(inputs[0])

            def launch():
                nonlocal calls
                calls += 1
                if mode == 'noop-after-two' and calls > 2:
                    return
                output[:] = inputs[0] + (1 if mode == 'corrupt-after-two' and calls > 2 else 0)

            return Workload(launch, output)

        exp = Experiment('T1.1', 'copy', (4,), {}, prepare)
        argv = ['test', '--output', str(directory)]
        if bench:
            argv += ['--bench', '--warmup', '1', '--repeat', '1', '--batches', '1']
        with patch.object(sys, 'argv', argv), patch('benchmarks.runner.adapter', cpu_adapter), \
             contextlib.redirect_stdout(io.StringIO()):
            if mode == 'good':
                run_cli([exp])
            else:
                with self.assertRaises(SystemExit):
                    run_cli([exp])
        return json.loads((directory / 'results.json').read_text())

    def test_functional_default_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = self.execute(out, 'good')
            self.assertEqual(result['passed'], 1)
            self.assertEqual(result['mode'], 'functional-only')
            self.assertEqual(len(result['cases'][0]['correctness']['repeats']), 2)
            with self.assertRaises(FileExistsError):
                self.execute(out, 'good')

    def test_timing_double_and_post_warmup_noop(self):
        for mode in ('good', 'noop-after-two', 'corrupt-after-two'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                result = self.execute(Path(tmp), mode, bench=True)
                record = result['cases'][0]
                self.assertEqual(record['status'], 'PASS' if mode == 'good' else 'FAIL')
                self.assertEqual(record['timing']['batch_average_ms'], [1.0])
                self.assertEqual(record['benchmark_correctness'][0]['status'], record['status'])
                actual = Path(tmp) / record['evidence'] / 'actual-benchmark-0.npy'
                self.assertTrue(actual.is_file())


if __name__ == '__main__':
    unittest.main()
