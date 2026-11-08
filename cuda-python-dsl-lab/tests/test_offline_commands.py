"""Offline command migration tests; synthetic inputs, no GPU or CUDA tools."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import command_support


def load(action):
    spec = importlib.util.spec_from_file_location('offline_' + action, ROOT / 'commands' / action / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfflineTests(unittest.TestCase):
    def test_static_reads_in_place_without_snapshot_or_hash(self):
        static = load('static')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / 'input/cutile/case'
            (case / 'artifacts').mkdir(parents=True)
            artifact = case / 'artifacts/kernel.tileirbc'
            artifact.write_bytes(b'synthetic TileIR, not runnable')
            (case.parent / 'results.json').write_text(json.dumps({'cases': [dict(
                id='U5.1', op='copy', shape=[1], config={}, status='PASS', evidence='case')]}))
            output = root / 'output'
            output.mkdir()

            def command(ev, flags, input_record=None):
                self.assertEqual(flags, ['--version'])
                return {'status': 'ok'}

            with patch.object(static.Evidence, 'command', command):
                report = static.collect([root / 'input'], output, ['test'])
            self.assertEqual(report['errors'], [])
            self.assertEqual(report['counts']['tileir'], 1)
            inputs = json.loads((output / 'files.json').read_text())
            self.assertEqual(inputs[0]['input_path'], str(artifact))
            self.assertNotIn('sha256', inputs[0])
            self.assertFalse((output / 'source').exists())
            self.assertFalse((output / 'cases').exists())
            self.assertEqual(artifact.read_bytes(), b'synthetic TileIR, not runnable')

    def test_static_missing_inputs_returns_nonzero(self):
        static = load('static')
        with tempfile.TemporaryDirectory() as tmp, patch.object(command_support, 'ROOT', Path(tmp)), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(static.main([str(Path(tmp) / 'missing')]), 1)
            status = json.loads((Path(tmp) / 'commands/static/log/exit.json').read_text())
            self.assertEqual(status['returncode'], 1)

    def test_comparison_reuses_recorded_input_identity(self):
        compare = load('compare')
        with tempfile.TemporaryDirectory() as tmp, patch.object(command_support, 'ROOT', Path(tmp)), \
             contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for name in ('triton', 'gluon'):
                directory = root / 'input' / name
                directory.mkdir(parents=True)
                (directory / 'results.json').write_text(json.dumps({'cases': [dict(
                    id='copy', op='copy', shape=[1], status='PASS', input_id='same-input', evidence='case')]}))
            with patch.object(sys, 'argv', ['compare', str(root / 'input')]):
                self.assertEqual(compare.main(), 0)
            result = json.loads((root / 'commands/compare/log/comparison.json').read_text())
            self.assertEqual(result['status'], 'PASS')
            self.assertNotIn('source_sha256', result)
            self.assertNotIn('sha256', result['inputs'][0])


if __name__ == '__main__':
    unittest.main()
