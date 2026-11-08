"""CPU-only checks for action logs and explicit-build requirements."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_support
import common
import session


class CommandTests(unittest.TestCase):
    def test_logs_replace_only_one_action_and_keep_history(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(session, 'ROOT', Path(tmp)), \
             contextlib.redirect_stdout(io.StringIO()):
            historical = Path(tmp) / 'results/old/log.txt'
            historical.parent.mkdir(parents=True)
            historical.write_text('history')
            other = session.Session('static')
            (other.path / 'keep.txt').write_text('other action')
            first = session.Session('run')
            (first.path / 'samples.csv').write_text('old samples')
            second = session.Session('run')
            self.assertEqual(first.path, second.path)
            self.assertFalse((second.path / 'samples.csv').exists())
            self.assertEqual(historical.read_text(), 'history')
            self.assertEqual((other.path / 'keep.txt').read_text(), 'other action')
            self.assertFalse((second.path / 'source').exists())
            code, _ = second.command([sys.executable, '-B', '-c', 'raise SystemExit(7)'], required=False)
            self.assertEqual(code, 7)
            command = json.loads((second.path / 'commands.jsonl').read_text())
            self.assertEqual(command['exit_code'], 7)

    def test_missing_build_never_invokes_tools(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(session, 'ROOT', Path(tmp)), \
             patch.object(build_support, 'ROOT', Path(tmp)), patch.dict(os.environ, CUDA_ARCH='sm_89'), \
             patch.object(session.subprocess, 'run') as command, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = common.execute('run', 'vector_add', lambda *args: self.fail('work must not run'))
            self.assertEqual(rc, 1)
            command.assert_not_called()
            status = (Path(tmp) / 'commands/run/log/status.txt').read_text()
            self.assertIn('commands/build/run.py', status)

    def test_host_build_location(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(build_support, 'ROOT', Path(tmp)):
            binary = Path(tmp) / 'build/host/host_tests'
            binary.parent.mkdir(parents=True)
            binary.touch()
            self.assertEqual(build_support.require_build([], host_only=True), binary.parent)

    def test_failed_explicit_build_invalidates_selected_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(build_support, 'ROOT', Path(tmp)), \
             patch.dict(os.environ, CUDA_ARCH='sm_89'):
            binary = Path(tmp) / 'build/sm_89/bin/vector_add'
            binary.parent.mkdir(parents=True)
            binary.write_text('old executable')

            class FakeSession:
                def command(self, args):
                    if args[0] == 'nvcc':
                        return 0, 'sm_89'
                    if '--build' in args:
                        binary.write_text('partial executable')
                        raise RuntimeError('compiler failed')
                    return 0, ''

            with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
                build_support.build(FakeSession(), selected=['vector_add'])
            self.assertFalse(binary.exists())


if __name__ == '__main__':
    unittest.main()
