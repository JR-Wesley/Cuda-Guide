"""Standard-library checks for teaching outputs without source/artifact fingerprints."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import cutile_suite, gluon_layouts


class MetadataTests(unittest.TestCase):
    def test_cutile_provenance_records_versions_without_reading_sources(self):
        with patch.object(cutile_suite, 'version', return_value='test-version') as version, \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('must not read package sources')):
            result = cutile_suite.api_provenance()
        self.assertEqual(set(result), {'python', 'packages'})
        self.assertEqual(set(result['packages']), {'cuda-tile', 'cupy-cuda13x', 'numpy'})
        self.assertEqual(version.call_count, 3)

    def test_cutile_export_keeps_bytecode_and_empty_output_check(self):
        compilation = ModuleType('cuda.tile.compilation')
        compilation.CallingConvention = SimpleNamespace(cutile_python_v1=lambda: 'test-convention')
        compilation.KernelSignature = SimpleNamespace(from_kernel_args=lambda *args: 'test-signature')
        modules = {'cuda': ModuleType('cuda'), 'cuda.tile': ModuleType('cuda.tile'),
                   'cuda.tile.compilation': compilation}
        for payload in (b'synthetic TileIR, not executable', b''):
            with self.subTest(empty=not payload):
                def export(kernel, signatures, buffer, **kwargs):
                    self.assertEqual(kwargs, {'output_format': 'tileir_bytecode', 'bytecode_version': '13.3'})
                    buffer.write(payload)

                compilation.export_kernel = export
                with patch.dict(sys.modules, modules), \
                     patch.object(cutile_suite, 'api_provenance', return_value={'packages': {}}):
                    artifacts = cutile_suite._artifact_export(object(), ())
                report = json.loads(artifacts['compiler-artifacts.json'])
                self.assertNotIn('sha256', report)
                if payload:
                    self.assertEqual(artifacts['kernel.tileirbc'], payload)
                    self.assertEqual(report['status'], 'exported')
                    self.assertEqual(report['jit_binary_identity'], 'not verified')
                else:
                    self.assertNotIn('kernel.tileirbc', artifacts)
                    self.assertEqual(report['status'], 'not measured')
                    self.assertIn('empty TileIR bytecode', report['error'])

    def test_gluon_cpu_layouts_keep_checks_without_source_copy(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('must not read source bytes')):
            output = Path(tmp)
            gluon_layouts.main(['--output', str(output)])
            summary = json.loads((output / 'summary.json').read_text())
            self.assertNotIn('source_sha256', summary)
            self.assertFalse((output / 'gluon_layouts.py').exists())
            self.assertEqual(summary['gpu_verification'], 'not requested')
            self.assertEqual(summary['exit_code'], 0)
            self.assertEqual(len(summary['cases']), 10)
            self.assertEqual(len(list(output.glob('*.csv'))), 10)
            self.assertTrue((output / 'maps.md').is_file())
            for case in summary['cases']:
                self.assertEqual(case['checks']['inverse_check'], 'passed')
                self.assertEqual(case['checks']['missing'], 0)
                self.assertEqual(case['checks']['duplicates'], 0)


if __name__ == '__main__':
    unittest.main()
