"""CPU contract tests, not CUDA/GPU validation.

Only current scripts, record.sh and sources are copied into a temporary tree.
Fake artifacts/reports and all action logs stay there; no real project build or
log is read/copied/written. Fake output satisfies file/grep checks solely to test
shell orchestration, never to establish compilation or GPU correctness.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = 'compile/artifacts/'
BUILD_OUTPUTS = {
    'build': ('vector_transpose',),
    'module-build': ('module-loading/runtime',),
    'module-driver-build': ('module-loading/driver',),
    'module-cubin-build': ('module-loading/image.cubin',),
    'module-ptx-build': ('module-loading/image.ptx',),
    'sanitizer-build': ('sanitizer/cases',),
    'compile-dryrun': (),
    'compile-ptx': (ARTIFACTS + 'probe.ptx',),
    'compile-cubin': (ARTIFACTS + 'probe.cubin',),
    'compile-fatbin': (ARTIFACTS + 'probe.fatbin',),
    'compile-object': (ARTIFACTS + 'probe.o',),
    'compile-host-link': (ARTIFACTS + 'probe',),
    'compile-rdc-compile': (ARTIFACTS + 'rdc_device.o', ARTIFACTS + 'rdc_main.o'),
    'compile-rdc-device-link': (ARTIFACTS + 'rdc_dlink.o',),
    'compile-rdc-host-link': (ARTIFACTS + 'rdc_app',),
    'compile-lto-compile': (ARTIFACTS + 'rdc_device_lto.o', ARTIFACTS + 'rdc_main_lto.o'),
    'compile-lto-device-link': (ARTIFACTS + 'rdc_lto_dlink.o',),
    'compile-lto-host-link': (ARTIFACTS + 'rdc_lto_app',),
}
CONSUMERS = {
    'execution': ('build/sm_89/vector_transpose',),
    'validation': ('build/sm_89/vector_transpose',),
    'validation-racecheck': ('build/sm_89/vector_transpose',),
    'ncu': ('build/sm_89/vector_transpose',),
    'ncu-memory': ('build/sm_89/vector_transpose',),
    'nsys': ('build/sm_89/vector_transpose',),
    'ncu-report': ('experiments/ncu/log/basic.ncu-repz',),
    'nsys-stats': ('experiments/nsys/log/timeline.nsys-rep',),
    'module-loading': ('build/sm_89/module-loading/runtime',),
    'module-driver-cubin': ('build/sm_89/module-loading/driver',
                            'build/sm_89/module-loading/image.cubin'),
    'module-driver-ptx': ('build/sm_89/module-loading/driver',
                          'build/sm_89/module-loading/image.ptx'),
    'sanitizer-static': ('build/sm_89/sanitizer/cases',),
    **{f'sanitizer-{tool}': ('build/sm_89/sanitizer/cases',)
       for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck')},
    **{f'binary-{action}': (f'build/sm_89/{ARTIFACTS}probe.{extension}',)
       for action, extension in (('elf', 'cubin'), ('list', 'fatbin'), ('ptx', 'fatbin'),
                                 ('resources', 'cubin'), ('sass', 'cubin'))},
    **{f'nvdisasm-{action}': (f'build/sm_89/{ARTIFACTS}probe.cubin',)
       for action in ('cfg', 'live-ranges', 'sass')},
}
TOOLS = ('nvcc', 'cuobjdump', 'nvdisasm', 'nvidia-smi', 'nsys', 'ncu',
         'compute-sanitizer', 'g++', 'readelf', 'nm', 'file', 'ptxas', 'nvlink')
DIAGNOSTICS = {
    'memcheck': 'Invalid __global__ write',
    'racecheck': 'Race reported',
    'initcheck': 'Uninitialized __global__ memory read',
    'synccheck': 'Barrier error detected. Invalid arguments',
}

FAKE = r'''
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ['FAKE_ROOT']).resolve()

def local(path):
    path = Path(path).resolve()
    path.relative_to(root)  # Refuse artifacts outside the disposable test tree.
    return path

with local(os.environ['TRACE']).open('a') as stream:
    stream.write(json.dumps({'name': name, 'args': args, 'cwd': str(Path.cwd())}) + '\n')

def value(flag):
    return args[args.index(flag) + 1]

def output(path, text):
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path

if name == 'nvcc':
    for index, arg in enumerate(args):
        if arg.endswith(('.cu', '.cpp', '.o')) and (index == 0 or args[index - 1] != '-o'):
            assert local(arg).is_file(), 'missing fake compiler input: ' + arg
    if '-o' in args and '--dryrun' not in args:
        # Executable fixtures trace any accidental launch during compile tests.
        path = output(value('-o'), local(os.environ['FAKE_PROGRAM']).read_text())
        path.chmod(0o755)

if (os.environ.get('FAKE_FAIL_TOOL') == name
        and (not os.environ.get('FAKE_FAIL_OUTPUT')
             or ('-o' in args and Path(value('-o')).name == os.environ['FAKE_FAIL_OUTPUT']))):
    print('FAKE failure stdout')
    print('FAKE failure stderr', file=sys.stderr)
    sys.exit(int(os.environ.get('FAKE_FAIL_RC', '37')))

if name == 'ncu':
    if '--import' in args:
        assert local(value('--import')).is_file()
    if '--export' in args:
        output(value('--export') + '.ncu-repz', 'FAKE NCU REPORT\n')
        print('FAKE correct=yes, errors=0,')
if name == 'nsys':
    if args[0] == 'profile':
        dest = next(a.split('=', 1)[1] for a in args if a.startswith('--output='))
        output(dest + '.nsys-rep', 'FAKE NSYS REPORT\n')
        print('FAKE correct=yes, errors=0,')
    elif args[0] == 'stats':
        assert local(args[-1]).is_file()
        output(value('--sqlite'), 'FAKE SQLITE (not a database)\n')
if name == 'compute-sanitizer':
    tool = value('--tool')
    assert value('--error-exitcode') == '86'
    if '--case' in args:
        print('FAKE correct=yes, errors=0,')
        print('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
              if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors')
    else:
        diagnostics = json.loads(os.environ['FAKE_DIAGNOSTICS'])
        print(os.environ.get('FAKE_DIAGNOSTIC', diagnostics[tool]))
        print(os.environ.get('FAKE_SUMMARY',
              'RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)'
              if tool == 'racecheck' else 'ERROR SUMMARY: 1 error'))
        sys.exit(int(os.environ.get('FAKE_SANITIZER_RC', '86')))
if name not in json.loads(os.environ['FAKE_TOOLS']):
    print('FAKE correct=yes, errors=0,')
print('FAKE CPU tool success; not GPU evidence')
'''


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='sdk-action-tests-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        scripts = [ROOT / 'experiments/record.sh',
                   *sorted((ROOT / 'experiments').glob('*/run.sh'))]
        sources = [p for p in (ROOT / 'experiments').glob('*/*')
                   if p.is_file() and p.suffix in ('.cu', '.cpp', '.h', '.cuh')]
        sources += list((ROOT / 'examples').glob('*.cu'))
        for source in scripts + sources:
            destination = self.root / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        tools = self.root / 'tools'
        tools.mkdir()
        # Do not append the host PATH: a missing fake must not fall back to CUDA.
        for name in ('bash', 'mkdir', 'rm', 'cat', 'grep', 'timeout'):
            executable = shutil.which(name)
            if executable is None:
                self.skipTest(f'CPU shell tests require {name}')
            (tools / name).symlink_to(executable)
        self.fake_program = '#!' + sys.executable + '\n' + FAKE
        for name in TOOLS:
            self.fixture(f'tools/{name}', self.fake_program, executable=True)
        self.env = {
            'PATH': str(tools), 'HOME': str(self.root), 'TMPDIR': str(self.root),
            'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_ARCH': 'sm_89',
            'TRACE': str(self.root / 'trace.jsonl'), 'FAKE_ROOT': str(self.root),
            'FAKE_PROGRAM': str(tools / 'nvcc'), 'FAKE_TOOLS': json.dumps(TOOLS),
            'FAKE_DIAGNOSTICS': json.dumps(DIAGNOSTICS),
            **{variable: str(tools / name) for variable, name in (
                ('NVCC', 'nvcc'), ('CUOBJDUMP', 'cuobjdump'), ('NVDISASM', 'nvdisasm'),
                ('NCU', 'ncu'), ('NSYS', 'nsys'), ('COMPUTE_SANITIZER', 'compute-sanitizer'))},
        }

    def fixture(self, relative, text='FAKE input, not a CUDA artifact\n', executable=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if executable:
            path.chmod(0o755)
        return path

    def seed_inputs(self, action):
        for relative in CONSUMERS[action]:
            self.fixture(relative, self.fake_program, executable=True)

    def seed_builds(self):
        for outputs in BUILD_OUTPUTS.values():
            for relative in outputs:
                self.fixture('build/sm_89/' + relative, self.fake_program, executable=True)

    def invoke(self, action, *args, extra_env=None):
        (self.root / 'trace.jsonl').unlink(missing_ok=True)
        return subprocess.run(
            [str(self.root / 'tools/bash'), 'run.sh', *args],
            cwd=self.root / 'experiments' / action,
            env={**self.env, **(extra_env or {})}, capture_output=True, text=True, timeout=20)

    def trace(self):
        path = self.root / 'trace.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def status(self, action):
        lines = (self.root / 'experiments' / action / 'log/status.tsv').read_text().splitlines()
        self.assertEqual(lines[0], 'step\texit_code')
        return [(step, int(code)) for step, code in (line.split('\t') for line in lines[1:])]

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def one_tool(self, name):
        commands = self.trace()
        self.assertEqual([command['name'] for command in commands], [name], commands)
        return commands[0]['args']

    def test_inventory_covers_every_current_run_action(self):
        actions = {p.parent.name for p in (self.root / 'experiments').glob('*/run.sh')}
        self.assertEqual(actions, set(BUILD_OUTPUTS) | set(CONSUMERS))
        self.assertTrue(set(BUILD_OUTPUTS).isdisjoint(CONSUMERS))

    def test_every_consumer_rejects_each_missing_input_without_compiling(self):
        for action, inputs in CONSUMERS.items():
            for missing in inputs:
                with self.subTest(action=action, missing=missing):
                    self.seed_inputs(action)
                    (self.root / missing).unlink()
                    result = self.invoke(action)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn('Missing ', result.stderr)
                    self.assertEqual(self.status(action), [('input', 1)])
                    self.assertEqual(self.trace(), [], 'Missing inputs must fail before tools run')

    def test_every_consumer_reuses_existing_inputs_without_compiling(self):
        for action in CONSUMERS:
            with self.subTest(action=action):
                self.seed_inputs(action)
                self.assert_success(self.invoke(action))
                self.assertTrue(self.trace(), 'Expected an actual fake consumer invocation')
                self.assertNotIn('nvcc', [command['name'] for command in self.trace()])
                self.assertTrue(self.status(action))
                self.assertFalse((self.root / 'experiments' / action / 'build').exists())

    def test_all_compile_and_link_actions_only_invoke_nvcc(self):
        # Dependency order is explicit; no action may secretly run its predecessors.
        for action, outputs in BUILD_OUTPUTS.items():
            with self.subTest(action=action):
                self.assert_success(self.invoke(action))
                commands = self.trace()
                self.assertEqual([c['name'] for c in commands], ['nvcc'] * max(1, len(outputs)))
                self.assertTrue(all(code == 0 for _, code in self.status(action)))
                for relative in outputs:
                    self.assertTrue((self.root / 'build/sm_89' / relative).is_file())
                if action == 'compile-dryrun':
                    self.assertIn('--dryrun', commands[0]['args'])
                    self.assertFalse((self.root / 'build/sm_89' / ARTIFACTS / 'probe_dryrun').exists())

    def test_failed_builds_remove_partial_and_stale_outputs_and_keep_real_code(self):
        self.seed_builds()
        sentinel = self.fixture('build/sm_89/unrelated', 'keep me\n')
        for action, outputs in BUILD_OUTPUTS.items():
            if not outputs:
                continue
            with self.subTest(action=action):
                self.seed_builds()
                # Fail on the second compile for paired RDC/LTO outputs, after the
                # first has succeeded. The fake creates the partial output first.
                result = self.invoke(action, extra_env={
                    'FAKE_FAIL_TOOL': 'nvcc', 'FAKE_FAIL_RC': '37',
                    'FAKE_FAIL_OUTPUT': Path(outputs[-1]).name,
                })
                self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
                self.assertEqual([c['name'] for c in self.trace()], ['nvcc'] * len(outputs))
                self.assertEqual([code for _, code in self.status(action)],
                                 [0] * (len(outputs) - 1) + [37])
                for relative in outputs:
                    self.assertFalse((self.root / 'build/sm_89' / relative).exists(), relative)
                self.assertEqual(sentinel.read_text(), 'keep me\n')

    def test_ncu_captures_one_filtered_kernel_without_import_or_compile(self):
        for action, report, sections in (
                ('ncu', 'basic', ['LaunchStats', 'SpeedOfLight']),
                ('ncu-memory', 'memory', ['MemoryWorkloadAnalysis_Tables'])):
            with self.subTest(action=action):
                self.seed_inputs(action)
                target = self.fixture(f'experiments/{action}/log/{report}.ncu-repz', 'stale\n')
                self.assert_success(self.invoke(action))
                args = self.one_tool('ncu')
                for flag, value in (('--kernel-name', 'vector_add'), ('--launch-skip', '1'),
                                    ('--launch-count', '1'), ('--case', 'vector'),
                                    ('--n', '257'), ('--repeat', '1'), ('--export', f'./log/{report}')):
                    self.assertEqual(args.count(flag), 1)
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertEqual([args[i + 1] for i, arg in enumerate(args) if arg == '--section'], sections)
                self.assertNotIn('--import', args)
                if action == 'ncu-memory':
                    for flag, value in (('--page', 'details'), ('--print-details', 'all')):
                        self.assertEqual(args.count(flag), 1)
                        self.assertEqual(args[args.index(flag) + 1], value)
                        self.assertLess(args.index(flag), args.index('../../build/sm_89/vector_transpose'))
                    self.assertNotIn('--csv', args)
                self.assertEqual(target.read_text(), 'FAKE NCU REPORT\n')
                self.assertEqual(self.status(action), [('profile', 0), ('correctness', 0)])

    def test_ncu_report_only_imports_and_preserves_capture(self):
        report = self.fixture('experiments/ncu/log/basic.ncu-repz', 'FAKE capture to preserve\n')
        self.assert_success(self.invoke('ncu-report'))
        self.assertEqual(self.one_tool('ncu'),
                         ['--import', '../ncu/log/basic.ncu-repz', '--page', 'details'])
        self.assertEqual(report.read_text(), 'FAKE capture to preserve\n')
        self.assertEqual(self.status('ncu-report'), [('details', 0)])
        self.assertFalse((self.root / 'build').exists())

    def test_nsys_only_profiles_and_does_not_run_stats(self):
        self.seed_inputs('nsys')
        report = self.fixture('experiments/nsys/log/timeline.nsys-rep', 'stale\n')
        self.assert_success(self.invoke('nsys'))
        args = self.one_tool('nsys')
        self.assertEqual(args, ['profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                                '--force-overwrite=true', '--output=./log/timeline',
                                '../../build/sm_89/vector_transpose', '--case', 'vector',
                                '--n', '257', '--repeat', '1'])
        self.assertFalse(list(self.root.rglob('*.sqlite')))
        self.assertEqual(report.read_text(), 'FAKE NSYS REPORT\n')
        self.assertEqual(self.status('nsys'), [('profile', 0), ('correctness', 0)])

    def test_nsys_stats_only_reads_report_and_keeps_sqlite_in_own_log(self):
        report = self.fixture('experiments/nsys/log/timeline.nsys-rep', 'FAKE capture to preserve\n')
        capture_db = self.fixture('experiments/nsys/log/timeline.sqlite', 'keep capture DB\n')
        database = self.fixture('experiments/nsys-stats/log/timeline.sqlite', 'stale stats DB\n')
        self.assert_success(self.invoke('nsys-stats'))
        self.assertEqual(self.one_tool('nsys'),
                         ['stats', '--report', 'cuda_gpu_kern_sum', '--report', 'cuda_api_sum',
                          '--sqlite', './log/timeline.sqlite', '../nsys/log/timeline.nsys-rep'])
        self.assertEqual(database.read_text(), 'FAKE SQLITE (not a database)\n')
        self.assertEqual(capture_db.read_text(), 'keep capture DB\n')
        self.assertEqual(report.read_text(), 'FAKE capture to preserve\n')
        self.assertEqual(self.status('nsys-stats'), [('stats', 0)])
        self.assertFalse((self.root / 'build').exists())

    def test_record_preserves_tool_failure_code_output_and_command(self):
        for action, tool, step in (('ncu', 'ncu', 'profile'), ('ncu-memory', 'ncu', 'profile'),
                                   ('ncu-report', 'ncu', 'details'), ('nsys', 'nsys', 'profile'),
                                   ('nsys-stats', 'nsys', 'stats'),
                                   ('execution', 'vector_transpose', 'vector')):
            with self.subTest(action=action):
                self.seed_inputs(action)
                result = self.invoke(action, extra_env={'FAKE_FAIL_TOOL': tool, 'FAKE_FAIL_RC': '29'})
                self.assertEqual(result.returncode, 29, result.stdout + result.stderr)
                self.assertEqual(self.status(action), [(step, 29)])
                self.one_tool(tool)
                log = self.root / 'experiments' / action / 'log'
                self.assertCountEqual((log / f'{step}.log').read_text().splitlines(),
                                      ['FAKE failure stdout', 'FAKE failure stderr'])
                self.assertIn(tool, (log / f'{step}.sh').read_text())

    def test_sanitizer_four_tools_require_expected_exit_diagnostic_and_summary(self):
        for tool in DIAGNOSTICS:
            action = 'sanitizer-' + tool
            with self.subTest(tool=tool):
                self.seed_inputs(action)
                self.assert_success(self.invoke(action))
                self.assertEqual(self.one_tool('compute-sanitizer'),
                                 ['--tool', tool, '--error-exitcode', '86',
                                  '../../build/sm_89/sanitizer/cases', tool])
                self.assertEqual(self.status(action),
                                 [(tool, 86), ('expected_exit', 0), ('diagnostic', 0), ('summary', 0)])

    def test_sanitizer_rejects_arbitrary_failures_and_incomplete_evidence(self):
        for tool in DIAGNOSTICS:
            action = 'sanitizer-' + tool
            zero = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                    if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors')
            cases = (
                ('wrong exit', {'FAKE_SANITIZER_RC': '42'}, 'expected_exit', 42),
                ('zero exit', {'FAKE_SANITIZER_RC': '0'}, 'expected_exit', 0),
                ('unrelated failure', {'FAKE_DIAGNOSTIC': 'CUDA driver unavailable',
                                       'FAKE_SUMMARY': ''}, 'diagnostic', 86),
                ('wrong diagnostic', {'FAKE_DIAGNOSTIC': 'unrelated diagnostic'}, 'diagnostic', 86),
                ('zero summary', {'FAKE_SUMMARY': zero}, 'summary', 86),
                ('missing summary', {'FAKE_SUMMARY': ''}, 'summary', 86),
            )
            for label, env, failed_step, raw_code in cases:
                with self.subTest(tool=tool, evidence=label):
                    self.seed_inputs(action)
                    result = self.invoke(action, extra_env=env)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    status = self.status(action)
                    self.assertEqual(status[0], (tool, raw_code))
                    self.assertEqual(status[-1], (failed_step, 1))
                    self.one_tool('compute-sanitizer')

    def test_racecheck_requires_nonzero_hazards_and_errors_not_just_warnings(self):
        action = 'sanitizer-racecheck'
        self.seed_inputs(action)
        for summary in ('RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)',
                        'RACECHECK SUMMARY: 0 hazards displayed (1 error, 0 warnings)',
                        'ERROR SUMMARY: 1 error'):
            with self.subTest(summary=summary):
                result = self.invoke(action, extra_env={'FAKE_SUMMARY': summary})
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(self.status(action),
                                 [('racecheck', 86), ('expected_exit', 0), ('diagnostic', 0), ('summary', 1)])


if __name__ == '__main__':
    unittest.main()
