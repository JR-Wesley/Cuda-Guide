"""CPU-only tests of independent commands. Fake tools never invoke CUDA/GPU tools."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent
BASE = TOOLS.parent
TOPICS = sorted(p.parent.parent for p in BASE.glob("*/*/commands/compile"))
SPEC = importlib.util.spec_from_file_location("lab_state", TOOLS / "lab_state.py")
state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(state)

FAKE = r'''
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
control = pathlib.Path(os.environ['FAKE_CONTROL'])
with (control / 'events').open('a') as f:
    f.write(json.dumps([name, args]) + '\n')
print('SIMULATED ONLY; NO GPU VALIDATION', name, flush=True)
if args == ['--version']:
    sys.exit(0)
if name in ('nvcc', 'ptxas'):
    output = pathlib.Path(args[args.index('-o') + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = (control / 'mode').read_text()
    if mode == 'missing': sys.exit(0)
    if mode == 'fail':
        output.write_text('partial')
        sys.exit(9)
    if name == 'nvcc' and any(a.startswith('-DPROBE_TYPE=') for a in args):
        typ = next(a.split('=',1)[1] for a in args if a.startswith('-DPROBE_TYPE='))
        op = int(next(a.split('=',1)[1] for a in args if a.startswith('-DPROBE_OP=')))
        if typ != 'unsigned' and op != 0 and mode != 'accept-invalid':
            print('unrelated tool failure' if mode == 'generic' else
                  'no matching function ' + ('plain','__ldg','__ldca','__stwb')[op])
            sys.exit(1)
    if name == 'ptxas' and 'probe_valid_' not in str(output):
        if mode != 'accept-invalid':
            print('unrelated tool failure' if mode == 'generic' else
                  'range between 24 and 256; multiple aligned 8; requires sm_90a; C7406 larger than')
            sys.exit(1)
    output.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("SIMULATED kernel",sys.argv[1:])\nsys.exit(int(os.environ.get("FAKE_KERNEL_EXIT","0")))\n')
    output.chmod(0o755)
    sys.exit(0)
if name == 'compute-sanitizer':
    i = args.index('--error-exitcode') + 2
    sys.exit(subprocess.run(args[i:]).returncode)
if name == 'cuda-gdb':
    print('width_debug_kernel; PASS checked_fields=0; exited normally')
    sys.exit(0)
if name == 'nvidia-smi': sys.exit(0)
if name == 'nsys':
    if args[0] != 'profile': sys.exit('Unexpected fake nsys arguments')
    if (control / 'mode').read_text() == 'profile-fail': sys.exit(9)
    import sqlite3
    prefix = pathlib.Path(args[args.index('-o') + 1])
    prefix.with_suffix('.nsys-rep').write_text('SIMULATED report; NO GPU VALIDATION')
    database = prefix.with_suffix('.sqlite')
    database.unlink(missing_ok=True)
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, streamId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER);
        """)
        for i, case in enumerate(('pinned_one_stream', 'pinned_three_streams', 'pageable_three_streams')):
            start = i * 1000
            db.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,NULL)', (start, start + 900, case))
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1)', (start + 300, start + 600))
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?)',
                           [(start + 100, start + 200), (start + 700, start + 800)])
    sys.exit(0)
if name == 'cuobjdump':
    print('deliberately invalid static parser fixture')
    sys.exit(0)
sys.exit('Unexpected simulated tool ' + name)
'''


class CommandTests(unittest.TestCase):
    def setUp(self):
        if not shutil.which('bash') or not shutil.which('flock'):
            self.skipTest('Bash and flock are required')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lab = self.root / 'lab'
        self.lab.mkdir()
        shutil.copytree(TOOLS, self.lab / 'tools', ignore=shutil.ignore_patterns('__pycache__', 'migrate*'))
        for topic in TOPICS:
            shutil.copytree(topic, self.lab / topic.relative_to(BASE),
                            ignore=shutil.ignore_patterns('build', 'results', 'runs', '*.log', '__pycache__'))
        self.control = self.root / 'control'
        self.control.mkdir()
        (self.control / 'mode').write_text('ok')
        (self.control / 'events').write_text('')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name in ('nvcc', 'ptxas', 'nvidia-smi', 'compute-sanitizer', 'cuda-gdb', 'cuobjdump', 'nsys'):
            tool = self.bin / name
            tool.write_text('#!' + sys.executable + '\n' + FAKE)
            tool.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        FAKE_CONTROL=str(self.control), PYTHONDONTWRITEBYTECODE='1')
        for var in ('NVCC', 'PTXAS', 'CUOBJDUMP', 'COMPUTE_SANITIZER', 'CUDA_GDB', 'NSYS',
                    'BUILD_MODE', 'TARGET', 'CUDA_ARCH'):
            self.env.pop(var, None)

    def topic(self, short):
        return next(self.lab / p.relative_to(BASE) for p in TOPICS if p.name == short)

    def call(self, short, action, expected=0, **env):
        topic = self.topic(short)
        result = subprocess.run(['bash', str(topic / 'commands' / action / 'run.sh')],
                                cwd=self.root, env=dict(self.env, **env),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=20)
        if expected is not None:
            self.assertEqual(result.returncode, expected, result.stdout)
        return result

    def compiles(self):
        import json
        return [row for row in map(json.loads, (self.control / 'events').read_text().splitlines())
                if row[0] in ('nvcc', 'ptxas') and '-o' in row[1]]

    def test_all_nine_compile_and_explicit_recompile(self):
        self.assertEqual(len(TOPICS), 9)
        for topic in TOPICS:
            with self.subTest(topic=topic.name):
                before = len(self.compiles())
                self.call(topic.name, 'compile')
                count = len(self.compiles()) - before
                self.assertGreater(count, 0)
                self.call(topic.name, 'compile')
                self.assertEqual(len(self.compiles()) - before, count * 2)
                self.assertFalse(list(self.topic(topic.name).rglob('.build-cache.json')))
                self.assertFalse(list(self.topic(topic.name).rglob('.state.json')))
                self.assertFalse(list(self.topic(topic.name).glob('commands/*/.steps-*')))

    def test_all_execution_actions_require_explicit_compile(self):
        for topic in TOPICS:
            for path in (topic / 'commands').glob('*/run.sh'):
                action = path.parent.name
                if action == 'compile' or action == 'types' or (topic.name == '01_setmaxnreg' and action == 'debug'):
                    continue
                with self.subTest(topic=topic.name, action=action):
                    result = self.call(topic.name, action, expected=2)
                    self.assertIn('commands/compile/run.sh first', result.stdout)
        self.assertEqual(self.compiles(), [])

    def test_release_reuse_and_separate_gdb_build(self):
        t = '04_warp_functions'
        self.call(t, 'compile', CUDA_ARCH='sm_89')
        count = len(self.compiles())
        for action in ('run', 'sanitizer', 'jit', 'debug'):
            self.call(t, action, CUDA_ARCH='sm_89')
        self.assertEqual(len(self.compiles()), count)
        self.call(t, 'gdb', expected=2, CUDA_ARCH='sm_89')
        self.call(t, 'compile', CUDA_ARCH='sm_89', BUILD_MODE='debug')
        self.assertEqual(len(self.compiles()), count + 1)
        self.call(t, 'gdb', CUDA_ARCH='sm_89')
        self.assertEqual(len(self.compiles()), count + 1)
        release = self.topic(t) / 'build/sm_89-release/bin/warp_functions'
        debug = self.topic(t) / 'build/sm_89-debug/bin/warp_functions'
        self.assertTrue(release.is_file() and debug.is_file())

    def test_compile_failure_blocks_old_binary(self):
        t = '04_warp_functions'
        self.call(t, 'compile')
        (self.control / 'mode').write_text('fail')
        self.call(t, 'compile', expected=9)
        self.call(t, 'run', expected=2)
        self.assertFalse((self.topic(t) / 'build/sm_80-release/bin/warp_functions').exists())
        self.assertFalse((self.topic(t) / 'build/sm_80-release/.main.compiled').exists())

    def test_missing_output_fails_compile(self):
        (self.control / 'mode').write_text('missing')
        self.call('04_warp_functions', 'compile', expected=1)
        self.call('04_warp_functions', 'run', expected=2)

    def test_missing_artifact_does_not_rebuild(self):
        t = '04_warp_functions'
        self.call(t, 'compile')
        (self.topic(t) / 'build/sm_80-release/main.ptx').unlink()
        count = len(self.compiles())
        self.call(t, 'static', expected=2)
        self.assertEqual(len(self.compiles()), count)

    def test_logs_replace_only_current_action_and_leave_history(self):
        t = '04_warp_functions'
        topic = self.topic(t)
        old = topic / 'results'
        old.mkdir()
        (old / 'sm_80-release.log').write_text('historical evidence')
        self.call(t, 'compile')
        compile_log = (topic / 'commands/compile/sm_80-release.log').read_text()
        self.call(t, 'run')
        self.call(t, 'run', expected=7, FAKE_KERNEL_EXIT='7')
        log = (topic / 'commands/run/sm_80-release.log').read_text()
        self.assertEqual(log.count('action_exit_code='), 1)
        self.assertIn('action_exit_code=7', log)
        self.assertEqual((topic / 'commands/compile/sm_80-release.log').read_text(), compile_log)
        self.assertEqual((old / 'sm_80-release.log').read_text(), 'historical evidence')
        self.assertFalse(list(topic.rglob('history')))

    def test_static_parser_failure_is_not_hidden_or_recompiled(self):
        t = '04_warp_functions'
        self.call(t, 'compile')
        count = len(self.compiles())
        result = self.call(t, 'static', expected=None)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('static_check', result.stdout)
        self.assertEqual(len(self.compiles()), count)

    def test_special_target_compilation_is_explicit(self):
        for t, target, action in [('02_async_barriers','tx90','tx90'),
                                  ('03_thread_scopes','cluster','cluster')]:
            with self.subTest(target=target):
                self.call(t, 'compile', TARGET=target)
                count = len(self.compiles())
                self.call(t, action)
                self.assertEqual(len(self.compiles()), count)
                self.call(t, 'run', expected=2, CUDA_ARCH='sm_90')

    def test_baseline_and_dynamic_arch_checks(self):
        t = '01_setmaxnreg'
        self.call(t, 'compile', BUILD_MODE='baseline', CUDA_ARCH='sm_89')
        count = len(self.compiles())
        self.call(t, 'baseline', CUDA_ARCH='sm_89')
        self.call(t, 'compile', expected=2, CUDA_ARCH='sm_90')
        self.assertEqual(len(self.compiles()), count)

    def test_type_probes_all_rerun(self):
        t = '03_load_store_intrinsics'
        for expected in (24, 48):
            self.call(t, 'types')
            self.assertEqual(len(self.compiles()), expected)

    def test_type_probes_reject_generic_failure_and_unexpected_success(self):
        for mode in ('generic', 'accept-invalid'):
            (self.control / 'mode').write_text(mode)
            self.call('03_load_store_intrinsics', 'types', expected=1)

    def test_instruction_probes_all_rerun(self):
        for expected in (8, 16):
            self.call('01_setmaxnreg', 'debug')
            self.assertEqual(len(self.compiles()), expected)

    def test_instruction_probes_reject_wrong_diagnostics(self):
        for mode in ('generic', 'accept-invalid'):
            (self.control / 'mode').write_text(mode)
            self.call('01_setmaxnreg', 'debug', expected=1)

    def test_profile_reports_are_local_and_preserve_historical_reports(self):
        t = '01_async_execution'
        self.call(t, 'compile', CUDA_ARCH='sm_89')
        topic = self.topic(t)
        build = topic / 'build/sm_89-release'
        historical = [build / (name + ext) for name in ('timeline', 'current_timeline')
                      for ext in ('.sqlite', '.nsys-rep')]
        for path in historical:
            path.write_bytes(b'historical report fixture')
        profile = topic / 'commands/profile'
        for arch in ('sm_89', 'sm_80'):
            if arch == 'sm_80':
                self.call(t, 'compile', CUDA_ARCH=arch)
            count = len(self.compiles())
            result = self.call(t, 'profile', CUDA_ARCH=arch)
            self.assertIn('PASS complete case ranges', result.stdout)
            self.assertEqual(len(self.compiles()), count)
            self.assertTrue((profile / f'{arch}-release-timeline.sqlite').is_file())
            self.assertTrue((profile / f'{arch}-release-timeline.nsys-rep').is_file())
            self.assertTrue((profile / f'{arch}-release.log').is_file())
            self.assertFalse(list((topic / 'build' / f'{arch}-release').glob('*-timeline.*')))
        other_variant = profile / 'sm_80-release-timeline.nsys-rep'
        other_variant.write_text('other variant fixture')
        current = profile / 'sm_89-release-timeline.nsys-rep'
        current.write_text('previous current report fixture')
        self.call(t, 'profile', CUDA_ARCH='sm_89')
        self.assertIn('SIMULATED report', current.read_text())
        self.assertEqual(other_variant.read_text(), 'other variant fixture')
        for path in historical:
            self.assertEqual(path.read_bytes(), b'historical report fixture')

    def test_failed_profile_does_not_analyze_stale_report(self):
        t = '01_async_execution'
        self.call(t, 'compile', CUDA_ARCH='sm_89')
        self.call(t, 'profile', CUDA_ARCH='sm_89')
        count = len(self.compiles())
        (self.control / 'mode').write_text('profile-fail')
        result = self.call(t, 'profile', expected=9, CUDA_ARCH='sm_89')
        self.assertNotIn('[trace_analysis]', result.stdout)
        self.assertNotIn('PASS complete case ranges', result.stdout)
        self.assertEqual(len(self.compiles()), count)
        log = self.topic(t) / 'commands/profile/sm_89-release.log'
        self.assertIn('action_exit_code=9', log.read_text())

    def test_topic_lock_preserves_owner_log(self):
        t = '04_warp_functions'
        self.call(t, 'compile')
        topic = self.topic(t)
        log = topic / 'commands/compile/sm_80-release.log'
        before = log.read_text()
        import fcntl
        with (topic / 'build/.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.call(t, 'compile', expected=2)
        self.assertEqual(log.read_text(), before)


class LayoutTests(unittest.TestCase):
    def test_no_dispatchers_or_hash_tracking(self):
        self.assertEqual(len(TOPICS), 9)
        self.assertFalse(list(BASE.glob('*/*/lab.sh')))
        scripts = list(BASE.glob('*/*/commands/*/run.sh'))
        self.assertEqual(len(scripts), 51)
        for path in scripts:
            text = path.read_text()
            self.assertNotIn('case "$action"', text)
            self.assertNotIn('sha256', text)
            self.assertNotIn('lab.sh', text)
            self.assertEqual(subprocess.run(['bash', '-n', str(path)]).returncode, 0)

    def test_missing_compiler_removes_old_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'old'
            out.write_text('stale')
            result = state.compile_command([str(Path(temp) / 'absent-nvcc'), '-o', str(out)])
            self.assertEqual(result.returncode, 127)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
