"""CPU-only tests for fixed action logs and real failure propagation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts import command_support as support


class RecordingTests(unittest.TestCase):
    def test_repeat_replaces_only_selected_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "runs" / "old.log"
            old.parent.mkdir()
            old.write_text("historical evidence")
            other = root / "commands" / "check" / "log" / "old.log"
            other.parent.mkdir(parents=True)
            other.write_text("other action")
            with patch.object(support, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                for code in (0, 7):
                    record = support.Record("run")
                    with patch.object(support.subprocess, "run", return_value=SimpleNamespace(returncode=code)):
                        self.assertEqual(record.call("sample", ["fake-python", "sample.py"]), code == 0)
                    self.assertEqual(record.finish(), int(code != 0))
                manifest = json.loads((record.path / "manifest.json").read_text())
                self.assertEqual(len(manifest["commands"]), 1)
                self.assertEqual(manifest["commands"][0]["exit_code"], 7)
                self.assertFalse((record.path / "source").exists())
            self.assertEqual(old.read_text(), "historical evidence")
            self.assertEqual(other.read_text(), "other action")

    def test_missing_command_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(support, "ROOT", Path(tmp)), \
             contextlib.redirect_stdout(io.StringIO()):
            record = support.Record("check")
            with patch.object(support.subprocess, "run", side_effect=FileNotFoundError("missing environment; run setup first")):
                self.assertFalse(record.call("imports", ["missing-python"]))
            self.assertEqual(record.finish(), 1)
            self.assertEqual(record.commands[0]["exit_code"], 127)


if __name__ == "__main__":
    unittest.main()
