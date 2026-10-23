"""Checks of the packaging boundary; no API, data or ML training needed."""

import importlib.util
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "build_submission", Path(__file__).resolve().parents[1] / "tools" / "build_submission.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
ENV_SPEC = importlib.util.spec_from_file_location(
    "check_environment", Path(__file__).resolve().parents[1] / "tools" / "check_environment.py"
)
environment = importlib.util.module_from_spec(ENV_SPEC)
ENV_SPEC.loader.exec_module(environment)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "reference").mkdir()
        self.requirements = self.root / "reference" / "requirements.txt"
        self.requirements.write_bytes(b"numpy==1.26.4\n")
        (self.root / "solution.py").write_text("import helper\n", encoding="utf-8")
        (self.root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        self.archive = self.root / "artifacts" / "solution.zip"

    def test_round_trip_only_allowed_files_and_exact_requirements(self):
        (self.root / ".env").write_text("not a real credential", encoding="utf-8")
        (self.root / "train.csv").write_text("ignored", encoding="utf-8")
        names = release.build_archive(self.root, self.archive)
        self.assertEqual(set(names), {"solution.py", "helper.py", "requirements.txt"})
        first = self.archive.read_bytes()
        release.build_archive(self.root, self.archive)
        self.assertEqual(first, self.archive.read_bytes())

    def test_missing_agent_is_rejected(self):
        (self.root / "solution.py").unlink()
        with self.assertRaises(release.ReleaseError):
            release.build_archive(self.root, self.archive)
        self.assertFalse(self.archive.exists())

    def test_invalid_python_preserves_previous_archive(self):
        release.build_archive(self.root, self.archive)
        previous = self.archive.read_bytes()
        (self.root / "helper.py").write_text("def broken(\n", encoding="utf-8")
        with self.assertRaises(release.ReleaseError):
            release.build_archive(self.root, self.archive)
        self.assertEqual(previous, self.archive.read_bytes())

    def test_credential_rejected_without_printing_value(self):
        fake_key = "sk-" + "x" * 30
        (self.root / "helper.py").write_text(f"KEY = {fake_key!r}\n", encoding="utf-8")
        with self.assertRaises(release.ReleaseError) as caught:
            release.build_archive(self.root, self.archive)
        self.assertNotIn(fake_key, str(caught.exception))

    def test_flattened_module_collision_is_rejected(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaises(release.ReleaseError):
            release.build_archive(self.root, self.archive, [Path("src/helper.py")])

    def test_existing_archive_rejects_bad_entries_and_modified_requirements(self):
        for extra, requirement in [("folder/helper.py", b"numpy==1.26.4\n"),
                                   ("data.csv", b"numpy==1.26.4\n"),
                                   ("helper.py", b"numpy==2\n")]:
            with self.subTest(extra=extra, requirement=requirement):
                with zipfile.ZipFile(self.root / "bad.zip", "w") as package:
                    package.writestr("solution.py", "pass\n")
                    package.writestr("requirements.txt", requirement)
                    package.writestr(extra, "pass\n")
                with self.assertRaises(release.ReleaseError):
                    release.audit_archive(self.root / "bad.zip", self.requirements)

    def test_archive_size_is_checked(self):
        release.build_archive(self.root, self.archive)
        with self.assertRaises(release.ReleaseError):
            release.audit_archive(self.archive, self.requirements, max_bytes=1)

    def test_clean_extraction_imports_only_packaged_modules(self):
        release.build_archive(self.root, self.archive)
        clean = self.root / "clean"
        command = [sys.executable, "-X", "utf8", str(Path(release.__file__)), "--root", str(self.root),
                   "--check", str(self.archive), "--extract-to", str(clean)]
        checked = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        child_env = os.environ.copy()
        child_env.pop("PYTHONPATH", None)
        imported = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", "import solution; assert solution.helper.VALUE == 1"],
            cwd=clean, env=child_env, capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        # Reusing a directory must not hide stale files from an earlier release.
        repeated = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("ERROR:", repeated.stderr)

    def test_explicit_nested_module_is_flattened(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "other.py").write_text("VALUE = 2\n", encoding="utf-8")
        names = release.build_archive(self.root, self.archive, [Path("src/other.py")])
        self.assertIn("other.py", names)
        self.assertFalse(any("/" in name for name in names))


class ApiDiagnosticTests(unittest.TestCase):
    def test_provider_exception_does_not_print_sensitive_message(self):
        sensitive = "must-not-appear-in-output"
        provider = SimpleNamespace(OpenAI=mock.Mock(side_effect=ValueError(sensitive)))
        with mock.patch.dict(sys.modules, {"openai": provider}), mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": sensitive, "OPENAI_BASE_URL": "https://example.invalid/v1",
                         "OPENAI_MODEL": "provided-model"}, clear=True
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(environment.check_api())
        self.assertIn("ValueError", output.getvalue())
        self.assertNotIn(sensitive, output.getvalue())


if __name__ == "__main__":
    unittest.main()
