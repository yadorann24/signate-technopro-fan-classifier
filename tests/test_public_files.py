import importlib.util
from pathlib import Path
import tempfile
import unittest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_public_files.py"
_SPEC = importlib.util.spec_from_file_location("publication_check", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class PublicFilesTests(unittest.TestCase):
    def test_rejects_data_even_when_it_is_ignored_by_git(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("Only documentation.")
            (root / "data").mkdir()
            (root / "data" / "private.csv").write_text("synthetic data")
            issues = _MODULE.inspect_public_files(root)
            self.assertEqual(len(issues), 1)
            self.assertIn("private.csv", issues[0])

    def test_reports_possible_secret_without_echoing_its_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic_token = "ghp_" + "X" * 36
            (root / "README.md").write_text(synthetic_token)
            issues = _MODULE.inspect_public_files(root)
            self.assertEqual(len(issues), 1)
            self.assertNotIn(synthetic_token, issues[0])


if __name__ == "__main__":
    unittest.main()
