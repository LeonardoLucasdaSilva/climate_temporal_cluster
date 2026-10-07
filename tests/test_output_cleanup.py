from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data.output_cleanup import remove_empty_directories


class OutputCleanupTests(unittest.TestCase):
    def test_removes_empty_directories_bottom_up_and_preserves_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            root = Path(temporary_directory)
            (root / "empty" / "nested").mkdir(parents=True)
            populated = root / "populated"
            populated.mkdir()
            (populated / "result.txt").write_text("result", encoding="utf-8")

            removed = remove_empty_directories(root)

            self.assertEqual(removed, 2)
            self.assertTrue(root.is_dir())
            self.assertFalse((root / "empty").exists())
            self.assertTrue(populated.is_dir())

    def test_missing_root_is_a_no_op(self) -> None:
        missing = Path("missing-output-cleanup-test-directory")

        self.assertEqual(remove_empty_directories(missing), 0)
