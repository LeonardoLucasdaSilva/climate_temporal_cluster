"""Tests for project configuration helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

from config import (
    PROJECT_ROOT,
    dated_output_root,
    output_date_folder,
    output_root_from_config,
)


class ConfigTests(unittest.TestCase):
    def test_output_date_folder_uses_day_month_two_digit_year(self) -> None:
        self.assertEqual(output_date_folder(date(2025, 7, 14)), "14_07_25")

    def test_dated_output_root_appends_daily_folder_once(self) -> None:
        root = Path("outputs")
        dated_root = dated_output_root(root, run_date=date(2025, 7, 14))

        self.assertEqual(dated_root, Path("outputs") / "14_07_25")
        self.assertEqual(
            dated_output_root(dated_root, run_date=date(2025, 7, 14)),
            dated_root,
        )

    def test_output_root_from_config_groups_relative_root_by_day(self) -> None:
        output_root = output_root_from_config({"output_root": "outputs"})

        self.assertEqual(output_root.parent, PROJECT_ROOT / "outputs")
        self.assertRegex(output_root.name, r"^\d{2}_\d{2}_\d{2}$")

    def test_output_root_from_config_can_disable_daily_grouping(self) -> None:
        self.assertEqual(
            output_root_from_config(
                {"output_root": "outputs", "group_outputs_by_day": False}
            ),
            PROJECT_ROOT / "outputs",
        )


if __name__ == "__main__":
    unittest.main()
