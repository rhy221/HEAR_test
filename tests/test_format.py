"""Verify submission.csv schema matches sample_submission.csv format."""
import ast
import csv
import os
from pathlib import Path

import pytest

EXPECTED_COLUMNS = ["id", "answer", "evidence_page_number"]


def _load_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return fieldnames, rows


def _get_csv_path(csv_path_fixture):
    if csv_path_fixture:
        return csv_path_fixture
    # Default: look for a generated submission in runs/manual/
    candidates = [
        Path(__file__).parent.parent / "runs" / "manual" / "submission.csv",
        Path(__file__).parent / "fixtures" / "sample_submission.csv",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    pytest.skip("No submission CSV found — pass --csv path/to/submission.csv")


class TestSubmissionFormat:
    def test_columns_exist(self, csv_path):
        path = _get_csv_path(csv_path)
        fieldnames, rows = _load_csv(path)
        assert fieldnames is not None, "CSV has no header"
        for col in EXPECTED_COLUMNS:
            assert col in fieldnames, f"Missing column: {col}"

    def test_columns_order(self, csv_path):
        path = _get_csv_path(csv_path)
        fieldnames, _ = _load_csv(path)
        assert list(fieldnames) == EXPECTED_COLUMNS, (
            f"Column order mismatch: got {fieldnames}, expected {EXPECTED_COLUMNS}"
        )

    def test_id_not_empty(self, csv_path):
        path = _get_csv_path(csv_path)
        _, rows = _load_csv(path)
        assert len(rows) > 0, "CSV has no data rows"
        for row in rows:
            assert row["id"].strip(), f"Empty id in row: {row}"

    def test_evidence_page_parseable(self, csv_path):
        path = _get_csv_path(csv_path)
        _, rows = _load_csv(path)
        for row in rows:
            ev = row["evidence_page_number"].strip()
            assert ev.startswith("[") and ev.endswith("]"), (
                f"evidence_page_number not a list string: {ev!r} in row id={row['id']}"
            )
            try:
                parsed = ast.literal_eval(ev)
                assert isinstance(parsed, list), f"Not a list: {ev!r}"
                for item in parsed:
                    assert isinstance(item, (int, str)), f"Invalid page item: {item!r}"
                    if isinstance(item, str):
                        assert item.strip().isdigit(), f"Non-integer page: {item!r}"
            except Exception as e:
                pytest.fail(f"Cannot parse evidence_page_number={ev!r}: {e}")

    def test_no_duplicate_ids(self, csv_path):
        path = _get_csv_path(csv_path)
        _, rows = _load_csv(path)
        ids = [row["id"] for row in rows]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {[i for i in ids if ids.count(i) > 1]}"

    def test_answer_not_null(self, csv_path):
        path = _get_csv_path(csv_path)
        _, rows = _load_csv(path)
        for row in rows:
            assert row["answer"] is not None, f"Null answer for id={row['id']}"
