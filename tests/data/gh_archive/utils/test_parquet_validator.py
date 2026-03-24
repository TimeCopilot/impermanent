import datetime as dt

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.gh_archive.utils.parquet_validator import (
    FileStatus,
    FileValidationResult,
    ParquetValidator,
    ValidationReport,
)


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_all_ok_when_no_problems(self):
        report = ValidationReport(
            results=[
                FileValidationResult(
                    path="a.parquet", status=FileStatus.OK, num_rows=10
                ),
                FileValidationResult(
                    path="b.parquet", status=FileStatus.OK, num_rows=5
                ),
            ],
        )
        assert report.all_ok is True
        assert report.total_files == 2
        assert report.ok_files == 2

    def test_not_ok_with_empty_files(self):
        report = ValidationReport(
            results=[
                FileValidationResult(
                    path="a.parquet", status=FileStatus.OK, num_rows=10
                ),
                FileValidationResult(path="b.parquet", status=FileStatus.EMPTY),
            ],
        )
        assert report.all_ok is False
        assert report.empty_files == 1

    def test_not_ok_with_missing_files(self):
        report = ValidationReport(
            results=[
                FileValidationResult(
                    path="gone.parquet",
                    status=FileStatus.MISSING,
                    error="file not found",
                ),
            ],
        )
        assert report.all_ok is False
        assert report.missing_files == 1

    def test_not_ok_with_unreadable_files(self):
        report = ValidationReport(
            results=[
                FileValidationResult(
                    path="bad.parquet",
                    status=FileStatus.UNREADABLE,
                    error="corrupt",
                ),
            ],
        )
        assert report.all_ok is False
        assert report.unreadable_files == 1

    def test_summary_lists_problem_files(self):
        report = ValidationReport(
            results=[
                FileValidationResult(
                    path="ok.parquet", status=FileStatus.OK, num_rows=10
                ),
                FileValidationResult(path="empty.parquet", status=FileStatus.EMPTY),
                FileValidationResult(
                    path="bad.parquet",
                    status=FileStatus.UNREADABLE,
                    error="corrupt header",
                ),
                FileValidationResult(
                    path="gone.parquet",
                    status=FileStatus.MISSING,
                    error="file not found",
                ),
            ],
        )
        summary = report.summary()
        assert "4 files scanned" in summary
        assert "empty.parquet" in summary
        assert "bad.parquet" in summary
        assert "corrupt header" in summary
        assert "gone.parquet" in summary

    def test_summary_all_ok(self):
        report = ValidationReport(
            results=[
                FileValidationResult(
                    path="ok.parquet", status=FileStatus.OK, num_rows=5
                ),
            ],
        )
        assert "All files OK" in report.summary()

    def test_merge_with_add(self):
        r1 = ValidationReport(
            results=[
                FileValidationResult(
                    path="a.parquet", status=FileStatus.OK, num_rows=10
                ),
            ]
        )
        r2 = ValidationReport(
            results=[
                FileValidationResult(path="b.parquet", status=FileStatus.EMPTY),
            ]
        )
        merged = r1 + r2
        assert merged.total_files == 2
        assert merged.ok_files == 1
        assert merged.empty_files == 1

    def test_merge_with_iadd(self):
        report = ValidationReport(
            results=[
                FileValidationResult(
                    path="a.parquet", status=FileStatus.OK, num_rows=3
                ),
            ]
        )
        report += ValidationReport(
            results=[
                FileValidationResult(
                    path="b.parquet",
                    status=FileStatus.MISSING,
                    error="file not found",
                ),
            ]
        )
        assert report.total_files == 2
        assert report.ok_files == 1
        assert report.missing_files == 1

    def test_empty_report_is_all_ok(self):
        report = ValidationReport()
        assert report.all_ok is True
        assert report.total_files == 0


class TestParquetValidator:
    """Tests for the standalone ParquetValidator class."""

    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Create a single valid parquet file with rows."""
        schema = pa.schema(
            [
                ("unique_id", pa.string()),
                ("ds", pa.timestamp("us")),
                ("y", pa.int64()),
            ]
        )
        table = pa.table(
            {
                "unique_id": ["repo:1", "repo:2"],
                "ds": [
                    dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
                ],
                "y": [10, 20],
            },
            schema=schema,
        )
        path = tmp_path / "valid.parquet"
        pq.write_table(table, str(path))
        return path

    @pytest.fixture
    def empty_parquet(self, tmp_path):
        """Create a valid parquet file with 0 rows."""
        schema = pa.schema(
            [
                ("unique_id", pa.string()),
                ("ds", pa.timestamp("us")),
                ("y", pa.int64()),
            ]
        )
        table = pa.table({"unique_id": [], "ds": [], "y": []}, schema=schema)
        path = tmp_path / "empty.parquet"
        pq.write_table(table, str(path))
        return path

    @pytest.fixture
    def corrupt_parquet(self, tmp_path):
        """Create a file with garbage bytes."""
        path = tmp_path / "corrupt.parquet"
        path.write_bytes(b"not a parquet file at all")
        return path

    def test_validate_ok_file(self, sample_parquet):
        v = ParquetValidator()
        result = v.validate_file(sample_parquet)
        assert result.status == FileStatus.OK
        assert result.num_rows == 2
        assert result.error is None

    def test_validate_empty_file(self, empty_parquet):
        v = ParquetValidator()
        result = v.validate_file(empty_parquet)
        assert result.status == FileStatus.EMPTY
        assert result.num_rows == 0

    def test_validate_missing_file(self, tmp_path):
        v = ParquetValidator()
        result = v.validate_file(tmp_path / "does_not_exist.parquet")
        assert result.status == FileStatus.MISSING
        assert result.error is not None and "file not found" in result.error

    def test_validate_corrupt_file(self, corrupt_parquet):
        v = ParquetValidator()
        result = v.validate_file(corrupt_parquet)
        assert result.status == FileStatus.UNREADABLE
        assert result.error is not None

    def test_validate_files_batch(
        self, sample_parquet, empty_parquet, corrupt_parquet, tmp_path
    ):
        missing = tmp_path / "gone.parquet"
        v = ParquetValidator()
        report = v.validate_files(
            [sample_parquet, empty_parquet, corrupt_parquet, missing]
        )
        assert report.total_files == 4
        assert report.ok_files == 1
        assert report.empty_files == 1
        assert report.unreadable_files == 1
        assert report.missing_files == 1
        assert report.all_ok is False

    def test_validate_files_all_ok(self, sample_parquet, tmp_path):
        # Create a second valid file
        schema = pa.schema([("x", pa.int64())])
        other = tmp_path / "other.parquet"
        pq.write_table(pa.table({"x": [1]}, schema=schema), str(other))

        v = ParquetValidator()
        report = v.validate_files([sample_parquet, other])
        assert report.all_ok is True
        assert report.total_files == 2
