import datetime as dt
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.gh_archive.utils import GHArchiveData
from src.data.gh_archive.utils.parquet_validator import FileStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DAILY_SCHEMA = pa.schema(
    [
        ("unique_id", pa.string()),
        ("ds", pa.timestamp("us")),
        ("y", pa.int64()),
    ]
)


@pytest.fixture()
def daily_base(tmp_path):
    """Create 14 days of daily parquet data (2026-01-01 … 2026-01-14).

    Two series: ``repo_a:stars`` and ``repo_b:stars``.
    """
    base = tmp_path / "processed"
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    for i in range(14):
        day = start + dt.timedelta(days=i)
        day_dir = (
            base
            / "daily"
            / f"year={day.year:04d}"
            / f"month={day.month:02d}"
            / f"day={day.day:02d}"
        )
        day_dir.mkdir(parents=True)

        table = pa.table(
            {
                "unique_id": ["repo_a:stars", "repo_b:stars"],
                "ds": [day.replace(tzinfo=None), day.replace(tzinfo=None)],
                "y": [i * 2, i * 3],
            },
            schema=DAILY_SCHEMA,
        )
        pq.write_table(table, str(day_dir / "series.parquet"))

    return base


@pytest.fixture()
def daily_data(daily_base):
    """Return a ``GHArchiveData`` instance pointing at the local fixture."""
    return GHArchiveData("daily", base_path=str(daily_base))


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestGHArchiveDataInit:
    """Tests for GHArchiveData initialization."""

    @pytest.mark.parametrize(
        "frequency,expected_h,expected_freq",
        [
            ("hourly", 24, "h"),
            ("daily", 7, "D"),
            ("weekly", 1, "W-SUN"),
            ("monthly", 1, "MS"),
        ],
    )
    def test_valid_frequencies(self, frequency, expected_h, expected_freq):
        with patch("boto3.client"):
            data = GHArchiveData(frequency)

        assert data.frequency == frequency
        assert data.h == expected_h
        assert data.freq == expected_freq

    @pytest.mark.parametrize(
        "invalid_frequency",
        ["invalid", "DAILY", "Weekly", "hour", "day", "week", "month", ""],
    )
    def test_invalid_frequencies(self, invalid_frequency):
        with pytest.raises(ValueError) as exc_info:
            GHArchiveData(invalid_frequency)

        assert f"Invalid frequency '{invalid_frequency}'" in str(exc_info.value)

    def test_init_with_base_path_sets_local_mode(self):
        data = GHArchiveData("daily", base_path="/test/path")
        assert data._use_s3 is False
        assert data._base_path is not None
        assert str(data._base_path) == "/test/path/daily"

    def test_init_without_base_path_sets_s3_mode(self):
        with patch("boto3.client"):
            data = GHArchiveData("daily")
        assert data._use_s3 is True
        assert data._base_path is None


# ---------------------------------------------------------------------------
# _parse_cutoff
# ---------------------------------------------------------------------------


class TestParseCutoff:
    """Tests for cutoff date parsing."""

    @pytest.fixture()
    def data(self):
        with patch("boto3.client"):
            return GHArchiveData("daily")

    @pytest.mark.parametrize(
        "cutoff_str,expected",
        [
            ("2026-01-15-12", dt.datetime(2026, 1, 15, 12, tzinfo=dt.timezone.utc)),
            ("2026-01-15", dt.datetime(2026, 1, 15, 0, tzinfo=dt.timezone.utc)),
            (
                "2026-01-15 14:30:00",
                dt.datetime(2026, 1, 15, 14, 30, 0, tzinfo=dt.timezone.utc),
            ),
        ],
    )
    def test_string_formats(self, data, cutoff_str, expected):
        assert data._parse_cutoff(cutoff_str) == expected

    def test_naive_datetime_gets_utc(self, data):
        naive = dt.datetime(2026, 1, 15, 12)
        result = data._parse_cutoff(naive)
        assert result.tzinfo == dt.timezone.utc

    def test_aware_datetime_preserved(self, data):
        aware = dt.datetime(2026, 1, 15, 12, tzinfo=dt.timezone.utc)
        assert data._parse_cutoff(aware) == aware

    def test_invalid_string_raises(self, data):
        with pytest.raises(ValueError, match="Could not parse cutoff"):
            data._parse_cutoff("invalid-date")


# ---------------------------------------------------------------------------
# get_df
# ---------------------------------------------------------------------------


class TestGetDf:
    """Tests for get_df with local fixtures."""

    def test_returns_correct_columns(self, daily_data):
        df = daily_data.get_df(cutoff="2026-01-10", max_window_size=7)
        assert list(df.columns) == ["unique_id", "ds", "y"]

    def test_returns_expected_rows(self, daily_data):
        df = daily_data.get_df(cutoff="2026-01-10", max_window_size=7)
        # 2 series × 7 days = 14 rows
        assert len(df) == 14

    def test_date_range_respects_cutoff(self, daily_data):
        df = daily_data.get_df(cutoff="2026-01-10", max_window_size=7)
        assert df["ds"].max() <= pd.Timestamp("2026-01-10")

    def test_date_range_spans_window(self, daily_data):
        df = daily_data.get_df(cutoff="2026-01-10", max_window_size=7)
        assert df["ds"].min() >= pd.Timestamp("2026-01-04")

    def test_smaller_window(self, daily_data):
        df = daily_data.get_df(cutoff="2026-01-14", max_window_size=3)
        assert len(df) == 6  # 2 series × 3 days

    def test_no_files_raises(self, daily_data):
        with pytest.raises(FileNotFoundError):
            daily_data.get_df(cutoff="2025-06-01", max_window_size=7)

    def test_sorted_by_unique_id_and_ds(self, daily_data):
        df = daily_data.get_df(cutoff="2026-01-14", max_window_size=5)
        assert df.equals(df.sort_values(["unique_id", "ds"]).reset_index(drop=True))


# ---------------------------------------------------------------------------
# get_actuals
# ---------------------------------------------------------------------------


class TestGetActuals:
    """Tests for get_actuals with local fixtures."""

    def test_returns_periods_after_cutoff(self, daily_data):
        # cutoff=Jan 5 → actuals for Jan 6..Jan 12 (h=7 for daily)
        df = daily_data.get_actuals(cutoff="2026-01-05")
        assert df["ds"].min() > pd.Timestamp("2026-01-05")

    def test_returns_h_periods(self, daily_data):
        df = daily_data.get_actuals(cutoff="2026-01-05")
        per_series = df.groupby("unique_id").size()
        assert (per_series == daily_data.h).all()

    def test_correct_columns(self, daily_data):
        df = daily_data.get_actuals(cutoff="2026-01-05")
        assert list(df.columns) == ["unique_id", "ds", "y"]

    def test_no_files_raises(self, daily_data):
        # All data ends at 2026-01-14; requesting actuals after that
        with pytest.raises(FileNotFoundError):
            daily_data.get_actuals(cutoff="2026-02-01")


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


class TestAttributes:
    """Tests for class-level attributes."""

    def test_freq_config_has_all_frequencies(self):
        assert set(GHArchiveData.FREQ_CONFIG.keys()) == {
            "hourly",
            "daily",
            "weekly",
            "monthly",
        }

    def test_freq_config_has_required_keys(self):
        for freq, config in GHArchiveData.FREQ_CONFIG.items():
            assert "h" in config, f"Missing 'h' for {freq}"
            assert "freq" in config, f"Missing 'freq' for {freq}"
            assert isinstance(config["h"], int)
            assert isinstance(config["freq"], str)


# ---------------------------------------------------------------------------
# validate_files
# ---------------------------------------------------------------------------


class TestValidateFiles:
    """Tests for validate_files (local only)."""

    @pytest.fixture()
    def validation_base(self, tmp_path):
        """Three daily files: 2 OK, 1 empty."""
        base = tmp_path / "processed"

        for day, num_rows in [(1, 5), (2, 3), (3, 0)]:
            day_dir = base / "daily" / "year=2026" / "month=01" / f"day={day:02d}"
            day_dir.mkdir(parents=True)

            if num_rows > 0:
                table = pa.table(
                    {
                        "unique_id": [f"repo:{i}" for i in range(num_rows)],
                        "ds": [dt.datetime(2026, 1, day)] * num_rows,
                        "y": list(range(num_rows)),
                    },
                    schema=DAILY_SCHEMA,
                )
            else:
                table = pa.table(
                    {"unique_id": [], "ds": [], "y": []}, schema=DAILY_SCHEMA
                )

            pq.write_table(table, str(day_dir / "series.parquet"))
        return base

    def test_all_files_found(self, validation_base):
        data = GHArchiveData("daily", base_path=str(validation_base))
        report = data.validate_files("2026-01-01", "2026-01-03")
        assert report.total_files == 3

    def test_detects_empty_file(self, validation_base):
        data = GHArchiveData("daily", base_path=str(validation_base))
        report = data.validate_files("2026-01-01", "2026-01-03")
        assert report.empty_files == 1
        assert report.ok_files == 2
        assert report.all_ok is False

        empty_paths = [r.path for r in report.results if r.status == FileStatus.EMPTY]
        assert len(empty_paths) == 1
        assert "day=03" in empty_paths[0]

    def test_detects_corrupt_file(self, validation_base):
        corrupt = (
            validation_base
            / "daily"
            / "year=2026"
            / "month=01"
            / "day=02"
            / "series.parquet"
        )
        corrupt.write_bytes(b"not a parquet file")

        data = GHArchiveData("daily", base_path=str(validation_base))
        report = data.validate_files("2026-01-01", "2026-01-03")
        assert report.unreadable_files == 1

    def test_missing_files_detected(self, validation_base):
        """Files that should exist but don't are reported as MISSING."""
        data = GHArchiveData("daily", base_path=str(validation_base))
        # Jan 4 and 5 have no files on disk
        report = data.validate_files("2026-01-01", "2026-01-05")
        assert report.missing_files == 2
        assert report.total_files == 5
