import tempfile

import pyarrow.parquet as pq
import pytest
from freezegun import freeze_time

from src.data.gh_archive.extract.core import RawGHArchiveHourExtractor


def test_extract_hour_creates_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        extractor = RawGHArchiveHourExtractor(output_dir=tmp_dir)
        hour = "2025-01-01-00"
        dt_hour = extractor.str_hour_to_dt(hour)
        result_path = extractor.extract_hour(hour)
        expected_path = extractor.hour_local_path(dt_hour)
        assert expected_path.exists()
        assert result_path == expected_path
        assert expected_path.name == "events.parquet"


def test_table_schema():
    with tempfile.TemporaryDirectory() as tmp_dir:
        extractor = RawGHArchiveHourExtractor(output_dir=tmp_dir)
        hour = "2025-01-01-00"
        result_path = extractor.extract_hour(hour)
        table = pq.read_table(result_path)
        assert table.column_names == [
            "event_id",
            "created_at",
            "event_type",
            "repo_id",
            "repo_name",
            "actor_id",
            "actor_login",
            "action",
            "push_distinct_size",
            "year",
            "month",
            "day",
            "hour",
        ]


def test_generate_str_hours():
    start_hour = "2025-01-01-00"
    end_hour = "2025-01-01-02"
    result = RawGHArchiveHourExtractor.generate_str_hours(start_hour, end_hour)
    assert result == ["2025-01-01-00", "2025-01-01-01", "2025-01-01-02"]


@freeze_time("2025-01-01-03")
@pytest.mark.parametrize("previous_hours", [2, 3])
def test_generate_previous_str_hours(previous_hours):
    result = RawGHArchiveHourExtractor.generate_previous_str_hours(previous_hours)
    assert len(result) == previous_hours
    assert result == [f"2025-01-01-{i:02d}" for i in range(3 - previous_hours, 3)]
