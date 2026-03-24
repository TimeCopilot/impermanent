from pathlib import Path

import pytest

from src.data.gh_archive.aggregate.core import PeriodAggregator
from src.data.gh_archive.extract.core import RawGHArchiveHourExtractor


@pytest.fixture
def tmp_dirs(tmp_path: Path):
    """Create temporary directories for testing."""
    hourly_dir = tmp_path / "hourly"
    aggregated_dir = tmp_path / "aggregated"
    repos_dir = tmp_path / "repos"
    first_events_dir = tmp_path / "first_events"
    hourly_dir.mkdir()
    aggregated_dir.mkdir()
    repos_dir.mkdir()
    first_events_dir.mkdir()
    return {
        "hourly_processed_dir": hourly_dir,
        "aggregated_events_dir": aggregated_dir,
        "repos_dir": repos_dir,
        "first_events_path": first_events_dir,
    }


@pytest.fixture
def aggregator(tmp_dirs):
    """Create a PeriodAggregator instance for testing."""
    return PeriodAggregator(
        hourly_processed_dir=tmp_dirs["hourly_processed_dir"],
        aggregated_events_dir=tmp_dirs["aggregated_events_dir"],
        repos_dir=tmp_dirs["repos_dir"],
        first_events_path=tmp_dirs["first_events_path"],
    )


@pytest.mark.parametrize(
    "freq,date,ex_period_start,ex_period_end,ex_period_ds",
    [
        ("D", "2026-01-02-01", "2026-01-01-00", "2026-01-01-23", "2026-01-01-00"),
        ("D", "2026-01-02-00", "2026-01-01-00", "2026-01-01-23", "2026-01-01-00"),
        ("M", "2026-01-15-00", "2025-12-01-00", "2025-12-31-23", "2025-12-01-00"),
        ("M", "2026-01-01-00", "2025-12-01-00", "2025-12-31-23", "2025-12-01-00"),
        ("W-SUN", "2023-01-10-00", "2023-01-02-00", "2023-01-08-23", "2023-01-08-00"),
        ("W-SUN", "2023-01-09-11", "2023-01-02-00", "2023-01-08-23", "2023-01-08-00"),
        ("W-SUN", "2023-01-08-23", "2022-12-26-00", "2023-01-01-23", "2023-01-01-00"),
    ],
)
def test_get_previous_period(
    aggregator,
    freq,
    date,
    ex_period_start,
    ex_period_end,
    ex_period_ds,
):
    """Test _get_previous_period method."""
    period_start, period_end, period_ds = aggregator.get_previous_period(
        RawGHArchiveHourExtractor.str_hour_to_dt(date),
        freq,
    )
    assert period_start == RawGHArchiveHourExtractor.str_hour_to_dt(ex_period_start)
    assert period_end == RawGHArchiveHourExtractor.str_hour_to_dt(ex_period_end)
    assert period_ds == RawGHArchiveHourExtractor.str_hour_to_dt(ex_period_ds)
