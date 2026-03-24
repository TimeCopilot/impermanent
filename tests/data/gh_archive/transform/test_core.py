import datetime as dt

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.gh_archive.extract.core import RawGHArchiveHourExtractor
from src.data.gh_archive.transform.core import HourTransformer

TEST_HOUR = "2025-01-01-00"
EXPECTED_COLUMNS = ["repo_id", "repo_name", "metric", "unique_id", "ds", "y"]
EXPECTED_METRICS = {"stars", "prs_opened", "issues_opened", "pushes"}


@pytest.fixture(scope="module")
def raw_events_dir(tmp_path_factory):
    """Download one hour of raw GH Archive events."""
    d = tmp_path_factory.mktemp("raw_events")
    extractor = RawGHArchiveHourExtractor(output_dir=d)
    extractor.extract_hour(TEST_HOUR)
    return d


@pytest.fixture(scope="module")
def two_repos(raw_events_dir):
    """Pick two real repo_ids from the downloaded raw events."""
    extractor = RawGHArchiveHourExtractor(output_dir=raw_events_dir)
    dt_hour = extractor.str_hour_to_dt(TEST_HOUR)
    path = extractor.hour_local_path(dt_hour)

    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"SELECT DISTINCT repo_id, repo_name "
        f"FROM read_parquet('{path}') "
        f"WHERE repo_id IS NOT NULL LIMIT 2"
    ).fetchall()
    con.close()

    assert len(rows) == 2, "Need at least 2 distinct repos in the test hour"
    return [(r[0], r[1]) for r in rows]


@pytest.fixture(scope="module")
def repos_dir(tmp_path_factory, two_repos):
    """Repos parquet with both test repos."""
    d = tmp_path_factory.mktemp("repos")
    table = pa.table(
        {
            "repo_id": pa.array([r[0] for r in two_repos], type=pa.int64()),
            "repo_name": pa.array([r[1] for r in two_repos]),
        }
    )
    pq.write_table(table, str(d / "repos.parquet"))
    return d


@pytest.fixture(scope="module")
def first_events_dir(tmp_path_factory, two_repos):
    """first_events parquet.

    Repo A: first_hour *before* test hour → included in output.
    Repo B: first_hour *after*  test hour → excluded from output.
    """
    d = tmp_path_factory.mktemp("first_events")
    dt_hour = RawGHArchiveHourExtractor.str_hour_to_dt(TEST_HOUR)

    before = (dt_hour - dt.timedelta(hours=1)).replace(tzinfo=None)
    after = (dt_hour + dt.timedelta(hours=1)).replace(tzinfo=None)

    table = pa.table(
        {
            "repo_id": pa.array([two_repos[0][0], two_repos[1][0]], type=pa.int64()),
            "first_hour": pa.array([before, after], type=pa.timestamp("us")),
        }
    )
    pq.write_table(table, str(d / "first_events.parquet"))
    return d


@pytest.fixture(scope="module")
def transformed_df(raw_events_dir, repos_dir, first_events_dir, tmp_path_factory):
    """Run HourTransformer.aggregate_hour and return the output DataFrame."""
    output_dir = tmp_path_factory.mktemp("processed")
    transformer = HourTransformer(
        raw_events_dir=raw_events_dir,
        processed_events_dir=output_dir,
        repos_dir=repos_dir,
        first_events_path=first_events_dir,
    )
    out_path = transformer.aggregate_hour(TEST_HOUR)
    return pd.read_parquet(out_path)


def test_output_columns(transformed_df):
    assert list(transformed_df.columns) == EXPECTED_COLUMNS


def test_output_metrics(transformed_df):
    assert set(transformed_df["metric"].unique()) == EXPECTED_METRICS


def test_included_repo_has_observations(transformed_df, two_repos):
    """Repo A (first_hour before test hour) should appear with all metrics."""
    included_id = two_repos[0][0]
    repo_df = transformed_df[transformed_df["repo_id"] == included_id]
    assert len(repo_df) > 0
    assert set(repo_df["metric"]) == EXPECTED_METRICS
    assert repo_df["repo_name"].iloc[0] == two_repos[0][1]


def test_excluded_repo_not_in_output(transformed_df, two_repos):
    """Repo B (first_hour after test hour) should NOT appear."""
    excluded_id = two_repos[1][0]
    repo_df = transformed_df[transformed_df["repo_id"] == excluded_id]
    assert len(repo_df) == 0
