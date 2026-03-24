import modal

app = modal.App(name="timecopilot-gh-archive-aggregate")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file(".python-version", "/root/.python-version", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .workdir("/root")
    .run_commands("uv pip install . --system --compile-bytecode")
)
secret = modal.Secret.from_name(
    "aws-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
volume = {
    "/s3-bucket": modal.CloudBucketMount(
        bucket_name="impermanent-benchmark",
        secret=secret,
    )
}


BASE_PATH = "/s3-bucket/v0.1.0/gh-archive"
HOURLY_DIR = f"{BASE_PATH}/processed-events/hourly"
REPOS_DIR = f"{BASE_PATH}/meta/repos"
FIRST_EVENTS_PATH = f"{BASE_PATH}/meta/first_events"


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 60,
    max_containers=50,
)
def aggregate_one_period(str_period: str, freq: str, overwrite: bool = False):
    import logging

    from ..extract.core import RawGHArchiveHourExtractor
    from .core import PeriodAggregator

    logging.basicConfig(level=logging.INFO)

    freq_type = PeriodAggregator.get_frequency_type(freq)
    aggregator = PeriodAggregator(
        hourly_processed_dir=HOURLY_DIR,
        aggregated_events_dir=f"{BASE_PATH}/processed-events/{freq_type}",
        repos_dir=REPOS_DIR,
        first_events_path=FIRST_EVENTS_PATH,
    )

    dt_period = RawGHArchiveHourExtractor.str_hour_to_dt(str_period)

    try:
        output_file, metadata = aggregator.aggregate_period(dt_period, freq, overwrite)
        logging.info(f"Aggregated period {str_period} to {output_file}")
        logging.info(f"Metadata: {metadata}")
    except ValueError as e:
        logging.error(f"Error aggregating period {str_period}: {e}")
        return None


def process_aggregate_periods(periods: list[str], freq: str, overwrite: bool = False):
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"Aggregating {len(periods)} periods "
        f"from {periods[0]} to {periods[-1]} (freq={freq})"
    )

    results = list(
        aggregate_one_period.starmap([(p, freq, overwrite) for p in periods])
    )
    errors = [r for r in results if r is not None]
    if errors:
        logging.error(f"Errors: {errors}")
    else:
        logging.info("All periods aggregated successfully")


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 5,
)
def load_validation_report(report_path: str):
    """Load a cached ValidationReport from the S3 mount."""
    from ..utils.parquet_validator import ValidationReport

    return ValidationReport.load(report_path)


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 5,
)
def save_validation_report(report_path: str, report):
    """Save a ValidationReport to the S3 mount."""
    report.save(report_path)


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 15,
)
def validate_aggregated_period(str_period: str, freq: str):
    """Validate a single aggregated period parquet file."""
    import logging

    logging.basicConfig(level=logging.INFO)

    from ..extract.core import RawGHArchiveHourExtractor
    from ..utils.parquet_validator import ParquetValidator
    from .core import PeriodAggregator

    freq_type = PeriodAggregator.get_frequency_type(freq)
    aggregator = PeriodAggregator(
        hourly_processed_dir=HOURLY_DIR,
        aggregated_events_dir=f"{BASE_PATH}/processed-events/{freq_type}",
        repos_dir=REPOS_DIR,
        first_events_path=FIRST_EVENTS_PATH,
    )

    dt_period = RawGHArchiveHourExtractor.str_hour_to_dt(str_period)
    period_start, _, _ = aggregator.get_previous_period(dt_period, freq)
    file = aggregator.get_output_path(period_start, freq)

    validator = ParquetValidator()
    result = validator.validate_file(file)
    result.metadata = {
        "str_period": str_period,
    }
    return result


KNOWN_BAD_PERIODS: set[str] = {  # type: ignore
    # Example: "2023-02-18-00",
}


def _fix_aggregated_periods(keys: list[str], freq: str) -> None:
    """Re-aggregate problematic periods."""
    process_aggregate_periods(keys, freq, overwrite=True)


@app.local_entrypoint()
def validate_aggregated_from_start_to_current(
    start_date: str = "2023-01-02-00",
    freq: str = "D",
    force_full: bool = False,
):
    """Validate aggregated periods, skipping those already OK.

    Args:
        start_date: Start date in format "YYYY-MM-DD-HH".
        freq: Pandas frequency string (e.g., 'D', 'W', 'M').
        force_full: If True, ignore cached report and re-validate all.
    """
    import functools
    import logging

    logging.basicConfig(level=logging.INFO)

    import pandas as pd

    from ..extract.core import RawGHArchiveHourExtractor
    from ..utils.parquet_validator import incremental_validate_and_fix
    from .core import PeriodAggregator

    last_date = RawGHArchiveHourExtractor.last_cutoff_str_hour()
    logging.info(f"Frequency: {freq}, Last date: {last_date}")
    periods = pd.date_range(
        start=start_date,
        end=last_date,
        freq=freq,
    )
    all_str_periods = [p.strftime("%Y-%m-%d-%H") for p in periods]
    logging.info(f"All periods: {all_str_periods}")

    freq_type = PeriodAggregator.get_frequency_type(freq)
    report_path = f"{BASE_PATH}/processed-events/{freq_type}/_validation_report.json"

    # validate_aggregated_period needs freq, but the orchestrator calls
    # validate_fn.map(keys) with just the key.  We use a wrapper that
    # binds freq via starmap.
    class _ValidateFnAdapter:
        """Adapter so incremental_validate_and_fix can call .map(keys)."""

        def map(self, keys):
            return validate_aggregated_period.starmap([(k, freq) for k in keys])

    incremental_validate_and_fix(
        all_keys=all_str_periods,
        validate_fn=_ValidateFnAdapter(),
        report_path=report_path,
        load_report_fn=load_validation_report.remote,
        save_report_fn=save_validation_report.remote,
        fix_fn=functools.partial(_fix_aggregated_periods, freq=freq),
        key_field="str_period",
        force_full=force_full,
        ignore_keys=KNOWN_BAD_PERIODS,
    )
