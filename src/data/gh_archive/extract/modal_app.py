import modal

RAW_EVENTS_DIR_KEY = "v0.1.0/gh-archive/raw-events"

app = modal.App(name="timecopilot-gh-archive-extract")
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


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 15,
)
def extract_hour(str_hour: str, overwrite: bool = False):
    import logging

    from .core import RawGHArchiveHourExtractor

    logging.basicConfig(level=logging.INFO)

    hour_extractor = RawGHArchiveHourExtractor(
        output_dir=f"/s3-bucket/{RAW_EVENTS_DIR_KEY}"
    )
    if hour_extractor.hour_exists(str_hour) and not overwrite:
        logging.info(f"Hour {str_hour} already exists, skipping extraction")
        return
    try:
        output_file = hour_extractor.extract_hour(str_hour)
        logging.info(f"Extracted hour {str_hour} to {output_file}")
    except Exception as e:
        logging.error(f"Error extracting hour {str_hour}: {e}")
        return None


def process_hours(hours: list[str], overwrite: bool = False):
    import logging

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Processing {len(hours)} hours from {hours[0]} to {hours[-1]}")

    results = list(extract_hour.starmap([(h, overwrite) for h in hours]))
    errors = [r for r in results if r is not None]
    if errors:
        logging.error(f"Errors: {errors}")
    else:
        logging.info("All hours processed successfully")


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 5,
)
def load_validation_report(report_path: str):
    """Load a cached ValidationReport from the S3 mount (runs on worker)."""
    from ..utils.parquet_validator import ValidationReport

    return ValidationReport.load(report_path)


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 5,
)
def save_validation_report(report_path: str, report):
    """Save a ValidationReport to the S3 mount (runs on worker)."""
    report.save(report_path)


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 15,
)
def validate_raw_events_hour(str_hour: str):
    import logging

    logging.basicConfig(level=logging.INFO)

    from ..utils.parquet_validator import ParquetValidator
    from .core import RawGHArchiveHourExtractor

    hour_extractor = RawGHArchiveHourExtractor(
        output_dir=f"/s3-bucket/{RAW_EVENTS_DIR_KEY}"
    )

    dt_hour = hour_extractor.str_hour_to_dt(str_hour)
    file = hour_extractor.hour_local_path(dt_hour)

    validator = ParquetValidator()
    result = validator.validate_file(file)
    result.metadata = {
        "str_hour": str_hour,
    }
    return result


KNOWN_BAD_HOURS: set[str] = {
    "2025-06-12-17",  # missing events.json file
    "2025-06-12-18",  # missing events.json file
    "2023-05-14-19",  # data format error in events.json file
}


def _fix_raw_events(keys: list[str]) -> None:
    """Re-extract raw event hours (wraps process_hours with overwrite)."""
    process_hours(keys, overwrite=True)


@app.local_entrypoint()
def validate_raw_events_from_start_to_current(
    start_date: str = "2023-01-01-00",
    force_full: bool = False,
):
    """Validate raw events, skipping hours already validated as OK.

    Args:
        start_date: Start date in format "YYYY-MM-DD-HH".
        force_full: If True, ignore the cached report and re-validate
            everything from scratch.
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    from ..utils.parquet_validator import incremental_validate_and_fix
    from .core import RawGHArchiveHourExtractor

    last_date = RawGHArchiveHourExtractor.last_cutoff_str_hour()
    logging.info(f"Last date: {last_date}")
    all_str_hours = RawGHArchiveHourExtractor.generate_str_hours(
        start_str_hour=start_date,
        end_str_hour=last_date,
    )

    incremental_validate_and_fix(
        all_keys=all_str_hours,
        validate_fn=validate_raw_events_hour,
        report_path=f"/s3-bucket/{RAW_EVENTS_DIR_KEY}/_validation_report.json",
        load_report_fn=load_validation_report.remote,
        save_report_fn=save_validation_report.remote,
        fix_fn=_fix_raw_events,
        key_field="str_hour",
        force_full=force_full,
        ignore_keys=KNOWN_BAD_HOURS,
    )
