import modal

app = modal.App(name="timecopilot-gh-archive-transform")
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
RAW_EVENTS_DIR = f"{BASE_PATH}/raw-events"
PROCESSED_DIR = f"{BASE_PATH}/processed-events/hourly"
REPOS_DIR = f"{BASE_PATH}/meta/repos"
FIRST_EVENTS_PATH = f"{BASE_PATH}/meta/first_events"


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 15,
    max_containers=50,
)
def transform_one_hour(str_hour: str, overwrite: bool = False):
    import logging

    from .core import HourTransformer

    logging.basicConfig(level=logging.INFO)

    transformer = HourTransformer(
        raw_events_dir=RAW_EVENTS_DIR,
        processed_events_dir=PROCESSED_DIR,
        repos_dir=REPOS_DIR,
        first_events_path=FIRST_EVENTS_PATH,
    )
    if transformer.hour_exists(str_hour) and not overwrite:
        logging.info(f"Hour {str_hour} already exists, skipping transformation")
        return
    try:
        output_file = transformer.aggregate_hour(str_hour)
        logging.info(f"Transformed hour {str_hour} to {output_file}")
    except Exception as e:
        logging.error(f"Error transforming hour {str_hour}: {e}")
        return None


def process_transform_hours(hours: list[str], overwrite: bool = False):
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Transforming {len(hours)} hours from {hours[0]} to {hours[-1]}")

    results = list(transform_one_hour.starmap([(h, overwrite) for h in hours]))
    errors = [r for r in results if r is not None]
    if errors:
        logging.error(f"Errors: {errors}")
    else:
        logging.info("All hours transformed successfully")


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
def validate_transformed_hour(str_hour: str):
    """Validate a single transformed hour parquet file."""
    import logging

    logging.basicConfig(level=logging.INFO)

    from ..utils.parquet_validator import ParquetValidator
    from .core import HourTransformer

    transformer = HourTransformer(
        raw_events_dir=RAW_EVENTS_DIR,
        processed_events_dir=PROCESSED_DIR,
        repos_dir=REPOS_DIR,
        first_events_path=FIRST_EVENTS_PATH,
    )

    dt_hour = transformer.str_hour_to_dt(str_hour)
    file = transformer.hour_local_path(dt_hour)

    validator = ParquetValidator()
    result = validator.validate_file(file)
    result.metadata = {
        "str_hour": str_hour,
    }
    return result


# Hours known to be bad at the origin (inherited from raw events).
KNOWN_BAD_HOURS: set[str] = {
    # Example: "2023-02-18-05",
    "2025-06-12-18",
    "2023-05-14-19",
    "2025-06-12-17",
}


def _fix_transformed_hours(keys: list[str]) -> None:
    """Re-transform problematic hours."""
    process_transform_hours(keys, overwrite=True)


@app.local_entrypoint()
def validate_transformed_from_start_to_current(
    start_date: str = "2023-01-01-00",
    force_full: bool = False,
):
    """Validate transformed events, skipping hours already OK.

    Args:
        start_date: Start date in format "YYYY-MM-DD-HH".
        force_full: If True, ignore the cached report and re-validate
            everything from scratch.
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    from ..extract.core import RawGHArchiveHourExtractor
    from ..utils.parquet_validator import incremental_validate_and_fix

    last_date = RawGHArchiveHourExtractor.last_cutoff_str_hour()
    logging.info(f"Last date: {last_date}")
    all_str_hours = RawGHArchiveHourExtractor.generate_str_hours(
        start_str_hour=start_date,
        end_str_hour=last_date,
    )

    incremental_validate_and_fix(
        all_keys=all_str_hours,
        validate_fn=validate_transformed_hour,
        report_path=f"{PROCESSED_DIR}/_validation_report.json",
        load_report_fn=load_validation_report.remote,
        save_report_fn=save_validation_report.remote,
        fix_fn=_fix_transformed_hours,
        key_field="str_hour",
        force_full=force_full,
        ignore_keys=KNOWN_BAD_HOURS,
    )
