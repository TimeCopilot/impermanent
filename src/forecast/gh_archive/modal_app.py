"""Modal app for running forecast models on GH Archive data."""

import modal

app = modal.App(name="impermanent-forecast")


def _base_image_setup(image: modal.Image) -> modal.Image:
    return (
        image.apt_install("git")
        .pip_install("uv")
        .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
        .add_local_file(".python-version", "/root/.python-version", copy=True)
        .add_local_file("uv.lock", "/root/uv.lock", copy=True)
        .add_local_dir("src", "/root/src", copy=True)
        .workdir("/root")
        .run_commands("uv pip install . --system --compile-bytecode")
    )


image_cpu = _base_image_setup(modal.Image.debian_slim(python_version="3.11"))
image_gpu = _base_image_setup(
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
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


BASE_PATH = "/s3-bucket/v0.1.0/gh-archive/processed-events"
FORECASTS_DIR = "/s3-bucket/v0.1.0/gh-archive/forecasts"


def _make_forecaster(frequency: str):
    from .core import GHArchiveForecaster

    return GHArchiveForecaster(
        frequency,
        base_path=BASE_PATH,
        forecasts_dir=FORECASTS_DIR,
    )


@app.function(
    image=image_cpu,
    volumes=volume,
    cpu=32,
    memory=16384,
    timeout=60 * 60 * 2,
    max_containers=25,
)
def run_cpu_model(
    model_name: str,
    cutoff: str,
    frequency: str,
) -> str | None:
    """Run a CPU-based model (statistical / ML) for one cutoff."""
    import logging

    logging.basicConfig(level=logging.INFO)
    forecaster = _make_forecaster(frequency)
    return forecaster.run_forecast(model_name, cutoff)


@app.function(
    image=image_gpu,
    volumes=volume,
    gpu="A10G",
    timeout=60 * 60 * 2,
    max_containers=25,
)
def run_gpu_model(
    model_name: str,
    cutoff: str,
    frequency: str,
) -> str | None:
    """Run a GPU-based model (neural / foundation) for one cutoff."""
    import logging

    logging.basicConfig(level=logging.INFO)
    forecaster = _make_forecaster(frequency)
    return forecaster.run_forecast(model_name, cutoff)


@app.function(image=image_cpu, volumes=volume, timeout=60 * 5)
def load_validation_report(report_path: str):
    """Load a cached ValidationReport from the S3 mount."""
    from src.data.gh_archive.utils.parquet_validator import ValidationReport

    return ValidationReport.load(report_path)


@app.function(image=image_cpu, volumes=volume, timeout=60 * 5)
def save_validation_report(report_path: str, report):
    """Save a ValidationReport to the S3 mount."""
    report.save(report_path)


@app.function(image=image_cpu, volumes=volume, timeout=60 * 15)
def validate_forecast(composite_key: str, frequency: str):
    """Validate a single forecast parquet file.

    *composite_key* is ``"model_name::cutoff"`` (see
    :meth:`GHArchiveForecaster.make_key`).
    """
    import logging

    from src.data.gh_archive.utils.parquet_validator import ParquetValidator

    logging.basicConfig(level=logging.INFO)

    from .core import GHArchiveForecaster

    model_name, cutoff = GHArchiveForecaster.split_key(composite_key)
    forecaster = _make_forecaster(frequency)

    path = forecaster.forecast_path(cutoff, model_name)

    validator = ParquetValidator()
    result = validator.validate_file(path)
    result.metadata = {"composite_key": composite_key}
    return result


KNOWN_BAD_KEYS: set[str] = set()


def _fix_forecasts(
    keys: list[str],
    frequency: str,
) -> None:
    """Re-run forecasts for problematic (model, cutoff) pairs."""
    import logging

    from .core import GHArchiveForecaster
    from src.forecast.forecast import is_gpu_model

    logging.basicConfig(level=logging.INFO)

    handles = []
    for key in keys:
        model_name, cutoff = GHArchiveForecaster.split_key(key)
        if is_gpu_model(model_name):
            handles.append(run_gpu_model.spawn(model_name, cutoff, frequency))
        else:
            handles.append(run_cpu_model.spawn(model_name, cutoff, frequency))

    for h in handles:
        h.get()


@app.local_entrypoint()
def validate_forecasts_from_start_to_current(
    start_date: str = "2023-01-08",
    frequency: str = "daily",
    force_full: bool = False,
):
    """Validate all forecast files, re-running missing ones.

    For each (model, cutoff) pair, checks the forecast parquet is
    present and readable.  Uses ``incremental_validate_and_fix``
    from the shared orchestrator.
    """
    import functools
    import logging

    logging.basicConfig(level=logging.INFO)

    from .core import GHArchiveForecaster
    from src.data.gh_archive.utils.parquet_validator import incremental_validate_and_fix
    from src.forecast.forecast import MODELS

    forecaster = GHArchiveForecaster(frequency)
    cutoffs = forecaster.cutoff_range(start_date=start_date)
    logging.info(f"Cutoffs: {cutoffs}")
    model_names = list(MODELS.keys())

    all_keys = [
        GHArchiveForecaster.make_key(m, c) for c in cutoffs for m in model_names
    ]

    report_path = f"{FORECASTS_DIR}/{frequency}/_validation_report.json"

    # Adapter: incremental_validate_and_fix calls validate_fn.map(keys)
    class _ValidateFnAdapter:
        def map(self, keys):
            return validate_forecast.starmap([(k, frequency) for k in keys])

    incremental_validate_and_fix(
        all_keys=all_keys,
        validate_fn=_ValidateFnAdapter(),
        report_path=report_path,
        load_report_fn=load_validation_report.remote,
        save_report_fn=save_validation_report.remote,
        fix_fn=functools.partial(_fix_forecasts, frequency=frequency),
        key_field="composite_key",
        force_full=force_full,
        ignore_keys=KNOWN_BAD_KEYS,
    )
