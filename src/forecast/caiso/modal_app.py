from __future__ import annotations

import modal

CAISO_DATA_ROOT = "/s3-bucket/v0.1.0/caiso"

CPU_MODELS = (
    "seasonal_naive",
    "historic_average",
    "auto_ets",
    "auto_ces",
    "dynamic_optimized_theta",
)

app = modal.App(name="timecopilot-caiso-forecast")
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
    timeout=60 * 30,
)
def run_forecast_model(cutoff: str, model: str) -> tuple[str, str, str | None]:
    import os

    os.environ["CAISO_DATA_ROOT"] = CAISO_DATA_ROOT

    from src.forecast.caiso.core import run_forecast

    try:
        forecast_path = run_forecast(
            cutoff=cutoff,
            model=model,
            h=24,
            max_window_size=48,
        )
        return model, str(forecast_path), None
    except Exception as exc:
        return model, "", repr(exc)


@app.function(
    image=image,
    volumes=volume,
    timeout=60 * 30,
)
def run_evaluation_model(cutoff: str, model: str) -> tuple[str, str, str | None]:
    import os

    os.environ["CAISO_DATA_ROOT"] = CAISO_DATA_ROOT

    from src.evaluation.caiso.core import run_evaluation

    try:
        metrics_path = run_evaluation(
            cutoff=cutoff,
            model=model,
            h=24,
            max_window_size=48,
        )
        return model, str(metrics_path), None
    except Exception as exc:
        return model, "", repr(exc)


def _print_results(results: list[tuple[str, str, str | None]]) -> None:
    failures = [result for result in results if result[2] is not None]

    for model, path, error in results:
        if error:
            print(f"FAILED {model}: {error}")
        else:
            print(f"OK {model}: {path}")

    if failures:
        raise RuntimeError(f"{len(failures)} CAISO model runs failed")


@app.local_entrypoint()
def forecast(cutoff: str):
    results = list(
        run_forecast_model.starmap([(cutoff, model) for model in CPU_MODELS])
    )
    _print_results(results)


@app.local_entrypoint()
def evaluate(cutoff: str):
    results = list(
        run_evaluation_model.starmap([(cutoff, model) for model in CPU_MODELS])
    )
    _print_results(results)
