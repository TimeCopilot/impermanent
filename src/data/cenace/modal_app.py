from __future__ import annotations

import modal

CENACE_DATA_ROOT = "/s3-bucket/v0.1.0/cenace"

app = modal.App(name="timecopilot-cenace-data")
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
def update_cenace_execution_date(execution_date: str) -> int:
    import os
    from datetime import datetime

    os.environ["CENACE_DATA_ROOT"] = CENACE_DATA_ROOT

    from src.data.cenace.pipeline import update_execution_date

    return update_execution_date(datetime.fromisoformat(execution_date))


@app.local_entrypoint()
def update(execution_date: str):
    n_written = update_cenace_execution_date.remote(execution_date)
    print(f"Done. Wrote {n_written} CENACE daily partitions.")
