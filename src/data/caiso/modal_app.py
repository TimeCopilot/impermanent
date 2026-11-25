from __future__ import annotations

import modal

CAISO_DATA_ROOT = "/s3-bucket/v0.1.0/caiso"

app = modal.App(name="timecopilot-caiso-data")
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
def update_caiso_range(start: str, end: str) -> int:
    import os
    from datetime import date

    os.environ["CAISO_DATA_ROOT"] = CAISO_DATA_ROOT

    from src.data.caiso.pipeline import update_date_range

    return update_date_range(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
    )


@app.local_entrypoint()
def update(start: str, end: str):
    n_written = update_caiso_range.remote(start, end)
    print(f"Done. Wrote {n_written} CAISO daily partitions.")
