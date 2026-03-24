"""Modal app for running forecast evaluations on GH Archive data."""

import modal

app = modal.App(name="impermanent-evaluate")


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
EVAL_DIR = "/s3-bucket/v0.1.0/gh-archive/evaluations"
# Athena UNLOAD (see ``src/data/gh_archive/athena/README.md``): per-series sparsity
SPARSITY_DIR = "/s3-bucket/v0.1.0/gh-archive/meta/series_sparsity"

# Zero-model scaling for leaderboard (see paper): reported score is v / max(b, τ₀),
# where b is the zero-model metric on that subdataset and τ₀ is the 10th percentile
# of strictly positive zero-model scores for that metric (per frequency/cutoff).
_ZERO_MODEL = "zero_model"
_TAU_DENOM_EPS = 1e-3

# Leaderboard includes only these processed-event frequencies (excludes hourly, etc.)
_LEADERBOARD_FREQUENCIES = ("daily", "weekly", "monthly")


def _make_evaluator(frequency: str):
    from .core import GHArchiveEvaluator

    return GHArchiveEvaluator(
        frequency,
        base_path=BASE_PATH,
        forecasts_dir=FORECASTS_DIR,
        eval_dir=EVAL_DIR,
    )


@app.function(
    image=image_cpu,
    volumes=volume,
    cpu=8,
    memory=8192,
    timeout=60 * 30,
    max_containers=125,
)
def evaluate_model(
    model_name: str,
    cutoff: str,
    frequency: str,
) -> str | None:
    """Evaluate a single model for one cutoff."""
    import logging

    logging.basicConfig(level=logging.INFO)
    evaluator = _make_evaluator(frequency)
    try:
        path = evaluator.run_evaluation(model_name, cutoff)
        logging.info(f"Evaluated {model_name} for {cutoff} at {path}")
        return None
    except Exception as e:
        logging.error(f"Error evaluating {model_name} for {cutoff}: {e}")
        return str(e)


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
def validate_evaluation(composite_key: str, frequency: str):
    """Validate a single evaluation metrics parquet file.

    *composite_key* is ``"model_name::cutoff"`` (see
    :meth:`GHArchiveForecaster.make_key`).
    """
    import logging

    from src.data.gh_archive.utils.parquet_validator import ParquetValidator
    from src.forecast.gh_archive.core import GHArchiveForecaster

    logging.basicConfig(level=logging.INFO)

    model_name, cutoff = GHArchiveForecaster.split_key(composite_key)
    evaluator = _make_evaluator(frequency)

    path = evaluator.eval_path(cutoff, model_name)

    validator = ParquetValidator()
    result = validator.validate_file(path)
    result.metadata = {"composite_key": composite_key}
    return result


KNOWN_BAD_KEYS: set[str] = set()


def _fix_evaluations(
    keys: list[str],
    frequency: str,
) -> None:
    """Re-run evaluations for problematic (model, cutoff) pairs."""
    import logging

    from src.forecast.gh_archive.core import GHArchiveForecaster

    logging.basicConfig(level=logging.INFO)

    handles = []
    for key in keys:
        model_name, cutoff = GHArchiveForecaster.split_key(key)
        handles.append(evaluate_model.spawn(model_name, cutoff, frequency))

    results = [h.get() for h in handles]
    errors = [r for r in results if r is not None]
    if errors:
        logging.error(f"Errors: {errors}")
    else:
        logging.info("All evaluations fixed successfully")


@app.local_entrypoint()
def validate_evaluations_from_start_to_current(
    start_date: str = "2023-01-08",
    frequency: str = "daily",
    force_full: bool = False,
):
    """Validate all evaluation files, re-running missing ones.

    For each (model, cutoff) pair, checks the metrics parquet is
    present and readable.  Uses ``incremental_validate_and_fix``
    from the shared orchestrator.
    """
    import functools
    import logging

    logging.basicConfig(level=logging.INFO)

    from src.data.gh_archive.utils.parquet_validator import incremental_validate_and_fix
    from src.forecast.forecast import MODELS
    from src.forecast.gh_archive.core import GHArchiveForecaster

    evaluator = _make_evaluator(frequency)
    cutoffs = evaluator.cutoff_range(start_date=start_date)
    logging.info(f"Cutoffs: {cutoffs}")
    model_names = list(MODELS.keys())

    all_keys = [
        GHArchiveForecaster.make_key(m, c) for c in cutoffs for m in model_names
    ]

    report_path = f"{EVAL_DIR}/{frequency}/_validation_report.json"

    class _ValidateFnAdapter:
        def map(self, keys):
            return validate_evaluation.starmap([(k, frequency) for k in keys])

    incremental_validate_and_fix(
        all_keys=all_keys,
        validate_fn=_ValidateFnAdapter(),
        report_path=report_path,
        load_report_fn=load_validation_report.remote,
        save_report_fn=save_validation_report.remote,
        fix_fn=functools.partial(_fix_evaluations, frequency=frequency),
        key_field="composite_key",
        force_full=force_full,
        ignore_keys=KNOWN_BAD_KEYS,
    )


@app.function(image=image_cpu, volumes=volume, timeout=60 * 15, cpu=4, memory=4096)
def aggregate_leaderboard() -> dict:
    """Read evaluation parquets across all frequencies and write one parquet.

    Only **daily**, **weekly**, and **monthly** evaluation runs are included
    (hourly and other frequencies are ignored).

    For every ``(frequency, cutoff)`` combination, guarantees that all
    subdatasets and models are present.  Missing combinations are filled
    with ``NA``.

    Per-series metrics are stored in ``metrics.parquet`` (one row per
    ``unique_id``). Sparsity tertiles are joined from ``meta/series_sparsity``
    on ``unique_id`` when building the leaderboard.
    For each ``(subdataset, sparsity_level, model, frequency, cutoff, metric)``,
    take ``median`` over series. Let that be *v*. Using only **zero-model**
    rows in the same group: let *M* be the median of zero-model **per-series**
    scores and *P* the 10th percentile of strictly positive zero-model
    per-series scores. Reported value is ``v / max(M, P)``. If there is no
    zero-model denominator for that group, the scaled ``value`` is ``NULL``.

    The output parquet has the long-format schema expected by the leaderboard:
    ``dataset, subdataset, sparsity_level, frequency, cutoff, metric, model,
    model_alias, value, median_zero, pct_z, denom`` (scaling stats are per
    subdataset, sparsity tertile, frequency, cutoff, and metric).

    Returns:
        Dict with ``path`` (str), ``rows`` (int), ``missing`` (int).
    """
    import logging
    import shutil
    import tempfile

    import duckdb

    logging.basicConfig(level=logging.INFO)

    glob_pattern = f"{EVAL_DIR}/**/metrics.parquet"
    output_path = f"{EVAL_DIR}/evaluation_results.parquet"
    freq_list = ", ".join(f"'{f}'" for f in _LEADERBOARD_FREQUENCIES)

    con = duckdb.connect(":memory:")
    con.execute("INSTALL parquet; LOAD parquet;")

    con.execute(
        f"""
        CREATE TABLE sparsity AS
        SELECT DISTINCT unique_id, sparsity_level
        FROM read_parquet('{SPARSITY_DIR}/*', union_by_name=true)
        """
    )

    # 1. Read per-series metrics.parquet rows and melt to long format (metrics.parquet
    #    has one row per unique_id with mase, scaled_crps). Only daily / weekly /
    #    monthly.
    #    Join sparsity tertiles (Athena UNLOAD) on unique_id.
    con.execute(
        f"""
        CREATE TABLE raw_long AS
        SELECT
            'gh-archive' AS dataset,
            coalesce(
                u.unique_id,
                u.subdataset || ':' || u.model || ':' || u.cutoff || ':' || u.frequency
            ) AS unique_id,
            u.subdataset,
            u.frequency,
            u.cutoff,
            u.metric,
            u.model,
            u.model_alias,
            u.value,
            coalesce(s.sparsity_level, 'unknown') AS sparsity_level
        FROM (
            SELECT * FROM (
                SELECT
                    unique_id,
                    model,
                    model_alias,
                    cutoff,
                    frequency,
                    subdataset,
                    mase,
                    scaled_crps
                FROM read_parquet('{glob_pattern}', union_by_name=true)
                WHERE frequency IN ({freq_list})
            ) AS wide
            UNPIVOT (value FOR metric IN (mase, scaled_crps))
        ) AS u
        LEFT JOIN sparsity s ON u.unique_id = s.unique_id
        """
    )

    # 1b. Median over series per (subdataset, sparsity_level, ...); scale v / max(M, P)
    #     where M and P are computed per (frequency, cutoff, metric, subdataset,
    #     sparsity_level) from zero-model per-series values only.
    con.execute(
        """
        CREATE TABLE agg_by_subdataset AS
        SELECT
            dataset,
            subdataset,
            sparsity_level,
            frequency,
            cutoff,
            metric,
            model,
            model_alias,
            median(value) AS v
        FROM raw_long
        GROUP BY
            dataset,
            subdataset,
            sparsity_level,
            frequency,
            cutoff,
            metric,
            model,
            model_alias
        """
    )

    con.execute(
        f"""
        CREATE TABLE denom AS
        WITH med AS (
            SELECT
                frequency,
                cutoff,
                metric,
                subdataset,
                sparsity_level,
                median(value) AS median_zero
            FROM raw_long
            WHERE model = '{_ZERO_MODEL}'
            GROUP BY frequency, cutoff, metric, subdataset, sparsity_level
        ),
        pct AS (
            SELECT
                frequency,
                cutoff,
                metric,
                subdataset,
                sparsity_level,
                quantile_cont(value, 0.1) AS pct_z
            FROM raw_long
            WHERE model = '{_ZERO_MODEL}'
              AND value > 0
            GROUP BY frequency, cutoff, metric, subdataset, sparsity_level
        )
        SELECT
            m.frequency,
            m.cutoff,
            m.metric,
            m.subdataset,
            m.sparsity_level,
            m.median_zero,
            p.pct_z,
            greatest(
                coalesce(m.median_zero, 0.0),
                coalesce(p.pct_z, {_TAU_DENOM_EPS})
            ) AS denom
        FROM med m
        LEFT JOIN pct p
            ON m.frequency = p.frequency
           AND m.cutoff = p.cutoff
           AND m.metric = p.metric
           AND m.subdataset = p.subdataset
           AND m.sparsity_level = p.sparsity_level
        """
    )

    con.execute(
        """
        CREATE TABLE scaled_long AS
        SELECT
            a.dataset,
            a.subdataset,
            a.sparsity_level,
            a.frequency,
            a.cutoff,
            a.metric,
            a.model,
            a.model_alias,
            d.median_zero,
            d.pct_z,
            d.denom,
            CASE
                WHEN d.denom IS NOT NULL AND d.denom > 0
                THEN a.v / d.denom
                ELSE NULL
            END AS value
        FROM agg_by_subdataset a
        LEFT JOIN denom d
            ON a.frequency = d.frequency
           AND a.cutoff = d.cutoff
           AND a.metric = d.metric
           AND a.subdataset = d.subdataset
           AND a.sparsity_level = d.sparsity_level
        """
    )

    # 2. Build the full grid per frequency:
    #    For each (frequency, cutoff), every (subdataset, sparsity_level, model, metric)
    #    must be present.  Left-join fills gaps with NULL.
    con.execute(
        """
        CREATE TABLE full_grid AS
        SELECT
            'gh-archive' AS dataset,
            s.subdataset,
            sp.sparsity_level,
            c.frequency,
            c.cutoff,
            m.metric,
            mo.model,
            mo.model_alias
        FROM
            (SELECT DISTINCT cutoff, frequency FROM scaled_long) c
            CROSS JOIN (SELECT DISTINCT subdataset FROM scaled_long) s
            CROSS JOIN (SELECT DISTINCT sparsity_level FROM scaled_long) sp
            CROSS JOIN (SELECT DISTINCT metric FROM scaled_long) m
            CROSS JOIN (SELECT DISTINCT model, model_alias FROM scaled_long) mo;
        """
    )

    # Write to a local temp file first, then copy to the volume.
    # DuckDB COPY uses atomic rename which fails on Modal volumes.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = f"{tmp_dir}/evaluation_results.parquet"

        con.execute(
            f"""
            COPY (
                SELECT
                    g.dataset,
                    g.subdataset,
                    g.sparsity_level,
                    g.frequency,
                    g.cutoff,
                    g.metric,
                    g.model,
                    g.model_alias,
                    r.value,
                    COALESCE(r.median_zero, d0.median_zero) AS median_zero,
                    COALESCE(r.pct_z, d0.pct_z) AS pct_z,
                    COALESCE(r.denom, d0.denom) AS denom
                FROM full_grid g
                LEFT JOIN scaled_long r
                    ON  g.cutoff     = r.cutoff
                    AND g.frequency  = r.frequency
                    AND g.subdataset = r.subdataset
                    AND g.sparsity_level = r.sparsity_level
                    AND g.metric     = r.metric
                    AND g.model      = r.model
                LEFT JOIN denom d0
                    ON  g.frequency = d0.frequency
                    AND g.cutoff = d0.cutoff
                    AND g.metric = d0.metric
                    AND g.subdataset = d0.subdataset
                    AND g.sparsity_level = d0.sparsity_level
                ORDER BY
                    g.frequency,
                    g.subdataset,
                    g.sparsity_level,
                    g.cutoff,
                    g.metric,
                    g.model
            ) TO '{tmp_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

        stats = con.execute(
            f"""
            SELECT
                count(*) AS total,
                count(*) - count(value) AS missing
            FROM read_parquet('{tmp_path}')
            """
        ).fetchone()
        total, missing = stats

        shutil.copy(tmp_path, output_path)

    logging.info(
        f"Wrote {total} rows to {output_path} ({missing} missing values)",
    )
    return {"path": output_path, "rows": total, "missing": missing}


@app.local_entrypoint()
def build_leaderboard():
    """Build a single leaderboard parquet from all evaluation parquets.

    Reads ``metrics.parquet`` for **daily**, **weekly**, and **monthly** only
    under ``{EVAL_DIR}/`` and writes a single ``evaluation_results.parquet``.
    Values are scaled as ``median / max(median, p10)`` of zero-model per-series
    scores within each subdataset (see :func:`aggregate_leaderboard`).
    Raises an error if any (frequency, cutoff, subdataset, sparsity_level,
    model, metric) combination has a missing value, but only after the file
    has been written.
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    result = aggregate_leaderboard.remote()
    logging.info(
        f"Leaderboard parquet written to {result['path']} "
        f"({result['rows']} rows, {result['missing']} missing)",
    )
    if result["missing"] > 0:
        logging.warning(
            f"ERROR: {result['missing']} missing values found in "
            f"{result['path']}. Some (frequency, cutoff, subdataset, "
            "model, metric) combinations have no evaluation result."
        )
