from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.cenace_pipeline import run_cenace_pipeline

DatasetRunner = Callable[..., tuple[str, str]]

DATASET_REGISTRY: dict[str, DatasetRunner] = {
    "cenace": run_cenace_pipeline,
}

PIPELINE_DATASET_CHOICES = sorted(DATASET_REGISTRY)


def run_dataset_pipeline(dataset: str, **kwargs: Any) -> tuple[str, str]:
    try:
        runner = DATASET_REGISTRY[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset: {dataset}. " f"Available: {PIPELINE_DATASET_CHOICES}"
        ) from exc

    return runner(**kwargs)
