"""Standalone parquet file validator, designed for parallel orchestration."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


class FileStatus(str, Enum):
    """Status of a validated file."""

    OK = "ok"
    MISSING = "missing"
    EMPTY = "empty"
    UNREADABLE = "unreadable"
    KNOWN_ISSUE = "known_issue"


@dataclass
class FileValidationResult:
    """Result of validating a single parquet file."""

    path: str
    status: FileStatus
    num_rows: int = 0
    error: str | None = None
    metadata: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FileValidationResult":
        d = d.copy()
        d["status"] = FileStatus(d["status"])
        return cls(**d)


@dataclass
class ValidationReport:
    """Aggregated report that can be built from individual results.

    Designed for parallel workflows: each worker produces a list of
    ``FileValidationResult`` objects; the orchestrator collects them into
    one or more ``ValidationReport`` instances and merges with ``+``.
    """

    results: list[FileValidationResult] = field(default_factory=list)

    # -- derived counts (computed on access) --------------------------------

    @property
    def total_files(self) -> int:
        return len(self.results)

    @property
    def ok_files(self) -> int:
        return sum(1 for r in self.results if r.status == FileStatus.OK)

    @property
    def missing_files(self) -> int:
        return sum(1 for r in self.results if r.status == FileStatus.MISSING)

    @property
    def empty_files(self) -> int:
        return sum(1 for r in self.results if r.status == FileStatus.EMPTY)

    @property
    def unreadable_files(self) -> int:
        return sum(1 for r in self.results if r.status == FileStatus.UNREADABLE)

    @property
    def known_issue_files(self) -> int:
        return sum(1 for r in self.results if r.status == FileStatus.KNOWN_ISSUE)

    _ACCEPTABLE = frozenset({FileStatus.OK, FileStatus.KNOWN_ISSUE})

    @property
    def all_ok(self) -> bool:
        """True when every result is either OK or a known issue."""
        return all(r.status in self._ACCEPTABLE for r in self.results)

    def __add__(self, other: "ValidationReport") -> "ValidationReport":
        """Merge two reports (e.g. from parallel workers)."""
        return ValidationReport(results=self.results + other.results)

    def __iadd__(self, other: "ValidationReport") -> "ValidationReport":
        self.results.extend(other.results)
        return self

    def get_missing_files(self) -> list[FileValidationResult]:
        return [r for r in self.results if r.status == FileStatus.MISSING]

    def get_empty_files(self) -> list[FileValidationResult]:
        return [r for r in self.results if r.status == FileStatus.EMPTY]

    def get_unreadable_files(self) -> list[FileValidationResult]:
        return [r for r in self.results if r.status == FileStatus.UNREADABLE]

    def get_known_issue_files(self) -> list[FileValidationResult]:
        return [r for r in self.results if r.status == FileStatus.KNOWN_ISSUE]

    def get_problem_files(self) -> list[FileValidationResult]:
        """Return results that are neither OK nor known issues."""
        return [r for r in self.results if r.status not in self._ACCEPTABLE]

    def mark_known_issues(
        self,
        keys: set[str],
        key_field: str = "str_hour",
        reason: str = "known upstream issue",
    ) -> int:
        """Reclassify problem results whose key is in *keys* as KNOWN_ISSUE.

        Only results with a non-OK status are reclassified; results already
        OK are left untouched.

        Args:
            keys: Set of key values to mark (e.g. ``{"2023-02-18-05"}``).
            key_field: Metadata field that holds the key.
            reason: Human-readable reason stored in ``error``.

        Returns:
            Number of results reclassified.
        """
        count = 0
        for r in self.results:
            if r.status in self._ACCEPTABLE:
                continue
            if r.metadata and r.metadata.get(key_field) in keys:
                r.status = FileStatus.KNOWN_ISSUE
                r.error = reason
                count += 1
        return count

    def get_ok_paths(self) -> set[str]:
        """Return the set of file paths that were validated as OK."""
        return {r.path for r in self.results if r.status == FileStatus.OK}

    def save(self, path: str | Path) -> None:
        """Save the report to a JSON file.

        Args:
            path: Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"results": [r.to_dict() for r in self.results]}
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ValidationReport":
        """Load a report from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A ValidationReport. Returns an empty report if the file
            does not exist.
        """
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        results = [FileValidationResult.from_dict(r) for r in data["results"]]
        return cls(results=results)

    def summary(self) -> str:
        lines = [
            f"Validation Report: {self.total_files} files scanned",
            f"  OK (non-empty):     {self.ok_files}",
            f"  Missing:            {self.missing_files}",
            f"  Empty (0 rows):     {self.empty_files}",
            f"  Unreadable/corrupt: {self.unreadable_files}",
            f"  Known issues:       {self.known_issue_files}",
        ]
        problems = self.get_problem_files()
        known = self.get_known_issue_files()
        if problems:
            lines.append("")
            lines.append("Problem files:")
            for r in problems:
                detail = r.error if r.error else r.status.value
                lines.append(f"  {r.path}: {detail}")
        if known:
            lines.append("")
            lines.append("Known issues (ignored):")
            for r in known:
                detail = r.error if r.error else r.status.value
                lines.append(f"  {r.path}: {detail}")
        if not problems and not known:
            lines.append("")
            lines.append("All files OK.")
        elif not problems:
            lines.append("")
            lines.append("All files OK (excluding known issues).")
        return "\n".join(lines)


class ParquetValidator:
    """Validate parquet files on the local filesystem.

    Designed to be called by an external orchestrator (e.g. Modal, asyncio,
    multiprocessing) that fans out ``validate_file`` calls in parallel, then
    collects the results.

    Example — sequential::

        validator = ParquetValidator()
        report = validator.validate_files(
            [
                "/data/year=2026/month=01/day=01/series.parquet",
                "/data/year=2026/month=01/day=02/series.parquet",
            ]
        )
        print(
            report.summary()
        )

    Example — parallel with an orchestrator::

        validator = ParquetValidator()

        # Fan-out (each call is independent, safe to run in parallel)
        results = orchestrator.map(
            validator.validate_file,
            all_paths,
        )

        # Collect
        report = ValidationReport(
            results=list(
                results
            )
        )
        print(
            report.summary()
        )
    """

    def validate_file(self, path: str | Path) -> FileValidationResult:
        """Validate a single local parquet file.

        Checks, in order:
        1. Does the file exist?  → ``MISSING``
        2. Can DuckDB read it?   → ``UNREADABLE``
        3. Does it have rows?    → ``EMPTY`` / ``OK``

        This method is **stateless** and safe to call from any thread or
        process, making it suitable for parallel fan-out.

        Args:
            path: Path to the parquet file.

        Returns:
            A single ``FileValidationResult``.
        """
        str_path = str(path)
        p = Path(path)

        if not p.exists():
            return FileValidationResult(
                path=str_path,
                status=FileStatus.MISSING,
                error="file not found",
            )

        try:
            con = duckdb.connect(database=":memory:")
            row_count = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{str_path}')"
            ).fetchone()[0]
            con.close()
        except Exception as e:
            logger.debug(f"Failed to read {str_path}: {e}")
            return FileValidationResult(
                path=str_path,
                status=FileStatus.UNREADABLE,
                error=str(e),
            )

        if row_count == 0:
            return FileValidationResult(
                path=str_path,
                status=FileStatus.EMPTY,
            )

        return FileValidationResult(
            path=str_path,
            status=FileStatus.OK,
            num_rows=row_count,
        )

    def validate_files(self, paths: list[str | Path]) -> ValidationReport:
        """Validate multiple local parquet files sequentially.

        For parallel validation, call ``validate_file`` per-path from your
        own orchestrator and collect the results into a ``ValidationReport``.

        Args:
            paths: List of file paths to validate.

        Returns:
            A ``ValidationReport`` with one result per path.
        """
        results = [self.validate_file(p) for p in paths]
        return ValidationReport(results=results)


def incremental_validate_and_fix(
    all_keys: list[str],
    validate_fn,
    report_path: str,
    load_report_fn,
    save_report_fn,
    fix_fn=None,
    key_field: str = "str_hour",
    force_full: bool = False,
    ignore_keys: set[str] | None = None,
    post_fix_delay_seconds: float = 30.0,
    post_fix_validation_retries: int = 1,
):
    """Reusable incremental validation loop.

    Framework-agnostic: works with any orchestrator (Modal, asyncio, etc.)
    that provides ``validate_fn.map(keys)``.

    Args:
        all_keys: Complete list of string keys to cover.
        validate_fn: Object whose ``.map(keys)`` fans out validation
            and returns an iterable of ``FileValidationResult``.
            Each result must store the key in
            ``result.metadata[key_field]``.
        report_path: Path where the JSON report is cached.
        load_report_fn: Callable ``(path) -> ValidationReport``.
        save_report_fn: Callable ``(path, report) -> None``.
        fix_fn: Optional ``(keys: list[str]) -> None`` that attempts
            to fix/re-create the data for problematic keys.
        key_field: Metadata field name that stores the key in each
            ``FileValidationResult``.
        force_full: If ``True``, discard the cached report and
            re-validate everything.
        ignore_keys: Set of keys whose failures are expected.
            After validation and fix, remaining problems matching
            these keys are reclassified as ``KNOWN_ISSUE``.
        post_fix_delay_seconds: Seconds to wait after fix_fn returns
            before re-validating. Use when writes (e.g. to S3 or a
            shared mount) are not immediately visible to validators.
        post_fix_validation_retries: If re-validation still has
            missing/empty after the delay, retry this many times
            (each time waiting post_fix_delay_seconds again).

    Returns:
        The merged ``ValidationReport`` (cached + new results).

    Raises:
        Exception: If there are still unexpected failures after fix +
            retry + known-issue filtering.
    """
    ignore_keys = ignore_keys or set()

    # 1. Load cached report
    cached_report = ValidationReport() if force_full else load_report_fn(report_path)

    # 2. Figure out which keys are already validated as OK or known issue
    cached_skip_keys: set[str] = set()
    acceptable = {FileStatus.OK, FileStatus.KNOWN_ISSUE}
    for r in cached_report.results:
        if r.status in acceptable and r.metadata and key_field in r.metadata:
            cached_skip_keys.add(r.metadata[key_field])

    keys_to_validate = [k for k in all_keys if k not in cached_skip_keys]

    n_cached = len(all_keys) - len(keys_to_validate)
    logger.info(
        f"Total: {len(all_keys)}, cached OK/known: {n_cached}, "
        f"to validate: {len(keys_to_validate)}",
    )

    if not keys_to_validate:
        logger.info("All items already validated as OK (or known issue), nothing to do")
        return cached_report

    new_results = list(validate_fn.map(keys_to_validate))
    new_report = ValidationReport(results=new_results)
    logger.info(new_report.summary())

    fixable = [
        r
        for r in new_report.get_missing_files() + new_report.get_empty_files()
        if not (r.metadata and r.metadata.get(key_field) in ignore_keys)
    ]
    if fixable and fix_fn is not None:
        fix_keys = [r.metadata[key_field] for r in fixable]  # type: ignore
        logger.info(f"Fixing {len(fix_keys)} items")
        fix_fn(fix_keys)

        # Allow time for writes to become visible (S3 eventual consistency,
        # mount sync). Retry validation a few times if still missing/empty.
        retry_report = None
        for attempt in range(post_fix_validation_retries + 1):
            if post_fix_delay_seconds > 0:
                logger.info(
                    f"Waiting {post_fix_delay_seconds:.1f}s for writes to become "
                    "visible before re-validation",
                )
                time.sleep(post_fix_delay_seconds)
            retry_results = list(validate_fn.map(fix_keys))
            retry_report = ValidationReport(results=retry_results)
            logger.info(retry_report.summary())
            if retry_report.missing_files == 0 and retry_report.empty_files == 0:
                break
            if attempt < post_fix_validation_retries:
                logger.warning(
                    f"Re-validation still had {retry_report.missing_files} missing, "
                    f"{retry_report.empty_files} empty; retrying after delay",
                )

        # Replace the original results for the retried items
        if retry_report is not None:
            retry_paths = {r.path for r in retry_report.results}
            kept = [r for r in new_report.results if r.path not in retry_paths]
            new_report = ValidationReport(results=kept + retry_report.results)

    if ignore_keys:
        marked = new_report.mark_known_issues(ignore_keys, key_field=key_field)
        if marked:
            logger.info(f"Marked {marked} results as known issues")

    merged_report = cached_report + new_report
    save_report_fn(report_path, merged_report)
    logger.info(
        f"Saved report to {report_path} ({merged_report.total_files} files, "
        f"{merged_report.ok_files} OK, {merged_report.known_issue_files} known issues)",
    )

    if not new_report.all_ok:
        problems = new_report.get_problem_files()
        raise Exception(
            f"Validation errors after fix: {len(problems)} unexpected failures "
            f"({new_report.unreadable_files} unreadable, "
            f"{new_report.missing_files} missing, "
            f"{new_report.empty_files} empty)"
        )

    return merged_report
