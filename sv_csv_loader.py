from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from pandas.errors import ParserError

VALID_SAMPLE_COLUMN_MODES = {"auto", "strict", "ragged"}


def normalize_sample_column_mode(
    value: str | None,
    context_label: str = "SAMPLE_COLUMN_MODE",
    logger: Callable[[str], None] | None = print,
) -> str:
    normalized = str(value).strip().lower() if value is not None else "auto"
    if normalized in VALID_SAMPLE_COLUMN_MODES:
        return normalized

    if logger is not None:
        logger(
            f"Warning: {context_label}={value!r} is invalid. "
            "Using 'auto' (supported: 'auto', 'strict', 'ragged')."
        )
    return "auto"


def _parse_sample_count(raw_value: str, line_no: int) -> int:
    try:
        sample_count_float = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid Sample_count value at line {line_no}: {raw_value!r}"
        ) from exc

    if not sample_count_float.is_integer():
        raise ValueError(
            f"Sample_count must be an integer value at line {line_no}: {raw_value!r}"
        )

    sample_count = int(sample_count_float)
    if sample_count < 0:
        raise ValueError(
            f"Sample_count must be non-negative at line {line_no}: {raw_value!r}"
        )
    return sample_count


def _parse_sample_value(raw_value: str, line_no: int, sample_idx: int) -> float:
    trimmed = raw_value.strip()
    if trimmed == "":
        return np.nan
    try:
        return float(trimmed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid sample value at line {line_no}, sample {sample_idx}: {raw_value!r}"
        ) from exc


def _read_ragged_with_validation(
    csv_path: Path,
    fixed_df: pd.DataFrame,
    fixed_column_names: Sequence[str],
    sample_column_names: Sequence[str],
) -> pd.DataFrame:
    fixed_col_count = len(fixed_column_names)
    max_samples = len(sample_column_names)

    sample_rows: list[list[float]] = []
    mismatch_examples: list[tuple[int, int, int]] = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Input CSV is empty: {csv_path}")
        if "Sample_count" not in header:
            raise ValueError(f"Missing required 'Sample_count' column in header: {csv_path}")

        sample_count_idx = header.index("Sample_count")

        for line_no, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue

            if sample_count_idx >= len(row):
                mismatch_examples.append((line_no, len(row), fixed_col_count))
                continue

            sample_count = _parse_sample_count(row[sample_count_idx], line_no)
            expected_len = fixed_col_count + sample_count
            row_len = len(row)
            if row_len != expected_len:
                mismatch_examples.append((line_no, row_len, expected_len))
                continue

            raw_samples = row[fixed_col_count:]
            parsed_samples = [
                _parse_sample_value(raw_sample, line_no, sample_idx + 1)
                for sample_idx, raw_sample in enumerate(raw_samples)
            ]
            if len(parsed_samples) < max_samples:
                parsed_samples.extend([np.nan] * (max_samples - len(parsed_samples)))
            sample_rows.append(parsed_samples)

    if mismatch_examples:
        snippets = ", ".join(
            f"line {line_no}: saw {row_len}, expected {expected_len}"
            for line_no, row_len, expected_len in mismatch_examples[:5]
        )
        raise ValueError(
            "Ragged parser validation failed: row width does not match "
            "fixed columns + Sample_count. "
            f"Examples: {snippets}"
        )

    if len(sample_rows) != len(fixed_df):
        raise ValueError(
            "Sample matrix row count does not match fixed-column row count "
            f"({len(sample_rows)} vs {len(fixed_df)})."
        )

    sample_matrix = np.asarray(sample_rows, dtype=np.float32)
    sample_df = pd.DataFrame(sample_matrix, columns=sample_column_names)

    return pd.concat([fixed_df.reset_index(drop=True), sample_df], axis=1)


def load_echoview_sv_csv(
    csv_path: str | Path,
    fixed_column_names: Sequence[str],
    sample_column_mode: str = "auto",
    low_memory: bool = False,
    logger: Callable[[str], None] | None = print,
) -> tuple[pd.DataFrame, int, str]:
    csv_path = Path(csv_path)
    fixed_column_names = list(fixed_column_names)
    fixed_df = pd.read_csv(csv_path, usecols=fixed_column_names)
    max_samples = int(fixed_df["Sample_count"].max())
    sample_column_names = [f"Sample_{idx}" for idx in range(1, max_samples + 1)]
    all_column_names = fixed_column_names + sample_column_names

    normalized_mode = normalize_sample_column_mode(
        sample_column_mode,
        context_label="sample_column_mode",
        logger=logger,
    )
    resolved_mode = normalized_mode

    if normalized_mode in {"auto", "strict"}:
        try:
            echogram_df = pd.read_csv(
                csv_path,
                names=all_column_names,
                header=0,
                low_memory=low_memory,
            )
            resolved_mode = "strict"
        except ParserError as parser_error:
            if normalized_mode == "strict":
                raise
            if logger is not None:
                logger(
                    "Strict sample-column parser failed; retrying with ragged parser. "
                    f"Parser error: {parser_error}"
                )
            echogram_df = _read_ragged_with_validation(
                csv_path=csv_path,
                fixed_df=fixed_df,
                fixed_column_names=fixed_column_names,
                sample_column_names=sample_column_names,
            )
            resolved_mode = "ragged"
    else:
        echogram_df = _read_ragged_with_validation(
            csv_path=csv_path,
            fixed_df=fixed_df,
            fixed_column_names=fixed_column_names,
            sample_column_names=sample_column_names,
        )
        resolved_mode = "ragged"

    return echogram_df, max_samples, resolved_mode
