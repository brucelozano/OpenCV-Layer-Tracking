from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layer_velocity_rollout_config import EXPERIMENT_LABEL, OUTPUT_ROOT_DIR, TARGET_DATASETS


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = (OUTPUT_ROOT_DIR / EXPERIMENT_LABEL).resolve()
OUTPUT_DIR = INPUT_ROOT / "cross_dataset_summary"
CSV_FILENAME = "dsl_layer_speed_vertical_summary.csv"
MAX_LAYER_RANK = 4
MIN_DATASETS_PER_RANK = 2

SPEED_METHODS = {
    "endpoint": "speed_m_per_min_display",
    "regression": "speed_m_per_min_regression_display",
    "windowed": "speed_m_per_min_windowed_display",
    "primary": "primary_speed_m_per_min_display",
}

DIRECTION_COLORS = {
    "downward": "#2f6db3",
    "upward": "#d58a2f",
    "unknown": "#7f7f7f",
}


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _infer_direction(dataset_name: str) -> str:
    upper_name = str(dataset_name).upper()
    cast_match = re.search(r"\bB\d+[DN]\b", upper_name)
    if cast_match:
        cast_token = cast_match.group(0)
        if cast_token.endswith("D"):
            return "downward"
        if cast_token.endswith("N"):
            return "upward"

    for token in upper_name.split("_"):
        if token.endswith("D") and any(ch.isdigit() for ch in token):
            return "downward"
        if token.endswith("N") and any(ch.isdigit() for ch in token):
            return "upward"
    return "unknown"


def _timestamp_to_clock_hour(series: pd.Series) -> pd.Series:
    timestamp_series = pd.to_datetime(series, errors="coerce", utc=True)
    return (
        timestamp_series.dt.hour
        + (timestamp_series.dt.minute / 60.0)
        + (timestamp_series.dt.second / 3600.0)
    )


def _load_dataset_main_layers(dataset_spec: dict) -> pd.DataFrame:
    dataset_name = str(dataset_spec["dataset_name"])
    cruise_name = str(dataset_spec["cruise_name"])
    csv_path = INPUT_ROOT / cruise_name / dataset_name / CSV_FILENAME
    if not csv_path.exists():
        print(f"Missing CSV, skipping: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if "layer_type" not in df.columns:
        print(f"Missing layer_type column, skipping: {csv_path}")
        return pd.DataFrame()

    direction = _infer_direction(dataset_name)
    direction_flag_col = (
        "primary_opposes_expected_direction"
        if "primary_opposes_expected_direction" in df.columns
        else "opposes_cast_direction"
    )
    if direction_flag_col not in df.columns:
        df[direction_flag_col] = False

    main_df = df[
        (df["layer_type"] == "main")
        & (~_coerce_bool(df[direction_flag_col]))
    ].copy()
    if main_df.empty:
        # Fallback to all main layers if filtering leaves nothing.
        main_df = df[df["layer_type"] == "main"].copy()
    if main_df.empty:
        print(f"No main layers available for dataset: {dataset_name}")
        return pd.DataFrame()

    if "contour_start_time_utc" in main_df.columns:
        main_df["contour_start_time_utc"] = pd.to_datetime(
            main_df["contour_start_time_utc"], errors="coerce", utc=True
        )
        main_df = main_df.sort_values("contour_start_time_utc", na_position="last")
    main_df = main_df.reset_index(drop=True)
    main_df["layer_rank"] = np.arange(1, len(main_df) + 1)
    main_df = main_df[main_df["layer_rank"] <= MAX_LAYER_RANK].copy()

    numeric_cols = [
        "vertical_movement_m",
        "contour_duration_min",
        "duration_min",
        "motion_duration_min",
        "speed_m_per_min_display",
        "speed_m_per_min_regression_display",
        "speed_m_per_min_windowed_display",
        "primary_speed_m_per_min_display",
    ]
    for col in numeric_cols:
        if col in main_df.columns:
            main_df[col] = pd.to_numeric(main_df[col], errors="coerce")
        else:
            main_df[col] = np.nan

    # Legacy fallback for contour duration.
    if "contour_duration_min" not in main_df.columns or main_df["contour_duration_min"].isna().all():
        main_df["contour_duration_min"] = pd.to_numeric(main_df["duration_min"], errors="coerce")

    main_df["distance_abs_m"] = pd.to_numeric(
        main_df["vertical_movement_m"], errors="coerce"
    ).abs()

    for time_col in [
        "contour_start_time_utc",
        "contour_end_time_utc",
        "motion_start_time_utc",
        "motion_end_time_utc",
    ]:
        if time_col in main_df.columns:
            main_df[time_col] = pd.to_datetime(main_df[time_col], errors="coerce", utc=True)
        else:
            main_df[time_col] = pd.NaT

    main_df["contour_start_clock_hr"] = _timestamp_to_clock_hour(main_df["contour_start_time_utc"])
    main_df["contour_end_clock_hr"] = _timestamp_to_clock_hour(main_df["contour_end_time_utc"])
    main_df["motion_start_clock_hr"] = _timestamp_to_clock_hour(main_df["motion_start_time_utc"])
    main_df["motion_end_clock_hr"] = _timestamp_to_clock_hour(main_df["motion_end_time_utc"])

    main_df["dataset"] = dataset_name
    main_df["cruise"] = cruise_name
    main_df["direction"] = direction

    keep_cols = [
        "dataset",
        "cruise",
        "direction",
        "layer_rank",
        "distance_abs_m",
        "contour_duration_min",
        "motion_duration_min",
        "contour_start_clock_hr",
        "contour_end_clock_hr",
        "motion_start_clock_hr",
        "motion_end_clock_hr",
    ] + list(SPEED_METHODS.values())
    return main_df[keep_cols].copy()


def _aggregate_metric(records_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if metric_col not in records_df.columns:
        return pd.DataFrame()

    metric_df = records_df.copy()
    metric_df[metric_col] = pd.to_numeric(metric_df[metric_col], errors="coerce")
    metric_df = metric_df[np.isfinite(metric_df[metric_col])].copy()
    if metric_df.empty:
        return pd.DataFrame()

    agg_df = (
        metric_df.groupby(["direction", "layer_rank"], as_index=False)
        .agg(
            dataset_count=("dataset", "nunique"),
            mean_value=(metric_col, "mean"),
            std_value=(metric_col, "std"),
        )
        .sort_values(["direction", "layer_rank"])
    )
    agg_df["sem_value"] = (
        agg_df["std_value"] / np.sqrt(agg_df["dataset_count"])
    ).fillna(0.0)
    return agg_df[agg_df["dataset_count"] >= MIN_DATASETS_PER_RANK].copy()


def _plot_direction_metric(
    agg_df: pd.DataFrame,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if agg_df.empty:
        print(f"Skipping plot (no eligible records): {title}")
        return

    rank_union = sorted(agg_df["layer_rank"].astype(int).unique().tolist())
    if not rank_union:
        print(f"Skipping plot (no rank labels): {title}")
        return

    direction_order = [
        direction for direction in ["downward", "upward", "unknown"]
        if direction in agg_df["direction"].unique()
    ]
    if not direction_order:
        print(f"Skipping plot (no direction groups): {title}")
        return

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rank_union))
    width = 0.24 if len(direction_order) >= 3 else 0.34

    for idx, direction in enumerate(direction_order):
        direction_df = agg_df[agg_df["direction"] == direction]
        mean_lookup = dict(zip(direction_df["layer_rank"], direction_df["mean_value"]))
        sem_lookup = dict(zip(direction_df["layer_rank"], direction_df["sem_value"]))
        means = np.array([mean_lookup.get(rank, np.nan) for rank in rank_union], dtype=float)
        sems = np.array([sem_lookup.get(rank, 0.0) for rank in rank_union], dtype=float)
        valid = np.isfinite(means)

        x_pos = x + (idx - (len(direction_order) - 1) / 2.0) * width
        ax.bar(
            x_pos[valid],
            means[valid],
            yerr=sems[valid],
            width=width,
            capsize=5,
            color=DIRECTION_COLORS.get(direction, "#7f7f7f"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            error_kw={"elinewidth": 1.1},
            label=direction.capitalize(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {rank}" for rank in rank_union])
    ax.set_title(title)
    ax.set_xlabel("Layer rank")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _build_speed_method_long_records(records_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_name, metric_col in SPEED_METHODS.items():
        if metric_col not in records_df.columns:
            continue
        method_df = records_df[["dataset", "direction", "layer_rank", metric_col]].copy()
        method_df["speed_value"] = pd.to_numeric(method_df[metric_col], errors="coerce")
        method_df = method_df[np.isfinite(method_df["speed_value"])].copy()
        if method_df.empty:
            continue
        method_df["method"] = method_name
        rows.append(method_df[["dataset", "direction", "layer_rank", "method", "speed_value"]])

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _plot_speed_method_comparison(speed_long_df: pd.DataFrame, output_path: Path) -> None:
    if speed_long_df.empty:
        print("Skipping method comparison plot (no speed records).")
        return

    summary_df = (
        speed_long_df.groupby(["direction", "layer_rank", "method"], as_index=False)
        .agg(
            dataset_count=("dataset", "nunique"),
            mean_speed=("speed_value", "mean"),
            std_speed=("speed_value", "std"),
        )
    )
    summary_df["sem_speed"] = (
        summary_df["std_speed"] / np.sqrt(summary_df["dataset_count"])
    ).fillna(0.0)
    summary_df = summary_df[summary_df["dataset_count"] >= MIN_DATASETS_PER_RANK].copy()
    if summary_df.empty:
        print("Skipping method comparison plot (insufficient n).")
        return

    directions = [d for d in ["downward", "upward", "unknown"] if d in summary_df["direction"].unique()]
    if not directions:
        return

    _apply_plot_style()
    fig, axes = plt.subplots(1, len(directions), figsize=(5.2 * len(directions), 4.6), sharey=True)
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        direction_df = summary_df[summary_df["direction"] == direction]
        rank_union = sorted(direction_df["layer_rank"].astype(int).unique().tolist())
        for method_name in SPEED_METHODS.keys():
            method_df = direction_df[direction_df["method"] == method_name]
            if method_df.empty:
                continue
            lookup_mean = dict(zip(method_df["layer_rank"], method_df["mean_speed"]))
            lookup_sem = dict(zip(method_df["layer_rank"], method_df["sem_speed"]))
            means = np.array([lookup_mean.get(rank, np.nan) for rank in rank_union], dtype=float)
            sems = np.array([lookup_sem.get(rank, 0.0) for rank in rank_union], dtype=float)
            valid = np.isfinite(means)
            if not np.any(valid):
                continue
            x = np.array(rank_union, dtype=float)
            ax.errorbar(
                x[valid],
                means[valid],
                yerr=sems[valid],
                marker="o",
                linewidth=1.8,
                capsize=4,
                label=method_name.capitalize(),
            )

        ax.set_title(f"{direction.capitalize()} casts")
        ax.set_xlabel("Layer rank")
        ax.set_xticks(rank_union)
        ax.set_xticklabels([f"Layer {rank}" for rank in rank_union])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Speed (m/min)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Speed metric comparison by layer rank")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    dataset_status = []
    for dataset_spec in TARGET_DATASETS:
        dataset_name = str(dataset_spec["dataset_name"])
        cruise_name = str(dataset_spec["cruise_name"])
        csv_path = INPUT_ROOT / cruise_name / dataset_name / CSV_FILENAME
        dataset_status.append(
            {
                "dataset": dataset_name,
                "cruise": cruise_name,
                "csv_path": str(csv_path),
                "csv_exists": csv_path.exists(),
            }
        )
        dataset_df = _load_dataset_main_layers(dataset_spec)
        if dataset_df.empty:
            continue
        records.append(dataset_df)

    status_df = pd.DataFrame(dataset_status)
    status_df.to_csv(OUTPUT_DIR / "dataset_input_status.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'dataset_input_status.csv'}")

    if not records:
        print("No dataset records available to summarize.")
        return 1

    combined_df = pd.concat(records, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "combined_main_layer_records.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'combined_main_layer_records.csv'}")

    metric_specs = [
        ("primary_speed_m_per_min_display", "Primary migration speed by layer rank", "Speed (m/min)"),
        ("speed_m_per_min_display", "Endpoint speed by layer rank", "Speed (m/min)"),
        ("speed_m_per_min_regression_display", "Regression trend speed by layer rank", "Speed (m/min)"),
        ("speed_m_per_min_windowed_display", "Windowed speed by layer rank", "Speed (m/min)"),
        ("distance_abs_m", "Distance migrated by layer rank", "Distance migrated (m)"),
        ("contour_duration_min", "Contour-span duration by layer rank", "Duration (min)"),
        ("motion_duration_min", "Motion-onset duration by layer rank", "Duration (min)"),
        ("motion_start_clock_hr", "Motion start time by layer rank", "UTC clock time (hours)"),
        ("motion_end_clock_hr", "Motion end time by layer rank", "UTC clock time (hours)"),
    ]

    metric_aggregate_frames = []
    for metric_col, title, ylabel in metric_specs:
        agg_df = _aggregate_metric(combined_df, metric_col)
        if agg_df.empty:
            continue
        agg_df["metric"] = metric_col
        metric_aggregate_frames.append(agg_df)
        agg_df.to_csv(OUTPUT_DIR / f"{metric_col}_by_direction_layer.csv", index=False)
        _plot_direction_metric(
            agg_df=agg_df,
            title=title,
            ylabel=ylabel,
            output_path=OUTPUT_DIR / f"{metric_col}_by_direction_layer.png",
        )

    if metric_aggregate_frames:
        all_metrics_df = pd.concat(metric_aggregate_frames, ignore_index=True)
        all_metrics_df.to_csv(OUTPUT_DIR / "all_metric_aggregates.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'all_metric_aggregates.csv'}")

    speed_long_df = _build_speed_method_long_records(combined_df)
    if not speed_long_df.empty:
        speed_long_df.to_csv(OUTPUT_DIR / "speed_method_long_records.csv", index=False)
        speed_method_summary = (
            speed_long_df.groupby(["direction", "layer_rank", "method"], as_index=False)
            .agg(
                dataset_count=("dataset", "nunique"),
                mean_speed=("speed_value", "mean"),
                std_speed=("speed_value", "std"),
            )
            .sort_values(["direction", "layer_rank", "method"])
        )
        speed_method_summary["sem_speed"] = (
            speed_method_summary["std_speed"] / np.sqrt(speed_method_summary["dataset_count"])
        ).fillna(0.0)
        speed_method_summary.to_csv(OUTPUT_DIR / "speed_method_summary.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'speed_method_summary.csv'}")
        _plot_speed_method_comparison(
            speed_long_df=speed_long_df,
            output_path=OUTPUT_DIR / "speed_method_comparison_by_direction.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
