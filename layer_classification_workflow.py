from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap

from layer_classification_config import (
    CLASS_WINDOWS,
    DEBUG_DELTA_VMAX,
    DEBUG_DELTA_VMIN,
    DEBUG_PLOT_DPI,
    EXPERIMENT_LABEL,
    INPUT_ROOT_DIR,
    INVALID_SV_MIN,
    MASK_SOURCE_FREQUENCY_KHZ,
    ONLY_PAIR_KEYS_CONTAINING,
    OUTPUT_ROOT_DIR,
    REUSE_REVIEWED_MASKS,
    REVIEWED_CONTOURS_ROOT,
    RUN_TRACKING_IF_MASK_MISSING,
    SAVE_DEBUG_PLOTS,
    SKIP_PAIR_KEYS_CONTAINING,
    TRACKING_PARAMS_SEARCH_DIRS,
    MAX_ALIGNMENT_DELTA_SECONDS,
)
from sv_csv_loader import load_echoview_sv_csv


FIXED_COLUMN_NAMES = [
    "Ping_index",
    "Distance_gps",
    "Distance_vl",
    "Ping_date",
    "Ping_time",
    "Ping_milliseconds",
    "Latitude",
    "Longitude",
    "Depth_start",
    "Depth_stop",
    "Range_start",
    "Range_stop",
    "Sample_count",
]


@dataclass(frozen=True)
class FrequencyPair:
    pair_key: str
    file_18: Path
    file_38: Path
    cruise_name: str | None


@dataclass(frozen=True)
class LoadedDataset:
    dataset_name: str
    file_path: Path
    dataframe: pd.DataFrame
    image_data: np.ndarray  # depth x ping
    timestamps_utc: pd.DatetimeIndex
    ping_values: np.ndarray
    depth_axis_m: np.ndarray


def strip_sv_csv_suffix(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".sv.csv"):
        return filename[:-7]
    return Path(filename).stem


def infer_frequency_khz_from_name(filename: str) -> int | None:
    match = re.search(r"(?<!\d)(18|38)\s*k?hz(?!\d)", filename, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def pair_key_from_dataset_name(dataset_name: str) -> str:
    # Remove the first frequency token from the dataset name.
    return re.sub(r"(?<!\d)[_-]?(18|38)\s*k?hz(?!\d)", "", dataset_name, count=1, flags=re.IGNORECASE).strip("_- ")


def infer_cruise_name_from_path(file_path: Path, input_root: Path) -> str | None:
    try:
        rel = file_path.relative_to(input_root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass

    match = re.search(r"\bDP\d{2}\b", file_path.name.upper())
    return match.group(0) if match else None


def pair_key_passes_filters(pair_key: str) -> bool:
    if ONLY_PAIR_KEYS_CONTAINING and not any(token in pair_key for token in ONLY_PAIR_KEYS_CONTAINING):
        return False
    if SKIP_PAIR_KEYS_CONTAINING and any(token in pair_key for token in SKIP_PAIR_KEYS_CONTAINING):
        return False
    return True


def discover_frequency_pairs(input_root: Path) -> list[FrequencyPair]:
    grouped: dict[str, dict[int, Path]] = {}
    cruise_by_key: dict[str, str | None] = {}

    for csv_path in sorted(input_root.rglob("*.sv.csv")):
        dataset_name = strip_sv_csv_suffix(csv_path.name)
        frequency_khz = infer_frequency_khz_from_name(dataset_name)
        if frequency_khz not in {18, 38}:
            continue

        key = pair_key_from_dataset_name(dataset_name)
        if not key or not pair_key_passes_filters(key):
            continue

        grouped.setdefault(key, {})[frequency_khz] = csv_path
        cruise_by_key.setdefault(key, infer_cruise_name_from_path(csv_path, input_root))

    pairs: list[FrequencyPair] = []
    for pair_key, file_map in grouped.items():
        if 18 not in file_map or 38 not in file_map:
            continue
        pairs.append(
            FrequencyPair(
                pair_key=pair_key,
                file_18=file_map[18],
                file_38=file_map[38],
                cruise_name=cruise_by_key.get(pair_key),
            )
        )
    return pairs


def parse_ping_timestamps_utc(echogram: pd.DataFrame) -> pd.DatetimeIndex:
    datetime_str = (
        echogram["Ping_date"].astype(str).str.strip()
        + " "
        + echogram["Ping_time"].astype(str).str.strip()
    )
    timestamps = pd.to_datetime(datetime_str, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Failed to parse one or more ping timestamps from Ping_date/Ping_time.")

    ping_time_has_fractional_seconds = echogram["Ping_time"].astype(str).str.contains(r"\.").any()
    if "Ping_milliseconds" in echogram.columns and not ping_time_has_fractional_seconds:
        ms = pd.to_numeric(echogram["Ping_milliseconds"], errors="coerce").fillna(0.0)
        ms = ms.where((ms >= 0.0) & (ms < 1000.0), 0.0)
        timestamps = timestamps + pd.to_timedelta(ms, unit="ms")

    if getattr(timestamps.dt, "tz", None) is None:
        timestamps = timestamps.dt.tz_localize("UTC")
    else:
        timestamps = timestamps.dt.tz_convert("UTC")
    return pd.DatetimeIndex(timestamps)


def sanitize_sv_values(image_data: np.ndarray, invalid_sv_min: float) -> np.ndarray:
    cleaned = np.asarray(image_data, dtype=np.float32).copy()
    cleaned[~np.isfinite(cleaned)] = np.nan
    cleaned[cleaned <= float(invalid_sv_min)] = np.nan
    return cleaned


def load_sv_dataset(csv_path: Path) -> LoadedDataset:
    echogram_df, _, _ = load_echoview_sv_csv(
        csv_path=csv_path,
        fixed_column_names=FIXED_COLUMN_NAMES,
        sample_column_mode="auto",
        low_memory=False,
    )

    sample_columns = [
        col for col in echogram_df.columns if col.startswith("Sample_") and col != "Sample_count"
    ]
    sample_matrix = echogram_df[sample_columns].to_numpy(dtype=np.float32).T
    sample_matrix = sanitize_sv_values(sample_matrix, invalid_sv_min=INVALID_SV_MIN)

    timestamps_utc = parse_ping_timestamps_utc(echogram_df)
    ping_values = echogram_df["Ping_index"].to_numpy(dtype=np.float64)

    depth_start = float(echogram_df.iloc[0]["Depth_start"])
    depth_stop = float(echogram_df.iloc[0]["Depth_stop"])
    depth_axis = np.linspace(depth_start, depth_stop, sample_matrix.shape[0], endpoint=False, dtype=np.float64)

    return LoadedDataset(
        dataset_name=strip_sv_csv_suffix(csv_path.name),
        file_path=csv_path,
        dataframe=echogram_df,
        image_data=sample_matrix,
        timestamps_utc=timestamps_utc,
        ping_values=ping_values,
        depth_axis_m=depth_axis,
    )


def _deserialize_contours(serialized_contours: Any) -> list[np.ndarray]:
    contours: list[np.ndarray] = []
    if serialized_contours is None:
        return contours
    for contour_points in serialized_contours:
        contour_array = np.asarray(contour_points, dtype=np.int32)
        if contour_array.size == 0:
            continue
        contour_array = contour_array.reshape(-1, 1, 2)
        contours.append(contour_array)
    return contours


def load_reviewed_contours_artifact(artifact_path: Path) -> dict[str, Any] | None:
    if not artifact_path.exists():
        return None
    try:
        with np.load(artifact_path, allow_pickle=False) as payload_npz:
            if "payload_json" not in payload_npz:
                return None
            payload_json_array = payload_npz["payload_json"]
            payload_json = (
                str(payload_json_array.tolist())
                if np.ndim(payload_json_array) == 0
                else str(payload_json_array[0])
            )
        payload = json.loads(payload_json)
    except Exception:
        return None

    return {
        "metadata": payload.get("metadata", {}),
        "main_contours": _deserialize_contours(payload.get("main_contours")),
        "diffuse_contours": _deserialize_contours(payload.get("diffuse_contours")),
        "diffuse_contours_original": _deserialize_contours(payload.get("diffuse_contours_original")),
    }


def find_reviewed_contours_artifact(dataset_name: str, cruise_name: str | None) -> Path | None:
    candidate_dirs: list[Path] = []
    if cruise_name:
        base = REVIEWED_CONTOURS_ROOT / cruise_name / dataset_name
    else:
        base = REVIEWED_CONTOURS_ROOT / dataset_name

    candidate_dirs.extend(
        [
            base,
            base / "reviewed",
            base / f"{dataset_name}_reviewed",
        ]
    )

    for candidate_dir in candidate_dirs:
        artifact_path = candidate_dir / "reviewed_contours.npz"
        if artifact_path.exists():
            return artifact_path
    return None


def dataset_id_from_name(dataset_name: str) -> str:
    parts = dataset_name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return dataset_name


def find_params_file_for_dataset(dataset_name: str) -> Path | None:
    candidate_names = [f"params_{dataset_name}.py", f"params_{dataset_id_from_name(dataset_name)}.py"]
    for root_text in TRACKING_PARAMS_SEARCH_DIRS:
        root = Path(root_text)
        if not root.is_absolute():
            root = Path(__file__).resolve().parent / root
        if not root.exists():
            continue
        for candidate_name in candidate_names:
            exact_match = root / candidate_name
            if exact_match.exists():
                return exact_match.resolve()
            recursive_matches = list(root.rglob(candidate_name))
            if recursive_matches:
                return recursive_matches[0].resolve()
    return None


def run_tracking_fallback_for_dataset(
    dataset_name: str,
    cruise_name: str | None,
    csv_path: Path,
) -> bool:
    params_file = find_params_file_for_dataset(dataset_name)
    if params_file is None:
        print(f"Tracking fallback unavailable for {dataset_name}: no params file found.")
        return False

    project_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["ECHOGRAM_PARAMS_FILE"] = str(params_file)
    env["ECHOGRAM_INPUT_FILE"] = str(csv_path)
    env["ECHOGRAM_DATASET_NAME"] = dataset_name
    if cruise_name:
        env["ECHOGRAM_CRUISE_NAME"] = cruise_name
    env["ECHOGRAM_GROUP_OUTPUTS_BY_CRUISE"] = "1"
    env["ECHOGRAM_FIGURES_DIR"] = str(REVIEWED_CONTOURS_ROOT.resolve())
    env["ECHOGRAM_ENABLE_LAYER_REVIEW"] = "0"
    env["ECHOGRAM_LOAD_REVIEWED_CONTOURS"] = "0"
    env["ECHOGRAM_SAVE_REVIEWED_CONTOURS"] = "1"
    env["ECHOGRAM_SKIP_LAYER_REVIEW_IF_LOADED"] = "1"
    env["ECHOGRAM_REVIEWED_CONTOURS_SUBDIR"] = f"{dataset_name}_reviewed"

    cmd = [sys.executable, str(project_root / "echogram_processing.py")]
    result = subprocess.run(cmd, cwd=str(project_root), env=env)
    if result.returncode != 0:
        print(f"Tracking fallback failed for {dataset_name} (exit code {result.returncode}).")
        return False
    return True


def ensure_mask_artifact_for_pair(
    pair: FrequencyPair,
    mask_dataset_name: str,
    mask_dataset_file: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    if not REUSE_REVIEWED_MASKS:
        return None, None

    artifact_path = find_reviewed_contours_artifact(mask_dataset_name, pair.cruise_name)
    payload = load_reviewed_contours_artifact(artifact_path) if artifact_path else None
    if payload is not None:
        return artifact_path, payload

    if not RUN_TRACKING_IF_MASK_MISSING:
        return None, None

    print(f"Reviewed contours missing for {mask_dataset_name}; running tracking fallback.")
    ran = run_tracking_fallback_for_dataset(
        dataset_name=mask_dataset_name,
        cruise_name=pair.cruise_name,
        csv_path=mask_dataset_file,
    )
    if not ran:
        return None, None

    artifact_path = find_reviewed_contours_artifact(mask_dataset_name, pair.cruise_name)
    payload = load_reviewed_contours_artifact(artifact_path) if artifact_path else None
    return artifact_path, payload


def pick_contours_from_payload(payload: dict[str, Any]) -> list[np.ndarray]:
    main_contours = payload.get("main_contours") or []
    if main_contours:
        return list(main_contours)
    diffuse_original = payload.get("diffuse_contours_original") or []
    if diffuse_original:
        return list(diffuse_original)
    diffuse = payload.get("diffuse_contours") or []
    return list(diffuse)


def nearest_ping_alignment_indices(
    reference_times: np.ndarray,
    target_times: np.ndarray,
    tolerance_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (matched_target_indices, abs_time_delta_seconds).
    Unmatched references are -1 and NaN respectively.
    """
    if reference_times.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    if target_times.size == 0:
        return (
            np.full(reference_times.shape, -1, dtype=np.int64),
            np.full(reference_times.shape, np.nan, dtype=np.float64),
        )

    insert_idx = np.searchsorted(target_times, reference_times)
    left_idx = np.clip(insert_idx - 1, 0, max(target_times.size - 1, 0))
    right_idx = np.clip(insert_idx, 0, max(target_times.size - 1, 0))

    left_delta = np.abs(reference_times - target_times[left_idx])
    right_delta = np.abs(reference_times - target_times[right_idx])
    choose_right = right_delta < left_delta
    best_idx = np.where(choose_right, right_idx, left_idx)
    best_delta = np.where(choose_right, right_delta, left_delta)

    matched_idx = best_idx.astype(np.int64, copy=True)
    matched_idx[best_delta > tolerance_seconds] = -1
    best_delta[best_delta > tolerance_seconds] = np.nan
    return matched_idx, best_delta


def select_reference_mask_frequencies() -> list[int]:
    """
    Parse mask source selection from config/env.

    Supported examples:
      - 18
      - 38
      - "18"
      - "38"
      - "both"
      - "18,38" / "38,18"
    """
    raw = MASK_SOURCE_FREQUENCY_KHZ
    if isinstance(raw, (int, np.integer)):
        value = int(raw)
        if value in {18, 38}:
            return [value]
        print(f"Warning: MASK_SOURCE_FREQUENCY_KHZ={raw} is invalid; using 38.")
        return [38]

    text = str(raw).strip().lower()
    if text in {"both", "all", "18+38", "38+18", "18|38", "38|18"}:
        return [38, 18]

    token_matches = re.findall(r"(18|38)", text)
    if token_matches:
        selected: list[int] = []
        for token in token_matches:
            value = int(token)
            if value not in selected:
                selected.append(value)
        return selected if selected else [38]

    print(f"Warning: MASK_SOURCE_FREQUENCY_KHZ={raw} is invalid; using 38.")
    return [38]


def select_reference_mask_frequency() -> int:
    return int(select_reference_mask_frequencies()[0])


def _depth_bounds_to_indices(depth_axis_m: np.ndarray, start_m: float, stop_m: float) -> tuple[int, int]:
    lower = min(float(start_m), float(stop_m))
    upper = max(float(start_m), float(stop_m))
    start_idx = int(np.searchsorted(depth_axis_m, lower, side="left"))
    end_idx = int(np.searchsorted(depth_axis_m, upper, side="right"))
    start_idx = max(0, min(start_idx, depth_axis_m.size))
    end_idx = max(start_idx + 1, min(end_idx, depth_axis_m.size))
    return start_idx, end_idx


def prepare_reference_crop(
    reference_dataset: LoadedDataset,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    min_ping = float(metadata.get("min_ping")) if metadata.get("min_ping") is not None else float(reference_dataset.ping_values[0])
    max_ping = float(metadata.get("max_ping")) if metadata.get("max_ping") is not None else float(reference_dataset.ping_values[-1])
    depth_start = float(metadata.get("depth_start")) if metadata.get("depth_start") is not None else float(reference_dataset.depth_axis_m[0])
    depth_stop = float(metadata.get("depth_stop")) if metadata.get("depth_stop") is not None else float(reference_dataset.depth_axis_m[-1])

    ping_mask = (reference_dataset.ping_values >= min_ping) & (reference_dataset.ping_values <= max_ping)
    ref_ping_indices = np.where(ping_mask)[0]
    if ref_ping_indices.size == 0:
        raise ValueError(
            f"No pings in mask source range [{min_ping}, {max_ping}] for dataset {reference_dataset.dataset_name}."
        )

    depth_start_idx, depth_end_idx = _depth_bounds_to_indices(reference_dataset.depth_axis_m, depth_start, depth_stop)

    cropped_image = reference_dataset.image_data[depth_start_idx:depth_end_idx, :][:, ref_ping_indices]
    cropped_depth_axis = reference_dataset.depth_axis_m[depth_start_idx:depth_end_idx]
    cropped_timestamps = reference_dataset.timestamps_utc[ref_ping_indices]
    cropped_ping_values = reference_dataset.ping_values[ref_ping_indices]

    expected_shape = metadata.get("pass1_image_shape")
    source_shape = tuple(int(v) for v in expected_shape) if expected_shape is not None else tuple(cropped_image.shape)

    return {
        "min_ping": min_ping,
        "max_ping": max_ping,
        "depth_start": depth_start,
        "depth_stop": depth_stop,
        "depth_start_idx": depth_start_idx,
        "depth_end_idx": depth_end_idx,
        "ref_ping_indices": ref_ping_indices,
        "cropped_image": cropped_image,
        "cropped_depth_axis": cropped_depth_axis,
        "cropped_timestamps": cropped_timestamps,
        "cropped_ping_values": cropped_ping_values,
        "source_shape": source_shape,
    }


def align_dataset_to_reference(
    reference_times_utc: pd.DatetimeIndex,
    target_dataset: LoadedDataset,
    target_depth_axis: np.ndarray,
    tolerance_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      aligned_image (depth x reference_ping_count),
      matched_target_indices,
      abs_time_delta_seconds
    """
    reference_seconds = reference_times_utc.asi8.astype(np.float64) / 1e9
    target_seconds = target_dataset.timestamps_utc.asi8.astype(np.float64) / 1e9
    matched_idx, abs_delta_s = nearest_ping_alignment_indices(
        reference_times=reference_seconds,
        target_times=target_seconds,
        tolerance_seconds=float(tolerance_seconds),
    )

    aligned = np.full((target_depth_axis.size, reference_seconds.size), np.nan, dtype=np.float32)
    if target_depth_axis.size == 0 or reference_seconds.size == 0:
        return aligned, matched_idx, abs_delta_s

    if (
        target_dataset.depth_axis_m.size == target_depth_axis.size
        and np.allclose(target_dataset.depth_axis_m, target_depth_axis, atol=1e-6, rtol=0.0)
    ):
        valid = matched_idx >= 0
        if np.any(valid):
            aligned[:, valid] = target_dataset.image_data[:, matched_idx[valid]]
        return aligned, matched_idx, abs_delta_s

    valid_positions = np.where(matched_idx >= 0)[0]
    for ref_col in valid_positions:
        target_col = int(matched_idx[ref_col])
        source_profile = target_dataset.image_data[:, target_col]
        finite = np.isfinite(source_profile)
        if finite.sum() < 2:
            continue
        aligned[:, ref_col] = np.interp(
            target_depth_axis,
            target_dataset.depth_axis_m[finite],
            source_profile[finite],
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)

    return aligned, matched_idx, abs_delta_s


def scale_contour_to_shape(contour: np.ndarray, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray:
    source_h, source_w = source_shape
    target_h, target_w = target_shape
    if source_h <= 0 or source_w <= 0:
        return contour.astype(np.int32, copy=True)
    if source_shape == target_shape:
        return contour.astype(np.int32, copy=True)

    scaled = contour.astype(np.float64, copy=True)
    scale_x = 1.0 if source_w <= 1 else (target_w - 1) / (source_w - 1)
    scale_y = 1.0 if source_h <= 1 else (target_h - 1) / (source_h - 1)
    scaled[:, :, 0] = np.round(scaled[:, :, 0] * scale_x)
    scaled[:, :, 1] = np.round(scaled[:, :, 1] * scale_y)
    scaled[:, :, 0] = np.clip(scaled[:, :, 0], 0, max(target_w - 1, 0))
    scaled[:, :, 1] = np.clip(scaled[:, :, 1], 0, max(target_h - 1, 0))
    return scaled.astype(np.int32)


def build_layer_masks(
    contours: list[np.ndarray],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masks: list[np.ndarray] = []
    adjusted_contours: list[np.ndarray] = []
    for contour in contours:
        adjusted = scale_contour_to_shape(contour, source_shape=source_shape, target_shape=target_shape)
        layer_mask = np.zeros(target_shape, dtype=np.uint8)
        cv2.drawContours(layer_mask, [adjusted], -1, 1, thickness=cv2.FILLED)
        masks.append(layer_mask.astype(bool))
        adjusted_contours.append(adjusted)
    return masks, adjusted_contours


def _axis_extent(ping_values: np.ndarray, depth_axis_m: np.ndarray) -> list[float]:
    if ping_values.size == 0 or depth_axis_m.size == 0:
        return [0.0, 1.0, 1.0, 0.0]
    x0 = float(ping_values[0])
    x1 = float(ping_values[-1])
    if x0 == x1:
        x1 = x0 + 1.0
    y0 = float(depth_axis_m[0])
    y1 = float(depth_axis_m[-1])
    if y0 == y1:
        y1 = y0 + 1.0
    return [x0, x1, y1, y0]


def _finite_percentile_limits(
    arrays: list[np.ndarray],
    lower_pct: float = 2.0,
    upper_pct: float = 98.0,
    fallback: tuple[float, float] = (-80.0, -55.0),
) -> tuple[float, float]:
    finite_chunks: list[np.ndarray] = []
    for array in arrays:
        finite = array[np.isfinite(array)]
        if finite.size > 0:
            finite_chunks.append(finite.astype(np.float32, copy=False))
    if not finite_chunks:
        return fallback
    pooled = np.concatenate(finite_chunks)
    vmin = float(np.percentile(pooled, lower_pct))
    vmax = float(np.percentile(pooled, upper_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return fallback
    return vmin, vmax


def _overlay_scaled_contours(
    ax: Any,
    scaled_contours: list[np.ndarray],
    ping_values: np.ndarray,
    depth_axis_m: np.ndarray,
    color: str = "white",
) -> None:
    if ping_values.size == 0 or depth_axis_m.size == 0:
        return
    max_x_idx = ping_values.size - 1
    max_y_idx = depth_axis_m.size - 1
    for contour in scaled_contours:
        points = contour.reshape(-1, 2)
        x_idx = np.clip(points[:, 0].astype(int), 0, max_x_idx)
        y_idx = np.clip(points[:, 1].astype(int), 0, max_y_idx)
        x_vals = ping_values[x_idx]
        y_vals = depth_axis_m[y_idx]
        ax.plot(x_vals, y_vals, color=color, linewidth=0.9, alpha=0.9)


def _save_echogram_debug_plot(
    image_data: np.ndarray,
    ping_values: np.ndarray,
    depth_axis_m: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    scaled_contours: list[np.ndarray] | None = None,
) -> None:
    extent = _axis_extent(ping_values, depth_axis_m)
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        image_data,
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Ping index")
    ax.set_ylabel("Depth (m)")
    if scaled_contours:
        _overlay_scaled_contours(ax, scaled_contours, ping_values, depth_axis_m, color="white")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DEBUG_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _save_class_map_plot(
    delta_sv: np.ndarray,
    layer_masks: list[np.ndarray],
    ping_values: np.ndarray,
    depth_axis_m: np.ndarray,
    output_path: Path,
) -> None:
    if layer_masks:
        combined_mask = np.any(np.stack(layer_masks, axis=0), axis=0)
    else:
        combined_mask = np.zeros(delta_sv.shape, dtype=bool)
    finite = np.isfinite(delta_sv)
    class_code_map = np.full(delta_sv.shape, -1, dtype=np.int16)  # -1=outside mask
    assigned = np.zeros(delta_sv.shape, dtype=bool)

    for class_index, window in enumerate(CLASS_WINDOWS):
        min_inclusive = float(window["min_inclusive"])
        max_exclusive = float(window["max_exclusive"])
        max_inclusive = bool(window.get("max_inclusive", False))
        lower_ok = delta_sv >= min_inclusive
        upper_ok = (delta_sv <= max_exclusive) if max_inclusive else (delta_sv < max_exclusive)
        this_mask = combined_mask & finite & lower_ok & upper_ok
        class_code_map[this_mask] = class_index
        assigned |= this_mask

    unclassified_mask = combined_mask & finite & (~assigned)
    class_code_map[unclassified_mask] = len(CLASS_WINDOWS)
    missing_in_mask = combined_mask & (~finite)
    class_code_map[missing_in_mask] = len(CLASS_WINDOWS) + 1

    # Shift codes to non-negative values so outside-mask can be shown too.
    plot_codes = class_code_map + 1
    labels = (
        ["outside_mask"]
        + [str(window["class_name"]) for window in CLASS_WINDOWS]
        + ["unclassified", "missing_in_mask"]
    )
    colors = [
        "#202020",  # outside_mask
        "#f4c430",  # swim bladdered
        "#4c78a8",  # large non-SB
        "#b279a2",  # small non-SB + crustaceans
        "#ff7f0e",  # unclassified
        "#9e9e9e",  # missing in mask
    ]
    cmap = ListedColormap(colors)

    extent = _axis_extent(ping_values, depth_axis_m)
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        plot_codes,
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(labels) - 0.5,
        interpolation="nearest",
    )
    ax.set_title("Delta Sv class map")
    ax.set_xlabel("Ping index")
    ax.set_ylabel("Depth (m)")

    legend_handles = [
        mpatches.Patch(color=colors[idx], label=labels[idx]) for idx in range(len(labels))
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Class index")
    cbar.set_ticks(list(range(len(labels))))
    cbar.set_ticklabels(labels)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DEBUG_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _save_layer_histograms(
    delta_sv: np.ndarray,
    layer_masks: list[np.ndarray],
    output_dir: Path,
) -> None:
    threshold_lines = [-14.0, -3.0, 0.0, 12.0]
    for layer_idx, layer_mask in enumerate(layer_masks, start=1):
        values = delta_sv[layer_mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(values, bins=120, color="#4c78a8", alpha=0.85)
        for threshold in threshold_lines:
            ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.0)
        ax.set_title(f"Layer {layer_idx} delta Sv distribution")
        ax.set_xlabel("Delta Sv (18kHz - 38kHz)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"debug_layer_{layer_idx}_delta_histogram.png", dpi=DEBUG_PLOT_DPI, bbox_inches="tight")
        plt.close(fig)


def save_debug_plots(
    pair_output_dir: Path,
    pair_key: str,
    reference_dataset_name: str,
    reference_khz: int,
    sv18: np.ndarray,
    sv38: np.ndarray,
    delta_sv: np.ndarray,
    ping_values: np.ndarray,
    depth_axis_m: np.ndarray,
    scaled_contours: list[np.ndarray],
    layer_masks: list[np.ndarray],
) -> None:
    debug_dir = pair_output_dir / "debug_plots"
    debug_dir.mkdir(parents=True, exist_ok=True)

    sv_vmin, sv_vmax = _finite_percentile_limits([sv18, sv38], lower_pct=2.0, upper_pct=98.0)
    _save_echogram_debug_plot(
        image_data=sv18,
        ping_values=ping_values,
        depth_axis_m=depth_axis_m,
        output_path=debug_dir / "debug_aligned_18khz_sv.png",
        title=f"{pair_key} aligned 18kHz Sv",
        colorbar_label="Sv (dB)",
        cmap="viridis",
        vmin=sv_vmin,
        vmax=sv_vmax,
    )
    _save_echogram_debug_plot(
        image_data=sv38,
        ping_values=ping_values,
        depth_axis_m=depth_axis_m,
        output_path=debug_dir / "debug_aligned_38khz_sv.png",
        title=f"{pair_key} aligned 38kHz Sv",
        colorbar_label="Sv (dB)",
        cmap="viridis",
        vmin=sv_vmin,
        vmax=sv_vmax,
    )
    _save_echogram_debug_plot(
        image_data=sv38 if reference_khz == 38 else sv18,
        ping_values=ping_values,
        depth_axis_m=depth_axis_m,
        output_path=debug_dir / "debug_mask_source_with_reviewed_contour.png",
        title=f"{pair_key} mask source ({reference_dataset_name}) with reviewed contour",
        colorbar_label="Sv (dB)",
        cmap="viridis",
        vmin=sv_vmin,
        vmax=sv_vmax,
        scaled_contours=scaled_contours,
    )
    _save_echogram_debug_plot(
        image_data=delta_sv,
        ping_values=ping_values,
        depth_axis_m=depth_axis_m,
        output_path=debug_dir / "debug_delta_sv_full.png",
        title=f"{pair_key} delta Sv (18kHz - 38kHz)",
        colorbar_label="Delta Sv (dB)",
        cmap="coolwarm",
        vmin=DEBUG_DELTA_VMIN,
        vmax=DEBUG_DELTA_VMAX,
        scaled_contours=scaled_contours,
    )

    if layer_masks:
        combined_mask = np.any(np.stack(layer_masks, axis=0), axis=0)
        masked_delta = np.where(combined_mask, delta_sv, np.nan)
        _save_echogram_debug_plot(
            image_data=masked_delta,
            ping_values=ping_values,
            depth_axis_m=depth_axis_m,
            output_path=debug_dir / "debug_delta_sv_masked.png",
            title=f"{pair_key} delta Sv inside layer mask",
            colorbar_label="Delta Sv (dB)",
            cmap="coolwarm",
            vmin=DEBUG_DELTA_VMIN,
            vmax=DEBUG_DELTA_VMAX,
            scaled_contours=scaled_contours,
        )

    _save_class_map_plot(
        delta_sv=delta_sv,
        layer_masks=layer_masks,
        ping_values=ping_values,
        depth_axis_m=depth_axis_m,
        output_path=debug_dir / "debug_class_map.png",
    )
    _save_layer_histograms(delta_sv=delta_sv, layer_masks=layer_masks, output_dir=debug_dir)


def save_dual_mask_class_map_comparison(pair_output_dir: Path) -> Path | None:
    """
    Build a vertically stacked class-map comparison image for dual-mask runs.
    Top panel: 18 kHz mask-source class map
    Bottom panel: 38 kHz mask-source class map
    """
    class_map_18 = pair_output_dir / "mask_source_18khz" / "debug_plots" / "debug_class_map.png"
    class_map_38 = pair_output_dir / "mask_source_38khz" / "debug_plots" / "debug_class_map.png"
    if not class_map_18.exists() or not class_map_38.exists():
        return None

    image_18 = plt.imread(class_map_18)
    image_38 = plt.imread(class_map_38)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    axes[0].imshow(image_18)
    axes[0].axis("off")
    axes[0].set_title("Mask source: 18 kHz", fontsize=13)

    axes[1].imshow(image_38)
    axes[1].axis("off")
    axes[1].set_title("Mask source: 38 kHz", fontsize=13)

    fig.suptitle("Delta Sv Class Map Comparison", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    comparison_path = pair_output_dir / "debug_class_map_comparison_18_top_38_bottom.png"
    fig.savefig(comparison_path, dpi=DEBUG_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return comparison_path


def classify_delta_values(delta_values: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    finite = np.isfinite(delta_values)
    class_masks: dict[str, np.ndarray] = {}
    assigned = np.zeros(delta_values.shape, dtype=bool)

    for window in CLASS_WINDOWS:
        class_name = str(window["class_name"])
        min_inclusive = float(window["min_inclusive"])
        max_exclusive = float(window["max_exclusive"])
        max_inclusive = bool(window.get("max_inclusive", False))

        lower_ok = delta_values >= min_inclusive
        if max_inclusive:
            upper_ok = delta_values <= max_exclusive
        else:
            upper_ok = delta_values < max_exclusive
        mask = finite & lower_ok & upper_ok
        class_masks[class_name] = mask
        assigned |= mask

    unclassified = finite & (~assigned)
    return class_masks, unclassified


def _safe_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size > 0 else float("nan")


def _safe_median(values: np.ndarray) -> float:
    return float(np.nanmedian(values)) if values.size > 0 else float("nan")


def summarize_layer(
    pair_key: str,
    layer_index: int,
    layer_mask: np.ndarray,
    delta_sv: np.ndarray,
    matched_count: int,
    unmatched_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    layer_values = delta_sv[layer_mask]
    finite_mask = np.isfinite(layer_values)
    finite_values = layer_values[finite_mask]

    class_masks, unclassified_mask = classify_delta_values(layer_values)
    class_value_map: dict[str, np.ndarray] = {}

    row: dict[str, Any] = {
        "pair_key": pair_key,
        "layer_index": layer_index,
        "masked_sample_count": int(layer_mask.sum()),
        "finite_delta_count": int(finite_mask.sum()),
        "missing_delta_count": int(layer_mask.sum() - finite_mask.sum()),
        "matched_reference_pings": int(matched_count),
        "unmatched_reference_pings": int(unmatched_count),
    }

    denom = float(finite_values.size) if finite_values.size > 0 else np.nan

    for class_name, class_mask in class_masks.items():
        class_values = layer_values[class_mask]
        class_value_map[class_name] = class_values
        row[f"{class_name}_count"] = int(class_values.size)
        row[f"{class_name}_percent"] = float(class_values.size / denom * 100.0) if np.isfinite(denom) else np.nan
        row[f"{class_name}_mean_delta_sv"] = _safe_mean(class_values)
        row[f"{class_name}_median_delta_sv"] = _safe_median(class_values)

    unclassified_values = layer_values[unclassified_mask]
    class_value_map["unclassified"] = unclassified_values
    row["unclassified_count"] = int(unclassified_values.size)
    row["unclassified_percent"] = (
        float(unclassified_values.size / denom * 100.0) if np.isfinite(denom) else np.nan
    )
    row["unclassified_mean_delta_sv"] = _safe_mean(unclassified_values)
    row["unclassified_median_delta_sv"] = _safe_median(unclassified_values)

    return row, class_value_map


def build_dataset_summary_row(
    pair_key: str,
    layer_rows: list[dict[str, Any]],
    class_values_by_name: dict[str, list[np.ndarray]],
    alignment_total: int,
    alignment_matched: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pair_key": pair_key,
        "layer_count": int(len(layer_rows)),
        "alignment_reference_ping_count": int(alignment_total),
        "alignment_matched_ping_count": int(alignment_matched),
        "alignment_unmatched_ping_count": int(alignment_total - alignment_matched),
        "alignment_match_percent": (
            float((alignment_matched / alignment_total) * 100.0) if alignment_total > 0 else np.nan
        ),
        "total_masked_sample_count": int(sum(int(v["masked_sample_count"]) for v in layer_rows)),
        "total_finite_delta_count": int(sum(int(v["finite_delta_count"]) for v in layer_rows)),
        "total_missing_delta_count": int(sum(int(v["missing_delta_count"]) for v in layer_rows)),
    }

    finite_total = row["total_finite_delta_count"]
    for class_name in [w["class_name"] for w in CLASS_WINDOWS] + ["unclassified"]:
        chunks = class_values_by_name.get(class_name, [])
        class_values = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        count = int(class_values.size)
        row[f"{class_name}_count"] = count
        row[f"{class_name}_percent"] = float((count / finite_total) * 100.0) if finite_total > 0 else np.nan
        row[f"{class_name}_mean_delta_sv"] = _safe_mean(class_values)
        row[f"{class_name}_median_delta_sv"] = _safe_median(class_values)
    return row


def classify_frequency_pair(
    pair: FrequencyPair,
    output_root: Path,
    reference_khz: int | None = None,
    pair_output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    dataset_18 = load_sv_dataset(pair.file_18)
    dataset_38 = load_sv_dataset(pair.file_38)

    if reference_khz is None:
        reference_khz = select_reference_mask_frequency()
    if reference_khz == 18:
        reference_dataset = dataset_18
        other_dataset = dataset_38
    else:
        reference_dataset = dataset_38
        other_dataset = dataset_18

    artifact_path, payload = ensure_mask_artifact_for_pair(
        pair=pair,
        mask_dataset_name=reference_dataset.dataset_name,
        mask_dataset_file=reference_dataset.file_path,
    )
    if payload is None:
        raise RuntimeError(
            f"No reviewed contour artifact available for mask source dataset {reference_dataset.dataset_name}."
        )

    contours = pick_contours_from_payload(payload)
    if not contours:
        raise RuntimeError(f"No contours found in reviewed artifact for pair {pair.pair_key}.")

    metadata = payload.get("metadata", {}) or {}
    crop_context = prepare_reference_crop(reference_dataset, metadata=metadata)

    ref_cropped = crop_context["cropped_image"]
    ref_times = crop_context["cropped_timestamps"]
    ref_depth_axis = crop_context["cropped_depth_axis"]
    source_shape = crop_context["source_shape"]
    target_shape = tuple(ref_cropped.shape)

    aligned_other, matched_other_idx, abs_delta_s = align_dataset_to_reference(
        reference_times_utc=ref_times,
        target_dataset=other_dataset,
        target_depth_axis=ref_depth_axis,
        tolerance_seconds=MAX_ALIGNMENT_DELTA_SECONDS,
    )

    if reference_khz == 18:
        sv18 = ref_cropped
        sv38 = aligned_other
    else:
        sv18 = aligned_other
        sv38 = ref_cropped
    delta_sv = sv18 - sv38

    layer_masks, scaled_contours = build_layer_masks(
        contours=contours,
        source_shape=source_shape,
        target_shape=target_shape,
    )
    matched_ping_count = int(np.sum(matched_other_idx >= 0))
    unmatched_ping_count = int(np.sum(matched_other_idx < 0))

    if pair_output_dir is None:
        pair_output_dir = output_root / pair.pair_key
    pair_output_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG_PLOTS:
        save_debug_plots(
            pair_output_dir=pair_output_dir,
            pair_key=pair.pair_key,
            reference_dataset_name=reference_dataset.dataset_name,
            reference_khz=int(reference_khz),
            sv18=sv18,
            sv38=sv38,
            delta_sv=delta_sv,
            ping_values=crop_context["cropped_ping_values"],
            depth_axis_m=ref_depth_axis,
            scaled_contours=scaled_contours,
            layer_masks=layer_masks,
        )

    layer_rows: list[dict[str, Any]] = []
    class_values_by_name: dict[str, list[np.ndarray]] = {
        w["class_name"]: [] for w in CLASS_WINDOWS
    }
    class_values_by_name["unclassified"] = []

    for layer_idx, layer_mask in enumerate(layer_masks, start=1):
        layer_row, class_values = summarize_layer(
            pair_key=pair.pair_key,
            layer_index=layer_idx,
            layer_mask=layer_mask,
            delta_sv=delta_sv,
            matched_count=matched_ping_count,
            unmatched_count=unmatched_ping_count,
        )
        layer_row["dataset_18"] = dataset_18.dataset_name
        layer_row["dataset_38"] = dataset_38.dataset_name
        layer_row["mask_source_dataset"] = reference_dataset.dataset_name
        layer_row["mask_source_frequency_khz"] = int(reference_khz)
        layer_row["reviewed_contours_artifact"] = str(artifact_path) if artifact_path else ""
        layer_rows.append(layer_row)

        for class_name, values in class_values.items():
            class_values_by_name[class_name].append(values.astype(np.float32, copy=False))

    dataset_row = build_dataset_summary_row(
        pair_key=pair.pair_key,
        layer_rows=layer_rows,
        class_values_by_name=class_values_by_name,
        alignment_total=int(len(matched_other_idx)),
        alignment_matched=matched_ping_count,
    )
    dataset_row["dataset_18"] = dataset_18.dataset_name
    dataset_row["dataset_38"] = dataset_38.dataset_name
    dataset_row["mask_source_dataset"] = reference_dataset.dataset_name
    dataset_row["mask_source_frequency_khz"] = int(reference_khz)
    dataset_row["reviewed_contours_artifact"] = str(artifact_path) if artifact_path else ""
    dataset_row["cruise_name"] = pair.cruise_name or ""

    alignment_rows: list[dict[str, Any]] = []
    ref_ping_values = crop_context["cropped_ping_values"]
    other_ping_values = other_dataset.ping_values
    for col_idx in range(len(ref_ping_values)):
        matched_idx = int(matched_other_idx[col_idx])
        alignment_rows.append(
            {
                "pair_key": pair.pair_key,
                "reference_dataset": reference_dataset.dataset_name,
                "target_dataset": other_dataset.dataset_name,
                "mask_source_frequency_khz": int(reference_khz),
                "reference_ping_index": float(ref_ping_values[col_idx]),
                "reference_time_utc": ref_times[col_idx].isoformat(),
                "matched": bool(matched_idx >= 0),
                "matched_ping_index": float(other_ping_values[matched_idx]) if matched_idx >= 0 else np.nan,
                "matched_time_utc": (
                    other_dataset.timestamps_utc[matched_idx].isoformat() if matched_idx >= 0 else ""
                ),
                "abs_time_delta_s": float(abs_delta_s[col_idx]) if np.isfinite(abs_delta_s[col_idx]) else np.nan,
                "tolerance_s": float(MAX_ALIGNMENT_DELTA_SECONDS),
            }
        )

    pd.DataFrame(layer_rows).to_csv(pair_output_dir / "layer_classification_layer_summary.csv", index=False)
    pd.DataFrame([dataset_row]).to_csv(pair_output_dir / "layer_classification_dataset_summary.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(pair_output_dir / "layer_classification_alignment_diagnostics.csv", index=False)

    return layer_rows, dataset_row, alignment_rows


def main() -> int:
    input_root = Path(INPUT_ROOT_DIR).resolve()
    if not input_root.exists():
        print(f"Layer classification input directory does not exist: {input_root}")
        return 1

    output_root = (Path(OUTPUT_ROOT_DIR) / EXPERIMENT_LABEL).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_frequency_pairs(input_root)
    if not pairs:
        print(f"No complete 18/38 pairs found under: {input_root}")
        return 1

    selected_mask_sources = select_reference_mask_frequencies()
    run_multi_mask_mode = len(selected_mask_sources) > 1

    print(
        f"Found {len(pairs)} frequency pairs for layer classification."
        f" Mask source(s): {selected_mask_sources}"
    )
    all_layer_rows: list[dict[str, Any]] = []
    all_dataset_rows: list[dict[str, Any]] = []
    all_alignment_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for idx, pair in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] Processing pair: {pair.pair_key}")
        for mask_khz in selected_mask_sources:
            scenario_name = f"{pair.pair_key} [mask {mask_khz}kHz]"
            scenario_output_dir = (
                output_root / pair.pair_key / f"mask_source_{mask_khz}khz"
                if run_multi_mask_mode
                else output_root / pair.pair_key
            )
            try:
                layer_rows, dataset_row, alignment_rows = classify_frequency_pair(
                    pair,
                    output_root=output_root,
                    reference_khz=int(mask_khz),
                    pair_output_dir=scenario_output_dir,
                )
            except Exception as exc:
                print(f"  FAILED: {scenario_name}: {exc}")
                errors.append(
                    {
                        "pair_key": pair.pair_key,
                        "mask_source_frequency_khz": str(mask_khz),
                        "error": str(exc),
                    }
                )
                continue

            all_layer_rows.extend(layer_rows)
            all_dataset_rows.append(dataset_row)
            all_alignment_rows.extend(alignment_rows)
            print(f"  Completed: {scenario_name} ({len(layer_rows)} layers)")

        if run_multi_mask_mode and SAVE_DEBUG_PLOTS:
            comparison_path = save_dual_mask_class_map_comparison(output_root / pair.pair_key)
            if comparison_path is not None:
                print(f"  Saved: {comparison_path}")

    layer_summary_path = output_root / "layer_classification_layer_summary.csv"
    dataset_summary_path = output_root / "layer_classification_dataset_summary.csv"
    alignment_summary_path = output_root / "layer_classification_alignment_diagnostics.csv"
    metadata_path = output_root / "layer_classification_run_metadata.json"

    pd.DataFrame(all_layer_rows).to_csv(layer_summary_path, index=False)
    pd.DataFrame(all_dataset_rows).to_csv(dataset_summary_path, index=False)
    pd.DataFrame(all_alignment_rows).to_csv(alignment_summary_path, index=False)

    metadata = {
        "input_root_dir": str(input_root),
        "output_root_dir": str(output_root),
        "pair_count_found": int(len(pairs)),
        "pair_count_completed": int(len(all_dataset_rows)),
        "pair_count_failed": int(len(errors)),
        "errors": errors,
        "mask_source_frequency_khz": int(select_reference_mask_frequency()),
        "mask_source_frequency_khz_raw": str(MASK_SOURCE_FREQUENCY_KHZ),
        "mask_source_frequencies_used": [int(v) for v in selected_mask_sources],
        "run_multi_mask_mode": bool(run_multi_mask_mode),
        "reuse_reviewed_masks": bool(REUSE_REVIEWED_MASKS),
        "run_tracking_if_mask_missing": bool(RUN_TRACKING_IF_MASK_MISSING),
        "save_debug_plots": bool(SAVE_DEBUG_PLOTS),
        "debug_plot_dpi": int(DEBUG_PLOT_DPI),
        "max_alignment_delta_seconds": float(MAX_ALIGNMENT_DELTA_SECONDS),
        "debug_delta_vmin": float(DEBUG_DELTA_VMIN),
        "debug_delta_vmax": float(DEBUG_DELTA_VMAX),
        "class_windows": CLASS_WINDOWS,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)

    print(f"Saved: {layer_summary_path}")
    print(f"Saved: {dataset_summary_path}")
    print(f"Saved: {alignment_summary_path}")
    print(f"Saved: {metadata_path}")

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
