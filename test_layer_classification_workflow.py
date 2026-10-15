from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import layer_classification_workflow as workflow


FIXED_HEADER = [
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


def _write_tiny_sv_csv(csv_path: Path, ping_times_ms: list[int], sample_rows: list[list[float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = len(sample_rows[0])
    sample_headers = [f"Sample_{idx}" for idx in range(1, sample_count + 1)]
    all_headers = FIXED_HEADER + sample_headers

    rows = []
    for idx, (ms, samples) in enumerate(zip(ping_times_ms, sample_rows), start=100):
        total_ms = int(ms)
        whole_seconds = total_ms // 1000
        subsecond_ms = total_ms % 1000
        ping_time_text = f"00:00:{whole_seconds:02d}"
        row = [
            idx,
            0.0,
            0.0,
            "2026-01-01",
            ping_time_text,
            float(subsecond_ms),
            0.0,
            0.0,
            0.0,
            4.0,
            0.0,
            4.0,
            sample_count,
        ] + samples
        rows.append(row)

    df = pd.DataFrame(rows, columns=all_headers)
    df.to_csv(csv_path, index=False)


def _write_reviewed_artifact(artifact_path: Path, contour_shape: tuple[int, int]) -> None:
    contour_h, contour_w = contour_shape
    contour = np.array(
        [
            [[0, 0]],
            [[contour_w - 1, 0]],
            [[contour_w - 1, contour_h - 1]],
            [[0, contour_h - 1]],
        ],
        dtype=np.int32,
    ).tolist()
    payload = {
        "metadata": {
            "dataset_name": "TESTCAST_38kHz",
            "scope": "main",
            "pass1_image_shape": [contour_h, contour_w],
            "pass2_image_shape": None,
            "min_ping": 100.0,
            "max_ping": 101.0,
            "second_pass_min_ping": None,
            "second_pass_max_ping": None,
            "depth_start": 0.0,
            "depth_stop": 4.0,
            "saved_utc": "2026-01-01T00:00:00+00:00",
        },
        "main_contours": [contour],
        "diffuse_contours": [],
        "diffuse_contours_original": [],
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, ensure_ascii=True)
    np.savez_compressed(artifact_path, payload_json=np.array([payload_json], dtype=np.str_))


def test_alignment_nearest_with_tolerance() -> None:
    ref = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    target = np.array([0.3, 5.2, 15.0], dtype=np.float64)
    matched, delta = workflow.nearest_ping_alignment_indices(ref, target, tolerance_seconds=1.0)
    assert matched.tolist() == [0, 1, -1]
    assert np.isfinite(delta[0]) and abs(delta[0] - 0.3) < 1e-6
    assert np.isfinite(delta[1]) and abs(delta[1] - 0.2) < 1e-6
    assert np.isnan(delta[2])


def test_threshold_boundaries_half_open_bins() -> None:
    values = np.array([-14.0, -3.0, 0.0, 12.0, -3.0001, 12.0001], dtype=np.float32)
    class_masks, unclassified = workflow.classify_delta_values(values)

    assert class_masks["small_non_sb_fish_crustaceans"].tolist() == [True, False, False, False, True, False]
    assert class_masks["large_non_sb_fish"].tolist() == [False, True, False, False, False, False]
    assert class_masks["swim_bladdered_fish"].tolist() == [False, False, True, True, False, False]
    assert unclassified.tolist() == [False, False, False, False, False, True]


def test_mask_source_selector_parses_both_mode() -> None:
    original = workflow.MASK_SOURCE_FREQUENCY_KHZ
    try:
        workflow.MASK_SOURCE_FREQUENCY_KHZ = "both"
        assert workflow.select_reference_mask_frequencies() == [38, 18]

        workflow.MASK_SOURCE_FREQUENCY_KHZ = "18,38"
        assert workflow.select_reference_mask_frequencies() == [18, 38]

        workflow.MASK_SOURCE_FREQUENCY_KHZ = "38"
        assert workflow.select_reference_mask_frequencies() == [38]
    finally:
        workflow.MASK_SOURCE_FREQUENCY_KHZ = original


def test_small_integration_pair_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        input_root = tmp_dir / "Layer Classification" / "DP00"
        file_18 = input_root / "TESTCAST_18kHz.sv.csv"
        file_38 = input_root / "TESTCAST_38kHz.sv.csv"

        sv38_rows = [
            [-70.0, -70.0, -70.0, -70.0],
            [-70.0, -70.0, -70.0, -70.0],
        ]
        sv18_rows = [
            [-69.0, -75.0, -70.0, -90.0],  # delta: 1, -5, 0, -20
            [-71.0, -80.0, -68.0, -72.0],  # delta: -1, -10, 2, -2
        ]

        _write_tiny_sv_csv(file_18, ping_times_ms=[500, 5500], sample_rows=sv18_rows)
        _write_tiny_sv_csv(file_38, ping_times_ms=[0, 5000], sample_rows=sv38_rows)

        reviewed_root = tmp_dir / "Figures"
        artifact_path = (
            reviewed_root
            / "DP00"
            / "TESTCAST_38kHz"
            / "TESTCAST_38kHz_reviewed"
            / "reviewed_contours.npz"
        )
        _write_reviewed_artifact(artifact_path, contour_shape=(4, 2))

        workflow.REVIEWED_CONTOURS_ROOT = reviewed_root
        workflow.RUN_TRACKING_IF_MASK_MISSING = False
        workflow.REUSE_REVIEWED_MASKS = True
        workflow.MAX_ALIGNMENT_DELTA_SECONDS = 2.5
        workflow.MASK_SOURCE_FREQUENCY_KHZ = 38

        pair = workflow.FrequencyPair(
            pair_key="TESTCAST",
            file_18=file_18,
            file_38=file_38,
            cruise_name="DP00",
        )
        output_root = tmp_dir / "Batch Results" / "layer_classification_v1"
        layer_rows, dataset_row, alignment_rows = workflow.classify_frequency_pair(pair, output_root=output_root)

        assert len(layer_rows) == 1
        row = layer_rows[0]
        assert int(row["swim_bladdered_fish_count"]) == 3
        assert int(row["large_non_sb_fish_count"]) == 2
        assert int(row["small_non_sb_fish_crustaceans_count"]) == 2
        assert int(row["unclassified_count"]) == 1
        assert int(dataset_row["total_finite_delta_count"]) == 8
        assert int(dataset_row["alignment_matched_ping_count"]) == 2
        assert len(alignment_rows) == 2


def main() -> None:
    test_alignment_nearest_with_tolerance()
    test_threshold_boundaries_half_open_bins()
    test_mask_source_selector_parses_both_mode()
    test_small_integration_pair_summary()
    print("layer classification workflow tests passed.")


if __name__ == "__main__":
    main()
